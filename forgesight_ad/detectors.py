"""Gradio-independent adapters for the supported ForgeSight detector families.

Adapters call the already validated family-specific inference functions. They
are a contract boundary, not a replacement scoring implementation.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
import json
from pathlib import Path
import threading
import time
from typing import Callable, Literal

from PIL import Image
import torch

from .autoencoder_pipeline import refine_predicted_mask
from .backbone_pipeline import ResNetBackboneConfig, build_resnet_extractor, build_transforms, compute_anomaly_maps
from .checkpoints import CheckpointCompatibilityError, load_checkpoint, validate_feature_statistics_checkpoint
from .patchcore_pipeline import (
    METHODOLOGY_ID as PATCHCORE_METHODOLOGY_ID,
    PatchCoreArtifactError,
    anomaly_map as patchcore_anomaly_map,
    load_artifact as load_patchcore_artifact,
)
from .v2_inference import METHODOLOGY_ID as V2_METHODOLOGY_ID, V2ArtifactError, infer_v2, load_v2_artifact


class DetectorCompatibilityError(ValueError):
    """A selected artifact cannot be used by the requested detector."""


DetectorFamily = Literal["historical", "v2", "patchcore"]


@dataclass
class Prediction:
    methodology: str
    backbone: str
    category: str
    device: str
    score: float
    threshold: float
    prediction: str
    anomaly_map: torch.Tensor
    mask: torch.Tensor
    inference_ms: float


def resolve_device(selection: str | torch.device = "auto") -> torch.device:
    """Resolve only the explicit project device options."""
    if isinstance(selection, torch.device):
        selection = str(selection)
    if selection == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if selection == "cuda":
        selection = "cuda:0"
    if selection == "cuda:0" and not torch.cuda.is_available():
        raise DetectorCompatibilityError("CUDA was selected but is not available")
    if selection not in {"cpu", "cuda:0"}:
        raise DetectorCompatibilityError("Device must be one of: auto, cpu, cuda, cuda:0")
    return torch.device(selection)


def _config_from_payload(raw: dict, device: torch.device) -> ResNetBackboneConfig:
    allowed = {item.name for item in fields(ResNetBackboneConfig)}
    values = {key: value for key, value in raw.items() if key in allowed}
    for key in ("project_root", "dataset_root", "artifacts_root"):
        if key in values:
            values[key] = Path(values[key])
    for key in ("feature_layers", "feature_blend_weights", "score_blend_weights"):
        if key in values:
            values[key] = tuple(values[key])
    values["device_name"] = str(device)
    try:
        return ResNetBackboneConfig(**values)
    except (TypeError, ValueError) as exc:
        raise DetectorCompatibilityError("Artifact has an invalid ResNet preprocessing configuration") from exc


def _image_tensor(image: Image.Image, image_size: int, device: torch.device) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        raise TypeError("Detector input must be a PIL.Image.Image")
    if image.width <= 0 or image.height <= 0:
        raise ValueError("Detector input image has invalid dimensions")
    transform, _ = build_transforms(image_size)
    return transform(image.convert("RGB")).unsqueeze(0).to(device, non_blocking=device.type == "cuda")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class DetectorAdapter:
    """Common adapter protocol; subclasses retain their native score semantics."""

    family: DetectorFamily

    def __init__(self, artifact: Path, backbone: str, category: str, device: str | torch.device = "auto"):
        self.artifact = Path(artifact)
        self.backbone = backbone
        self.category = category
        self.device = resolve_device(device)

    def predict(self, image: Image.Image) -> Prediction:  # pragma: no cover - interface only
        raise NotImplementedError

    def _validated_prediction(
        self, methodology: str, score: torch.Tensor, threshold: float, anomaly: torch.Tensor, mask: torch.Tensor, elapsed_ms: float
    ) -> Prediction:
        if anomaly.shape != mask.shape or anomaly.ndim != 4 or anomaly.shape[1] != 1:
            raise DetectorCompatibilityError("Detector returned invalid anomaly-map or mask dimensions")
        if not torch.isfinite(anomaly).all() or not torch.isfinite(score).all():
            raise DetectorCompatibilityError("Detector returned non-finite anomaly values")
        scalar = float(score.reshape(-1)[0].detach().cpu())
        if not torch.isfinite(torch.tensor(threshold)):
            raise DetectorCompatibilityError("Detector threshold is not finite")
        return Prediction(
            methodology=methodology, backbone=self.backbone, category=self.category, device=str(self.device),
            score=scalar, threshold=float(threshold), prediction="FAIL" if scalar > threshold else "PASS",
            anomaly_map=anomaly.detach().cpu(), mask=mask.detach().cpu(), inference_ms=elapsed_ms,
        )


class HistoricalFeatureStatisticsAdapter(DetectorAdapter):
    family: DetectorFamily = "historical"

    def __init__(self, artifact: Path, backbone: str, category: str, device: str | torch.device = "auto"):
        super().__init__(artifact, backbone, category, device)
        try:
            self.payload = load_checkpoint(self.artifact, self.device)
            validate_feature_statistics_checkpoint(self.payload, backbone)
        except (CheckpointCompatibilityError, FileNotFoundError) as exc:
            raise DetectorCompatibilityError(str(exc)) from exc
        if self.payload.get("category") != category:
            raise DetectorCompatibilityError("Selected category does not match historical checkpoint")
        self.config = _config_from_payload(self.payload["config"], self.device)
        if self.config.backbone_name != backbone:
            raise DetectorCompatibilityError("Historical configuration backbone does not match selection")
        self.extractor, _ = build_resnet_extractor(backbone, self.device, allow_random_init=False)
        self.extractor.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        images = _image_tensor(image, self.config.image_size, self.device)
        _synchronize(self.device)
        start = time.perf_counter()
        anomaly, scores = compute_anomaly_maps(self.extractor, images, self.payload["statistics"], self.config)
        mask = refine_predicted_mask(
            anomaly, float(self.payload["pixel_threshold"]), self.config.mask_smoothing_kernel,
            self.config.mask_min_area_ratio, self.config.max_mask_components,
        )
        _synchronize(self.device)
        return self._validated_prediction(
            str(self.payload.get("pipeline", "feature_statistics_legacy")), scores, float(self.payload["image_threshold"]),
            anomaly, mask, (time.perf_counter() - start) * 1000,
        )


class V2FeatureStatisticsAdapter(DetectorAdapter):
    family: DetectorFamily = "v2"

    def __init__(self, artifact: Path, backbone: str, category: str, device: str | torch.device = "auto"):
        super().__init__(artifact, backbone, category, device)
        try:
            self.payload, self.config, self.extractor = load_v2_artifact(self.artifact, self.device, backbone)
        except (V2ArtifactError, FileNotFoundError) as exc:
            raise DetectorCompatibilityError(str(exc)) from exc
        provenance = self.payload.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("category") != category:
            raise DetectorCompatibilityError("Selected category does not match v2 artifact provenance")
        if self.payload.get("methodology_id") != V2_METHODOLOGY_ID:
            raise DetectorCompatibilityError("Artifact is not a v2 feature-statistics artifact")

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        images = _image_tensor(image, self.config.image_size, self.device)
        _synchronize(self.device)
        start = time.perf_counter()
        anomaly, scores, mask = infer_v2(self.payload, self.config, self.extractor, images)
        _synchronize(self.device)
        return self._validated_prediction(
            V2_METHODOLOGY_ID, scores, float(self.payload["image_threshold"]), anomaly, mask,
            (time.perf_counter() - start) * 1000,
        )


class PatchCoreAdapter(DetectorAdapter):
    family: DetectorFamily = "patchcore"

    def __init__(self, artifact: Path, backbone: str, category: str, device: str | torch.device = "auto", calibration: Path | None = None):
        super().__init__(artifact, backbone, category, device)
        try:
            self.payload = load_patchcore_artifact(self.artifact, self.device, backbone)
        except (PatchCoreArtifactError, FileNotFoundError) as exc:
            raise DetectorCompatibilityError(str(exc)) from exc
        # Immutable bottle-v1 artifacts predate the category field; newer
        # category artifacts carry explicit provenance.
        if self.payload.get("category", "bottle") != category:
            raise DetectorCompatibilityError("Selected category does not match PatchCore artifact provenance")
        if self.payload.get("methodology_id") != PATCHCORE_METHODOLOGY_ID:
            raise DetectorCompatibilityError("Artifact is not a PatchCore v1 artifact")
        self.threshold = self._load_threshold(calibration)
        self.extractor, _ = build_resnet_extractor(backbone, self.device, allow_random_init=False)
        self.extractor.eval()

    def _load_threshold(self, calibration: Path | None) -> float:
        if calibration is None:
            calibration = self.artifact.parent.with_name(self.artifact.parent.name + "_evaluation") / "results.json"
        try:
            data = json.loads(Path(calibration).read_text(encoding="utf-8"))
            value = float(data["threshold"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise DetectorCompatibilityError("PatchCore requires a valid normal-only calibration results JSON") from exc
        if data.get("methodology_id") != PATCHCORE_METHODOLOGY_ID or not torch.isfinite(torch.tensor(value)):
            raise DetectorCompatibilityError("PatchCore calibration sidecar is incompatible or has a non-finite threshold")
        return value

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        images = _image_tensor(image, int(self.payload["image_size"]), self.device)
        _synchronize(self.device)
        start = time.perf_counter()
        anomaly = patchcore_anomaly_map(self.extractor, images, self.payload["memory_bank"], tuple(self.payload["feature_layers"]))
        scores = anomaly.flatten(start_dim=1).max(dim=1).values
        mask = (anomaly > self.threshold).to(dtype=anomaly.dtype)
        _synchronize(self.device)
        return self._validated_prediction(
            PATCHCORE_METHODOLOGY_ID, scores, self.threshold, anomaly, mask, (time.perf_counter() - start) * 1000,
        )


def build_detector(
    family: DetectorFamily, artifact: Path, backbone: str, category: str, device: str | torch.device = "auto", **kwargs,
) -> DetectorAdapter:
    """Build exactly the requested family; cross-family fallbacks are forbidden."""
    adapters: dict[str, Callable[..., DetectorAdapter]] = {
        "historical": HistoricalFeatureStatisticsAdapter,
        "v2": V2FeatureStatisticsAdapter,
        "patchcore": PatchCoreAdapter,
    }
    try:
        return adapters[family](artifact, backbone, category, device, **kwargs)
    except KeyError as exc:
        raise DetectorCompatibilityError(f"Unsupported detector family: {family}") from exc


class DetectorCache:
    """Small LRU cache with cheap artifact-identity invalidation."""
    def __init__(self, max_entries: int = 2):
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._items: OrderedDict[tuple, tuple[tuple[int, int], object]] = OrderedDict()
        self._lock = threading.Lock()

    def get_or_load(self, key: tuple, artifact: Path, loader: Callable[[], object]):
        artifact = Path(artifact)
        stamp = (artifact.stat().st_mtime_ns, artifact.stat().st_size)
        with self._lock:
            value = self._items.pop(key, None)
            if value is not None and value[0] == stamp:
                self._items[key] = value
                return value[1]
            loaded = loader()  # Lock prevents duplicate concurrent GPU initialization.
            self._items[key] = (stamp, loaded)
            while len(self._items) > self.max_entries:
                _, (_, evicted) = self._items.popitem(last=False)
                del evicted
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return loaded
