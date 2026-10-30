"""ONNX-accelerated feature extraction with unchanged PatchCore scoring.

Only the frozen ResNet extractor runs in ONNX Runtime. Patch construction,
nearest-neighbour scoring, thresholding, and rendering intentionally remain in
the validated Python/PyTorch path. This is therefore an ONNX-backed prototype,
not a fully ONNX-native detector.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from .autoencoder_pipeline import imagenet_normalize
from .backbone_pipeline import build_transforms
from .detectors import Prediction
from .patchcore_pipeline import (
    METHODOLOGY_ID,
    PatchCoreArtifactError,
    load_artifact,
    nearest_distances,
    patch_embeddings,
)


class OnnxPatchCoreError(ValueError):
    """An ONNX graph, provider, or PatchCore artifact is incompatible."""


def available_execution_providers() -> tuple[str, ...]:
    """Return installed ONNX Runtime providers without making one mandatory at import."""
    try:
        import onnxruntime as ort
    except ImportError:
        return ()
    return tuple(ort.get_available_providers())


class OnnxPatchCoreAdapter:
    """PatchCore adapter whose backbone features come from ONNX Runtime CPU/CUDA."""

    def __init__(
        self,
        graph: Path,
        artifact: Path,
        calibration: Path,
        category: str = "bottle",
        provider: str = "CPUExecutionProvider",
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise OnnxPatchCoreError("onnxruntime is required for ONNX-backed inference") from exc
        self.graph = Path(graph)
        if not self.graph.is_file():
            raise OnnxPatchCoreError(f"ONNX graph does not exist: {self.graph}")
        available = tuple(ort.get_available_providers())
        if provider not in available:
            raise OnnxPatchCoreError(f"Requested ONNX Runtime provider is unavailable: {provider}; available: {available}")
        try:
            self.payload = load_artifact(Path(artifact), torch.device("cpu"), "resnet18")
        except (PatchCoreArtifactError, OSError) as exc:
            raise OnnxPatchCoreError("PatchCore artifact is incompatible") from exc
        if self.payload.get("methodology_id") != METHODOLOGY_ID:
            raise OnnxPatchCoreError("Artifact is not PatchCore v1")
        if self.payload.get("category", "bottle") != category:
            raise OnnxPatchCoreError("PatchCore artifact category does not match selection")
        try:
            sidecar = json.loads(Path(calibration).read_text(encoding="utf-8"))
            self.threshold = float(sidecar["threshold"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise OnnxPatchCoreError("PatchCore calibration sidecar is invalid") from exc
        if sidecar.get("methodology_id") != METHODOLOGY_ID or not np.isfinite(self.threshold):
            raise OnnxPatchCoreError("PatchCore calibration methodology or threshold is invalid")
        self.category = category
        self.provider = provider
        self.layers = tuple(self.payload["feature_layers"])
        if self.layers != ("layer2", "layer3"):
            raise OnnxPatchCoreError("The prototype graph is only compatible with layer2/layer3 PatchCore artifacts")
        self.session = ort.InferenceSession(str(self.graph), providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        output_names = tuple(output.name for output in self.session.get_outputs())
        expected = ("layer1", "layer2", "layer3", "layer4")
        if output_names != expected:
            raise OnnxPatchCoreError(f"Unexpected ONNX output order: {output_names}")
        image_shape = self.session.get_inputs()[0].shape
        expected_size = int(self.payload["image_size"])
        if image_shape != [1, 3, expected_size, expected_size]:
            raise OnnxPatchCoreError(f"Graph input shape {image_shape} does not match artifact image size {expected_size}")
        self.image_size = expected_size
        self.bank = self.payload["memory_bank"].cpu().contiguous()
        self.last_timings_ms: dict[str, float] = {}

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("ONNX PatchCore input must be a PIL.Image.Image")
        transform, _ = build_transforms(self.image_size)
        return transform(image.convert("RGB")).unsqueeze(0)

    def extract_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        normalized = imagenet_normalize(images).numpy().astype(np.float32, copy=False)
        values = self.session.run(None, {self.input_name: normalized})
        return {name: torch.from_numpy(value) for name, value in zip(("layer1", "layer2", "layer3", "layer4"), values)}

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> Prediction:
        total_start = time.perf_counter()
        start = total_start
        images = self._preprocess(image)
        preprocess_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        features = self.extract_features(images)
        extraction_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        patches, shape = patch_embeddings(features, self.layers)
        distances = nearest_distances(patches, self.bank)
        anomaly = F.interpolate(
            distances.reshape(images.shape[0], 1, *shape), size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        score = anomaly.flatten(start_dim=1).max(dim=1).values
        mask = (anomaly > self.threshold).to(dtype=anomaly.dtype)
        scoring_ms = (time.perf_counter() - start) * 1000
        if not torch.isfinite(anomaly).all() or not torch.isfinite(score).all():
            raise OnnxPatchCoreError("ONNX-backed PatchCore generated non-finite output")
        self.last_timings_ms = {
            "preprocess": preprocess_ms,
            "onnx_feature_extraction": extraction_ms,
            "python_patchcore_scoring": scoring_ms,
            "total": (time.perf_counter() - total_start) * 1000,
        }
        scalar = float(score.item())
        return Prediction(
            methodology=METHODOLOGY_ID,
            backbone="resnet18",
            category=self.category,
            device=f"onnxruntime:{self.provider}",
            score=scalar,
            threshold=self.threshold,
            prediction="FAIL" if scalar > self.threshold else "PASS",
            anomaly_map=anomaly.cpu(),
            mask=mask.cpu(),
            inference_ms=self.last_timings_ms["total"],
        )
