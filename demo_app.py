"""Local-only ForgeSight Gradio application with explicit detector selection."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import json
import os
import sys

import gradio as gr
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forgesight_ad.backbone_pipeline import build_resnet_extractor, build_transforms, compute_anomaly_maps
from forgesight_ad.checkpoints import load_checkpoint, validate_feature_statistics_checkpoint
from forgesight_ad.detectors import DetectorCache, DetectorCompatibilityError, build_detector, resolve_device
from forgesight_ad.networking import ensure_loopback_no_proxy


# The historical results live beside the canonical repository; current artifacts
# are deliberately separate so their provenance cannot be confused.
HISTORICAL_ARTIFACTS_ROOT = Path(
    os.environ.get("FORGESIGHT_ARTIFACTS_ROOT", PROJECT_ROOT.parent / "artifacts_resnet_compare")
)
DEVICE = resolve_device(os.environ.get("FORGESIGHT_DEVICE", "auto"))
DETECTOR_CACHE = DetectorCache(max_entries=2)


@dataclass(frozen=True)
class DetectorConfiguration:
    family: str
    label: str
    backbone: str
    category: str
    artifact: Path


FAMILY_LABELS = {
    "historical": "Historical feature statistics",
    "v2": "Corrected feature statistics v2",
    "patchcore": "PatchCore",
}


def historical_artifact_path(backbone: str, category: str) -> Path:
    return HISTORICAL_ARTIFACTS_ROOT / backbone / category / f"{backbone}_feature_statistics" / "model.pt"


def artifact_model_path(backbone: str, category: str) -> Path:
    """Backward-compatible name for the historical artifact resolver."""
    return historical_artifact_path(backbone, category)


@lru_cache(maxsize=1)
def detector_registry() -> tuple[DetectorConfiguration, ...]:
    """Return only artifact combinations which physically exist and are supported."""
    entries: list[DetectorConfiguration] = []
    if HISTORICAL_ARTIFACTS_ROOT.is_dir():
        for backbone_dir in sorted(HISTORICAL_ARTIFACTS_ROOT.iterdir()):
            if not backbone_dir.is_dir():
                continue
            for category_dir in sorted(backbone_dir.iterdir()):
                artifact = historical_artifact_path(backbone_dir.name, category_dir.name)
                if artifact.is_file():
                    entries.append(DetectorConfiguration(
                        "historical", FAMILY_LABELS["historical"], backbone_dir.name, category_dir.name, artifact
                    ))
    v2 = PROJECT_ROOT / "artifacts" / "v2_experiments" / "resnet50_bottle_v2_seed42" / "model.pt"
    if v2.is_file():
        entries.append(DetectorConfiguration("v2", FAMILY_LABELS["v2"], "resnet50", "bottle", v2))
    patchcore = PROJECT_ROOT / "artifacts" / "patchcore_experiments" / "resnet18_bottle_seed42" / "model.pt"
    if patchcore.is_file():
        entries.append(DetectorConfiguration("patchcore", FAMILY_LABELS["patchcore"], "resnet18", "bottle", patchcore))
    # Generic category artifacts appear only after their matching evaluation is
    # complete. This avoids advertising interrupted fitting runs in the UI.
    generic = PROJECT_ROOT / "artifacts" / "category_experiments"
    for method, family, backbone in (("v2", "v2", "resnet50"), ("patchcore", "patchcore", "resnet18")):
        for artifact in (generic / method).glob("*/model.pt") if (generic / method).is_dir() else ():
            metrics = artifact.parent / "metrics.json" if family == "v2" else artifact.parent.with_name(artifact.parent.name + "_evaluation") / "results.json"
            try:
                record = json.loads(metrics.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            category = record.get("category")
            if record.get("status") == "complete" and isinstance(category, str):
                entries.append(DetectorConfiguration(family, FAMILY_LABELS[family], backbone, category, artifact))
    return tuple(entries)


def supported_configurations(family: str | None = None) -> tuple[DetectorConfiguration, ...]:
    entries = detector_registry()
    return tuple(entry for entry in entries if family is None or entry.family == family)


def detector_families() -> list[str]:
    families = {entry.family for entry in supported_configurations()}
    return [family for family in FAMILY_LABELS if family in families]


def get_supported_configuration(family: str, backbone: str, category: str) -> DetectorConfiguration:
    for entry in supported_configurations(family):
        if entry.backbone == backbone and entry.category == category:
            return entry
    raise DetectorCompatibilityError(
        f"Unsupported selection: {FAMILY_LABELS.get(family, family)} / {backbone} / {category}."
    )


def available_backbones(family: str = "historical") -> list[str]:
    return sorted({entry.backbone for entry in supported_configurations(family)})


def available_categories(backbone: str, family: str = "historical") -> list[str]:
    return sorted({entry.category for entry in supported_configurations(family) if entry.backbone == backbone})


def update_detector_choices(family: str):
    backbones = available_backbones(family)
    backbone = backbones[0] if backbones else None
    categories = available_categories(backbone, family) if backbone else []
    return (
        gr.Dropdown(choices=backbones, value=backbone),
        gr.Dropdown(choices=categories, value=(categories[0] if categories else None)),
    )


def update_category_choices(backbone: str, family: str = "historical"):
    categories = available_categories(backbone, family)
    return gr.Dropdown(choices=categories, value=(categories[0] if categories else None))


def safe_build_inference_transforms(image_size: int):
    """Compatibility wrapper retained for the historical diagnostic runner."""
    try:
        return build_transforms(image_size, train=False, enable_augmentation=False, augmentation_level="none")
    except TypeError:
        return build_transforms(image_size)


def load_bundle(backbone: str, category: str):
    """Legacy read-only loader retained for existing historical diagnostics."""
    model_path = historical_artifact_path(backbone, category)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")
    bundle = load_checkpoint(model_path, DEVICE)
    validate_feature_statistics_checkpoint(bundle, selected_backbone=backbone)
    extractor, _ = build_resnet_extractor(backbone, DEVICE, allow_random_init=False)
    extractor.eval()
    from forgesight_ad.detectors import _config_from_payload
    runtime_cfg = _config_from_payload(bundle["config"], DEVICE)
    return bundle, extractor, bundle["statistics"], runtime_cfg


def normalize_map(heatmap: np.ndarray) -> np.ndarray:
    return heatmap / max(float(heatmap.max()), 1e-8)


def mask_outline(mask: np.ndarray) -> np.ndarray:
    binary = mask > 0
    outline = np.zeros_like(binary, dtype=bool)
    outline[:-1, :] |= binary[:-1, :] != binary[1:, :]
    outline[1:, :] |= binary[1:, :] != binary[:-1, :]
    outline[:, :-1] |= binary[:, :-1] != binary[:, 1:]
    outline[:, 1:] |= binary[:, 1:] != binary[:, :-1]
    return outline & binary


def build_heatmap_overlay(image_array: np.ndarray, normalized_heatmap: np.ndarray) -> np.ndarray:
    """Render the existing warm heatmap with fewer full-image temporaries.

    The channel arithmetic and uint8 conversion deliberately match
    ``_build_heatmap_overlay_reference`` below.  Keep that reference for
    regression comparison before changing this routine.
    """
    overlay = image_array.astype(np.float32, copy=True)
    heatmap = np.asarray(normalized_heatmap, dtype=np.float32)
    alpha = np.clip(heatmap ** 0.85, 0, 1) * 0.65
    inverse_alpha = 1.0 - alpha
    overlay[..., 0] = inverse_alpha * overlay[..., 0] + alpha * (255.0 * np.clip(heatmap * 1.8, 0, 1))
    overlay[..., 1] = inverse_alpha * overlay[..., 1] + alpha * (255.0 * np.clip(heatmap * 0.65, 0, 1))
    overlay[..., 2] = inverse_alpha * overlay[..., 2] + alpha * (255.0 * np.clip(heatmap * 0.1, 0, 1))
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _build_heatmap_overlay_reference(image_array: np.ndarray, normalized_heatmap: np.ndarray) -> np.ndarray:
    """Pre-optimization implementation retained solely for equivalence tests."""
    overlay = image_array.astype(np.float32).copy()
    hot = np.stack([
        np.clip(normalized_heatmap * 1.8, 0, 1), np.clip(normalized_heatmap * 0.65, 0, 1),
        np.clip(normalized_heatmap * 0.1, 0, 1),
    ], axis=-1)
    alpha = np.clip(normalized_heatmap ** 0.85, 0, 1)[..., None] * 0.65
    return np.clip((1 - alpha) * overlay + alpha * (255 * hot), 0, 255).astype(np.uint8)


def build_mask_overlay(image_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render the existing mask overlay without redundant source-array copying."""
    overlay = image_array.astype(np.float32).copy()
    fill = mask > 0
    overlay[fill] = 0.7 * overlay[fill] + 0.3 * np.array([255, 120, 70], dtype=np.float32)
    outline = mask_outline(mask)
    overlay[outline] = np.array([255, 245, 80], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _cached_adapter(configuration: DetectorConfiguration, device: str | torch.device = DEVICE):
    selected_device = resolve_device(device)
    key = (configuration.family, configuration.artifact.resolve(), configuration.backbone, configuration.category, str(selected_device))
    return DETECTOR_CACHE.get_or_load(
        key, configuration.artifact,
        lambda: build_detector(configuration.family, configuration.artifact, configuration.backbone, configuration.category, selected_device),
    )


def infer_detector(family: str, backbone: str, category: str, input_image: Image.Image, device: str | torch.device = DEVICE):
    """Four-output Gradio callback for all supported detector families."""
    if input_image is None:
        raise gr.Error("Upload an image first.")
    if not family or not backbone or not category:
        raise gr.Error("Select a detector, backbone, and category first.")
    try:
        configuration = get_supported_configuration(family, backbone, category)
        prediction = _cached_adapter(configuration, device).predict(input_image)
        heatmap = prediction.anomaly_map.squeeze(0).squeeze(0).numpy()
        mask = prediction.mask.squeeze(0).squeeze(0).numpy()
        image_size = (heatmap.shape[1], heatmap.shape[0])
        # RGB uploads are already in the required presentation mode.  Avoid an
        # otherwise redundant PIL conversion while retaining non-RGB behavior.
        source_image = input_image if input_image.mode == "RGB" else input_image.convert("RGB")
        resized_input = source_image.resize(image_size)
        image_array = np.asarray(resized_input)
        heatmap_overlay = build_heatmap_overlay(image_array, normalize_map(heatmap))
        mask_overlay = build_mask_overlay(image_array, mask)
        summary = {
            "detector_family": family, "methodology": prediction.methodology, "backbone": prediction.backbone,
            "category": prediction.category, "prediction": prediction.prediction, "image_score": round(prediction.score, 6),
            "image_threshold": round(prediction.threshold, 6),
            # PatchCore's image and mask threshold are intentionally the same operational value.
            "pixel_threshold": round(prediction.threshold, 6), "mask_area_ratio": round(float(mask.mean()), 6),
            "device": prediction.device, "inference_ms": round(prediction.inference_ms, 6),
        }
        return resized_input, heatmap_overlay, mask_overlay, json.dumps(summary, indent=2, allow_nan=False)
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Inference failed: {exc}") from exc


def infer(backbone: str, category: str, input_image: Image.Image):
    """Original historical callback signature retained for callers and diagnostics."""
    return infer_detector("historical", backbone, category, input_image)


initial_families = detector_families()
initial_family = initial_families[0] if initial_families else None
initial_backbones = available_backbones(initial_family) if initial_family else []
initial_backbone = initial_backbones[0] if initial_backbones else None
initial_categories = available_categories(initial_backbone, initial_family) if initial_backbone else []
initial_category = initial_categories[0] if initial_categories else None

def build_demo() -> gr.Blocks:
    """Construct the UI only when launching it, not when importing callbacks."""
    with gr.Blocks(title="ForgeSight AD Demo") as app:
        gr.Markdown("# ForgeSight AD Demo\nUpload an industrial image and inspect the detector-specific anomaly output.")
        with gr.Row():
            detector_family = gr.Dropdown(choices=initial_families, label="Detector family", value=initial_family)
            backbone = gr.Dropdown(choices=initial_backbones, label="Backbone", value=initial_backbone)
            category = gr.Dropdown(choices=initial_categories, label="Category", value=initial_category)
            image_input = gr.Image(type="pil", label="Upload image")
        detector_family.change(fn=update_detector_choices, inputs=detector_family, outputs=[backbone, category])
        backbone.change(fn=update_category_choices, inputs=[backbone, detector_family], outputs=category)
        run_button = gr.Button("Detect anomalies")
        with gr.Row():
            input_output = gr.Image(label="Input")
            heatmap_output = gr.Image(label="Heatmap Overlay")
            mask_output = gr.Image(label="Anomaly Highlight")
        summary_output = gr.Code(label="Prediction summary", language="json")
        run_button.click(
            fn=infer_detector, inputs=[detector_family, backbone, category, image_input],
            outputs=[input_output, heatmap_output, mask_output, summary_output],
        )
    return app


def launch_demo() -> None:
    """Launch a localhost-only UI with visible callback errors."""
    ensure_loopback_no_proxy()
    entries = detector_registry()
    if not entries:
        raise RuntimeError(
            "No supported detector artifacts were found. Mount or configure the artifacts directory before starting ForgeSight."
        )
    # Containers listen on all container interfaces so Docker can publish the
    # port to the host loopback binding. The default remains loopback for a
    # direct local-Python launch; `share=False` is never relaxed.
    host = os.environ.get("FORGESIGHT_SERVER_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "0.0.0.0", "::1"}:
        raise RuntimeError("FORGESIGHT_SERVER_HOST must be a local loopback address or 0.0.0.0 for a Docker port mapping")
    launch_options = {
        "server_port": int(os.environ.get("FORGESIGHT_SERVER_PORT", "7860")),
        "share": False,
        "show_error": True,
    }
    if host == "127.0.0.1":
        # Preserve the direct-Python local-only contract. Docker overrides this
        # only inside its isolated network namespace and publishes host loopback.
        build_demo().launch(server_name="127.0.0.1", **launch_options)
    else:
        build_demo().launch(server_name=host, **launch_options)


if __name__ == "__main__":
    launch_demo()
