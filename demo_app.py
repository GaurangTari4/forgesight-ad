from __future__ import annotations

from pathlib import Path
import json
import sys

import gradio as gr
import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resnet_compare.resnet_backbone_comparison_pipeline import (  # noqa: E402
    ResNetBackboneConfig,
    build_resnet_extractor,
    build_transforms,
    compute_anomaly_maps,
    refine_predicted_mask,
)


ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts_resnet_compare"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    overlay = image_array.astype(np.float32).copy()
    hot = np.stack(
        [
            np.clip(normalized_heatmap * 1.8, 0, 1),
            np.clip(normalized_heatmap * 0.65, 0, 1),
            np.clip(normalized_heatmap * 0.1, 0, 1),
        ],
        axis=-1,
    )
    alpha = np.clip(normalized_heatmap ** 0.85, 0, 1)[..., None] * 0.65
    return np.clip((1 - alpha) * overlay + alpha * (255 * hot), 0, 255).astype(np.uint8)


def build_mask_overlay(image_array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image_array.astype(np.float32).copy()
    fill = mask > 0
    overlay[fill] = 0.7 * overlay[fill] + 0.3 * np.array([255, 120, 70], dtype=np.float32)
    outline = mask_outline(mask)
    overlay[outline] = np.array([255, 245, 80], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def artifact_model_path(backbone: str, category: str) -> Path:
    pipeline_name = f"{backbone}_feature_statistics"
    return ARTIFACTS_ROOT / backbone / category / pipeline_name / "model.pt"


def safe_torch_load(path: Path, device: torch.device):
    # PyTorch >= 2.6 changed default torch.load behavior to weights_only=True.
    # Our artifact is a full dictionary payload, so we force weights_only=False.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # Older versions do not expose weights_only.
        return torch.load(path, map_location=device)


def safe_build_inference_transforms(image_size: int):
    # Compatibility wrapper: different pipeline versions expose different
    # build_transforms signatures.
    try:
        return build_transforms(image_size, train=False, enable_augmentation=False, augmentation_level="none")
    except TypeError:
        return build_transforms(image_size)


def available_backbones() -> list[str]:
    if not ARTIFACTS_ROOT.exists():
        return []
    backbones: list[str] = []
    for backbone_dir in ARTIFACTS_ROOT.iterdir():
        if not backbone_dir.is_dir():
            continue
        has_model = any(
            (category_dir / f"{backbone_dir.name}_feature_statistics" / "model.pt").exists()
            for category_dir in backbone_dir.iterdir()
            if category_dir.is_dir()
        )
        if has_model:
            backbones.append(backbone_dir.name)
    return sorted(backbones)


def available_categories(backbone: str) -> list[str]:
    if not backbone:
        return []
    backbone_root = ARTIFACTS_ROOT / backbone
    if not backbone_root.exists():
        return []
    categories = []
    for category_dir in backbone_root.iterdir():
        if category_dir.is_dir() and artifact_model_path(backbone, category_dir.name).exists():
            categories.append(category_dir.name)
    return sorted(categories)


def update_category_choices(backbone: str):
    categories = available_categories(backbone)
    return gr.Dropdown(choices=categories, value=(categories[0] if categories else None))


def build_runtime_config(cfg_dict: dict, device: torch.device) -> ResNetBackboneConfig:
    return ResNetBackboneConfig(
        project_root=Path(cfg_dict["project_root"]),
        dataset_root=Path(cfg_dict["dataset_root"]),
        artifacts_root=Path(cfg_dict["artifacts_root"]),
        image_size=int(cfg_dict["image_size"]),
        batch_size=int(cfg_dict["batch_size"]),
        num_epochs=int(cfg_dict["num_epochs"]),
        learning_rate=float(cfg_dict["learning_rate"]),
        latent_channels=int(cfg_dict["latent_channels"]),
        image_threshold_percentile=float(cfg_dict["image_threshold_percentile"]),
        pixel_threshold_percentile=float(cfg_dict["pixel_threshold_percentile"]),
        top_k_ratio=float(cfg_dict["top_k_ratio"]),
        heatmap_blur_kernel=int(cfg_dict["heatmap_blur_kernel"]),
        heatmap_blur_sigma=float(cfg_dict["heatmap_blur_sigma"]),
        mask_smoothing_kernel=int(cfg_dict["mask_smoothing_kernel"]),
        mask_min_area_ratio=float(cfg_dict["mask_min_area_ratio"]),
        max_mask_components=int(cfg_dict.get("max_mask_components", 2)),
        mse_weight=float(cfg_dict["mse_weight"]),
        ssim_weight=float(cfg_dict["ssim_weight"]),
        feature_map_weight=float(cfg_dict["feature_map_weight"]),
        num_workers=int(cfg_dict["num_workers"]),
        seed=int(cfg_dict["seed"]),
        device_name=str(device),
        backbone_name=str(cfg_dict.get("backbone_name", "resnet50")),
        feature_layers=tuple(cfg_dict.get("feature_layers", ["layer1", "layer2", "layer3", "layer4"])),
        feature_blend_weights=tuple(cfg_dict.get("feature_blend_weights", [0.4, 0.3, 0.2, 0.1])),
        score_blend_weights=tuple(cfg_dict.get("score_blend_weights", [0.4, 0.3, 0.2, 0.1])),
        embedding_eps=float(cfg_dict.get("embedding_eps", 1e-6)),
        suppress_background_anomalies=bool(cfg_dict.get("suppress_background_anomalies", True)),
        background_tolerance=float(cfg_dict.get("background_tolerance", 0.08)),
        background_min_coverage=float(cfg_dict.get("background_min_coverage", 0.05)),
    )


def load_bundle(backbone: str, category: str):
    model_path = artifact_model_path(backbone, category)
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")

    bundle = safe_torch_load(model_path, DEVICE)
    extractor, _ = build_resnet_extractor(backbone, DEVICE)
    extractor.eval()
    statistics = bundle["statistics"]
    runtime_cfg = build_runtime_config(bundle["config"], DEVICE)
    return bundle, extractor, statistics, runtime_cfg


@torch.no_grad()
def infer(backbone: str, category: str, input_image: Image.Image):
    if input_image is None:
        raise gr.Error("Upload an image first.")
    if not backbone or not category:
        raise gr.Error("Select a backbone and category first.")

    try:
        bundle, extractor, statistics, runtime_cfg = load_bundle(backbone, category)
        transform, _ = safe_build_inference_transforms(int(runtime_cfg.image_size))
        image_tensor = transform(input_image.convert("RGB")).unsqueeze(0).to(DEVICE)

        anomaly_map, image_score = compute_anomaly_maps(extractor, image_tensor, statistics, runtime_cfg)
        predicted_mask = refine_predicted_mask(
            anomaly_map,
            float(bundle["pixel_threshold"]),
            runtime_cfg.mask_smoothing_kernel,
            runtime_cfg.mask_min_area_ratio,
            runtime_cfg.max_mask_components,
        )

        heatmap = anomaly_map.squeeze(0).squeeze(0).cpu().numpy()
        mask = predicted_mask.squeeze(0).squeeze(0).cpu().numpy()
        normalized_heatmap = normalize_map(heatmap)
        image_size = (heatmap.shape[1], heatmap.shape[0])
        resized_input = input_image.convert("RGB").resize(image_size)
        image_array = np.asarray(resized_input)

        heatmap_overlay = build_heatmap_overlay(image_array, normalized_heatmap)
        mask_overlay = build_mask_overlay(image_array, mask)

        summary = {
            "backbone": backbone,
            "category": category,
            "prediction": "damaged" if float(image_score.item()) > float(bundle["image_threshold"]) else "good",
            "image_score": round(float(image_score.item()), 6),
            "image_threshold": round(float(bundle["image_threshold"]), 6),
            "pixel_threshold": round(float(bundle["pixel_threshold"]), 6),
            "mask_area_ratio": round(float(mask.mean()), 6),
            "device": str(DEVICE),
        }
        return resized_input, heatmap_overlay, mask_overlay, json.dumps(summary, indent=2)
    except Exception as exc:
        raise gr.Error(f"Inference failed: {exc}") from exc


initial_backbones = available_backbones()
initial_backbone = initial_backbones[0] if initial_backbones else None
initial_categories = available_categories(initial_backbone) if initial_backbone else []
initial_category = initial_categories[0] if initial_categories else None

with gr.Blocks(title="ForgeSight AD Demo") as demo:
    gr.Markdown(
        """
        # ForgeSight AD Demo
        Upload an industrial image, choose backbone/category, and inspect anomaly heatmap plus highlighted mask.
        """
    )
    with gr.Row():
        backbone = gr.Dropdown(choices=initial_backbones, label="Backbone", value=initial_backbone)
        category = gr.Dropdown(choices=initial_categories, label="Category", value=initial_category)
        image_input = gr.Image(type="pil", label="Upload image")

    backbone.change(fn=update_category_choices, inputs=backbone, outputs=category)
    run_button = gr.Button("Detect anomalies")

    with gr.Row():
        input_output = gr.Image(label="Input")
        heatmap_output = gr.Image(label="Heatmap Overlay")
        mask_output = gr.Image(label="Anomaly Highlight")
    summary_output = gr.Code(label="Prediction summary", language="json")

    run_button.click(
        fn=infer,
        inputs=[backbone, category, image_input],
        outputs=[input_output, heatmap_output, mask_output, summary_output],
    )


if __name__ == "__main__":
    demo.launch()

