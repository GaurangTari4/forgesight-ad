from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from functools import partial
import os
from pathlib import Path
import csv
import json
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from .autoencoder_pipeline import (
    ExperimentConfig,
    MVTecTestDataset,
    MVTecTrainDataset,
    gaussian_blur,
    imagenet_normalize,
    refine_predicted_mask,
    resize_with_padding,
    set_seed,
    topk_mean_scores,
)
from .experiment_registry import evaluation_context, rebuild_leaderboard, write_experiment_record


SUPPORTED_BACKBONES = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")

BACKBONE_SPECS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "resnet101": (models.resnet101, models.ResNet101_Weights.DEFAULT),
    "resnet152": (models.resnet152, models.ResNet152_Weights.DEFAULT),
}


@dataclass
class ResNetBackboneConfig(ExperimentConfig):
    backbone_name: str = "resnet50"
    feature_layers: tuple[str, ...] = ("layer1", "layer2", "layer3", "layer4")
    feature_blend_weights: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)
    score_blend_weights: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)
    embedding_eps: float = 1e-6
    suppress_background_anomalies: bool = True
    background_tolerance: float = 0.08
    background_min_coverage: float = 0.05
    allow_random_init: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["dataset_root"] = str(self.dataset_root)
        payload["artifacts_root"] = str(self.artifacts_root)
        payload["device"] = str(self.device)
        payload["feature_layers"] = list(self.feature_layers)
        payload["feature_blend_weights"] = list(self.feature_blend_weights)
        payload["score_blend_weights"] = list(self.score_blend_weights)
        return payload


class ResNetMultiScaleExtractor(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.stem(x)
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return {
            "layer1": layer1,
            "layer2": layer2,
            "layer3": layer3,
            "layer4": layer4,
        }


def build_resnet_extractor(backbone_name: str, device: torch.device, allow_random_init: bool = False):
    if backbone_name not in BACKBONE_SPECS:
        raise ValueError(f"Unsupported backbone '{backbone_name}'. Supported: {SUPPORTED_BACKBONES}")

    constructor, default_weights = BACKBONE_SPECS[backbone_name]
    model_name = backbone_name.upper()
    # Docker sets this guard after a deliberate, separate cache-preparation
    # step. TorchVision otherwise tries to download a missing DEFAULT weight at
    # request time, which is unacceptable for a reproducible inference service.
    if os.environ.get("FORGESIGHT_OFFLINE_WEIGHTS", "").strip().lower() in {"1", "true", "yes"}:
        filename = Path(default_weights.url).name
        cache_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not cache_path.is_file():
            raise RuntimeError(
                f"Required pretrained {model_name} weights are missing from the read-only Torch cache: {cache_path}. "
                "Prepopulate the cache before deployment; ForgeSight will not download weights at inference time."
            )
    try:
        backbone = constructor(weights=default_weights)
        source = "pretrained"
    except Exception as exc:
        if not allow_random_init:
            raise RuntimeError(
                f"Could not load pretrained {model_name} weights. Refusing to silently use random features. "
                "Install/cache the torchvision weights or pass allow_random_init=True explicitly."
            ) from exc
        backbone = constructor(weights=None)
        source = "random_init_explicit"

    extractor = ResNetMultiScaleExtractor(backbone).to(device)
    extractor.eval()
    for parameter in extractor.parameters():
        parameter.requires_grad = False
    return extractor, source


def build_transforms(image_size: int):
    bilinear_resize = transforms.Lambda(
        partial(resize_with_padding, image_size=image_size, interpolation=InterpolationMode.BILINEAR)
    )
    nearest_resize = transforms.Lambda(
        partial(resize_with_padding, image_size=image_size, interpolation=InterpolationMode.NEAREST)
    )
    image_transform = transforms.Compose([bilinear_resize, transforms.ToTensor()])
    mask_transform = transforms.Compose([nearest_resize, transforms.ToTensor()])
    return image_transform, mask_transform


def create_dataloaders(config: ResNetBackboneConfig, category: str):
    image_transform, mask_transform = build_transforms(config.image_size)
    train_dataset = MVTecTrainDataset(config.dataset_root, category, transform=image_transform)
    test_dataset = MVTecTestDataset(
        config.dataset_root,
        category,
        image_transform=image_transform,
        mask_transform=mask_transform,
        image_size=config.image_size,
    )
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.device.type == "cuda",
    }
    if config.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_dataset, test_dataset, train_loader, test_loader


def _normalize_weights(weights: tuple[float, ...], expected_length: int) -> list[float]:
    if len(weights) != expected_length:
        raise ValueError(f"Expected {expected_length} blend weights, got {len(weights)}")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Blend weights must sum to a positive value")
    return [float(weight / total) for weight in weights]


def estimate_foreground_mask(images: torch.Tensor, tolerance: float, min_coverage: float) -> torch.Tensor:
    gray = images.mean(dim=1, keepdim=True)
    _, _, height, width = gray.shape
    border_width = max(6, min(height, width) // 24)

    border_pixels = torch.cat(
        [
            gray[:, :, :border_width, :].flatten(start_dim=2),
            gray[:, :, -border_width:, :].flatten(start_dim=2),
            gray[:, :, :, :border_width].flatten(start_dim=2),
            gray[:, :, :, -border_width:].flatten(start_dim=2),
        ],
        dim=2,
    )
    background_level = border_pixels.median(dim=2).values.view(-1, 1, 1, 1)
    difference = torch.abs(gray - background_level)

    raw_foreground = (difference > tolerance).float()
    expanded = F.max_pool2d(raw_foreground, kernel_size=7, stride=1, padding=3)
    softened = F.avg_pool2d(expanded, kernel_size=11, stride=1, padding=5)
    foreground_mask = (softened > 0.2).float()

    coverage = foreground_mask.flatten(start_dim=1).mean(dim=1)
    reliable = (coverage > min_coverage).float().view(-1, 1, 1, 1)
    return reliable * foreground_mask + (1.0 - reliable)


@torch.no_grad()
def collect_feature_statistics(extractor, dataloader, config: ResNetBackboneConfig):
    normalized_feature_weights = _normalize_weights(config.feature_blend_weights, len(config.feature_layers))
    sums = {}
    sq_sums = {}
    count = 0
    target_size = None

    start_time = time.time()
    for images, _ in dataloader:
        images = images.to(config.device)
        features = extractor(imagenet_normalize(images))

        selected_maps = [features[layer] for layer in config.feature_layers]
        if target_size is None:
            target_size = selected_maps[0].shape[-2:]

        resized_maps = []
        for feature_map in selected_maps:
            if feature_map.shape[-2:] != target_size:
                feature_map = F.interpolate(feature_map, size=target_size, mode="bilinear", align_corners=False)
            resized_maps.append(feature_map)

        for layer_name, weight, feature_map in zip(config.feature_layers, normalized_feature_weights, resized_maps):
            weighted_map = feature_map * weight
            if layer_name not in sums:
                sums[layer_name] = weighted_map.sum(dim=0)
                sq_sums[layer_name] = (weighted_map ** 2).sum(dim=0)
            else:
                sums[layer_name] += weighted_map.sum(dim=0)
                sq_sums[layer_name] += (weighted_map ** 2).sum(dim=0)
        count += images.size(0)

    if count == 0:
        raise ValueError("No training samples were available to fit feature statistics")

    statistics = {"target_size": target_size, "layers": {}}
    for layer_name in config.feature_layers:
        mean = sums[layer_name] / count
        variance = (sq_sums[layer_name] / count) - mean.square()
        statistics["layers"][layer_name] = {
            "mean": mean.cpu(),
            "std": torch.sqrt(torch.clamp(variance, min=config.embedding_eps)).cpu(),
        }

    summary = {
        "num_samples": count,
        "target_size": tuple(int(value) for value in target_size),
        "time_seconds": time.time() - start_time,
    }
    return statistics, summary


@torch.no_grad()
def compute_anomaly_maps(extractor, images: torch.Tensor, statistics: dict, config: ResNetBackboneConfig):
    normalized_score_weights = _normalize_weights(config.score_blend_weights, len(config.feature_layers))
    features = extractor(imagenet_normalize(images))
    target_size = tuple(int(value) for value in statistics["target_size"])
    per_layer_maps = []

    for layer_name, weight in zip(config.feature_layers, normalized_score_weights):
        feature_map = features[layer_name]
        if feature_map.shape[-2:] != target_size:
            feature_map = F.interpolate(feature_map, size=target_size, mode="bilinear", align_corners=False)

        layer_stats = statistics["layers"][layer_name]
        mean = layer_stats["mean"].to(images.device)
        std = layer_stats["std"].to(images.device)
        normalized_distance = ((feature_map - mean) / (std + config.embedding_eps)) ** 2
        anomaly_map = normalized_distance.mean(dim=1, keepdim=True)
        anomaly_map = F.interpolate(anomaly_map, size=images.shape[-2:], mode="bilinear", align_corners=False)
        per_layer_maps.append(weight * anomaly_map)

    combined_map = torch.stack(per_layer_maps, dim=0).sum(dim=0)
    smoothed_map = gaussian_blur(combined_map, config.heatmap_blur_kernel, config.heatmap_blur_sigma)

    padded_mask = (images.max(dim=1, keepdim=True).values <= (0.5 / 255.0)).float()
    if torch.any(padded_mask > 0):
        padded_mask = F.max_pool2d(padded_mask, kernel_size=15, stride=1, padding=7)
        smoothed_map = smoothed_map * (1.0 - padded_mask)

    if config.suppress_background_anomalies:
        foreground_mask = estimate_foreground_mask(
            images,
            tolerance=float(config.background_tolerance),
            min_coverage=float(config.background_min_coverage),
        )
        smoothed_map = smoothed_map * foreground_mask

    border_margin = max(12, images.shape[-1] // 12)
    smoothed_map[..., :border_margin, :] = 0
    smoothed_map[..., -border_margin:, :] = 0
    smoothed_map[..., :, :border_margin] = 0
    smoothed_map[..., :, -border_margin:] = 0
    image_scores = topk_mean_scores(smoothed_map, config.top_k_ratio)
    return smoothed_map, image_scores


def binary_iou(prediction_mask: np.ndarray, target_mask: np.ndarray) -> float:
    prediction = prediction_mask.astype(bool)
    target = target_mask.astype(bool)
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(prediction, target).sum()
    return float(intersection / union)


def update_top_records(bucket: list[dict], record: dict, limit: int = 3) -> None:
    bucket.append(record)
    bucket.sort(key=lambda item: item["score"], reverse=True)
    if len(bucket) > limit:
        del bucket[limit:]


@torch.no_grad()
def collect_train_statistics(extractor, statistics: dict, dataloader, config: ResNetBackboneConfig):
    image_scores = []
    pixel_scores = []
    for images, _ in dataloader:
        images = images.to(config.device)
        anomaly_maps, batch_scores = compute_anomaly_maps(extractor, images, statistics, config)
        image_scores.extend(batch_scores.cpu().numpy().tolist())
        pixel_scores.extend(anomaly_maps.flatten(start_dim=1).cpu().numpy().reshape(-1).tolist())
    return np.array(image_scores, dtype=np.float32), np.array(pixel_scores, dtype=np.float32)


@torch.no_grad()
def evaluate_model(extractor, statistics: dict, dataloader, image_threshold: float, pixel_threshold: float, config: ResNetBackboneConfig):
    records = []
    normal_examples = []
    anomalous_examples = []
    pixel_label_chunks = []
    pixel_score_chunks = []
    has_positive_pixel = False
    has_negative_pixel = False

    if config.device.type == "cuda":
        torch.cuda.synchronize(config.device)
        torch.cuda.reset_peak_memory_stats(config.device)
    evaluation_start = time.perf_counter()

    for batch in dataloader:
        images = batch["image"].to(config.device)
        anomaly_maps, image_scores = compute_anomaly_maps(extractor, images, statistics, config)
        predicted_masks = refine_predicted_mask(
            anomaly_maps,
            pixel_threshold,
            config.mask_smoothing_kernel,
            config.mask_min_area_ratio,
            config.max_mask_components,
        )

        for idx in range(images.size(0)):
            score = float(image_scores[idx].cpu().item())
            label = int(batch["label"][idx].cpu().item())
            anomaly_map = anomaly_maps[idx].cpu()
            predicted_mask = predicted_masks[idx].cpu()
            mask = batch["mask"][idx].cpu()
            mask_array = (mask.numpy().reshape(-1) > 0.5).astype(np.uint8)
            pixel_label_chunks.append(mask_array)
            pixel_score_chunks.append(anomaly_map.numpy().reshape(-1).astype(np.float32, copy=False))
            has_positive_pixel = has_positive_pixel or bool(mask_array.any())
            has_negative_pixel = has_negative_pixel or bool((mask_array == 0).any())

            record = {
                "path": batch["path"][idx],
                "defect_type": batch["defect_type"][idx],
                "label": label,
                "prediction": int(score > image_threshold),
                "score": score,
                "iou": binary_iou(predicted_mask.numpy() > 0.5, mask.numpy() > 0.5),
            }
            records.append(record)

            example_record = {
                **record,
                "image": images[idx].cpu(),
                "anomaly_map": anomaly_map,
                "predicted_mask": predicted_mask,
                "mask": mask,
            }
            if label == 0:
                update_top_records(normal_examples, example_record)
            else:
                update_top_records(anomalous_examples, example_record)

    labels = np.array([record["label"] for record in records], dtype=np.int64)
    scores = np.array([record["score"] for record in records], dtype=np.float32)
    predictions = np.array([record["prediction"] for record in records], dtype=np.int64)
    anomalous_ious = [record["iou"] for record in records if record["label"] == 1]

    pixel_roc_auc = float("nan")
    if has_positive_pixel and has_negative_pixel:
        pixel_labels = np.concatenate(pixel_label_chunks)
        pixel_scores = np.concatenate(pixel_score_chunks)
        pixel_roc_auc = float(roc_auc_score(pixel_labels, pixel_scores))

    if config.device.type == "cuda":
        torch.cuda.synchronize(config.device)
    metrics = {
        "image_accuracy": float((predictions == labels).mean()),
        "image_roc_auc": float(roc_auc_score(labels, scores)) if labels.min() != labels.max() else float("nan"),
        "pixel_roc_auc": pixel_roc_auc,
        "mean_iou": float(np.mean(anomalous_ious)) if anomalous_ious else float("nan"),
        "evaluation_time_seconds": float(time.perf_counter() - evaluation_start),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(config.device)) if config.device.type == "cuda" else 0,
    }
    return records, metrics, normal_examples + anomalous_examples


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_feature_statistics_summary(summary: dict, category: str, backbone_name: str, output_path: Path | None = None) -> None:
    plt.figure(figsize=(5, 3))
    plt.bar(["train_samples"], [summary["num_samples"]], color="#4468b3")
    plt.title(f"{category} {backbone_name.upper()} Fit Summary")
    plt.ylabel("Samples")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def show_predictions(records: list[dict], output_path: Path, num_examples: int = 6) -> None:
    selected = records[:num_examples]
    if not selected:
        return

    fig, axes = plt.subplots(len(selected), 4, figsize=(13, 3 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_axes, record in zip(axes, selected):
        row_axes[0].imshow(record["image"].permute(1, 2, 0).numpy())
        row_axes[0].set_title(f"Input\n{record['defect_type']}")
        row_axes[1].imshow(record["anomaly_map"].squeeze(0).numpy(), cmap="hot")
        row_axes[1].set_title(f"Anomaly map\nscore={record['score']:.4f}")
        row_axes[2].imshow(record["predicted_mask"].squeeze(0).numpy(), cmap="gray")
        row_axes[2].set_title(f"Predicted mask\nIoU={record['iou']:.3f}")
        row_axes[3].imshow(record["mask"].squeeze(0).numpy(), cmap="gray")
        row_axes[3].set_title(f"Ground truth\nlabel={record['label']}")
        for ax in row_axes:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def serialize_records(records: list[dict]) -> list[dict]:
    payload = []
    for record in records:
        payload.append({
            "path": record["path"],
            "defect_type": record["defect_type"],
            "label": int(record["label"]),
            "prediction": int(record["prediction"]),
            "score": float(record["score"]),
            "iou": float(record["iou"]),
        })
    return payload


def summarize_results(rows: list[dict]) -> dict:
    return {
        "image_accuracy_mean": float(np.mean([row["image_accuracy"] for row in rows])) if rows else float("nan"),
        "image_roc_auc_mean": float(np.mean([row["image_roc_auc"] for row in rows])) if rows else float("nan"),
        "pixel_roc_auc_mean": float(np.mean([row["pixel_roc_auc"] for row in rows])) if rows else float("nan"),
        "mean_iou_mean": float(np.mean([row["mean_iou"] for row in rows])) if rows else float("nan"),
        "evaluation_time_seconds_total": float(np.sum([row["evaluation_time_seconds"] for row in rows])) if rows else 0.0,
        "peak_cuda_memory_bytes_max": int(max((row["peak_cuda_memory_bytes"] for row in rows), default=0)),
        "num_categories": len(rows),
    }


def build_comparison_markdown(overall_rows: list[dict], per_category_rows: list[dict]) -> str:
    lines = [
        "# ResNet Backbone Comparison Results",
        "",
        "Backbones compared using the same feature-statistics anomaly pipeline.",
        "",
        "## Overall Backbone Ranking (Mean Across Categories)",
        "",
        "| backbone | categories | image_accuracy_mean | image_roc_auc_mean | pixel_roc_auc_mean | mean_iou_mean | evaluation_seconds | peak_cuda_memory_bytes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(overall_rows, key=lambda item: item["image_roc_auc_mean"], reverse=True):
        lines.append(
            f"| {row['backbone']} | {row['num_categories']} | {row['image_accuracy_mean']:.4f} | "
            f"{row['image_roc_auc_mean']:.4f} | {row['pixel_roc_auc_mean']:.4f} | {row['mean_iou_mean']:.4f} | "
            f"{row['evaluation_time_seconds_total']:.3f} | {row['peak_cuda_memory_bytes_max']} |"
        )

    lines.extend([
        "",
        "## Per-Category Comparison",
        "",
        "| category | backbone | image_accuracy | image_roc_auc | pixel_roc_auc | mean_iou | evaluation_seconds | peak_cuda_memory_bytes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in sorted(per_category_rows, key=lambda item: (item["category"], -item["image_roc_auc"])):
        lines.append(
            f"| {row['category']} | {row['backbone']} | {row['image_accuracy']:.4f} | "
            f"{row['image_roc_auc']:.4f} | {row['pixel_roc_auc']:.4f} | {row['mean_iou']:.4f} | "
            f"{row['evaluation_time_seconds']:.3f} | {row['peak_cuda_memory_bytes']} |"
        )
    return "\n".join(lines)


def run_category_experiment(config: ResNetBackboneConfig, category: str):
    pipeline_name = f"{config.backbone_name}_feature_statistics"
    category_dir = config.artifacts_root / config.backbone_name / category / pipeline_name
    category_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n===== Backbone: {config.backbone_name} | Category: {category} =====")

    train_dataset, test_dataset, train_loader, test_loader = create_dataloaders(config, category)
    print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    extractor, backbone_source = build_resnet_extractor(
        config.backbone_name, config.device, allow_random_init=config.allow_random_init
    )
    statistics, fit_summary = collect_feature_statistics(extractor, train_loader, config)
    print(f"Feature fitting complete in {fit_summary['time_seconds']:.1f}s")

    train_image_scores, train_pixel_scores = collect_train_statistics(extractor, statistics, train_loader, config)
    image_threshold = float(np.percentile(train_image_scores, config.image_threshold_percentile))
    pixel_threshold = float(np.percentile(train_pixel_scores, config.pixel_threshold_percentile))
    print(f"Image threshold ({config.image_threshold_percentile}th): {image_threshold:.6f}")
    print(f"Pixel threshold ({config.pixel_threshold_percentile}th): {pixel_threshold:.6f}")

    test_records, test_metrics, example_records = evaluate_model(
        extractor,
        statistics,
        test_loader,
        image_threshold,
        pixel_threshold,
        config,
    )
    print(test_metrics)

    model_payload = {
        "category": category,
        "pipeline": pipeline_name,
        "backbone_name": config.backbone_name,
        "config": config.to_dict(),
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "backbone_source": backbone_source,
        "statistics": statistics,
    }
    torch.save(model_payload, category_dir / "model.pt")

    np.savez(category_dir / "thresholds.npz", image_threshold=image_threshold, pixel_threshold=pixel_threshold)
    plot_feature_statistics_summary(fit_summary, category, config.backbone_name, category_dir / "fit_summary.png")
    save_json(category_dir / "fit_summary.json", fit_summary)

    ordered_records = sorted(test_records, key=lambda record: record["score"], reverse=True)
    show_predictions(example_records, category_dir / "qualitative_examples.png")
    save_json(category_dir / "predictions.json", serialize_records(ordered_records))

    metrics_payload = {
        "category": category,
        "pipeline": pipeline_name,
        "backbone_name": config.backbone_name,
        "backbone_source": backbone_source,
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        **test_metrics,
    }
    save_json(category_dir / "metrics.json", metrics_payload)

    compact_summary = {
        "backbone": config.backbone_name,
        "category": category,
        "pipeline": pipeline_name,
        "image_accuracy": float(test_metrics["image_accuracy"]),
        "image_roc_auc": float(test_metrics["image_roc_auc"]),
        "pixel_roc_auc": float(test_metrics["pixel_roc_auc"]),
        "mean_iou": float(test_metrics["mean_iou"]),
        "evaluation_time_seconds": float(test_metrics["evaluation_time_seconds"]),
        "peak_cuda_memory_bytes": int(test_metrics["peak_cuda_memory_bytes"]),
    }
    return compact_summary


def run_backbone_experiments(config: ResNetBackboneConfig, categories: list[str]):
    per_category_rows = []
    for category in categories:
        per_category_rows.append(run_category_experiment(config, category))

    overall = summarize_results(per_category_rows)
    overall_row = {
        "backbone": config.backbone_name,
        **overall,
    }
    return per_category_rows, overall_row


def run_comparison(
    base_config: ResNetBackboneConfig,
    categories: list[str],
    backbones: list[str],
):
    set_seed(base_config.seed)
    # Historical artifacts are immutable. Every new evaluation receives an isolated run directory.
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    registry_root = base_config.artifacts_root / "experiment_registry"
    run_root = base_config.artifacts_root / "runs" / run_id
    run_config = replace(base_config, artifacts_root=run_root)
    run_root.mkdir(parents=True, exist_ok=False)

    all_per_category_rows = []
    overall_rows = []

    for backbone_name in backbones:
        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone '{backbone_name}'. Supported: {SUPPORTED_BACKBONES}")

        backbone_config = replace(run_config, backbone_name=backbone_name)
        per_category_rows, overall_row = run_backbone_experiments(backbone_config, categories)
        all_per_category_rows.extend(per_category_rows)
        overall_rows.append(overall_row)

    save_json(run_root / "resnet_backbone_comparison_per_category.json", all_per_category_rows)
    save_csv(
        run_root / "resnet_backbone_comparison_per_category.csv",
        all_per_category_rows,
        ["backbone", "category", "pipeline", "image_accuracy", "image_roc_auc", "pixel_roc_auc", "mean_iou", "evaluation_time_seconds", "peak_cuda_memory_bytes"],
    )
    save_json(run_root / "resnet_backbone_comparison_overall.json", overall_rows)
    save_csv(
        run_root / "resnet_backbone_comparison_overall.csv",
        overall_rows,
        [
            "backbone",
            "num_categories",
            "image_accuracy_mean",
            "image_roc_auc_mean",
            "pixel_roc_auc_mean",
            "mean_iou_mean",
            "evaluation_time_seconds_total",
            "peak_cuda_memory_bytes_max",
        ],
    )

    markdown = build_comparison_markdown(overall_rows, all_per_category_rows)
    (run_root / "RESNET_BACKBONE_COMPARISON.md").write_text(markdown, encoding="utf-8")
    context = evaluation_context(run_config, categories)
    write_experiment_record(registry_root, run_id, context, overall_rows, all_per_category_rows, run_root)
    rebuild_leaderboard(registry_root, base_config.artifacts_root / "leaderboards" / context["pipeline"], context)

    return {
        "per_category": all_per_category_rows,
        "overall": overall_rows,
        "run_id": run_id,
        "run_root": run_root,
    }


def default_categories(dataset_root: Path) -> list[str]:
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


def parse_args():
    parser = argparse.ArgumentParser(description="Run feature-statistics anomaly detection with multiple ResNet backbones.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--artifacts-root", type=Path, default=None)
    parser.add_argument(
        "--backbones",
        type=str,
        default=",".join(SUPPORTED_BACKBONES),
        help=f"Comma-separated backbones from: {', '.join(SUPPORTED_BACKBONES)}",
    )
    parser.add_argument("--categories", type=str, default="", help="Comma-separated categories. Empty means all.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k-ratio", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = args.project_root.resolve()
    dataset_root = args.dataset_root.resolve() if args.dataset_root else project_root / "mvtec-ad" / "versions" / "2"
    artifacts_root = args.artifacts_root.resolve() if args.artifacts_root else project_root / "artifacts_resnet_compare"

    backbones = [item.strip() for item in args.backbones.split(",") if item.strip()]
    categories = [item.strip() for item in args.categories.split(",") if item.strip()]
    if not categories:
        categories = default_categories(dataset_root)

    config = ResNetBackboneConfig(
        project_root=project_root,
        dataset_root=dataset_root,
        artifacts_root=artifacts_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k_ratio=args.top_k_ratio,
        seed=args.seed,
    )

    print(f"Project root: {project_root}")
    print(f"Dataset root: {dataset_root}")
    print(f"Artifacts root: {artifacts_root}")
    print(f"Device: {config.device}")
    print(f"Backbones: {backbones}")
    print(f"Categories: {len(categories)}")

    run_comparison(config, categories, backbones)


if __name__ == "__main__":
    main()
