"""Read-only inventory and CPU smoke validation for trusted ForgeSight checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from forgesight_ad.backbone_pipeline import (
    ResNetBackboneConfig,
    build_resnet_extractor,
    build_transforms,
    compute_anomaly_maps,
    refine_predicted_mask,
)
from forgesight_ad.checkpoints import (
    CheckpointCompatibilityError,
    load_checkpoint,
    validate_feature_statistics_checkpoint,
)


def checkpoint_summary(path: Path, device: torch.device) -> dict:
    record = {"path": str(path), "size_bytes": path.stat().st_size}
    try:
        payload = load_checkpoint(path, device)
        validate_feature_statistics_checkpoint(payload, str(payload.get("backbone_name", "")))
        config = payload["config"]
        stats = payload["statistics"]
        record.update(
            status="compatible",
            backbone=payload["backbone_name"],
            pipeline=payload.get("pipeline"),
            image_size=config["image_size"],
            feature_layers=config.get("feature_layers"),
            target_size=list(stats["target_size"]),
            backbone_source=payload.get("backbone_source"),
            keys=sorted(payload),
        )
    except (CheckpointCompatibilityError, KeyError, TypeError, ValueError) as exc:
        record.update(status="incompatible", error=str(exc))
    return record


def inventory(root: Path, output: Path, device: torch.device) -> list[dict]:
    existing = json.loads(output.read_text(encoding="utf-8")).get("records", []) if output.exists() else []
    by_path = {record["path"]: record for record in existing}
    paths = sorted(root.rglob("model.pt"))
    for index, path in enumerate(paths, 1):
        if str(path) not in by_path:
            by_path[str(path)] = checkpoint_summary(path, device)
            print(f"[{index}/{len(paths)}] {path.name}: {by_path[str(path)]['status']}", flush=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"records": list(by_path.values())}, indent=2), encoding="utf-8")
    records = [by_path[str(path)] for path in paths]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_root": str(root),
        "loader": "torch.load(weights_only=True)",
        "records": records,
        "summary": {
            "total": len(records),
            "compatible": sum(item["status"] == "compatible" for item in records),
            "incompatible": sum(item["status"] == "incompatible" for item in records),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return records


def runtime_config(payload: dict, device: torch.device) -> ResNetBackboneConfig:
    cfg = payload["config"]
    return ResNetBackboneConfig(
        project_root=Path(cfg["project_root"]), dataset_root=Path(cfg["dataset_root"]),
        artifacts_root=Path(cfg["artifacts_root"]), image_size=int(cfg["image_size"]),
        batch_size=int(cfg["batch_size"]), num_epochs=int(cfg["num_epochs"]),
        learning_rate=float(cfg["learning_rate"]), latent_channels=int(cfg["latent_channels"]),
        image_threshold_percentile=float(cfg["image_threshold_percentile"]),
        pixel_threshold_percentile=float(cfg["pixel_threshold_percentile"]),
        top_k_ratio=float(cfg["top_k_ratio"]), heatmap_blur_kernel=int(cfg["heatmap_blur_kernel"]),
        heatmap_blur_sigma=float(cfg["heatmap_blur_sigma"]), mask_smoothing_kernel=int(cfg["mask_smoothing_kernel"]),
        mask_min_area_ratio=float(cfg["mask_min_area_ratio"]), max_mask_components=int(cfg.get("max_mask_components", 2)),
        mse_weight=float(cfg["mse_weight"]), ssim_weight=float(cfg["ssim_weight"]),
        feature_map_weight=float(cfg["feature_map_weight"]), num_workers=0, seed=int(cfg["seed"]),
        device_name=str(device), backbone_name=str(cfg["backbone_name"]),
        feature_layers=tuple(cfg["feature_layers"]), feature_blend_weights=tuple(cfg["feature_blend_weights"]),
        score_blend_weights=tuple(cfg["score_blend_weights"]), embedding_eps=float(cfg.get("embedding_eps", 1e-6)),
        suppress_background_anomalies=bool(cfg.get("suppress_background_anomalies", True)),
        background_tolerance=float(cfg.get("background_tolerance", 0.08)),
        background_min_coverage=float(cfg.get("background_min_coverage", 0.05)),
    )


def smoke(checkpoint: Path, image_path: Path, output_dir: Path, device: torch.device) -> None:
    payload = load_checkpoint(checkpoint, device)
    validate_feature_statistics_checkpoint(payload, "resnet18")
    config = runtime_config(payload, device)
    extractor, source = build_resnet_extractor("resnet18", device, allow_random_init=False)
    transform, _ = build_transforms(config.image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    ended = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    if started: started.record()
    with torch.inference_mode():
        anomaly_map, score = compute_anomaly_maps(extractor, tensor, payload["statistics"], config)
    if ended:
        ended.record(); torch.cuda.synchronize(device)
    mask = refine_predicted_mask(anomaly_map, float(payload["pixel_threshold"]), config.mask_smoothing_kernel, config.mask_min_area_ratio, config.max_mask_components)
    if not torch.isfinite(anomaly_map).all() or not torch.isfinite(score).all():
        raise RuntimeError("Smoke inference generated non-finite values.")
    output_dir.mkdir(parents=True, exist_ok=False)
    heatmap = anomaly_map.squeeze().cpu().numpy()
    np.save(output_dir / "anomaly_map.npy", heatmap)
    Image.fromarray((mask.squeeze().cpu().numpy() * 255).astype(np.uint8)).save(output_dir / "mask.png")
    plt.imsave(output_dir / "heatmap.png", heatmap, cmap="inferno")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "input_image": str(image_path),
        "input_tensor_shape": list(tensor.shape), "anomaly_map_shape": list(anomaly_map.shape),
        "mask_shape": list(mask.shape), "score": float(score.item()), "finite": True,
        "prediction": "damaged" if float(score.item()) > float(payload["image_threshold"]) else "good",
        "backbone_source": source, "device": str(device),
        "execution_time_ms": started.elapsed_time(ended) if started else None,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "note": "Technical smoke test only; it does not establish anomaly-detection accuracy.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--smoke-checkpoint", type=Path)
    parser.add_argument("--smoke-image", type=Path)
    parser.add_argument("--smoke-output", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = torch.device("cuda:0" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        raise RuntimeError("CUDA was requested but is unavailable.")
    records = inventory(args.checkpoint_root, args.inventory, device)
    print(f"Inventory: {sum(item['status'] == 'compatible' for item in records)}/{len(records)} compatible")
    if args.smoke_checkpoint:
        if not args.smoke_image or not args.smoke_output:
            raise ValueError("--smoke-image and --smoke-output are required with --smoke-checkpoint")
        smoke(args.smoke_checkpoint, args.smoke_image, args.smoke_output, device)
        print(f"Smoke outputs: {args.smoke_output}")


if __name__ == "__main__":
    main()
