"""Resumable GPU evaluation of trusted, precomputed ForgeSight checkpoints.

This never fits statistics or writes beside historical checkpoints.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forgesight_ad.autoencoder_pipeline import MVTecTestDataset
from forgesight_ad.backbone_pipeline import ResNetBackboneConfig, build_resnet_extractor, build_transforms, compute_anomaly_maps
from forgesight_ad.checkpoints import load_checkpoint, validate_feature_statistics_checkpoint
from forgesight_ad.experiment_registry import evaluation_context, write_experiment_record


BACKBONES = ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_config(payload: dict, dataset_root: Path, output_root: Path, device: torch.device, batch_size: int) -> ResNetBackboneConfig:
    values = dict(payload["config"])
    allowed = {item.name for item in fields(ResNetBackboneConfig)}
    values = {key: value for key, value in values.items() if key in allowed}
    values.update(project_root=ROOT, dataset_root=dataset_root, artifacts_root=output_root, device_name=str(device), batch_size=batch_size, num_workers=0)
    for key in ("feature_layers", "feature_blend_weights", "score_blend_weights"):
        if key in values:
            values[key] = tuple(values[key])
    return ResNetBackboneConfig(**values)


def evaluate(backbone: str, category: str, checkpoint: Path, dataset_root: Path, output_root: Path, device: torch.device, batch_size: int) -> dict:
    payload = load_checkpoint(checkpoint, device)
    validate_feature_statistics_checkpoint(payload, backbone)
    config = runtime_config(payload, dataset_root, output_root, device, batch_size)
    if config.image_size != 512:
        raise ValueError(f"{backbone} checkpoint requires image_size={config.image_size}, not the requested 512.")

    image_transform, mask_transform = build_transforms(config.image_size)
    dataset = MVTecTestDataset(dataset_root, category, image_transform, mask_transform, config.image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    extractor, source = build_resnet_extractor(backbone, device, allow_random_init=False)
    extractor.eval()

    image_labels: list[np.ndarray] = []
    image_scores: list[np.ndarray] = []
    pixel_labels: list[np.ndarray] = []
    pixel_scores: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            maps, scores = compute_anomaly_maps(extractor, images, payload["statistics"], config)
            if not torch.isfinite(maps).all() or not torch.isfinite(scores).all():
                raise RuntimeError(f"{backbone} produced non-finite inference values")
            image_labels.append(batch["label"].numpy().astype(np.uint8, copy=False))
            image_scores.append(scores.cpu().numpy().astype(np.float32, copy=False))
            pixel_labels.append((batch["mask"].numpy().reshape(-1) > 0.5).astype(np.uint8, copy=False))
            pixel_scores.append(maps.cpu().numpy().reshape(-1).astype(np.float32, copy=False))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    labels, scores = np.concatenate(image_labels), np.concatenate(image_scores)
    pixel_y, pixel_s = np.concatenate(pixel_labels), np.concatenate(pixel_scores)
    metrics = {
        "backbone": backbone, "category": category, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "backbone_source": source, "device": str(device), "batch_size": batch_size, "image_size": config.image_size,
        "feature_layers": list(config.feature_layers), "image_threshold": float(payload["image_threshold"]), "pixel_threshold": float(payload["pixel_threshold"]),
        "images_evaluated": int(labels.size), "defective_images": int(labels.sum()), "normal_images": int((labels == 0).sum()),
        "image_roc_auc": float(roc_auc_score(labels, scores)), "pixel_roc_auc": float(roc_auc_score(pixel_y, pixel_s)),
        "image_average_precision": float(average_precision_score(labels, scores)), "pixel_average_precision": float(average_precision_score(pixel_y, pixel_s)),
        "evaluation_duration_seconds": duration, "end_to_end_evaluation_ms_per_image": duration * 1000 / labels.size,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "thresholds_source": "checkpoint training-statistics percentiles; test labels were not used for fitting",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    del extractor
    if device.type == "cuda": torch.cuda.empty_cache()
    return metrics


def write_outputs(run_root: Path, rows: list[dict]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "per_backbone.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    if rows:
        with (run_root / "per_backbone.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    macro = []
    for backbone in BACKBONES:
        subset = [row for row in rows if row["backbone"] == backbone]
        if subset:
            macro.append({"backbone": backbone, "categories_completed": len(subset), "image_roc_auc_macro": float(np.mean([x["image_roc_auc"] for x in subset])), "pixel_roc_auc_macro": float(np.mean([x["pixel_roc_auc"] for x in subset])), "image_average_precision_macro": float(np.mean([x["image_average_precision"] for x in subset])), "pixel_average_precision_macro": float(np.mean([x["pixel_average_precision"] for x in subset])), "evaluation_duration_seconds_total": float(sum(x["evaluation_duration_seconds"] for x in subset)), "end_to_end_evaluation_ms_per_image": float(sum(x["evaluation_duration_seconds"] for x in subset) * 1000 / sum(x["images_evaluated"] for x in subset)), "peak_cuda_memory_bytes_max": int(max(x["peak_cuda_memory_bytes"] for x in subset))})
    ordered = sorted(macro, key=lambda row: row["image_roc_auc_macro"], reverse=True)
    (run_root / "leaderboard.json").write_text(json.dumps(ordered, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# ForgeSight Existing-Checkpoint Benchmark", "", "All thresholds come from training statistics; test labels are only used for metrics.", "", "## Macro leaderboard", "", "| rank | backbone | categories | image AUROC | pixel AUROC | image AP | pixel AP | eval s | eval ms/image | peak CUDA bytes |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for rank, row in enumerate(ordered, 1): lines.append(f"| {rank} | {row['backbone']} | {row['categories_completed']} | {row['image_roc_auc_macro']:.6f} | {row['pixel_roc_auc_macro']:.6f} | {row['image_average_precision_macro']:.6f} | {row['pixel_average_precision_macro']:.6f} | {row['evaluation_duration_seconds_total']:.3f} | {row['end_to_end_evaluation_ms_per_image']:.3f} | {row['peak_cuda_memory_bytes_max']} |")
    (run_root / "BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if macro:
        figure, axis = plt.subplots(figsize=(8, 4)); axis.bar([row["backbone"] for row in ordered], [row["image_roc_auc_macro"] for row in ordered]); axis.set_ylim(0, 1); axis.set_ylabel("Macro image AUROC"); figure.tight_layout(); figure.savefig(run_root / "image_auroc.png", dpi=160); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=WORKSPACE / "mvtec-ad" / "versions" / "2")
    parser.add_argument("--checkpoint-root", type=Path, default=WORKSPACE / "artifacts_resnet_compare")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "benchmark_runs")
    parser.add_argument("--run-id", default=datetime.now(timezone.utc).strftime("existing_checkpoints_%Y%m%dT%H%M%SZ"))
    parser.add_argument("--backbones", default=",".join(BACKBONES))
    parser.add_argument("--categories", default="", help="Comma-separated categories; empty evaluates every category.")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA is required for this benchmark.")
    run_root = args.output_root / args.run_id; run_root.mkdir(parents=True, exist_ok=True)
    result_file = run_root / "per_backbone.json"; completed = json.loads(result_file.read_text()).copy() if result_file.exists() else []
    done = {(row["backbone"], row["category"]) for row in completed}; requested = [item.strip() for item in args.backbones.split(",") if item.strip()]
    categories = [item.strip() for item in args.categories.split(",") if item.strip()] or sorted(path.name for path in args.dataset_root.iterdir() if path.is_dir())
    for category in categories:
      for backbone in requested:
        if (backbone, category) in done: print(f"Skipping completed {backbone}/{category}", flush=True); continue
        checkpoint = args.checkpoint_root / backbone / category / f"{backbone}_feature_statistics" / "model.pt"
        print(f"Evaluating {backbone}/{category}: {checkpoint}", flush=True)
        completed.append(evaluate(backbone, category, checkpoint, args.dataset_root, run_root, device, args.batch_size)); write_outputs(run_root, completed)
        print(f"Completed {backbone}/{category}", flush=True)
    write_outputs(run_root, completed)
    if {(row["backbone"], row["category"]) for row in completed} >= {(backbone, category) for backbone in BACKBONES for category in categories}:
        reference = load_checkpoint(args.checkpoint_root / "resnet18" / "bottle" / "resnet18_feature_statistics" / "model.pt", device)
        context = evaluation_context(runtime_config(reference, args.dataset_root, run_root, device, args.batch_size), categories)
        overall = json.loads((run_root / "leaderboard.json").read_text(encoding="utf-8"))
        for row in overall: row.update(num_categories=row.pop("categories_completed"), image_roc_auc_mean=row.pop("image_roc_auc_macro"), pixel_roc_auc_mean=row.pop("pixel_roc_auc_macro"), mean_iou_mean=None, image_accuracy_mean=None, evaluation_time_seconds_total=row["evaluation_duration_seconds_total"], peak_cuda_memory_bytes_max=row["peak_cuda_memory_bytes_max"])
        write_experiment_record(args.output_root.parent / "experiment_registry", args.run_id, context, overall, completed, run_root)
    print(run_root, flush=True)


if __name__ == "__main__": main()
