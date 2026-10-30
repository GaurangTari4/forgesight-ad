"""Evaluate a frozen ForgeSight detector on a manifest's untouched final split.

This command rejects MVTec test reuse through the manifest validator. It does
not fit, calibrate, select a threshold, or overwrite a result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forgesight_ad.detectors import build_detector
from forgesight_ad.backbone_pipeline import build_transforms
from forgesight_ad.independent_data import validate_manifest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(events: int, total: int, z: float = 1.96) -> list[float | None]:
    if total == 0:
        return [None, None]
    value = events / total
    denominator = 1 + z * z / total
    centre = (value + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(value * (1 - value) / total + z * z / (4 * total * total)) / denominator
    return [float(max(0.0, centre - radius)), float(min(1.0, centre + radius))]


def safe_metric(fn, labels: np.ndarray, scores: np.ndarray) -> float | None:
    return float(fn(labels, scores)) if len(np.unique(labels)) == 2 else None


def aligned_binary_mask(image: Image.Image, image_size: int = 512) -> np.ndarray:
    """Apply precisely the detector's letterbox transform to a ground-truth mask."""
    _, mask_transform = build_transforms(image_size)
    return (mask_transform(image.convert("L")).numpy()[0] > 0).astype(np.uint8)


def summarize(rows: list[dict]) -> dict:
    image_labels = np.asarray([row["image_label"] for row in rows], dtype=np.uint8)
    image_scores = np.asarray([row["score"] for row in rows], dtype=float)
    predictions = np.asarray([row["prediction"] == "FAIL" for row in rows], dtype=bool)
    tn = int(np.sum((image_labels == 0) & ~predictions)); fp = int(np.sum((image_labels == 0) & predictions))
    fn = int(np.sum((image_labels == 1) & ~predictions)); tp = int(np.sum((image_labels == 1) & predictions))
    pixel_labels = np.concatenate([np.asarray(row["mask"], dtype=np.uint8).reshape(-1) for row in rows])
    pixel_scores = np.concatenate([np.asarray(row["anomaly_map"], dtype=float).reshape(-1) for row in rows])
    pixel_predictions = np.concatenate([np.asarray(row["predicted_mask"], dtype=np.uint8).reshape(-1) for row in rows])
    intersection = int(np.sum((pixel_labels == 1) & (pixel_predictions == 1)))
    union = int(np.sum((pixel_labels == 1) | (pixel_predictions == 1)))
    denom = int(pixel_labels.sum() + pixel_predictions.sum())
    latency = np.asarray([row["total_call_ms"] for row in rows], dtype=float)
    return {
        "image_roc_auc": safe_metric(roc_auc_score, image_labels, image_scores),
        "image_average_precision": safe_metric(average_precision_score, image_labels, image_scores),
        "pixel_roc_auc": safe_metric(roc_auc_score, pixel_labels, pixel_scores),
        "pixel_average_precision": safe_metric(average_precision_score, pixel_labels, pixel_scores),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "sensitivity": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "fpr": fp / (fp + tn) if fp + tn else None,
        "fnr": fn / (fn + tp) if fn + tp else None,
        "fpr_wilson_95": wilson_interval(fp, fp + tn),
        "fnr_wilson_95": wilson_interval(fn, fn + tp),
        "pixel_iou": intersection / union if union else 1.0,
        "pixel_dice": 2 * intersection / denom if denom else 1.0,
        "warm_total_call_ms": {"mean": float(latency.mean()), "median": float(np.median(latency)), "p95": float(np.quantile(latency, 0.95))},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "final_evaluation"), default="final_evaluation")
    parser.add_argument("--split-view", type=Path, help="Required for development to avoid reading final records")
    parser.add_argument("--family", choices=("historical", "v2", "patchcore"), required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration", type=Path, help="Required PatchCore calibration sidecar when non-default")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation result: {args.output}")
    if args.warmups < 0:
        parser.error("--warmups must be non-negative")
    if args.split == "development":
        if args.split_view is None:
            raise ValueError("Development evaluation requires --split-view so sealed final records are not loaded")
        view = json.loads(args.split_view.read_text(encoding="utf-8"))
        if view.get("permitted_splits") != ["development"] or set(view.get("splits", ())) != {"development"}:
            raise ValueError("Development split view is malformed or exposes disallowed partitions")
        dataset = {"dataset_id": view["dataset_id"], "manifest_path": str(args.split_view), "manifest_sha256": view["parent_manifest_sha256"], "provenance": {"split_view": "development-only"}, "splits": view["splits"]}
    else:
        dataset = validate_manifest(args.manifest)
    kwargs = {"calibration": args.calibration} if args.family == "patchcore" and args.calibration else {}
    detector = build_detector(args.family, args.artifact, args.backbone, args.category, args.device, **kwargs)
    rows: list[dict] = []
    selected_rows = dataset["splits"][args.split]
    for row in selected_rows[: min(args.warmups, len(selected_rows))]:
        with Image.open(row.get("image_path", row["image"])) as image:
            detector.predict(image.convert("RGB"))
    for record in selected_rows:
        image_path = Path(record.get("image_path", record["image"]))
        with Image.open(image_path) as image:
            started = time.perf_counter(); prediction = detector.predict(image.convert("RGB")); total_ms = (time.perf_counter() - started) * 1000
        height, width = prediction.anomaly_map.shape[-2:]
        if record.get("mask_path") or record.get("mask"):
            with Image.open(record.get("mask_path", record["mask"])) as image:
                # Match detector letterboxing exactly; direct square resize would
                # distort non-square VisA masks and invalidate pixel metrics.
                mask = aligned_binary_mask(image)
        else:
            mask = np.zeros((height, width), dtype=np.uint8)
        rows.append({
            "image_sha256": record.get("image_sha256", sha256(image_path)), "mask_sha256": record.get("mask_sha256") or (sha256(Path(record["mask"])) if record.get("mask") else None), "image_label": int(record["image_label"] == "defect"),
            "defect_subtype": record.get("defect_subtype", "normal"), "score": prediction.score, "threshold": prediction.threshold,
            "prediction": prediction.prediction, "anomaly_map": prediction.anomaly_map.numpy().astype(np.float32)[0, 0].tolist(),
            "predicted_mask": prediction.mask.numpy().astype(np.uint8)[0, 0].tolist(), "mask": mask.tolist(),
            "adapter_inference_ms": prediction.inference_ms, "total_call_ms": total_ms,
        })
    subtype = {name: summarize(group) for name, group in {name: [row for row in rows if row["defect_subtype"] == name] for name in sorted({row["defect_subtype"] for row in rows})}.items()}
    persisted_rows = [{key: value for key, value in row.items() if key not in {"anomaly_map", "predicted_mask", "mask"}} for row in rows]
    report = {
        "protocol": "independent final evaluation only; frozen artifact and threshold; no fitting/calibration/model selection",
        "dataset": {key: dataset[key] for key in ("dataset_id", "manifest_path", "manifest_sha256", "provenance")},
        "detector": {"family": args.family, "artifact": str(args.artifact), "artifact_sha256": sha256(args.artifact), "backbone": args.backbone, "category": args.category, "device": args.device},
        "evaluated_split": args.split, "metrics": summarize(rows), "defect_subtype_breakdown": subtype, "records": persisted_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
