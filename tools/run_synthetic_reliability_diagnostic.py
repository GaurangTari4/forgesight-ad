"""Run a prespecified synthetic-only reliability diagnostic for PatchCore.

This tool intentionally reads only training-good images listed in an existing
artifact's calibration split.  Its synthetic changes are diagnostic proxies,
not real defects, and its results must never select a model or operational
threshold.  It does not modify the loaded artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forgesight_ad.backbone_pipeline import build_resnet_extractor, build_transforms
from forgesight_ad.patchcore_pipeline import anomaly_map, load_artifact
from tools.run_representation_experiments import synthetic


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(records: list[dict], metric_data: dict[str, list[tuple[np.ndarray, np.ndarray]]] | None = None) -> dict:
    """Summarize paired normal/synthetic proxy responses without labels."""
    result: dict[str, dict] = {}
    for kind in sorted({record["synthetic_type"] for record in records}):
        rows = [record for record in records if record["synthetic_type"] == kind]
        deltas = np.asarray([row["score_delta"] for row in rows], dtype=float)
        ratios = np.asarray([row["region_to_global_max_ratio"] for row in rows], dtype=float)
        result[kind] = {
            "samples": len(rows),
            "score_delta_mean": float(deltas.mean()),
            "score_delta_median": float(np.median(deltas)),
            "score_delta_positive_fraction": float((deltas > 0).mean()),
            "region_to_global_max_ratio_mean": float(ratios.mean()),
            "argmax_inside_synthetic_region_fraction": float(
                np.mean([row["argmax_inside_synthetic_region"] for row in rows])
            ),
            "inside_mean_over_outside_mean": float(np.mean([row["inside_mean_over_outside_mean"] for row in rows])),
            "threshold_mask_iou_mean": float(np.mean([row["threshold_mask_iou"] for row in rows])),
            "threshold_mask_dice_mean": float(np.mean([row["threshold_mask_dice"] for row in rows])),
            "background_false_activation_rate_mean": float(np.mean([row["background_false_activation_rate"] for row in rows])),
            "synthetic_area_pixels_mean": float(np.mean([row["synthetic_area_pixels"] for row in rows])),
            "synthetic_area_ratio_mean": float(np.mean([row["synthetic_area_ratio"] for row in rows])),
        }
        if metric_data is not None:
            labels = np.concatenate([item[0] for item in metric_data[kind]])
            scores = np.concatenate([item[1] for item in metric_data[kind]])
            result[kind]["pixel_roc_auc"] = float(roc_auc_score(labels, scores))
            result[kind]["pixel_average_precision"] = float(average_precision_score(labels, scores))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "reliability_experiments")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    resources = json.loads(args.resources.read_text(encoding="utf-8"))
    frozen_threshold = float(resources["normal_calibration"]["q98_higher"])
    manifest_path = args.artifact.parent / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fit_paths = {Path(path).resolve() for path in manifest["fit_paths"]}
    calibration_paths = [Path(path).resolve() for path in manifest["calibration_paths"]]
    if fit_paths.intersection(calibration_paths):
        raise ValueError("artifact split manifest leaks calibration images into fitting")
    if args.samples > len(calibration_paths):
        raise ValueError("requested samples exceed the reserved calibration-normal split")

    payload = load_artifact(args.artifact, device, backbone="resnet18")
    if payload.get("category") != resources.get("category"):
        raise ValueError("artifact and resources category provenance differ")
    if tuple(payload["feature_layers"]) != ("layer2", "layer3"):
        raise ValueError("diagnostic is prespecified only for layer2/layer3 PatchCore")
    if payload["split_manifest_sha256"] != sha256(manifest_path):
        raise ValueError("artifact split-manifest digest does not match")
    image_size = int(payload["image_size"])
    transform, _ = build_transforms(image_size)
    extractor, source = build_resnet_extractor("resnet18", device, allow_random_init=False)
    kinds = ("small_local", "low_contrast", "thin_scratch", "color_shift", "localized_change")
    records: list[dict] = []
    metric_data: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {kind: [] for kind in kinds}
    for index, source_path in enumerate(calibration_paths[: args.samples]):
        image = Image.open(source_path).convert("RGB")
        baseline = transform(image).unsqueeze(0).to(device)
        with torch.inference_mode():
            base_map = anomaly_map(extractor, baseline, payload["memory_bank"], tuple(payload["feature_layers"]))
        base_score = float(base_map.max())
        for kind in kinds:
            changed, box = synthetic(image, kind, args.seed + index)
            tensor = transform(changed).unsqueeze(0).to(device)
            with torch.inference_mode():
                mapped = anomaly_map(extractor, tensor, payload["memory_bank"], tuple(payload["feature_layers"]))
            score = float(mapped.max())
            x0, y0, x1, y1 = box
            map_2d = mapped[0, 0].detach().cpu().numpy()
            synthetic_mask = np.zeros(map_2d.shape, dtype=np.uint8)
            synthetic_mask[y0 : y1 + 1, x0 : x1 + 1] = 1
            threshold_mask = (map_2d > frozen_threshold).astype(np.uint8)
            y_peak, x_peak = np.unravel_index(int(np.argmax(map_2d)), map_2d.shape)
            region_max = float(map_2d[y0 : y1 + 1, x0 : x1 + 1].max())
            inside = map_2d[synthetic_mask == 1]
            outside = map_2d[synthetic_mask == 0]
            intersection = int(np.logical_and(synthetic_mask, threshold_mask).sum())
            union = int(np.logical_or(synthetic_mask, threshold_mask).sum())
            dice_denominator = int(synthetic_mask.sum() + threshold_mask.sum())
            metric_data[kind].append((synthetic_mask.reshape(-1), map_2d.reshape(-1)))
            records.append(
                {
                    "source_calibration_path": str(source_path),
                    "synthetic_type": kind,
                    "region_xyxy": list(box),
                    "baseline_score": base_score,
                    "synthetic_score": score,
                    "score_delta": score - base_score,
                    "region_to_global_max_ratio": region_max / max(score, np.finfo(float).eps),
                    "argmax_inside_synthetic_region": bool(x0 <= x_peak <= x1 and y0 <= y_peak <= y1),
                    "inside_mean_over_outside_mean": float(inside.mean() / max(outside.mean(), np.finfo(float).eps)),
                    "threshold_mask_iou": intersection / union if union else 1.0,
                    "threshold_mask_dice": 2 * intersection / dice_denominator if dice_denominator else 1.0,
                    "background_false_activation_rate": float(threshold_mask[synthetic_mask == 0].mean()),
                    "synthetic_area_pixels": int(synthetic_mask.sum()),
                    "synthetic_area_ratio": float(synthetic_mask.mean()),
                }
            )

    output = args.output_root / f"patchcore_r18_bank512_synthetic_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol": "synthetic diagnostic only; no MVTec test data, no model/threshold selection",
        "artifact": {"path": str(args.artifact), "sha256": sha256(args.artifact)},
        "resources": {"path": str(args.resources), "sha256": sha256(args.resources)},
        "split_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "category": payload["category"],
        "backbone": payload["backbone_name"],
        "feature_layers": payload["feature_layers"],
        "image_size": image_size,
        "device": str(device),
        "backbone_source": source,
        "normal_source": "reserved disjoint calibration-good images only",
        "test_data_accessed": False,
        "operational_threshold_changed": False,
        "frozen_diagnostic_threshold": frozen_threshold,
        "records": records,
        "summary_by_synthetic_type": summarize(records, metric_data),
    }
    (output / "synthetic_reliability.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": report["summary_by_synthetic_type"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
