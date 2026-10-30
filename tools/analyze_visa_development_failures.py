"""Audit VisA development scores without reading the sealed-final partition.

This is strictly post-hoc descriptive analysis.  It neither fits a detector nor
changes a calibration threshold. ``--view`` must be the development-only view.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forgesight_ad.autoencoder_pipeline import resize_with_padding
from forgesight_ad.backbone_pipeline import build_transforms
from forgesight_ad.detectors import build_detector


def distribution(values: list[float] | np.ndarray) -> dict[str, float | int]:
    """Return named quantiles and reject empty samples instead of hiding them."""
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty score distribution")
    return {
        "count": int(array.size), "min": float(array.min()),
        "q05": float(np.quantile(array, 0.05)), "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)), "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)), "max": float(array.max()),
        "mean": float(array.mean()), "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
    }


def render_overlays(image: Image.Image, anomaly_map: np.ndarray, mask: np.ndarray, stem: Path) -> dict[str, str]:
    """Render on the same letterboxed 512-square canvas used for inference."""
    base = np.asarray(resize_with_padding(image.convert("RGB"), 512, InterpolationMode.BILINEAR), dtype=np.float32)
    if anomaly_map.shape != base.shape[:2] or mask.shape != base.shape[:2]:
        raise ValueError("Visualization image, anomaly map, and mask are not spatially aligned")
    heat = (anomaly_map / max(float(anomaly_map.max()), 1e-12) * 255).clip(0, 255).astype(np.uint8)
    colour = np.zeros_like(base); colour[:, :, 0] = heat; colour[:, :, 1] = 255 - heat
    heatmap_path = stem.with_name(stem.name + "_heatmap.png")
    Image.fromarray((0.6 * base + 0.4 * colour).astype(np.uint8)).save(heatmap_path)
    overlay = base.copy()
    overlay[mask > 0] = 0.45 * overlay[mask > 0] + 0.55 * np.array([0, 255, 0])
    mask_path = stem.with_name(stem.name + "_ground_truth_mask.png")
    Image.fromarray(overlay.astype(np.uint8)).save(mask_path)
    return {"heatmap": str(heatmap_path), "ground_truth_mask_overlay": str(mask_path)}


def validate_development_view(view: dict) -> list[dict]:
    if view.get("permitted_splits") != ["development"] or set(view.get("splits", ())) != {"development"}:
        raise ValueError("Refusing analysis: --view is not a development-only split view")
    rows = view["splits"]["development"]
    if not rows:
        raise ValueError("Development-only view contains no rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-visualizations", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite analysis output: {args.output}")
    if args.max_visualizations < 0:
        parser.error("--max-visualizations must be non-negative")

    view = json.loads(args.view.read_text(encoding="utf-8"))
    rows = validate_development_view(view)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration_scores = np.asarray(calibration["scores"], dtype=float)
    threshold = float(calibration["threshold"])
    if not np.isfinite(calibration_scores).all() or not np.isfinite(threshold):
        raise ValueError("Calibration contains non-finite scores or threshold")
    recomputed = float(np.quantile(calibration_scores, 0.98, method="higher"))
    if not np.isclose(threshold, recomputed, rtol=0, atol=1e-12):
        raise ValueError("Stored threshold does not match q98-higher calibration scores")

    detector = build_detector("patchcore", args.artifact, "resnet18", view["dataset_id"], args.device, calibration=args.calibration)
    _, mask_transform = build_transforms(512)
    records: list[dict] = []
    visualizations: list[dict] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        with Image.open(row["image"]) as handle:
            image = handle.convert("RGB").copy()
        prediction = detector.predict(image)
        anomaly_map = prediction.anomaly_map.detach().cpu().numpy()[0, 0]
        if anomaly_map.shape != (512, 512) or not np.isfinite(anomaly_map).all():
            raise ValueError(f"Invalid anomaly map for {row['image']}")
        label = row["image_label"]
        mask = np.zeros_like(anomaly_map, dtype=np.uint8)
        if label == "defect":
            with Image.open(row["mask"]) as handle:
                mask = (mask_transform(handle.convert("L")).numpy()[0] > 0).astype(np.uint8)
            foreground, background = anomaly_map[mask > 0], anomaly_map[mask == 0]
            foreground_mean, foreground_max = float(foreground.mean()), float(foreground.max())
        else:
            foreground = np.empty(0, dtype=float); background = anomaly_map.reshape(-1)
            foreground_mean = foreground_max = None
        background_mean = float(background.mean())
        record = {
            "path": row["image"], "label": label, "subtype": row.get("defect_subtype", "normal"),
            "score": float(prediction.score), "threshold": threshold, "prediction": prediction.prediction,
            "margin_to_threshold": float(threshold - prediction.score), "foreground_mean": foreground_mean,
            "foreground_max": foreground_max, "background_mean": background_mean,
            "background_q95": float(np.quantile(background, 0.95)),
            "foreground_background_ratio": float(foreground_mean / max(background_mean, 1e-12)) if foreground_mean is not None else None,
        }
        records.append(record)
        if label == "defect" and prediction.prediction == "PASS" and len(visualizations) < args.max_visualizations:
            stem = args.output.parent / f"false_negative_{len(visualizations) + 1}_{Path(row['image']).stem}"
            visualizations.append({"record": record, **render_overlays(image, anomaly_map, mask, stem)})

    normal_scores = [r["score"] for r in records if r["label"] == "normal"]
    defect_scores = [r["score"] for r in records if r["label"] == "defect"]
    labels = np.asarray([r["label"] == "defect" for r in records], dtype=np.uint8)
    scores = np.asarray([r["score"] for r in records], dtype=float)
    false_negatives = [r for r in records if r["label"] == "defect" and r["prediction"] == "PASS"]
    foreground_means = [r["foreground_mean"] for r in records if r["foreground_mean"] is not None]
    background_means = [r["background_mean"] for r in records if r["foreground_mean"] is not None]
    result = {
        "scope": "development-only; sealed-final images and masks were not loaded",
        "method": "post-hoc analysis only; no refitting, recalibration, threshold selection, or final evaluation",
        "threshold_provenance": {
            "stored_image_threshold": threshold, "recomputed_q98_higher": recomputed,
            "calibration_score_count": int(calibration_scores.size), "image_and_pixel_threshold_same": True,
            "implementation": "PatchCoreAdapter predicted_mask = anomaly_map > image threshold",
        },
        "score_distributions": {
            "calibration_normal": distribution(calibration_scores), "development_normal": distribution(normal_scores),
            "development_defect": distribution(defect_scores),
            "defect_fraction_at_or_below_threshold": float(np.mean(np.asarray(defect_scores) <= threshold)),
            "defect_median_margin_below_threshold": float(np.median(threshold - np.asarray(defect_scores))),
        },
        "separability": {"image_auroc": float(roc_auc_score(labels, scores)), "image_average_precision": float(average_precision_score(labels, scores))},
        "localization": {
            "defect_foreground_mean": distribution(foreground_means), "defect_background_mean": distribution(background_means),
            "mean_foreground_background_ratio": float(np.mean(np.asarray(foreground_means) / np.maximum(background_means, 1e-12))),
            "mask_transform": "same 512x512 resize_with_padding transform as detector input",
        },
        "false_negatives": {"count": len(false_negatives), "by_subtype": {subtype: sum(r["subtype"] == subtype for r in false_negatives) for subtype in sorted({r["subtype"] for r in false_negatives})}},
        "records": records, "visualizations": visualizations, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
