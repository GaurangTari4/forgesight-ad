"""Validate ONNX-backed PatchCore against the unchanged PyTorch CPU path."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_app import build_heatmap_overlay, build_mask_overlay, normalize_map
from forgesight_ad.autoencoder_pipeline import imagenet_normalize
from forgesight_ad.detectors import PatchCoreAdapter
from forgesight_ad.onnx_patchcore import OnnxPatchCoreAdapter, available_execution_providers


def percentile(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def render(image: Image.Image, prediction, destination: Path, stem: str) -> dict[str, list[int]]:
    anomaly = prediction.anomaly_map.squeeze().numpy()
    mask = prediction.mask.squeeze().numpy()
    resized = image.convert("RGB").resize((anomaly.shape[1], anomaly.shape[0]))
    raw = np.asarray(resized)
    Image.fromarray(raw).save(destination / f"{stem}_input.png")
    Image.fromarray(build_heatmap_overlay(raw, normalize_map(anomaly))).save(destination / f"{stem}_heatmap.png")
    Image.fromarray(build_mask_overlay(raw, mask)).save(destination / f"{stem}_mask.png")
    return {"input": list(raw.shape), "map": list(prediction.anomaly_map.shape), "mask": list(prediction.mask.shape)}


def e2e(adapter, image: Image.Image, render_dir: Path | None = None) -> tuple[float, object]:
    start = time.perf_counter()
    prediction = adapter.predict(image)
    anomaly = prediction.anomaly_map.squeeze().numpy()
    mask = prediction.mask.squeeze().numpy()
    raw = np.asarray(image.convert("RGB").resize((anomaly.shape[1], anomaly.shape[0])))
    # Same output-boundary work as the Gradio callback, excluding JSON/UI HTTP.
    build_heatmap_overlay(raw, normalize_map(anomaly))
    build_mask_overlay(raw, mask)
    return (time.perf_counter() - start) * 1000, prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=ROOT / "artifacts" / "patchcore_experiments" / "resnet18_bottle_seed42" / "model.pt")
    parser.add_argument("--calibration", type=Path, default=ROOT / "artifacts" / "patchcore_experiments" / "resnet18_bottle_seed42_evaluation" / "results.json")
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=10)
    args = parser.parse_args()
    if args.repetitions < 1 or args.warmups < 1:
        parser.error("repetitions and warmups must be positive")
    dataset = ROOT.parent / "mvtec-ad" / "versions" / "2" / "bottle" / "test"
    normal_path = sorted((dataset / "good").glob("*"))[0]
    defect_folder = next(path for path in sorted(dataset.iterdir()) if path.is_dir() and path.name != "good")
    defect_path = sorted(defect_folder.glob("*"))[0]
    normal, defective = Image.open(normal_path).convert("RGB"), Image.open(defect_path).convert("RGB")

    output = ROOT / "validation_runs" / f"onnx_patchcore_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    pytorch = PatchCoreAdapter(args.artifact, "resnet18", "bottle", "cpu", calibration=args.calibration)
    onnx = OnnxPatchCoreAdapter(args.graph, args.artifact, args.calibration, "bottle")
    samples = {}
    feature_errors: dict[str, float] = {}
    for label, image in (("normal", normal), ("defective", defective)):
        tensor = onnx._preprocess(image)
        with torch.inference_mode():
            torch_features = pytorch.extractor(imagenet_normalize(tensor))
        ort_features = onnx.extract_features(tensor)
        feature_errors[label] = max(
            float((torch_features[layer] - ort_features[layer]).abs().max())
            for layer in ("layer1", "layer2", "layer3", "layer4")
        )
        reference = pytorch.predict(image)
        candidate = onnx.predict(image)
        samples[label] = {
            "input": str(normal_path if label == "normal" else defect_path),
            "pytorch": {"score": reference.score, "prediction": reference.prediction},
            "onnx_backed": {"score": candidate.score, "prediction": candidate.prediction},
            "score_absolute_error": abs(reference.score - candidate.score),
            "map_max_absolute_error": float((reference.anomaly_map - candidate.anomaly_map).abs().max()),
            "map_mean_absolute_error": float((reference.anomaly_map - candidate.anomaly_map).abs().mean()),
            "decision_agreement": reference.prediction == candidate.prediction,
            "mask_exact_agreement": bool(torch.equal(reference.mask, candidate.mask)),
            "output_dimensions": render(image, candidate, output, label),
        }

    for _ in range(args.warmups):
        e2e(pytorch, normal)
        e2e(onnx, normal)
    torch_ms, onnx_ms, stages = [], [], {"preprocess": [], "onnx_feature_extraction": [], "python_patchcore_scoring": []}
    for _ in range(args.repetitions):
        elapsed, _ = e2e(pytorch, normal)
        torch_ms.append(elapsed)
        elapsed, _ = e2e(onnx, normal)
        onnx_ms.append(elapsed)
        for key in stages:
            stages[key].append(onnx.last_timings_ms[key])
    report = {
        "scope": "ONNX Runtime frozen feature extraction plus unchanged Python/PyTorch PatchCore scoring and visualization",
        "graph": str(args.graph),
        "artifact": str(args.artifact),
        "calibration": str(args.calibration),
        "provider": onnx.provider,
        "available_providers": list(available_execution_providers()),
        "feature_max_absolute_error": feature_errors,
        "samples": samples,
        "latency_ms": {
            "boundary": "preloaded adapter: preprocessing + feature extraction + PatchCore scoring + overlays; excludes JSON, HTTP, browser rendering",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "pytorch_patchcore_cpu": percentile(torch_ms),
            "onnx_backed_patchcore_cpu": percentile(onnx_ms),
            "onnx_internal_stage_means": {key: float(np.mean(values)) for key, values in stages.items()},
        },
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output / "validation.json", flush=True)


if __name__ == "__main__":
    main()
