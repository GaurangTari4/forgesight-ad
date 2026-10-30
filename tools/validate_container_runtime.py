"""Capture immutable-artifact callback evidence inside or outside a container.

The caller supplies read-only normal/defective images. This tool performs no
fitting, calibration, or artifact writes. It writes a JSON report and an NPZ
file with the anomaly maps/masks needed for native-versus-container parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_app import _cached_adapter, get_supported_configuration, infer_detector


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def statistics(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "p95": float(np.quantile(values, 0.95)), "p99": float(np.quantile(values, 0.99))}


def callback_capture(family: str, backbone: str, category: str, image: Image.Image, device: str) -> tuple[dict, object]:
    """Return the Gradio callback contract plus cached-adapter tensor outputs."""
    outputs = infer_detector(family, backbone, category, image, device=device)
    summary = json.loads(outputs[3])
    configuration = get_supported_configuration(family, backbone, category)
    prediction = _cached_adapter(configuration, device).predict(image)
    if not np.isclose(summary["image_score"], prediction.score, rtol=1e-5, atol=1e-5):
        raise RuntimeError("Gradio summary score differs from the cached adapter prediction")
    return {
        "summary": summary,
        "input_dimensions": list(np.asarray(outputs[0]).shape),
        "heatmap_dimensions": list(np.asarray(outputs[1]).shape),
        "mask_overlay_dimensions": list(np.asarray(outputs[2]).shape),
        "anomaly_map_dimensions": list(prediction.anomaly_map.shape),
        "predicted_mask_dimensions": list(prediction.mask.shape),
    }, prediction


def callback_only(family: str, backbone: str, category: str, image: Image.Image, device: str) -> None:
    """Exercise precisely the Python function used by Gradio's click callback."""
    outputs = infer_detector(family, backbone, category, image, device=device)
    json.loads(outputs[3])


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--defective", type=Path, required=True)
    parser.add_argument("--family", default="patchcore", choices=("historical", "v2", "patchcore"))
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite deployment validation report: {args.output}")
    normal, defective = Image.open(args.normal).convert("RGB"), Image.open(args.defective).convert("RGB")
    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for deployment validation but is unavailable")
    configuration = get_supported_configuration(args.family, args.backbone, args.category)
    adapter = _cached_adapter(configuration, args.device)
    for _ in range(args.warmups): callback_only(args.family, args.backbone, args.category, normal, args.device)
    if requested_device.type == "cuda": torch.cuda.reset_peak_memory_stats(requested_device)
    callback_ms, detector_ms = [], []
    for _ in range(args.repetitions):
        sync(requested_device); start = time.perf_counter()
        callback_only(args.family, args.backbone, args.category, normal, args.device)
        sync(requested_device); callback_ms.append((time.perf_counter() - start) * 1000)
        detector_ms.append(float(adapter.predict(normal).inference_ms))
    normal_record, normal_prediction = callback_capture(args.family, args.backbone, args.category, normal, args.device)
    defective_record, defective_prediction = callback_capture(args.family, args.backbone, args.category, defective, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = args.output.with_suffix(".npz")
    np.savez_compressed(arrays_path, normal_map=normal_prediction.anomaly_map.numpy(), normal_mask=normal_prediction.mask.numpy(), defective_map=defective_prediction.anomaly_map.numpy(), defective_mask=defective_prediction.mask.numpy())
    report = {
        "timing_boundaries": {
            "detector_inference_ms": "adapter-reported model/scoring/mask interval; semantics are existing family-specific behavior",
            "python_callback_ms": "preloaded Gradio callback: preprocessing, inference, overlays, JSON; excludes HTTP and browser rendering",
            "http_browser": "not measured by this tool",
        },
        "family": args.family, "backbone": args.backbone, "category": args.category, "device": args.device,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda, "cuda_device": torch.cuda.get_device_name(requested_device) if requested_device.type == "cuda" else None},
        "inputs": {"normal": {"path": str(args.normal), "sha256": sha256(args.normal)}, "defective": {"path": str(args.defective), "sha256": sha256(args.defective)}},
        "artifact": {"path": str(configuration.artifact), "sha256": sha256(configuration.artifact)},
        "normal": normal_record, "defective": defective_record, "arrays": str(arrays_path),
        "warmups": args.warmups, "repetitions": args.repetitions,
        "latency_ms": {"detector_inference": statistics(detector_ms), "python_callback": statistics(callback_ms)},
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(requested_device)) if requested_device.type == "cuda" else 0,
    }
    with args.output.open("x", encoding="utf-8") as handle: json.dump(report, handle, indent=2)
    print(args.output, flush=True)


if __name__ == "__main__": main()
