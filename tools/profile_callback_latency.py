"""Profile warm, local Python callback latency without changing detector behavior.

This intentionally measures the Gradio Python callback only.  It does not
measure HTTP transport or browser rendering.  Stage values are diagnostic
instrumentation around the same adapters used by ``demo_app.infer_detector``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def summary(values: list[float]) -> dict[str, float]:
    """Return stable millisecond summary statistics for a non-empty sample."""
    if not values:
        raise ValueError("Cannot summarize an empty timing series")
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)), "p99": float(np.percentile(array, 99)),
        "min": float(array.min()), "max": float(array.max()),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StageInstrumentation:
    """Temporary timing hooks, restored after each detector family."""

    def __init__(self, adapter, demo_app, detectors_module):
        self.adapter = adapter
        self.demo_app = demo_app
        self.detectors_module = detectors_module
        self.samples: dict[str, list[float]] = {
            "image_conversion_ms": [], "cpu_preprocess_ms": [], "host_to_device_ms": [],
            "feature_extraction_ms": [], "heatmap_render_ms": [], "mask_render_ms": [], "json_ms": [],
        }

    def __enter__(self):
        self._image_tensor = self.detectors_module._image_tensor
        self._forward = self.adapter.extractor.forward
        self._heatmap = self.demo_app.build_heatmap_overlay
        self._mask = self.demo_app.build_mask_overlay
        self._json_dumps = self.demo_app.json.dumps

        def image_tensor(image: Image.Image, image_size: int, device: torch.device):
            start = time.perf_counter(); converted = image.convert("RGB")
            self.samples["image_conversion_ms"].append((time.perf_counter() - start) * 1000)
            transform, _ = self.detectors_module.build_transforms(image_size)
            start = time.perf_counter(); tensor = transform(converted)
            self.samples["cpu_preprocess_ms"].append((time.perf_counter() - start) * 1000)
            _sync(device); start = time.perf_counter()
            tensor = tensor.unsqueeze(0).to(device, non_blocking=device.type == "cuda")
            _sync(device); self.samples["host_to_device_ms"].append((time.perf_counter() - start) * 1000)
            return tensor

        def forward(*args, **kwargs):
            _sync(self.adapter.device); start = time.perf_counter()
            result = self._forward(*args, **kwargs)
            _sync(self.adapter.device)
            self.samples["feature_extraction_ms"].append((time.perf_counter() - start) * 1000)
            return result

        def heatmap(*args, **kwargs):
            start = time.perf_counter(); result = self._heatmap(*args, **kwargs)
            self.samples["heatmap_render_ms"].append((time.perf_counter() - start) * 1000)
            return result

        def mask(*args, **kwargs):
            start = time.perf_counter(); result = self._mask(*args, **kwargs)
            self.samples["mask_render_ms"].append((time.perf_counter() - start) * 1000)
            return result

        def dumps(*args, **kwargs):
            start = time.perf_counter(); result = self._json_dumps(*args, **kwargs)
            self.samples["json_ms"].append((time.perf_counter() - start) * 1000)
            return result

        self.detectors_module._image_tensor = image_tensor
        self.adapter.extractor.forward = forward
        self.demo_app.build_heatmap_overlay = heatmap
        self.demo_app.build_mask_overlay = mask
        self.demo_app.json.dumps = dumps
        return self

    def __exit__(self, *_):
        self.detectors_module._image_tensor = self._image_tensor
        self.adapter.extractor.forward = self._forward
        self.demo_app.build_heatmap_overlay = self._heatmap
        self.demo_app.build_mask_overlay = self._mask
        self.demo_app.json.dumps = self._json_dumps


def profile_family(demo_app, detectors_module, family: str, backbone: str, image: Image.Image, iterations: int, warmups: int) -> dict:
    configuration = demo_app.get_supported_configuration(family, backbone, "bottle")
    # The isolated profiler uses one cache entry, so peak memory represents this
    # detector rather than multiple simultaneously cached GPU models.
    torch.cuda.empty_cache()
    load_start = time.perf_counter()
    adapter = demo_app._cached_adapter(configuration, "cuda:0")
    cold_load_ms = (time.perf_counter() - load_start) * 1000
    lookup_times = []
    for _ in range(10):
        start = time.perf_counter()
        assert demo_app._cached_adapter(configuration, "cuda:0") is adapter
        lookup_times.append((time.perf_counter() - start) * 1000)
    for _ in range(warmups):
        demo_app.infer_detector(family, backbone, "bottle", image, "cuda:0")
    _sync(adapter.device)
    torch.cuda.reset_peak_memory_stats(adapter.device)
    totals: list[float] = []
    inference: list[float] = []
    with StageInstrumentation(adapter, demo_app, detectors_module) as stages:
        for _ in range(iterations):
            _sync(adapter.device); start = time.perf_counter()
            _, _, _, text = demo_app.infer_detector(family, backbone, "bottle", image, "cuda:0")
            _sync(adapter.device); totals.append((time.perf_counter() - start) * 1000)
            inference.append(float(json.loads(text)["inference_ms"]))
    per_stage = {name: summary(values) for name, values in stages.samples.items()}
    accounted_mean = sum(per_stage[name]["mean"] for name in per_stage)
    # Detector inference includes native anomaly scoring and output tensor
    # materialization.  Its feature-extractor portion is reported separately.
    scoring_mean = max(0.0, summary(inference)["mean"] - per_stage["feature_extraction_ms"]["mean"])
    residual_mean = max(0.0, summary(totals)["mean"] - accounted_mean - scoring_mean)
    return {
        "family": family, "backbone": backbone, "iterations": iterations, "warmups": warmups,
        "cold_adapter_load_ms": cold_load_ms, "warm_cache_lookup_ms": summary(lookup_times),
        "total_python_callback_ms": summary(totals), "adapter_inference_ms": summary(inference),
        "stage_ms": per_stage,
        "derived_detector_scoring_and_tensor_materialization_mean_ms": scoring_mean,
        "unattributed_callback_overhead_mean_ms": residual_mean,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(adapter.device)),
        "notes": [
            "Total is the local Python callback: it excludes HTTP transport and browser rendering.",
            "Feature extraction is timed with CUDA synchronization. Scoring is adapter inference minus feature extraction.",
            "Residual includes map/mask device-to-host materialization, normalization, mask-area calculation, and Python dispatch not separately exposed by the adapter contract.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--families", nargs="+", choices=("historical", "v2", "patchcore"), default=("historical", "v2", "patchcore"))
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0:
        raise ValueError("iterations must be positive and warmups non-negative")
    import demo_app
    import forgesight_ad.detectors as detectors_module
    # Do not retain unrelated live models during a measurement family.
    demo_app.DETECTOR_CACHE = detectors_module.DetectorCache(max_entries=1)
    image = Image.open(ROOT.parent / "mvtec-ad" / "versions" / "2" / "bottle" / "test" / "good" / "000.png").convert("RGB")
    output = ROOT / "validation_runs" / f"callback_latency_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    # Historical ResNet-18 is selected for this full-callback profile so all
    # three families can coexist with other Windows GPU consumers on 6 GB VRAM.
    # It remains an historical detector-family measurement, not a ResNet-50
    # feature-statistics comparison.
    profiles = []
    backbones = {"historical": "resnet18", "v2": "resnet50", "patchcore": "resnet18"}
    for family in args.families:
        backbone = backbones[family]
        print(f"profile:{family}:{backbone}:begin", flush=True)
        try:
            profiles.append(profile_family(demo_app, detectors_module, family, backbone, image, args.iterations, args.warmups))
            print(f"profile:{family}:{backbone}:complete", flush=True)
        except Exception as exc:
            profiles.append({"family": family, "backbone": backbone, "status": "failed", "exception": repr(exc)})
            print(f"profile:{family}:{backbone}:failed:{type(exc).__name__}:{exc}", flush=True)
    report = {"protocol": "real bottle image; CUDA batch=1; preloaded warm callback; no HTTP/browser timing", "profiles": profiles}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
