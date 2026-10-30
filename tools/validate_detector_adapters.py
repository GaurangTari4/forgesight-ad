"""Run the three concrete adapters against immutable bottle artifacts.

This is a read-only integration validation, not an evaluation or calibration run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forgesight_ad.detectors import build_detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "cuda:0"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "validation_runs" / f"detector_adapters_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    image = next((ROOT.parent / "mvtec-ad" / "versions" / "2" / "bottle" / "test" / "good").glob("*.png"))
    artifacts = (
        ("historical", ROOT.parent / "artifacts_resnet_compare" / "resnet50" / "bottle" / "resnet50_feature_statistics" / "model.pt", "resnet50"),
        ("v2", ROOT / "artifacts" / "v2_experiments" / "resnet50_bottle_v2_seed42" / "model.pt", "resnet50"),
        ("patchcore", ROOT / "artifacts" / "patchcore_experiments" / "resnet18_bottle_seed42" / "model.pt", "resnet18"),
    )
    results = []
    for family, artifact, backbone in artifacts:
        detector = build_detector(family, artifact, backbone, "bottle", args.device)
        prediction = detector.predict(Image.open(image).convert("RGB"))
        results.append({
            "family": family, "artifact": str(artifact), "methodology": prediction.methodology,
            "backbone": prediction.backbone, "category": prediction.category, "device": prediction.device,
            "score": prediction.score, "threshold": prediction.threshold, "prediction": prediction.prediction,
            "map_shape": list(prediction.anomaly_map.shape), "mask_shape": list(prediction.mask.shape),
            "finite": bool(torch.isfinite(prediction.anomaly_map).all() and torch.isfinite(prediction.mask).all()),
            "inference_ms": prediction.inference_ms,
        })
    report = {"kind": "read_only_adapter_validation", "input_image": str(image), "results": results}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
