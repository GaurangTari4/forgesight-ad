"""Export only the frozen feature extractor when ONNX tooling is available.

This is intentionally scoped: feature-statistics scoring, PatchCore nearest
neighbour search, PIL preprocessing, and visualization remain Python-side.
No successful run of this script is evidence of a complete detector export.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forgesight_ad.backbone_pipeline import build_resnet_extractor


def runtime_availability() -> dict[str, bool]:
    """Return availability without importing optional deployment packages."""
    return {
        "onnx": importlib.util.find_spec("onnx") is not None,
        "onnxruntime": importlib.util.find_spec("onnxruntime") is not None,
        "openvino": importlib.util.find_spec("openvino") is not None,
    }


class ExtractorOutputs(nn.Module):
    """ONNX-friendly tuple wrapper around the dictionary-returning extractor."""

    def __init__(self, extractor: nn.Module):
        super().__init__()
        self.extractor = extractor

    def forward(self, images: torch.Tensor):
        features = self.extractor(images)
        return tuple(features[name] for name in ("layer1", "layer2", "layer3", "layer4"))


def write_report(destination: Path, report: dict) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "onnx_export_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="resnet18", choices=("resnet18", "resnet50"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "deployment_exports")
    args = parser.parse_args()
    availability = runtime_availability()
    output = args.output_root / f"feature_extractor_{args.backbone}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    base = {
        "scope": "frozen feature extractor only; detector scoring/preprocessing/postprocessing are not exported",
        "backbone": args.backbone,
        "image_size": args.image_size,
        "opset": args.opset,
        "runtime_availability": availability,
        "openvino_conversion": "not attempted; requires a validated ONNX artifact and OpenVINO runtime/tooling",
    }
    if not availability["onnx"] or not availability["onnxruntime"]:
        base.update(
            status="blocked_missing_optional_dependencies",
            reason="Both onnx and onnxruntime are required for export plus numerical agreement validation.",
        )
        print(write_report(output, base), flush=True)
        return

    device = torch.device(args.device)
    extractor, source = build_resnet_extractor(args.backbone, device, allow_random_init=False)
    wrapper = ExtractorOutputs(extractor).eval()
    fixed_input = torch.linspace(0, 1, 3 * args.image_size * args.image_size, device=device).reshape(
        1, 3, args.image_size, args.image_size
    )
    onnx_path = output / "feature_extractor.onnx"
    output.mkdir(parents=True, exist_ok=False)
    torch.onnx.export(
        wrapper,
        fixed_input,
        onnx_path,
        input_names=["images"],
        output_names=["layer1", "layer2", "layer3", "layer4"],
        opset_version=args.opset,
        dynamo=False,
    )
    import onnxruntime as ort

    with torch.inference_mode():
        reference = [value.detach().cpu().numpy() for value in wrapper(fixed_input)]
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    exported = session.run(None, {"images": fixed_input.cpu().numpy()})
    errors = [float(np.max(np.abs(left - right))) for left, right in zip(reference, exported)]
    base.update(
        status="exported_and_verified",
        pretrained_weight_source=source,
        artifact=str(onnx_path),
        output_shapes=[list(value.shape) for value in reference],
        max_absolute_error_per_output=errors,
        max_absolute_error=max(errors),
    )
    (output / "onnx_export_report.json").write_text(json.dumps(base, indent=2), encoding="utf-8")
    print(output / "onnx_export_report.json", flush=True)


if __name__ == "__main__":
    main()
