"""Staged, unbuffered diagnostics for the ForgeSight demo callback."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def marker(log: Path, stage: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {stage}"
    print(line, flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n"); handle.flush(); os.fsync(handle.fileno())

def describe_output(value):
    """Serialize callback values without assuming every ``.size`` is iterable."""
    import numpy as np
    import torch
    record = {"type": type(value).__name__}
    if isinstance(value, Image.Image):
        record.update(kind="pil_image", size=list(value.size), mode=value.mode)
    elif isinstance(value, np.ndarray):
        record.update(kind="numpy_array", shape=list(value.shape), finite=bool(np.isfinite(value).all()))
    elif isinstance(value, torch.Tensor):
        record.update(kind="torch_tensor", shape=list(value.shape), finite=bool(torch.isfinite(value).all().item()))
    elif isinstance(value, np.generic):
        scalar = value.item()
        record.update(kind="scalar", value=scalar)
        if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
            record["finite"] = bool(np.isfinite(scalar))
    elif isinstance(value, (int, float, bool, str)) or value is None:
        record.update(kind="scalar", value=value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            record["finite"] = bool(np.isfinite(value))
    else:
        raise TypeError(f"Unsupported callback output type: {type(value).__name__}")
    return record


def validate_callback_result(result):
    """Validate the Gradio callback contract before writing its report."""
    import numpy as np

    if not isinstance(result, tuple) or len(result) != 4:
        received_length = len(result) if hasattr(result, "__len__") else "unknown"
        raise TypeError(
            "Expected four callback outputs, received "
            f"{type(result).__name__} of length {received_length}"
        )

    input_image, heatmap_overlay, mask_overlay, summary_text = result
    if not isinstance(input_image, Image.Image):
        raise TypeError(f"Callback output 0 must be PIL.Image.Image, got {type(input_image).__name__}")
    for index, overlay in ((1, heatmap_overlay), (2, mask_overlay)):
        if not isinstance(overlay, np.ndarray):
            raise TypeError(f"Callback output {index} must be numpy.ndarray, got {type(overlay).__name__}")
        if overlay.ndim != 3 or overlay.shape[2] != 3:
            raise ValueError(f"Callback output {index} must be an HxWx3 image, got {list(overlay.shape)}")
        if not np.isfinite(overlay).all():
            raise ValueError(f"Callback output {index} contains non-finite values")
    if heatmap_overlay.shape != mask_overlay.shape:
        raise ValueError("Heatmap and mask overlay dimensions differ")
    if input_image.size != (heatmap_overlay.shape[1], heatmap_overlay.shape[0]):
        raise ValueError("Input image and overlay dimensions differ")
    if not isinstance(summary_text, str):
        raise TypeError(f"Callback output 3 must be JSON text, got {type(summary_text).__name__}")
    summary = json.loads(summary_text)
    for key in ("image_score", "image_threshold", "pixel_threshold", "mask_area_ratio"):
        if key not in summary or not np.isfinite(summary[key]):
            raise ValueError(f"Summary field {key!r} must be finite")
    return summary

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["startup", "gradio", "app", "bundle", "canonical", "callback"], default="callback")
    parser.add_argument("--log", type=Path, default=ROOT / "validation_runs" / f"demo_diagnostic_{int(time.time())}.log")
    args = parser.parse_args(); args.log.parent.mkdir(parents=True, exist_ok=True)
    marker(args.log, "startup:begin")
    if args.stage == "startup": marker(args.log, "startup:complete"); return
    marker(args.log, "gradio:before-import"); import gradio  # noqa
    marker(args.log, "gradio:after-import")
    if args.stage == "gradio": marker(args.log, "complete"); return
    os.environ["FORGESIGHT_ARTIFACTS_ROOT"] = str(WORKSPACE / "artifacts_resnet_compare")
    marker(args.log, "demo_app:before-import"); import demo_app
    marker(args.log, "demo_app:after-import")
    if args.stage == "app": marker(args.log, "complete"); return
    checkpoint = WORKSPACE / "artifacts_resnet_compare" / "resnet18" / "bottle" / "resnet18_feature_statistics" / "model.pt"
    marker(args.log, "bundle:before-load"); bundle, extractor, stats, cfg = demo_app.load_bundle("resnet18", "bottle")
    marker(args.log, "bundle:after-load")
    if args.stage == "bundle": marker(args.log, "complete"); return
    image = Image.open(WORKSPACE / "mvtec-ad" / "versions" / "2" / "bottle" / "test" / "good" / "000.png")
    if args.stage == "canonical":
        marker(args.log, "canonical:before-inference"); transform, _ = demo_app.safe_build_inference_transforms(cfg.image_size)
        demo_app.compute_anomaly_maps(extractor, transform(image.convert("RGB")).unsqueeze(0).to(demo_app.DEVICE), stats, cfg)
        marker(args.log, "canonical:after-inference"); return
    marker(args.log, "callback:before-inference"); result = demo_app.infer("resnet18", "bottle", image)
    marker(args.log, "callback:after-inference")
    summary = validate_callback_result(result)
    report = {"output_count": len(result), "outputs": [describe_output(value) for value in result], "summary": summary}
    with (args.log.parent / (args.log.stem + ".json")).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    marker(args.log, "callback:return-complete")
if __name__ == "__main__": main()
