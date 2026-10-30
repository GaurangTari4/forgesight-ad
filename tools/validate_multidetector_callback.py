"""Read-only direct Gradio-callback validation for the three detector families."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    print("callback-validation:before-demo-import", flush=True)
    import demo_app
    print("callback-validation:after-demo-import", flush=True)

    output = ROOT / "validation_runs" / f"multidetector_callback_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False)
    image = Image.open(ROOT.parent / "mvtec-ad" / "versions" / "2" / "bottle" / "test" / "good" / "000.png").convert("RGB")
    results = []
    for family, backbone in (("historical", "resnet50"), ("v2", "resnet50"), ("patchcore", "resnet18")):
        adapter = demo_app._cached_adapter(demo_app.get_supported_configuration(family, backbone, "bottle"), "cuda:0")
        direct = adapter.predict(image)
        callback = demo_app.infer_detector(family, backbone, "bottle", image, "cuda:0")
        summary = json.loads(callback[3])
        results.append({
            "family": family, "backbone": backbone, "callback_output_count": len(callback),
            "input_size": list(callback[0].size), "heatmap_shape": list(callback[1].shape), "mask_shape": list(callback[2].shape),
            "finite_overlays": bool(np.isfinite(callback[1]).all() and np.isfinite(callback[2]).all()),
            "callback_score": summary["image_score"], "adapter_score": direct.score,
            "score_absolute_error": abs(summary["image_score"] - direct.score),
            "threshold": summary["image_threshold"], "adapter_threshold": direct.threshold,
            "prediction": summary["prediction"], "adapter_prediction": direct.prediction,
            "cache_reused": demo_app._cached_adapter(demo_app.get_supported_configuration(family, backbone, "bottle"), "cuda:0") is adapter,
        })
    report = {"kind": "direct_cuda_gradio_callback_validation", "device": "cuda:0", "results": results}
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
