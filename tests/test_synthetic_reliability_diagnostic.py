from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "tools" / "run_synthetic_reliability_diagnostic.py"
SPEC = spec_from_file_location("synthetic_reliability", PATH)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_synthetic_reliability_summary_is_grouped_and_pairwise():
    records = [
        {"synthetic_type": "scratch", "score_delta": 2.0, "region_to_global_max_ratio": 1.0, "argmax_inside_synthetic_region": True, "inside_mean_over_outside_mean": 2.0, "threshold_mask_iou": 0.5, "threshold_mask_dice": 0.6, "background_false_activation_rate": 0.1, "synthetic_area_pixels": 5, "synthetic_area_ratio": 0.1},
        {"synthetic_type": "scratch", "score_delta": -1.0, "region_to_global_max_ratio": 0.5, "argmax_inside_synthetic_region": False, "inside_mean_over_outside_mean": 1.0, "threshold_mask_iou": 0.25, "threshold_mask_dice": 0.4, "background_false_activation_rate": 0.2, "synthetic_area_pixels": 5, "synthetic_area_ratio": 0.1},
    ]
    report = MODULE.summarize(records)["scratch"]
    assert report["samples"] == 2
    assert report["score_delta_positive_fraction"] == 0.5
    assert report["argmax_inside_synthetic_region_fraction"] == 0.5
