"""Regression coverage for the callback diagnostic report serialization."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


RUNNER_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_demo_callback.py"
SPEC = spec_from_file_location("run_demo_callback", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_describe_output_uses_correct_dimension_attribute():
    assert runner.describe_output(Image.new("RGB", (11, 7)))["size"] == [11, 7]
    assert runner.describe_output(np.zeros((7, 11, 3), dtype=np.uint8))["shape"] == [7, 11, 3]
    assert runner.describe_output(np.float32(1.5)) == {
        "type": "float32", "kind": "scalar", "value": 1.5, "finite": True
    }


def test_validate_callback_result_accepts_the_demo_contract():
    result = (
        Image.new("RGB", (11, 7)),
        np.zeros((7, 11, 3), dtype=np.uint8),
        np.ones((7, 11, 3), dtype=np.uint8),
        '{"image_score": 1.0, "image_threshold": 2.0, "pixel_threshold": 3.0, "mask_area_ratio": 0.5}',
    )
    assert runner.validate_callback_result(result)["image_score"] == 1.0


def test_validate_callback_result_rejects_non_finite_overlay():
    result = (
        Image.new("RGB", (2, 2)),
        np.full((2, 2, 3), np.nan),
        np.zeros((2, 2, 3)),
        '{"image_score": 1.0, "image_threshold": 2.0, "pixel_threshold": 3.0, "mask_area_ratio": 0.5}',
    )
    with pytest.raises(ValueError, match="non-finite"):
        runner.validate_callback_result(result)
