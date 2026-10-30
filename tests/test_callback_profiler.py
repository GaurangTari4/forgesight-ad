"""Tests for deterministic latency-report calculations without CUDA hardware."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "tools" / "profile_callback_latency.py"
SPEC = spec_from_file_location("profile_callback_latency", PATH)
assert SPEC and SPEC.loader
profiler = module_from_spec(SPEC)
SPEC.loader.exec_module(profiler)


def test_summary_reports_expected_percentiles():
    result = profiler.summary([1.0, 2.0, 3.0, 4.0])
    assert result["mean"] == 2.5
    assert result["median"] == 2.5
    assert result["p95"] > result["median"]
    assert result["min"] == 1.0 and result["max"] == 4.0


def test_summary_rejects_empty_series():
    with pytest.raises(ValueError, match="empty"):
        profiler.summary([])
