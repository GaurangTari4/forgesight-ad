from pathlib import Path

import pytest

from forgesight_ad.category_workflows import discover_categories, macro_average, normal_split, write_json_new


def test_category_discovery_and_deterministic_isolated_split(tmp_path: Path):
    (tmp_path / "bottle" / "train" / "good").mkdir(parents=True)
    (tmp_path / "bottle" / "test").mkdir()
    assert discover_categories(tmp_path) == ["bottle"]
    fit, calibration = normal_split(10, 42)
    assert set(fit).isdisjoint(calibration)
    assert sorted(fit + calibration) == list(range(10))
    assert normal_split(10, 42) == (fit, calibration)


def test_new_records_are_immutable_and_macro_uses_completed_only(tmp_path: Path):
    path = tmp_path / "record.json"
    write_json_new(path, {"status": "complete"})
    with pytest.raises(FileExistsError):
        write_json_new(path, {})
    record = {"status": "complete", "image_roc_auc": 1.0, "pixel_roc_auc": 0.9, "image_average_precision": 1.0, "pixel_average_precision": 0.8, "fpr": 0.0, "fnr": 0.1}
    result = macro_average([record, {"status": "failed"}])
    assert result["completed_categories"] == 1 and result["image_roc_auc"] == 1.0
