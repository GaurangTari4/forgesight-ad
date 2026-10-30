import pytest

from forgesight_ad.semi_supervised_calibration import select_threshold
from tools.run_mixed_validation_calibration import load_scores


def test_prespecified_constraint_and_tie_break():
    result = select_threshold([0.1, 0.2, 0.3], [0.25, 0.8], 0.34)
    assert result["validation_fpr"] <= 0.34
    assert result["validation_tpr"] == 1.0
    assert result["normal_validation_count"] == 3
    assert result["defective_validation_count"] == 2


def test_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        select_threshold([], [1.0])
    with pytest.raises(ValueError):
        select_threshold([0.0], [1.0], 1)


def test_reads_synthetic_development_score_fixture(tmp_path):
    source = tmp_path / "scores.json"
    source.write_text('{"scores": [0.1, 0.2]}', encoding="utf-8")
    assert load_scores(source) == [0.1, 0.2]
