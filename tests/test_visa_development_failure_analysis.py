import pytest

from tools.analyze_visa_development_failures import distribution, validate_development_view


def test_development_failure_analysis_rejects_a_view_that_can_expose_final_rows():
    with pytest.raises(ValueError, match="development-only"):
        validate_development_view({"permitted_splits": ["development", "final_evaluation"], "splits": {"development": []}})


def test_distribution_reports_tail_statistics_and_rejects_empty_samples():
    result = distribution([0.1, 0.2, 0.3])
    assert result["count"] == 3
    assert result["max"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="empty"):
        distribution([])
