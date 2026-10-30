import pytest

torch = pytest.importorskip("torch")

from forgesight_ad.feature_statistics_v2 import METHODOLOGY_ID, fit_statistics, score


class Extractor:
    def __call__(self, images):
        return {"layer1": images[:, :1]}


def test_v2_fits_and_scores_the_same_unweighted_representation():
    normal = torch.ones(2, 3, 4, 4)
    stats = fit_statistics(Extractor(), [normal], ("layer1",), 1e-6)
    assert stats["methodology_id"] == METHODOLOGY_ID
    anomaly_map = score(Extractor(), normal, stats, ("layer1",), (1.0,), 1e-6)
    assert torch.allclose(anomaly_map, torch.zeros_like(anomaly_map), atol=1e-6)


def test_v2_rejects_legacy_statistics():
    with pytest.raises(ValueError, match="v2 statistics"):
        score(Extractor(), torch.ones(1, 3, 4, 4), {"target_size": (4, 4), "layers": {}}, ("layer1",), (1.0,), 1e-6)
