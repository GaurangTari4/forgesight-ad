from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
from forgesight_ad.v2_inference import V2ArtifactError, load_v2_artifact


def test_v2_loader_rejects_legacy_or_malformed_artifact(tmp_path: Path):
    path = tmp_path / "bad.pt"
    torch.save({"statistics": {}}, path)
    with pytest.raises(V2ArtifactError, match="not feature_statistics"):
        load_v2_artifact(path, torch.device("cpu"))


def test_v2_loader_rejects_backbone_mismatch_before_model_loading(tmp_path: Path):
    path = tmp_path / "mismatch.pt"
    torch.save({"methodology_id": "feature_statistics_v2_unweighted_layer_stats", "statistics": {"methodology_id": "feature_statistics_v2_unweighted_layer_stats"}, "config": {"backbone_name": "resnet50", "image_size": 512, "feature_layers": ["layer1"], "score_blend_weights": [1.0], "embedding_eps": 1e-6}}, path)
    with pytest.raises(V2ArtifactError, match="Selected backbone"):
        load_v2_artifact(path, torch.device("cpu"), "resnet18")
