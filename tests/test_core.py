from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from forgesight_ad import backbone_pipeline as pipeline
from forgesight_ad.checkpoints import CheckpointCompatibilityError, validate_feature_statistics_checkpoint
from forgesight_ad.experiment_registry import evaluation_context, rebuild_leaderboard, write_experiment_record


def config(tmp_path: Path, **kwargs):
    return pipeline.ResNetBackboneConfig(
        project_root=tmp_path,
        dataset_root=tmp_path / "data",
        artifacts_root=tmp_path / "artifacts",
        image_size=32,
        batch_size=1,
        heatmap_blur_kernel=3,
        mask_smoothing_kernel=3,
        **kwargs,
    )


def checkpoint_payload(backbone: str = "resnet18"):
    return {
        "backbone_name": backbone,
        "config": {"image_size": 32, "feature_layers": ["layer1"]},
        "statistics": {"target_size": [4, 4], "layers": {"layer1": {"mean": torch.zeros(2, 4, 4), "std": torch.ones(2, 4, 4)}}},
        "image_threshold": 1.0,
        "pixel_threshold": 1.0,
    }


def test_supported_backbone_configuration(tmp_path):
    assert "resnet50" in pipeline.SUPPORTED_BACKBONES
    assert config(tmp_path).to_dict()["backbone_name"] == "resnet50"


def test_pretrained_failure_requires_explicit_random_authorization(monkeypatch):
    def broken_constructor(*args, **kwargs):
        raise OSError("weights unavailable")
    monkeypatch.setitem(pipeline.BACKBONE_SPECS, "resnet18", (broken_constructor, object()))
    with pytest.raises(RuntimeError, match="Refusing to silently"):
        pipeline.build_resnet_extractor("resnet18", torch.device("cpu"))


def test_offline_weight_guard_refuses_a_missing_torchvision_cache(monkeypatch, tmp_path):
    class Weights:
        url = "https://weights.invalid/resnet18-test.pth"

    def must_not_download(*args, **kwargs):
        raise AssertionError("offline guard must run before a TorchVision download")

    monkeypatch.setenv("FORGESIGHT_OFFLINE_WEIGHTS", "1")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "torch-cache"))
    monkeypatch.setitem(pipeline.BACKBONE_SPECS, "resnet18", (must_not_download, Weights()))
    with pytest.raises(RuntimeError, match="will not download weights"):
        pipeline.build_resnet_extractor("resnet18", torch.device("cpu"))


def test_checkpoint_compatibility_rejects_wrong_backbone():
    with pytest.raises(CheckpointCompatibilityError, match="does not match"):
        validate_feature_statistics_checkpoint(checkpoint_payload(), "resnet50")


def test_preprocessing_preserves_requested_square_shape(tmp_path):
    image = pytest.importorskip("PIL.Image").new("RGB", (20, 10), color="white")
    transform, _ = pipeline.build_transforms(32)
    assert tuple(transform(image).shape) == (3, 32, 32)


def test_feature_extractor_output_shapes(monkeypatch):
    constructor = lambda weights=None: pytest.importorskip("torchvision.models").resnet18(weights=None)
    monkeypatch.setitem(pipeline.BACKBONE_SPECS, "resnet18", (constructor, None))
    extractor, source = pipeline.build_resnet_extractor("resnet18", torch.device("cpu"), allow_random_init=True)
    features = extractor(torch.randn(1, 3, 64, 64))
    assert source == "pretrained"
    assert set(features) == {"layer1", "layer2", "layer3", "layer4"}
    assert features["layer1"].ndim == 4


def test_anomaly_score_shape(tmp_path):
    class Extractor:
        def __call__(self, images):
            return {"layer1": torch.zeros(images.shape[0], 2, 4, 4)}
    cfg = config(tmp_path, feature_layers=("layer1",), feature_blend_weights=(1.0,), score_blend_weights=(1.0,), suppress_background_anomalies=False)
    statistics = {"target_size": (4, 4), "layers": {"layer1": {"mean": torch.zeros(2, 4, 4), "std": torch.ones(2, 4, 4)}}}
    anomaly_map, score = pipeline.compute_anomaly_maps(Extractor(), torch.ones(1, 3, 32, 32), statistics, cfg)
    assert tuple(anomaly_map.shape) == (1, 1, 32, 32)
    assert tuple(score.shape) == (1,)


def test_registry_persists_only_comparable_records(tmp_path):
    cfg = config(tmp_path, feature_layers=("layer1",), feature_blend_weights=(1.0,), score_blend_weights=(1.0,))
    context = evaluation_context(cfg, ["bottle"])
    registry = tmp_path / "registry"
    overall = [{"backbone": "resnet18", "image_roc_auc_mean": 0.8}]
    write_experiment_record(registry, "run_a", context, overall, [], tmp_path / "run_a")
    leaderboard = rebuild_leaderboard(registry, tmp_path / "leaderboard", context)
    assert leaderboard == overall
    assert (tmp_path / "leaderboard" / "verified_leaderboard.json").exists()
