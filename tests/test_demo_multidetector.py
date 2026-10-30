"""Unit coverage for the multi-detector Gradio callback contract."""
from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
import pytest
import torch

import demo_app
from forgesight_ad.detectors import DetectorCompatibilityError, Prediction


def test_registry_advertises_only_available_validated_families():
    families = demo_app.detector_families()
    assert {"historical", "v2", "patchcore"}.issubset(families)
    assert all(item.artifact.is_file() for item in demo_app.supported_configurations())
    v2 = demo_app.get_supported_configuration("v2", "resnet50", "bottle")
    assert v2.family == "v2"
    with pytest.raises(DetectorCompatibilityError, match="Unsupported selection"):
        demo_app.get_supported_configuration("v2", "resnet18", "bottle")


def test_detector_dropdown_choices_follow_the_selected_family():
    backbones, categories = demo_app.update_detector_choices("v2")
    assert backbones.value == "resnet50" and categories.value == "bottle"
    assert {"bottle", "cable"}.issubset(demo_app.available_categories("resnet18", "patchcore"))


def test_callback_preserves_four_output_contract_with_adapter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    config = demo_app.DetectorConfiguration("v2", "V2", "resnet50", "bottle", tmp_path / "model.pt")
    config.artifact.write_bytes(b"test")
    prediction = Prediction(
        methodology="feature_statistics_v2_unweighted_layer_stats", backbone="resnet50", category="bottle", device="cpu",
        score=1.25, threshold=2.0, prediction="PASS", anomaly_map=torch.ones((1, 1, 8, 8)),
        mask=torch.zeros((1, 1, 8, 8)), inference_ms=3.5,
    )
    monkeypatch.setattr(demo_app, "get_supported_configuration", lambda *args: config)
    monkeypatch.setattr(demo_app, "_cached_adapter", lambda *args: type("Adapter", (), {"predict": lambda self, image: prediction})())
    result = demo_app.infer_detector("v2", "resnet50", "bottle", Image.new("RGB", (11, 7)), "cpu")
    assert isinstance(result[0], Image.Image)
    assert result[1].shape == result[2].shape == (8, 8, 3)
    assert np.isfinite(result[1]).all() and np.isfinite(result[2]).all()
    summary = json.loads(result[3])
    assert summary["prediction"] == "PASS" and summary["image_score"] == 1.25


def test_callback_rejects_missing_image():
    with pytest.raises(Exception, match="Upload an image"):
        demo_app.infer_detector("v2", "resnet50", "bottle", None, "cpu")


def test_rapid_detector_switching_and_concurrent_mocked_callbacks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    configurations = {
        "historical": demo_app.DetectorConfiguration("historical", "Historical", "resnet18", "bottle", tmp_path / "h.pt"),
        "v2": demo_app.DetectorConfiguration("v2", "V2", "resnet50", "bottle", tmp_path / "v.pt"),
        "patchcore": demo_app.DetectorConfiguration("patchcore", "PatchCore", "resnet18", "bottle", tmp_path / "p.pt"),
    }
    for configuration in configurations.values():
        configuration.artifact.write_bytes(b"test")

    def config_for(family, *_):
        return configurations[family]

    def cached(config, *_):
        prediction = Prediction(
            methodology=config.family, backbone=config.backbone, category="bottle", device="cpu", score=0.1,
            threshold=0.2, prediction="PASS", anomaly_map=torch.ones((1, 1, 8, 8)),
            mask=torch.zeros((1, 1, 8, 8)), inference_ms=1.0,
        )
        return type("Adapter", (), {"predict": lambda self, image: prediction})()

    monkeypatch.setattr(demo_app, "get_supported_configuration", config_for)
    monkeypatch.setattr(demo_app, "_cached_adapter", cached)
    image = Image.new("RGB", (10, 10))
    families = ["historical", "v2", "patchcore", "historical", "patchcore"]
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(lambda family: demo_app.infer_detector(family, configurations[family].backbone, "bottle", image, "cpu"), families))
    assert all(json.loads(result[3])["prediction"] == "PASS" for result in results)
