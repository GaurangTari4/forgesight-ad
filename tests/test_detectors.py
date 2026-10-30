from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
import pytest
import torch

from forgesight_ad.detectors import (
    DetectorAdapter,
    DetectorCache,
    DetectorCompatibilityError,
    build_detector,
    resolve_device,
)
def test_cache_hit_invalidation_and_eviction(tmp_path:Path):
 p=tmp_path/'a';p.write_text('a');c=DetectorCache(1);n=[];f=lambda:(n.append(1) or object());x=c.get_or_load(('a',),p,f);assert c.get_or_load(('a',),p,f) is x and len(n)==1;p.write_text('ab');assert c.get_or_load(('a',),p,f) is not x;c.get_or_load(('b',),p,f);assert len(c._items)==1


def test_prediction_contract_rejects_nonfinite_and_bad_dimensions(tmp_path: Path):
    adapter = DetectorAdapter(tmp_path / "unused.pt", "resnet18", "bottle", "cpu")
    good = torch.zeros((1, 1, 8, 8))
    result = adapter._validated_prediction("test", torch.tensor([0.25]), 0.5, good, good, 1.0)
    assert result.prediction == "PASS" and result.anomaly_map.shape == (1, 1, 8, 8)
    with pytest.raises(DetectorCompatibilityError, match="non-finite"):
        adapter._validated_prediction("test", torch.tensor([float("nan")]), 0.5, good, good, 1.0)
    with pytest.raises(DetectorCompatibilityError, match="dimensions"):
        adapter._validated_prediction("test", torch.tensor([0.25]), 0.5, good, torch.zeros((1, 8, 8)), 1.0)


def test_invalid_family_and_artifact_are_rejected(tmp_path: Path):
    artifact = tmp_path / "wrong.pt"
    torch.save({"methodology_id": "not-v2"}, artifact)
    with pytest.raises(DetectorCompatibilityError, match="Unsupported detector family"):
        build_detector("unknown", artifact, "resnet18", "bottle", "cpu")
    with pytest.raises(DetectorCompatibilityError):
        build_detector("v2", artifact, "resnet50", "bottle", "cpu")


def test_device_selection_is_explicit():
    assert str(resolve_device("cpu")) == "cpu"
    with pytest.raises(DetectorCompatibilityError):
        resolve_device("mps")


def test_cache_serializes_concurrent_initialization(tmp_path: Path):
    artifact = tmp_path / "artifact"
    artifact.write_text("identity")
    cache = DetectorCache(2)
    loads = []

    def loader():
        loads.append(object())
        time.sleep(0.01)
        return loads[-1]

    with ThreadPoolExecutor(max_workers=6) as executor:
        values = list(executor.map(lambda _: cache.get_or_load(("v2", "cpu"), artifact, loader), range(6)))
    assert len(loads) == 1
    assert all(value is values[0] for value in values)
