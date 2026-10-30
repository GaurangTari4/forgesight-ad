import json
from pathlib import Path

import numpy as np
import pytest

from tools.compare_deployment_parity import compare


def _write_capture(path: Path, *, score: float = 1.0, value: float = 0.5, artifact_hash: str = "artifact") -> None:
    sample = {
        "summary": {"image_score": score, "prediction": "PASS"},
        "anomaly_map_dimensions": [1, 1, 2, 2],
    }
    record = {
        "family": "patchcore",
        "backbone": "resnet18",
        "category": "bottle",
        "artifact": {"sha256": artifact_hash},
        "inputs": {"normal": {"sha256": "normal"}, "defective": {"sha256": "defect"}},
        "normal": sample,
        "defective": sample,
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    np.savez_compressed(
        path.with_suffix(".npz"),
        normal_map=np.full((1, 1, 2, 2), value, dtype=np.float32),
        defective_map=np.full((1, 1, 2, 2), value, dtype=np.float32),
        normal_mask=np.zeros((1, 1, 2, 2), dtype=np.uint8),
        defective_mask=np.zeros((1, 1, 2, 2), dtype=np.uint8),
    )


def test_deployment_parity_accepts_declared_cross_environment_tolerance(tmp_path):
    native, container = tmp_path / "native.json", tmp_path / "container.json"
    _write_capture(native)
    _write_capture(container, score=1.00005, value=0.5001)
    result = compare(native, container)
    assert result["all_passed"]
    assert result["samples"]["normal"]["mask_exact_agreement"]


def test_deployment_parity_rejects_different_artifacts(tmp_path):
    native, container = tmp_path / "native.json", tmp_path / "container.json"
    _write_capture(native)
    _write_capture(container, artifact_hash="other")
    with pytest.raises(ValueError, match="Artifact hashes differ"):
        compare(native, container)
