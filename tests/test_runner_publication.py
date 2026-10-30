from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "tools" / "run_category_detectors.py"
SPEC = spec_from_file_location("run_category_detectors", PATH)
assert SPEC and SPEC.loader
runner = module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_v2_atomic_publication_rejects_overwrite(tmp_path: Path):
    path = tmp_path / "partial" / "model.pt"
    runner.atomic_torch_save_new(path, {"marker": 1})
    assert path.is_file()
    with pytest.raises(FileExistsError):
        runner.atomic_torch_save_new(path, {"marker": 2})
    assert not list(path.parent.glob("*.tmp"))
