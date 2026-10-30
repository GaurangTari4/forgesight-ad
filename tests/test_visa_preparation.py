from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "tools" / "prepare_visa_capsules_manifest.py"
SPEC = spec_from_file_location("visa_preparation", PATH)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_prespecified_visa_partition_is_deterministic_and_disjoint():
    rows = [{"image_label": "normal", "id": f"n{i}"} for i in range(40)]
    rows += [{"image_label": "defect", "defect_subtype": "scratch", "id": f"d{i}"} for i in range(10)]
    left, right = MODULE.partition([row.copy() for row in rows], 42), MODULE.partition([row.copy() for row in rows], 42)
    assert left == right
    identifiers = [row["id"] for values in left.values() for row in values]
    assert len(identifiers) == len(set(identifiers)) == len(rows)
    assert len(left["development"]) and len(left["final_evaluation"])


def test_prespecified_partition_rejects_unsplittable_defect_subtype():
    rows = [{"image_label": "normal", "id": f"n{i}"} for i in range(20)]
    rows += [{"image_label": "defect", "defect_subtype": "rare", "id": "d"}]
    with pytest.raises(ValueError, match="cannot be split"):
        MODULE.partition(rows, 42)
