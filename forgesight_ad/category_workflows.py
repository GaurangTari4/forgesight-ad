"""Shared category discovery, split, audit, and aggregation helpers."""
from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Iterable

from PIL import Image


def discover_categories(dataset_root: Path) -> list[str]:
    """Return MVTec-style categories only when their normal training split exists."""
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir() and (path / "train" / "good").is_dir() and (path / "test").is_dir())


def normal_split(count: int, seed: int, calibration_fraction: float = 0.2) -> tuple[list[int], list[int]]:
    if count < 2:
        raise ValueError("At least two normal training images are required for disjoint fit/calibration splits")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be strictly between zero and one")
    indices = list(range(count)); random.Random(seed).shuffle(indices)
    calibration_count = min(count - 1, max(1, round(count * calibration_fraction)))
    calibration = sorted(indices[:calibration_count]); fitting = sorted(indices[calibration_count:])
    if set(fitting).intersection(calibration):
        raise RuntimeError("Normal-only split overlap detected")
    return fitting, calibration


def _images(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


def audit_category(dataset_root: Path, category: str, verify_images: bool = True) -> dict:
    root = dataset_root / category
    train = _images(root / "train" / "good")
    test_good = _images(root / "test" / "good")
    test_defect = [path for path in _images(root / "test") if path.parent.name != "good"]
    masks = _images(root / "ground_truth")
    corrupt: list[str] = []
    mask_issues: list[str] = []
    if verify_images:
        for path in train + test_good + test_defect + masks:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                corrupt.append(str(path))
        for image_path in test_defect:
            expected = root / "ground_truth" / image_path.parent.name / f"{image_path.stem}_mask.png"
            if not expected.is_file():
                mask_issues.append(f"missing:{expected}")
                continue
            try:
                with Image.open(image_path) as image, Image.open(expected) as mask:
                    if image.size != mask.size:
                        mask_issues.append(f"dimension_mismatch:{image_path}:{image.size}!={mask.size}")
            except Exception:
                mask_issues.append(f"unreadable_mask_pair:{image_path}")
    train_set = {path.resolve() for path in train}; test_set = {path.resolve() for path in test_good + test_defect}
    return {
        "category": category, "train_good": len(train), "test_good": len(test_good), "test_defect": len(test_defect),
        "test_total": len(test_good) + len(test_defect), "ground_truth_masks": len(masks),
        "train_test_overlap": sorted(str(path) for path in train_set.intersection(test_set)), "corrupt_files": corrupt, "mask_issues": mask_issues,
        "valid": bool(train and test_good and test_defect and not corrupt and not mask_issues and not train_set.intersection(test_set)),
    }


def write_json_new(path: Path, payload: dict) -> None:
    """Write a new isolated record; overwrite is always an error."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def macro_average(records: Iterable[dict]) -> dict:
    completed = [record for record in records if record.get("status", "complete") == "complete"]
    keys = ("image_roc_auc", "pixel_roc_auc", "image_average_precision", "pixel_average_precision", "fpr", "fnr")
    return {key: sum(float(record[key]) for record in completed) / len(completed) for key in keys if completed and all(key in record for record in completed)} | {"completed_categories": len(completed)}
