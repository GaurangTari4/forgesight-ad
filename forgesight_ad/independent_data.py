"""Strict manifest validation for future independent inspection datasets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


SPLITS = ("fitting", "calibration", "development", "final_evaluation")


class DatasetManifestError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(root: Path, value: str, field: str) -> Path:
    candidate = (root / value).resolve()
    if not candidate.is_file():
        raise DatasetManifestError(f"Missing {field}: {candidate}")
    parts = {part.lower() for part in candidate.parts}
    if "mvtec-ad" in parts and "test" in parts:
        raise DatasetManifestError("Previously examined MVTec test data cannot be declared independent")
    return candidate


def validate_manifest(path: Path) -> dict[str, Any]:
    """Validate explicit split assignments and return resolved, hashed records."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetManifestError(f"Could not read dataset manifest: {path}") from exc
    if raw.get("schema_version") != "forgesight-independent-dataset-v1":
        raise DatasetManifestError("Unsupported or missing manifest schema_version")
    provenance = raw.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("independent_from_historical_mvtec_tests", False):
        raise DatasetManifestError("Manifest must attest independence from previously examined MVTec tests")
    if not raw.get("final_evaluation_immutable", False):
        raise DatasetManifestError("final_evaluation_immutable must be true")
    if not isinstance(raw.get("final_evaluation_sealed_at"), str) or not raw["final_evaluation_sealed_at"].strip():
        raise DatasetManifestError("final_evaluation_sealed_at must record when the final partition was frozen")
    split_data = raw.get("splits")
    if not isinstance(split_data, dict) or set(split_data) != set(SPLITS):
        raise DatasetManifestError(f"splits must contain exactly: {', '.join(SPLITS)}")
    seen: dict[str, str] = {}
    source_groups: dict[str, str] = {}
    resolved: dict[str, list[dict]] = {}
    root = path.parent
    for split in SPLITS:
        rows = split_data[split]
        if not isinstance(rows, list) or not rows:
            raise DatasetManifestError(f"Split '{split}' must be a non-empty list")
        resolved[split] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or row.get("image_label") not in {"normal", "defect"}:
                raise DatasetManifestError(f"{split}[{index}] needs image_label normal or defect")
            image = _path(root, str(row.get("image", "")), f"{split}[{index}].image")
            try:
                with Image.open(image) as decoded:
                    image_size = decoded.size
                    decoded.verify()
            except Exception as exc:
                raise DatasetManifestError(f"Unreadable image: {image}") from exc
            digest = sha256(image)
            if digest in seen:
                raise DatasetManifestError(f"Duplicate image content across splits: {split}[{index}] duplicates {seen[digest]}")
            seen[digest] = f"{split}[{index}]"
            source_group = row.get("source_group")
            if source_group:
                prior_group_split = source_groups.get(str(source_group))
                if prior_group_split and prior_group_split != split:
                    raise DatasetManifestError(
                        f"Source group crosses partitions: {source_group!r} in {prior_group_split} and {split}"
                    )
                source_groups[str(source_group)] = split
            mask_value = row.get("mask")
            if row["image_label"] == "defect" and not mask_value:
                raise DatasetManifestError(f"Defect record {split}[{index}] requires a pixel mask")
            mask_path = None
            mask_digest = None
            if mask_value:
                mask_path = _path(root, str(mask_value), f"{split}[{index}].mask")
                try:
                    with Image.open(mask_path) as decoded:
                        if decoded.size != image_size:
                            raise DatasetManifestError(f"Mask dimensions differ from image: {mask_path}")
                        decoded.verify()
                except DatasetManifestError:
                    raise
                except Exception as exc:
                    raise DatasetManifestError(f"Unreadable mask: {mask_path}") from exc
                mask_digest = sha256(mask_path)
            if row["image_label"] == "defect" and not row.get("defect_subtype"):
                raise DatasetManifestError(f"Defect record {split}[{index}] requires defect_subtype")
            resolved[split].append({
                **row, "image_path": str(image), "image_sha256": digest,
                "mask_path": str(mask_path) if mask_path else None, "mask_sha256": mask_digest,
                "image_size": list(image_size),
            })
    return {"manifest_path": str(path.resolve()), "manifest_sha256": sha256(path), "dataset_id": raw.get("dataset_id"), "provenance": provenance, "splits": resolved}
