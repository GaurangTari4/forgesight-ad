"""Freeze a prespecified four-way VisA capsules manifest before fitting.

The split is deterministic and uses official annotation labels only to stratify
the development/final partitions. It never invokes a detector or reads an
evaluation outcome. The generated final partition is sealed in its manifest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def field(row: dict[str, str], *names: str) -> str:
    lowered = {key.lower(): value for key, value in row.items()}
    for name in names:
        if value := lowered.get(name.lower()):
            return value.strip()
    return ""


def resolve_under(root: Path, value: str) -> Path:
    relative = Path(value.replace("\\", "/"))
    # Official VisA CSV paths are rooted at `capsules/`, while selective
    # extraction makes the category itself the supplied root.
    if relative.parts and relative.parts[0].lower() == root.name.lower():
        relative = Path(*relative.parts[1:])
    candidate = (root / relative).resolve()
    if not candidate.is_file() or root.resolve() not in candidate.parents:
        raise FileNotFoundError(f"Annotation references missing or unsafe file: {value}")
    return candidate


def partition(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """70/15/7.5/7.5 normal and 50/50 stratified defect split, predeclared."""
    normal = [row for row in rows if row["image_label"] == "normal"]
    defect: dict[str, list[dict]] = {}
    for row in rows:
        if row["image_label"] == "defect":
            defect.setdefault(row["defect_subtype"], []).append(row)
    rng = random.Random(seed); rng.shuffle(normal)
    count = len(normal); fit_end = int(count * 0.70); calibration_end = fit_end + int(count * 0.15); development_end = calibration_end + int(count * 0.075)
    output = {"fitting": normal[:fit_end], "calibration": normal[fit_end:calibration_end], "development": normal[calibration_end:development_end], "final_evaluation": normal[development_end:]}
    for subtype, group in sorted(defect.items()):
        rng.shuffle(group); pivot = len(group) // 2
        if pivot == 0 or pivot == len(group):
            raise ValueError(f"Defect subtype '{subtype}' cannot be split into development and final partitions")
        output["development"].extend(group[:pivot]); output["final_evaluation"].extend(group[pivot:])
    if not all(output.values()):
        raise ValueError("Prespecified partition produced an empty split")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category-root", type=Path, required=True, help="Extracted raw/capsules directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    root = args.category_root.resolve(); annotation = root / "image_anno.csv"
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {args.output}")
    if not annotation.is_file():
        raise FileNotFoundError(f"Missing official annotation CSV: {annotation}")
    rows: list[dict] = []
    with annotation.open("r", newline="", encoding="utf-8-sig") as handle:
        for source_row in csv.DictReader(handle):
            image_value = field(source_row, "image_path", "image", "path")
            label = field(source_row, "label", "image_label", "class").lower()
            if not label:
                raise ValueError("Official VisA annotation has an empty label")
            image = resolve_under(root, image_value)
            # VisA stores the defect subtype directly in `label` (for example
            # bubble), rather than using a generic anomaly label.
            is_defect = label != "normal"
            mask_value = field(source_row, "mask_path", "mask", "mask_name")
            mask = resolve_under(root, mask_value) if is_defect else None
            if is_defect and mask is None:
                raise ValueError(f"Defect annotation lacks mask for {image_value}")
            subtype = field(source_row, "defect_type", "anomaly_type", "subtype") or label
            rows.append({
                "image": str(image), "image_label": "defect" if is_defect else "normal",
                "mask": str(mask) if mask else None,
                "defect_subtype": subtype if is_defect else None,
                "source_group": field(source_row, "object_id", "session", "group") or sha256(image),
                "official_annotation_row": source_row,
            })
    splits = partition(rows, args.seed)
    for records in splits.values():
        for record in records:
            record.pop("official_annotation_row", None)
            if record["mask"] is None:
                record.pop("mask")
            if record["defect_subtype"] is None:
                record.pop("defect_subtype")
    manifest = {
        "schema_version": "forgesight-independent-dataset-v1", "dataset_id": "visa_20220922_capsules_independent_v1",
        "final_evaluation_immutable": True, "final_evaluation_sealed_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {"source": "official Amazon Science VisA", "official_archive_sha256": "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362", "annotation_csv_sha256": sha256(annotation), "independent_from_historical_mvtec_tests": True, "split_policy": "prespecified deterministic seed-stratified VisA split before detector fitting/evaluation", "seed": args.seed, "physical_group_metadata": "official annotation lacks guaranteed physical-object/session grouping; source_group falls back to image SHA-256"},
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    training_view = args.output.with_name(args.output.stem + "_training_view.json")
    development_view = args.output.with_name(args.output.stem + "_development_view.json")
    for view, permitted in ((training_view, ("fitting", "calibration")), (development_view, ("development",))):
        if view.exists():
            raise FileExistsError(f"Refusing to overwrite split view: {view}")
        view.write_text(json.dumps({"parent_manifest": str(args.output), "parent_manifest_sha256": sha256(args.output), "dataset_id": manifest["dataset_id"], "permitted_splits": list(permitted), "splits": {name: splits[name] for name in permitted}}, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "training_view": str(training_view), "development_view": str(development_view), "counts": {name: len(items) for name, items in splits.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
