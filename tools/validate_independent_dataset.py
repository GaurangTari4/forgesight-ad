"""Validate a future independent inspection dataset manifest without mutation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forgesight_ad.independent_data import DatasetManifestError, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        validated = validate_manifest(args.manifest)
    except DatasetManifestError as exc:
        raise SystemExit(f"Independent dataset validation failed: {exc}") from exc
    report = {
        "status": "valid",
        "dataset_id": validated["dataset_id"],
        "manifest_path": validated["manifest_path"],
        "manifest_sha256": validated["manifest_sha256"],
        "split_counts": {name: len(rows) for name, rows in validated["splits"].items()},
        "provenance": validated["provenance"],
        "records": validated["splits"],
    }
    if args.report:
        if args.report.exists():
            raise FileExistsError(f"Refusing to overwrite validation report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
