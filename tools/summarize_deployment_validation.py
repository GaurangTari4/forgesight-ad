"""Create a non-destructive index for a ForgeSight Docker validation run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _records(directory: Path) -> dict[str, dict]:
    if not directory.exists():
        return {}
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))}


def build_summary(run_dir: Path) -> dict:
    captures = _records(run_dir / "container")
    profiles = _records(run_dir / "container_profiles")
    native_profiles = _records(run_dir / "native_profiles")
    parity = _records(run_dir / "parity")
    first = next(iter(captures.values()), None)
    return {
        "purpose": "Read-only Docker deployment parity and warmed-callback validation",
        "environment": first["environment"] if first else None,
        "captured_configurations": sorted(captures),
        "input_hashes_and_artifacts": {
            key: {"inputs": record["inputs"], "artifact": record["artifact"]}
            for key, record in captures.items()
        },
        "parity": parity,
        "container_warm_profiles": profiles,
        "native_warm_profiles": native_profiles,
        "scope": {
            "artifact_mutation": "None; detector artifacts were mounted read-only.",
            "http_and_browser_timing": "Not measured by the callback verifier.",
            "visualization_semantics": "Heatmaps are contrast-normalized per image; PASS/FAIL uses raw image score versus immutable threshold.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    output = args.run_dir / "deployment_validation_summary.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite validation summary: {output}")
    output.write_text(json.dumps(build_summary(args.run_dir), indent=2), encoding="utf-8")
    print(output, flush=True)


if __name__ == "__main__":
    main()
