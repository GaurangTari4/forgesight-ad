"""Compare native and container ForgeSight deployment validation records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_record(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    return record, {key: value for key, value in np.load(path.with_suffix(".npz")).items()}


def compare(native: Path, container: Path, score_atol: float = 1e-4, map_atol: float = 2e-4) -> dict:
    left, left_arrays = load_record(native)
    right, right_arrays = load_record(container)
    for key in ("family", "backbone", "category"):
        if left[key] != right[key]: raise ValueError(f"Configuration mismatch for {key}: {left[key]} != {right[key]}")
    if left["artifact"]["sha256"] != right["artifact"]["sha256"]: raise ValueError("Artifact hashes differ")
    result = {"configuration": {key: left[key] for key in ("family", "backbone", "category")}, "samples": {}}
    for sample in ("normal", "defective"):
        if left["inputs"][sample]["sha256"] != right["inputs"][sample]["sha256"]: raise ValueError(f"Input hash differs for {sample}")
        l_summary, r_summary = left[sample]["summary"], right[sample]["summary"]
        l_map, r_map = left_arrays[f"{sample}_map"], right_arrays[f"{sample}_map"]
        l_mask, r_mask = left_arrays[f"{sample}_mask"], right_arrays[f"{sample}_mask"]
        error = np.abs(l_map - r_map)
        score_error = abs(float(l_summary["image_score"]) - float(r_summary["image_score"]))
        result["samples"][sample] = {
            "score_absolute_error": score_error, "map_max_absolute_error": float(error.max()), "map_mean_absolute_error": float(error.mean()),
            "mask_exact_agreement": bool(np.array_equal(l_mask, r_mask)), "decision_agreement": l_summary["prediction"] == r_summary["prediction"],
            "output_dimension_agreement": left[sample]["anomaly_map_dimensions"] == right[sample]["anomaly_map_dimensions"],
            "within_tolerance": bool(score_error <= score_atol and error.max() <= map_atol),
        }
    result["tolerances"] = {"score_atol": score_atol, "map_atol": map_atol, "rationale": "Cross-environment CUDA/cuDNN reductions can differ slightly; decisions and masks must agree exactly."}
    result["all_passed"] = all(value["within_tolerance"] and value["mask_exact_agreement"] and value["decision_agreement"] and value["output_dimension_agreement"] for value in result["samples"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True); parser.add_argument("--container", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"Refusing to overwrite parity report: {args.output}")
    result = compare(args.native, args.container)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output, flush=True)
    if not result["all_passed"]: raise SystemExit("Native/container parity exceeded declared tolerance")


if __name__ == "__main__": main()
