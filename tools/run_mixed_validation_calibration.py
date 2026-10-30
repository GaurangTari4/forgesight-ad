"""Freeze an optional mixed-validation threshold from distinct score files.

This tool never fits a detector and never evaluates a final test set.  Inputs
must be scores from separately labelled development-good and development-defect
sets; their provenance labels are persisted with the threshold decision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forgesight_ad.semi_supervised_calibration import select_threshold


def load_scores(path: Path) -> list[float]:
    """Read either a JSON score list or an object containing ``scores``."""
    value = json.loads(path.read_text(encoding="utf-8"))
    scores = value["scores"] if isinstance(value, dict) and "scores" in value else value
    if not isinstance(scores, list):
        raise ValueError(f"{path} must contain a JSON score list or a 'scores' list")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-scores", type=Path, required=True)
    parser.add_argument("--defect-scores", type=Path, required=True)
    parser.add_argument("--normal-provenance", required=True)
    parser.add_argument("--defect-provenance", required=True)
    parser.add_argument("--max-fpr", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.normal_scores.resolve() == args.defect_scores.resolve():
        parser.error("normal and defective score files must be distinct")
    if not args.normal_provenance.strip() or not args.defect_provenance.strip():
        parser.error("explicit normal and defective provenance labels are required")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite calibration record: {args.output}")
    result = select_threshold(load_scores(args.normal_scores), load_scores(args.defect_scores), args.max_fpr)
    result.update(
        {
            "calibration_mode": "mixed_labelled_development_optional",
            "normal_score_file": str(args.normal_scores),
            "defect_score_file": str(args.defect_scores),
            "normal_provenance": args.normal_provenance,
            "defect_provenance": args.defect_provenance,
            "final_evaluation_used": False,
            "threshold_status": "frozen_for_subsequent_independent_evaluation",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
