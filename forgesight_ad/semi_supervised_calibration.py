"""Optional mixed-validation threshold selection; fitting remains normal-only.

This module deliberately has no access to fitting or final-evaluation data.
Callers must supply a distinct, labelled development set, and freeze its
threshold before starting an independent final evaluation.
"""
from __future__ import annotations

import numpy as np


def select_threshold(normal_scores, defective_scores, max_fpr: float = 0.02) -> dict:
    """Maximize development-set sensitivity subject to a prespecified FPR cap.

    A sample is classified defective when ``score > threshold``. Equal-TPR
    candidates use the larger threshold, yielding the lower-FPR operating
    point. This is not a test-set optimization utility.
    """
    normal = np.asarray(normal_scores, dtype=float).reshape(-1)
    defect = np.asarray(defective_scores, dtype=float).reshape(-1)
    if len(normal) < 1 or len(defect) < 1 or not np.isfinite(normal).all() or not np.isfinite(defect).all():
        raise ValueError("finite normal and defective validation scores are required")
    if not 0 <= max_fpr < 1:
        raise ValueError("max_fpr must be in [0, 1)")

    # The next representable value gives the valid all-normal-PASS candidate
    # even when all observed scores are equal.
    candidates = np.unique(
        np.concatenate([normal, defect, [np.nextafter(max(normal.max(), defect.max()), np.inf)]])
    )
    rows = []
    for threshold in candidates:
        fpr = float((normal > threshold).mean())
        tpr = float((defect > threshold).mean())
        rows.append((float(threshold), fpr, tpr))
    feasible = [row for row in rows if row[1] <= max_fpr]
    if not feasible:
        raise RuntimeError("No threshold satisfies the prespecified validation FPR constraint")
    threshold, fpr, tpr = max(feasible, key=lambda row: (row[2], row[0]))
    return {
        "threshold": threshold,
        "validation_fpr": fpr,
        "validation_tpr": tpr,
        "validation_fnr": 1 - tpr,
        "normal_validation_count": int(len(normal)),
        "defective_validation_count": int(len(defect)),
        "max_fpr_constraint": max_fpr,
        "objective": "maximize validation TPR subject to validation FPR <= max_fpr; ties use larger threshold",
        "candidate_count": len(rows),
    }
