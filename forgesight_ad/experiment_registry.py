"""Append-only experiment records and comparable-run leaderboards."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import torch


def evaluation_context(config: Any, categories: list[str]) -> dict[str, Any]:
    return {
        "pipeline": "feature_statistics_v1",
        "categories": sorted(categories),
        "image_size": config.image_size,
        "top_k_ratio": config.top_k_ratio,
        "threshold_percentiles": [config.image_threshold_percentile, config.pixel_threshold_percentile],
        "feature_layers": list(config.feature_layers),
        "feature_weights": list(config.feature_blend_weights),
        "score_weights": list(config.score_blend_weights),
        "background_suppression": config.suppress_background_anomalies,
    }


def context_id(context: dict[str, Any]) -> str:
    encoded = json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def software_environment() -> dict[str, str]:
    return {"python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__}


def write_experiment_record(
    registry_root: Path, run_id: str, context: dict[str, Any], overall: list[dict], per_category: list[dict], run_root: Path
) -> Path:
    registry_root.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "status": "verified_new_run",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "context": context,
        "context_id": context_id(context),
        "software_environment": software_environment(),
        "run_root": str(run_root),
        "overall": overall,
        "per_category": per_category,
    }
    path = registry_root / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    return path


def rebuild_leaderboard(registry_root: Path, output_root: Path, context: dict[str, Any]) -> list[dict]:
    target_context = context_id(context)
    rows: list[dict] = []
    for path in sorted(registry_root.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") == "verified_new_run" and record.get("context_id") == target_context:
            rows.extend(record.get("overall", []))
    latest_by_backbone = {row["backbone"]: row for row in rows}
    leaderboard = sorted(latest_by_backbone.values(), key=lambda row: row["image_roc_auc_mean"], reverse=True)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "verified_leaderboard.json").write_text(json.dumps(leaderboard, indent=2, allow_nan=False), encoding="utf-8")
    return leaderboard
