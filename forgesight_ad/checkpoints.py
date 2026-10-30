"""Safe, explicit compatibility checks for ForgeSight feature-statistic checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot be used by the selected runtime."""


REQUIRED_KEYS = frozenset({"backbone_name", "config", "statistics", "image_threshold", "pixel_threshold"})


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint without enabling arbitrary-object unpickling on supported PyTorch versions."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.6
        payload = torch.load(path, map_location=device)
    except Exception as exc:
        raise CheckpointCompatibilityError(
            f"Could not safely load checkpoint '{path.name}'. Use a trusted ForgeSight checkpoint."
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError("Checkpoint payload must be a dictionary.")
    return payload


def validate_feature_statistics_checkpoint(payload: dict[str, Any], selected_backbone: str | None = None) -> None:
    missing = REQUIRED_KEYS.difference(payload)
    if missing:
        raise CheckpointCompatibilityError(f"Checkpoint is missing required keys: {', '.join(sorted(missing))}.")
    checkpoint_backbone = str(payload["backbone_name"])
    if selected_backbone and checkpoint_backbone != selected_backbone:
        raise CheckpointCompatibilityError(
            f"Selected backbone '{selected_backbone}' does not match checkpoint backbone '{checkpoint_backbone}'."
        )
    config = payload["config"]
    if not isinstance(config, dict) or int(config.get("image_size", 0)) <= 0:
        raise CheckpointCompatibilityError("Checkpoint has no valid preprocessing image_size.")
    statistics = payload["statistics"]
    if not isinstance(statistics, dict) or not statistics:
        raise CheckpointCompatibilityError("Checkpoint has no feature-statistics payload.")
    layers = tuple(config.get("feature_layers", ("layer1", "layer2", "layer3", "layer4")))
    layer_statistics = statistics.get("layers") if isinstance(statistics, dict) else None
    if not isinstance(layer_statistics, dict) or "target_size" not in statistics:
        raise CheckpointCompatibilityError("Checkpoint feature statistics have an unsupported format.")
    missing_layers = [layer for layer in layers if layer not in layer_statistics]
    if missing_layers:
        raise CheckpointCompatibilityError(
            f"Checkpoint statistics do not contain configured layers: {', '.join(missing_layers)}."
        )
