from __future__ import annotations

import argparse
import json
from pathlib import Path


BACKBONE_ORDER = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

BACKBONE_PROFILES = {
    "resnet18": {
        "relative_speed": "very_fast",
        "relative_memory": "very_low",
        "vram_recommended_gb": 4,
        "quality_tendency": "baseline",
    },
    "resnet34": {
        "relative_speed": "fast",
        "relative_memory": "low",
        "vram_recommended_gb": 6,
        "quality_tendency": "slightly_better",
    },
    "resnet50": {
        "relative_speed": "medium",
        "relative_memory": "medium",
        "vram_recommended_gb": 8,
        "quality_tendency": "strong_default",
    },
    "resnet101": {
        "relative_speed": "slow",
        "relative_memory": "high",
        "vram_recommended_gb": 10,
        "quality_tendency": "high",
    },
    "resnet152": {
        "relative_speed": "very_slow",
        "relative_memory": "very_high",
        "vram_recommended_gb": 12,
        "quality_tendency": "highest_depth",
    },
}


def recommend_backbone(vram_gb: float, priority: str, latency_target_ms: int | None) -> str:
    # Advisory scoring, not hard blocking: users may still run larger models
    # by reducing batch size / image size.
    scores: dict[str, float] = {}
    for name in BACKBONE_ORDER:
        profile = BACKBONE_PROFILES[name]
        score = 0.0

        # Soft VRAM preference.
        vram_gap = profile["vram_recommended_gb"] - vram_gb
        if vram_gap <= 0:
            score += 2.0
        else:
            score -= (0.4 * vram_gap)

        if priority == "speed":
            score += {
                "resnet18": 4.0,
                "resnet34": 3.0,
                "resnet50": 1.5,
                "resnet101": 0.5,
                "resnet152": 0.0,
            }[name]
        elif priority == "accuracy":
            score += {
                "resnet18": 0.0,
                "resnet34": 1.0,
                "resnet50": 2.0,
                "resnet101": 3.0,
                "resnet152": 4.0,
            }[name]
        else:
            # balanced
            score += {
                "resnet18": 1.5,
                "resnet34": 2.5,
                "resnet50": 3.0,
                "resnet101": 2.4,
                "resnet152": 1.8,
            }[name]

        if latency_target_ms is not None:
            if latency_target_ms <= 80 and name in {"resnet101", "resnet152"}:
                score -= 2.5
            elif latency_target_ms <= 150 and name == "resnet152":
                score -= 1.5

        scores[name] = score

    return max(scores, key=scores.get)


def explain_tradeoffs(chosen: str) -> dict:
    idx = BACKBONE_ORDER.index(chosen)
    smaller = BACKBONE_ORDER[:idx]
    larger = BACKBONE_ORDER[idx + 1 :]
    return {
        "chosen": chosen,
        "expected_changes_if_you_switch_smaller": {
            "faster": True,
            "lower_memory": True,
            "possible_quality_drop": bool(smaller),
            "alternatives": smaller,
        },
        "expected_changes_if_you_switch_larger": {
            "slower": bool(larger),
            "higher_memory": bool(larger),
            "possible_quality_gain": bool(larger),
            "alternatives": larger,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend ResNet backbone from hardware and requirements.")
    parser.add_argument("--vram-gb", type=float, required=True, help="Available GPU VRAM in GB")
    parser.add_argument("--priority", choices=["speed", "balanced", "accuracy"], default="balanced")
    parser.add_argument("--latency-target-ms", type=int, default=None)
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chosen = recommend_backbone(args.vram_gb, args.priority, args.latency_target_ms)

    result = {
        "input": {
            "vram_gb": args.vram_gb,
            "priority": args.priority,
            "latency_target_ms": args.latency_target_ms,
        },
        "recommendation": {
            "backbone": chosen,
            "profile": BACKBONE_PROFILES[chosen],
            "note": "This is a recommendation, not a hard constraint. You can run larger models by reducing batch size/image size.",
        },
        "tradeoffs": explain_tradeoffs(chosen),
    }

    print(json.dumps(result, indent=2))
    if args.save_json:
        args.save_json.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
