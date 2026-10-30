from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from choose_backbone import recommend_backbone  # noqa: E402
from forgesight_ad.backbone_pipeline import (
    ResNetBackboneConfig,
    SUPPORTED_BACKBONES,
    run_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-entry pipeline runner for recommendation, comparison, and demo.")
    parser.add_argument("--workflow", choices=["quick_check", "full_compare", "demo"], default="quick_check")
    parser.add_argument("--vram-gb", type=float, default=8.0)
    parser.add_argument("--priority", choices=["speed", "balanced", "accuracy"], default="balanced")
    parser.add_argument("--latency-target-ms", type=int, default=200)
    parser.add_argument("--backbone", type=str, default="", help="Optional override; if empty, auto recommendation is used.")
    parser.add_argument("--categories", type=str, default="", help="Comma-separated categories. Empty means all.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2, help="Sequential-evaluation default suitable for a 6 GB GPU.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k-ratio", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=Path, required=False)
    parser.add_argument("--artifacts-root", type=Path, required=False)
    parser.add_argument("--allow-random-init", action="store_true")
    return parser.parse_args()


def resolve_categories(dataset_root: Path, categories_csv: str) -> list[str]:
    categories = [item.strip() for item in categories_csv.split(",") if item.strip()]
    if categories:
        return categories
    return sorted(path.name for path in dataset_root.iterdir() if path.is_dir())


def resolve_backbone(args: argparse.Namespace) -> str:
    if args.backbone:
        if args.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone '{args.backbone}'. Supported: {SUPPORTED_BACKBONES}")
        return args.backbone
    return recommend_backbone(args.vram_gb, args.priority, args.latency_target_ms)


def build_config(args: argparse.Namespace) -> ResNetBackboneConfig:
    return ResNetBackboneConfig(
        project_root=PROJECT_ROOT,
        dataset_root=args.dataset_root,
        artifacts_root=args.artifacts_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k_ratio=args.top_k_ratio,
        seed=args.seed,
        allow_random_init=args.allow_random_init,
    )


def run_quick_check(args: argparse.Namespace) -> None:
    chosen_backbone = resolve_backbone(args)
    categories = resolve_categories(args.dataset_root, args.categories)
    if not categories:
        raise ValueError(f"No categories found at {args.dataset_root}")

    # Keep quick check lightweight by defaulting to one category if user did not pass categories.
    if not args.categories:
        categories = [categories[0]]

    print(f"Recommended/Selected backbone: {chosen_backbone}")
    print(f"Quick-check categories: {categories}")
    config = build_config(args)
    run_comparison(config, categories, [chosen_backbone])


def run_full_compare(args: argparse.Namespace) -> None:
    categories = resolve_categories(args.dataset_root, args.categories)
    if not categories:
        raise ValueError(f"No categories found at {args.dataset_root}")
    config = build_config(args)
    print(f"Running full comparison on backbones: {list(SUPPORTED_BACKBONES)}")
    print(f"Categories: {len(categories)}")
    run_comparison(config, categories, list(SUPPORTED_BACKBONES))


def run_demo() -> None:
    from demo_app import launch_demo  # imported lazily so non-demo workflows do not initialize gradio

    launch_demo()


def main() -> None:
    args = parse_args()
    if args.workflow != "demo" and (args.dataset_root is None or args.artifacts_root is None):
        raise ValueError("--dataset-root and --artifacts-root are required for evaluation workflows.")
    args.dataset_root = args.dataset_root.resolve() if args.dataset_root else None
    args.artifacts_root = args.artifacts_root.resolve() if args.artifacts_root else None

    if args.workflow != "demo" and not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    print(f"Workflow: {args.workflow}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Artifacts root: {args.artifacts_root}")

    if args.workflow == "quick_check":
        run_quick_check(args)
    elif args.workflow == "full_compare":
        run_full_compare(args)
    else:
        run_demo()


if __name__ == "__main__":
    main()
