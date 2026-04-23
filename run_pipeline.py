from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from resnet_compare.resnet_backbone_comparison_pipeline import (  # noqa: E402
    ResNetBackboneConfig,
    SUPPORTED_BACKBONES,
    run_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configurable ResNet backbone pipeline.")
    parser.add_argument("--mode", choices=["single", "compare"], default="single")
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--backbones", type=str, default=",".join(SUPPORTED_BACKBONES))
    parser.add_argument("--categories", type=str, default="")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k-ratio", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "mvtec-ad" / "versions" / "2")
    parser.add_argument("--artifacts-root", type=Path, default=PROJECT_ROOT / "artifacts_resnet_compare")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.artifacts_root = args.artifacts_root.resolve()

    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {args.dataset_root}")

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    if not categories:
        categories = sorted(path.name for path in args.dataset_root.iterdir() if path.is_dir())
    if not categories:
        raise ValueError(f"No categories found in dataset root: {args.dataset_root}")

    missing_categories = [c for c in categories if not (args.dataset_root / c).is_dir()]
    if missing_categories:
        raise ValueError(f"Categories not found under dataset root {args.dataset_root}: {missing_categories}")

    if args.mode == "single":
        if args.backbone not in SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone '{args.backbone}'. Supported: {SUPPORTED_BACKBONES}")
        backbone_list = [args.backbone]
    else:
        backbone_list = [b.strip() for b in args.backbones.split(",") if b.strip()]
        for b in backbone_list:
            if b not in SUPPORTED_BACKBONES:
                raise ValueError(f"Unsupported backbone '{b}'. Supported: {SUPPORTED_BACKBONES}")

    cfg = ResNetBackboneConfig(
        project_root=PROJECT_ROOT,
        dataset_root=args.dataset_root,
        artifacts_root=args.artifacts_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k_ratio=args.top_k_ratio,
        seed=args.seed,
    )

    print(f"Mode: {args.mode}")
    print(f"Backbones: {backbone_list}")
    print(f"Categories: {len(categories)}")
    print(f"Category list: {categories}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Artifacts: {args.artifacts_root}")

    run_comparison(cfg, categories, backbone_list)


if __name__ == "__main__":
    main()
