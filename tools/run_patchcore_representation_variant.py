"""Run one prespecified, normal-only PatchCore representation diagnostic.

This script intentionally never reads MVTec test samples.  It is not a model
selection utility: outputs are isolated resource and calibration diagnostics
for a configuration fixed in ``configs/representation_experiments.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forgesight_ad.autoencoder_pipeline import MVTecTrainDataset, imagenet_normalize
from forgesight_ad.backbone_pipeline import build_resnet_extractor, build_transforms
from forgesight_ad.patchcore_pipeline import (
    METHODOLOGY_ID,
    anomaly_map,
    greedy_coreset,
    patch_embeddings,
    save_artifact,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bottle")
    parser.add_argument("--bank-size", type=int, default=512)
    parser.add_argument("--candidate-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warm-repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.bank_size <= 0 or args.candidate_size <= 0 or args.warm_repetitions <= 0:
        parser.error("bank-size, candidate-size, and warm-repetitions must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dataset_root = ROOT.parent / "mvtec-ad" / "versions" / "2"
    source_manifest = ROOT / "artifacts" / "v2_experiments" / f"resnet50_{args.category}_v2_seed42" / "split_manifest.json"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    fitting_indices = list(manifest["fit_indices"])
    calibration_indices = list(manifest["calibration_indices"])
    if set(fitting_indices).intersection(calibration_indices):
        raise ValueError("source split manifest leaks calibration samples into fitting")

    transform, _ = build_transforms(512)
    train = MVTecTrainDataset(dataset_root, args.category, transform)
    if sorted(fitting_indices + calibration_indices) != list(range(len(train))):
        raise ValueError("source split manifest does not partition training-good images")

    output = ROOT / "artifacts" / "representation_experiments" / (
        f"patchcore_{args.category}_r18_l23_bank{args.bank_size}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    output.mkdir(parents=True, exist_ok=False)
    split_manifest = {
        "protocol_id": "representation_experiments_v1_prespecified",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "seed": args.seed,
        "fit_indices": fitting_indices,
        "calibration_indices": calibration_indices,
        "fit_paths": [str(train.image_paths[index]) for index in fitting_indices],
        "calibration_paths": [str(train.image_paths[index]) for index in calibration_indices],
        "test_data_accessed": False,
    }
    split_path = output / "split_manifest.json"
    split_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    model, source = build_resnet_extractor("resnet18", device, allow_random_init=False)
    layers = ("layer2", "layer3")
    loader = DataLoader(
        Subset(train, fitting_indices), batch_size=2, shuffle=False, num_workers=0, pin_memory=device.type == "cuda"
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    fit_start = time.perf_counter()
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, _ in loader:
            features = model(imagenet_normalize(images.to(device, non_blocking=device.type == "cuda")))
            patches, _ = patch_embeddings(features, layers)
            chunks.append(patches.cpu())
    patches = torch.cat(chunks)
    candidate_count = min(args.candidate_size, len(patches))
    candidate_indices = torch.linspace(0, len(patches) - 1, candidate_count).long()
    candidates = patches[candidate_indices].to(device)
    bank = candidates[greedy_coreset(candidates, args.bank_size, args.seed)].cpu().contiguous()
    sync(device)
    fit_seconds = time.perf_counter() - fit_start

    payload = {
        "methodology_id": METHODOLOGY_ID,
        "category": args.category,
        "backbone_name": "resnet18",
        "backbone_source": source,
        "feature_layers": list(layers),
        "image_size": 512,
        "embedding_normalization": "l2",
        "memory_bank": bank,
        "coreset": {
            "algorithm": "deterministic_greedy_kcenter_on_uniform_candidate_subset",
            "candidate_count": candidate_count,
            "selected_count": len(bank),
        },
        "seed": args.seed,
        "split_manifest_sha256": sha256(split_path),
        "calibration": {"status": "normal_only_diagnostic", "indices": calibration_indices},
        "fit_images": len(fitting_indices),
        "total_patches": len(patches),
        "fit_seconds": fit_seconds,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "experiment_note": "prespecified representation diagnostic; never rank with previously examined MVTec test labels",
    }
    artifact = output / "model.pt"
    save_artifact(artifact, payload)

    calibration_loader = DataLoader(
        Subset(train, calibration_indices), batch_size=1, shuffle=False, num_workers=0, pin_memory=device.type == "cuda"
    )
    calibration_scores: list[float] = []
    first_image = None
    with torch.inference_mode():
        for images, _ in calibration_loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            if first_image is None:
                first_image = images
            score = anomaly_map(model, images, bank, layers).flatten(1).max(1).values.item()
            calibration_scores.append(float(score))
    assert first_image is not None
    for _ in range(3):
        anomaly_map(model, first_image, bank, layers)
    sync(device)
    elapsed_ms = []
    for _ in range(args.warm_repetitions):
        sync(device)
        start = time.perf_counter()
        anomaly_map(model, first_image, bank, layers)
        sync(device)
        elapsed_ms.append((time.perf_counter() - start) * 1000)

    resources = {
        "protocol": "prespecified representation diagnostic; fitting/calibration use disjoint training-good samples only",
        "category": args.category,
        "bank_size": args.bank_size,
        "candidate_size": candidate_count,
        "fit_images": len(fitting_indices),
        "calibration_images": len(calibration_indices),
        "fit_seconds": fit_seconds,
        "peak_cuda_memory_bytes": payload["peak_cuda_memory_bytes"],
        "artifact_bytes": artifact.stat().st_size,
        "layers": list(layers),
        "backbone_source": source,
        "normal_calibration": {
            "scores": calibration_scores,
            "mean": float(np.mean(calibration_scores)),
            "std": float(np.std(calibration_scores)),
            "q98_higher": float(np.quantile(calibration_scores, 0.98, method="higher")),
        },
        "warm_gpu_inference_ms": {
            "repetitions": len(elapsed_ms),
            "mean": float(np.mean(elapsed_ms)),
            "median": float(np.median(elapsed_ms)),
            "p95": float(np.quantile(elapsed_ms, 0.95)),
        },
        "test_data_accessed": False,
    }
    (output / "resources.json").write_text(json.dumps(resources, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **resources}, indent=2), flush=True)


if __name__ == "__main__":
    main()
