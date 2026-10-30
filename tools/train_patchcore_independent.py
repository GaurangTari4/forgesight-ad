"""Fit and normal-only calibrate a new frozen-backbone PatchCore experiment.

Consumes only a preparation-generated fitting/calibration view; this prevents
the fitting process from reading development or sealed-final image records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from forgesight_ad.backbone_pipeline import build_resnet_extractor, build_transforms
from forgesight_ad.patchcore_pipeline import METHODOLOGY_ID, anomaly_map, greedy_coreset, patch_embeddings, save_artifact


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


def tensor(path: str, transform, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB")).unsqueeze(0).to(device, non_blocking=device.type == "cuda")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "independent_experiments")
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bank-size", type=int, default=512)
    parser.add_argument("--candidate-size", type=int, default=8192)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    view = json.loads(args.training_view.read_text(encoding="utf-8"))
    if set(view.get("permitted_splits", ())) != {"fitting", "calibration"} or set(view.get("splits", ())) != {"fitting", "calibration"}:
        raise ValueError("Training view must expose exactly fitting and calibration splits")
    fitting, calibration = view["splits"]["fitting"], view["splits"]["calibration"]
    if any(row["image_label"] != "normal" for row in fitting + calibration): raise ValueError("PatchCore fitting/calibration must be normal-only")
    if {row["image"] for row in fitting}.intersection(row["image"] for row in calibration): raise ValueError("Training view leaks calibration image into fitting")
    transform, _ = build_transforms(512); extractor, source = build_resnet_extractor("resnet18", device, allow_random_init=False); layers = ("layer2", "layer3")
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter(); chunks = []
    with torch.inference_mode():
        for row in fitting:
            features = extractor(tensor(row["image"], transform, device))
            patches, _ = patch_embeddings(features, layers); chunks.append(patches.cpu())
    patches = torch.cat(chunks); candidates = patches[torch.linspace(0, len(patches) - 1, min(args.candidate_size, len(patches))).long()].to(device)
    bank = candidates[greedy_coreset(candidates, min(args.bank_size, len(candidates)), args.seed)].cpu().contiguous()
    if device.type == "cuda": torch.cuda.synchronize(device)
    output = args.output_root / f"{view['dataset_id']}_patchcore_r18_bank{len(bank)}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    output.mkdir(parents=True, exist_ok=False); view_digest = digest(args.training_view)
    payload = {"methodology_id": METHODOLOGY_ID, "category": view["dataset_id"], "backbone_name": "resnet18", "backbone_source": source, "feature_layers": list(layers), "image_size": 512, "embedding_normalization": "l2", "memory_bank": bank, "coreset": {"algorithm": "deterministic_greedy_kcenter_on_uniform_candidate_subset", "candidate_count": len(candidates), "selected_count": len(bank)}, "seed": args.seed, "split_manifest_sha256": view_digest, "parent_manifest_sha256": view["parent_manifest_sha256"], "fit_images": len(fitting), "calibration_images": len(calibration), "provenance": "independent VisA training view; no development/final records exposed"}
    model_path = output / "model.pt"; save_artifact(model_path, payload)
    scores = []
    with torch.inference_mode():
        for row in calibration: scores.append(float(anomaly_map(extractor, tensor(row["image"], transform, device), bank, layers).max()))
    threshold = float(np.quantile(scores, 0.98, method="higher"))
    calibration_record = {"methodology_id": METHODOLOGY_ID, "calibration_mode": "normal_only_q98_higher", "threshold": threshold, "score_count": len(scores), "scores": scores, "training_view_sha256": view_digest, "artifact_sha256": digest(model_path), "frozen": True, "test_or_development_data_accessed": False}
    (output / "calibration.json").write_text(json.dumps(calibration_record, indent=2), encoding="utf-8")
    report = {"artifact": str(model_path), "artifact_sha256": digest(model_path), "calibration": str(output / "calibration.json"), "threshold": threshold, "fit_seconds": time.perf_counter() - started, "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0, "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device)}}
    (output / "fit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
