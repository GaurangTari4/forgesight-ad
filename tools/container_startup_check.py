"""Fail early with actionable errors before a ForgeSight container starts."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def startup_issues(
    requested_device: str,
    cuda_available: bool,
    artifact_root: Path,
    historical_root: Path,
    families: set[str] | None = None,
) -> list[str]:
    """Return actionable deployment preflight errors without launching Gradio."""
    issues: list[str] = []
    if requested_device in {"cuda", "cuda:0"} and not cuda_available:
        issues.append(
            "CUDA was requested but is unavailable; enable WSL2/NVIDIA GPU support or set FORGESIGHT_DEVICE=cpu."
        )
    if not artifact_root.is_dir():
        issues.append("/opt/forgesight/artifacts is missing; mount the repository artifacts read-only.")
    if not historical_root.is_dir():
        issues.append("historical artifact mount is missing; set FORGESIGHT_ARTIFACTS_ROOT to its read-only mount path.")
    if families is not None:
        missing = {"historical", "v2", "patchcore"}.difference(families)
        if missing:
            issues.append(f"required detector artifacts are unavailable for {sorted(missing)}.")
    return issues


def main() -> None:
    import torch

    requested = os.environ.get("FORGESIGHT_DEVICE", "auto")
    artifact_root = ROOT / "artifacts"
    historical_root = Path(os.environ.get("FORGESIGHT_ARTIFACTS_ROOT", ROOT.parent / "artifacts_resnet_compare"))
    basic_issues = startup_issues(requested, torch.cuda.is_available(), artifact_root, historical_root)
    if basic_issues:
        raise SystemExit("ForgeSight startup failed: " + " ".join(basic_issues))
    # Import after mount checks so errors identify the actual deployment cause.
    from demo_app import detector_families, detector_registry

    entries = detector_registry()
    families = detector_families()
    issues = startup_issues(requested, torch.cuda.is_available(), artifact_root, historical_root, set(families))
    if issues or not entries:
        raise SystemExit("ForgeSight startup failed: " + " ".join(issues or ["no detector artifact configurations were discovered."]))
    print(f"ForgeSight startup check passed: {len(entries)} configurations; device={requested}; families={families}", flush=True)


if __name__ == "__main__":
    main()
