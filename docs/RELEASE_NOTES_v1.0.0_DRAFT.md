# ForgeSight v1.0.0 — draft release notes

## Research-prototype release

ForgeSight v1.0.0 packages the reproducible source, detector adapters,
localhost-only Gradio application, Docker-first native PyTorch/CUDA prototype,
tests, and experimental documentation for one-class visual anomaly detection.

### Included

- Historical feature-statistics, corrected v2 feature-statistics, and
  PatchCore adapters with compatibility checks.
- Read-only artifact Docker mounts, offline wheelhouse support, and fail-closed
  pretrained-weight caching.
- Reproducible MVTec and external VisA experimental records and methodology
  documentation.
- 72 local regression tests at release-preparation time.

### Evidence and limitations

- MVTec 15-category macro image/pixel AUROC: v2 `0.948426/0.919948`, PatchCore
  `0.896620/0.889385`.
- VisA capsules frozen PatchCore transfer: image AUROC `0.635550`, FNR `41/51`.
- Historical VisA sealed-final pixel metrics are alignment-limited and were not
  rerun.
- No production qualification is claimed. Accuracy, operational FPR/FNR,
  independent industrial validation, final localization validation, browser/
  concurrency testing, and SBOM/vulnerability review remain incomplete.

### Upgrade notes

The Docker service requires trusted external artifacts and a deliberately
populated read-only TorchVision weight cache. Missing weights fail clearly;
the service does not download weights, train, calibrate, or rewrite artifacts
at startup.
