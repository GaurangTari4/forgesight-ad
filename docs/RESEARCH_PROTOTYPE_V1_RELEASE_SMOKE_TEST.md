# ForgeSight research-prototype v1 release smoke test

Executed 2026-10-31 against `forgesight:local` rebuilt from the pinned
PyTorch 2.5.1 / CUDA 12.4 image. This is deployment functionality evidence,
not a detector-performance or production-qualification claim.

## Preconditions and explicit supported set

The read-only TorchVision cache contained only official ResNet-18
`resnet18-f37072fd.pth` and ResNet-50 `resnet50-11ad3fa6.pth`. Therefore this
smoke test deliberately supported only:

| Detector family | Backbone | Category | Artifact SHA-256 |
| --- | --- | --- | --- |
| Historical feature statistics | ResNet-18 | bottle | `f46f7b00e26f748c585f136f01886a30eff42f48b3657a2ca9b832ba0090a67b` |
| Corrected feature statistics v2 | ResNet-50 | bottle | `3872a29dbfb4b863d18b051a447f6c51314c9459e2533cbf6153a75fe4a40594` |
| PatchCore | ResNet-18 | bottle | `a6398bc4a24e4f49ff543bc90191ee33da8e260d36df4120d325504c50360354` |

Historical ResNet-34/101/152 are intentionally not supported by this
hermetic cache until their official TorchVision weights are deliberately
prepopulated according to `DOCKER_DEPLOYMENT.md`.

## Container checks

- `docker compose up --build -d`: completed using the offline wheelhouse.
- Startup preflight: passed with `105 configurations`, `device=cuda:0`, and
  the three registered detector families.
- Docker health: `healthy`.
- HTTP request: `200`, `text/html; charset=utf-8` at `127.0.0.1:7860`.
- Compose inspection: GPU reservation present; artifacts, historical
  artifacts, and TorchVision cache mounted with `rw=false`.
- Standard startup did not train, calibrate, write checkpoints, or download
  pretrained weights.

## Real CUDA callback smoke results

Each execution used the same read-only MVTec bottle `test/good/000.png` and
`test/broken_large/000.png`, a preloaded CUDA adapter, and generated 512x512
anomaly maps/masks. JSON and NPZ evidence are in `validation_runs/`.

| Family | Normal decision | Defective decision | Mean callback ms (3 reps) | Peak CUDA bytes | Evidence |
| --- | --- | --- | ---: | ---: | --- |
| Historical R18 | PASS | FAIL | 67.95 | 337,443,840 | `release_smoke_20261031_historical_resnet18.json` |
| Corrected v2 R50 | PASS | FAIL | 87.13 | 1,239,848,448 | `release_smoke_20261031_v2_resnet50.json` |
| PatchCore R18 | PASS | FAIL | 37.53 | 93,518,848 | `release_smoke_20261031_patchcore_resnet18.json` |

The records include source-image and artifact SHA-256 values, image score,
threshold, decision, output dimensions, CUDA environment, and detector versus
callback timing boundaries. The small three-repetition timing sample is smoke
evidence only, not a new latency benchmark.

## Offline-negative-path check

Requesting historical ResNet-34 bottle produced the explicit error:

> Required pretrained RESNET34 weights are missing from the read-only Torch
> cache ... Prepopulate the cache before deployment; ForgeSight will not
> download weights at inference time.

The command exited nonzero as intended, and before/after SHA-256 inventories
of the cache were identical. This verifies fail-closed behavior without
downloading a model.

## Regression and release checks

- Full pytest suite with an isolated workspace-local base temp: 72 passed.
- `python -m pip check`: passed.
- `python -m compileall -q forgesight_ad tools tests demo_app.py`: passed.
- `CITATION.cff`: not changed. Its repository URL is a placeholder and its
  release date has not been independently confirmed; neither has authoritative
  replacement metadata in this workspace.

## Decision

**Recommend release only as “ForgeSight research-prototype v1”.** Do not call
it production-ready. The known accuracy, FPR/FNR, VisA localization-alignment,
independent-data, security/SBOM, and browser/concurrency limitations in
`RESEARCH_PROTOTYPE_V1_RELEASE_READINESS.md` remain release caveats.
