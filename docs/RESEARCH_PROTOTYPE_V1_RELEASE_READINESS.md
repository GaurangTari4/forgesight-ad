# ForgeSight research-prototype v1 release-readiness audit

Assessment date: 2026-10-31. This is a release-readiness audit for a
reproducible research prototype, **not** a production-qualification claim.

## Detector scope

| Family | Methodological boundary | Evidence status |
| --- | --- | --- |
| Historical feature statistics | Original per-layer feature-statistics checkpoints, five ResNet backbones, historical threshold/mask semantics. | Functional and Docker-parity checked for representative ResNet-18 cases; legacy results are not a complete multi-backbone leaderboard. |
| Corrected feature statistics v2 | ResNet-50, validated methodology ID `feature_statistics_v2_unweighted_layer_stats`, normal-only fitting/calibration. | 15 MVTec-category exploratory benchmark and Docker parity evidence. |
| PatchCore | Frozen ImageNet ResNet features, deterministic normal-only patch coreset and nearest-neighbour maps. | 15 MVTec-category exploratory benchmark, VisA capsules transfer experiment, and Docker parity evidence. |

Artifacts are loaded with methodology/backbone/category checks. The UI does
not silently substitute a family when a selected artifact is incompatible.

## Evidence summary

| Criterion | Status | Verified evidence / limitation |
| --- | --- | --- |
| Reproducible native CUDA container | Verified | Pinned PyTorch 2.5.1/CUDA 12.4 base, offline hashed Linux CPython 3.11 wheelhouse path, `pip check`, import gate, GPU preflight, loopback-only Compose binding. |
| Artifact safety | Verified with local-trust boundary | Artifacts are externally mounted read-only; startup discovers 105 configurations. Host filesystem permissions and provenance remain the operator's responsibility. |
| Docker numerical parity | Verified, representative only | Six native Windows/Docker comparisons (normal/defective bottle and screw across the three families) had captured exact parity. Declared future-compatible tolerances: score `1e-4`, map `2e-4`. |
| MVTec performance | Exploratory; targets unmet overall | v2 macro image/pixel AUROC `0.948426/0.919948`; PatchCore `0.896620/0.889385`. These are previously examined test-set results, not model-selection or production evidence. |
| VisA capsules external transfer | Verified negative image-level result | Frozen PatchCore final image AUROC `0.635550`; FNR `41/51`; observed FPR `0/46` has 95% Wilson upper bound 7.71%. |
| VisA pixel performance | Alignment-limited historical evidence | The first evaluator square-resized 1500x1000 masks although inference letterboxes. Historical sealed-final pixel metrics remain immutable but cannot substantiate pixel-localization performance. The final partition was not rerun. |
| VisA calibration | Verified provenance; ineffective operating point | Normal-only q98-higher threshold `0.9470425248146057` exactly reproduces from 90 calibration scores. Development-only review found 40/49 defects at/below it. No post-hoc threshold change was made. |
| Latency | Verified under a narrow boundary | Container warm Python callback means: historical R18 68.13 ms, v2 R50 79.63 ms, PatchCore R18 41.88 ms. Includes Python preprocessing/inference/rendering/JSON; excludes HTTP and browser rendering. |
| Production reliability | Unmet | All-category MVTec FPR/FNR and VisA FNR miss original goals. No large, access-controlled independent production validation or soak test exists. |

## Docker release contract

The image never trains, recalibrates, writes artifacts, or downloads weights at
startup. Application and detector artifacts are not copied into the image;
Compose mounts them read-only. `FORGESIGHT_OFFLINE_WEIGHTS=1` rejects a missing
TorchVision cached weight before TorchVision can download it. A clean release
operator must provide:

1. The documented Linux CPython 3.11 wheelhouse generated from the hashed lock.
2. Trusted external artifact mounts, read-only.
3. A deliberately populated, read-only TorchVision cache for each backbone to
   be served.
4. Docker Desktop/WSL2 NVIDIA GPU support, or the documented CPU Compose path.

## Git and packaging audit

`.gitignore` excludes environments, artifacts, generated runs, datasets,
archives, model/export binaries, local Torch cache, and wheelhouse directories.
`.dockerignore` excludes Git state, datasets, artifacts, outputs, tests and
documentation while including only the needed Linux wheelhouse in the build
context. The source lock and wheelhouse builder are the reproducible dependency
specification; do not publish an unreviewed local wheelhouse or dataset archive.

Before publication, scan the staged tree for secrets and oversized files,
confirm the release date before adding it to `CITATION.cff`, and review
third-party dataset and pretrained-weight licenses.

## Release decision

**Suitable as a clearly labelled research-prototype v1 after the operator
provides trusted external assets and reviews the working tree. Not suitable for
production qualification.** Blocking production evidence includes independent
industrial data, fixed-protocol sensitivity/specificity confidence intervals,
valid pixel-localization evaluation, browser/HTTP latency, concurrency/soak
testing, artifact-signing/access controls, a complete deployment SBOM or
vulnerability review.
