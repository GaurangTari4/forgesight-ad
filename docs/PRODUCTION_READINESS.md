# ForgeSight production-readiness assessment

Assessment date: 2026-10-30. This document distinguishes verified bottle-only
evidence from general production claims. No detector was retrained, refit, or
recalibrated for this assessment.

The v1 release audit is in `RESEARCH_PROTOTYPE_V1_RELEASE_READINESS.md`.
ForgeSight is suitable only as a clearly labelled research prototype; this
document is not a production-qualification certificate.

| Requirement | Current evidence | Status | Remaining validation |
| --- | --- | --- | --- |
| Image AUROC > 0.95 | Bottle: historical ResNet-50 1.000000; corrected v2 ResNet-50 1.000000; PatchCore ResNet-18 0.987302. | Met for exploratory bottle evaluation | Blind evaluation across representative production data and all intended categories. |
| Pixel AUROC > 0.95 | Bottle: historical ResNet-50 0.975086; v2 ResNet-50 0.976032; PatchCore ResNet-18 0.951566. | Met for exploratory bottle evaluation | Per-category blind validation; acceptable defect-localization review. |
| Operational FPR < 2% | Bottle normal test evaluation: all three reported 0/20 false positives. Thresholds came from normal-only calibration. | Insufficient evidence | Confidence interval and longer normal-production run; 20 normal tests cannot establish a low operational FPR. |
| Operational FNR < 0.5% | Bottle: v2 missed 8/63 defects (12.70%); PatchCore missed 9/63 (14.29%). | Not met | New detector/calibration work, followed by blinded validation. Do not tune with this test set. |
| Warm Python callback < 100 ms | After pixel-equivalent rendering optimization: historical ResNet-18 mean 85.23 ms/p95 89.81; v2 ResNet-50 mean 112.76 ms/p95 136.79; PatchCore ResNet-18 mean 54.90 ms/p95 60.22. | Met for historical ResNet-18 and PatchCore under this boundary; not met for v2 | Measure HTTP and browser-rendering latency on deployment hardware; optimize only after defining production SLOs. |
| GPU-only latency | Earlier CUDA-event detector-path measurement: v2 mean 55.85 ms/p95 63.28; PatchCore mean 6.10 ms/p95 6.98. It excludes preprocessing, transfers, overlays, and UI work. | Evidence only | Do not substitute this number for end-to-end latency. |
| Artifact safety | Checkpoints/artifacts use `weights_only=True` where supported; methodology, backbone, category, and preprocessing checks reject incompatible selections. | Met for validated artifacts | Maintain provenance/signatures and trusted-artifact controls in deployment. |
| Gradio detector reliability | Direct CUDA callbacks passed for historical, v2, and PatchCore. User-supplied Windows screenshots confirm manual localhost switching/inference for historical ResNet-18/34/50/101/152, v2 ResNet-50, and PatchCore ResNet-18. `share=False`, `127.0.0.1` binding. | Met for manual local validation | Automated browser regression test and soak/concurrency testing. Manual screenshots are not automated browser evidence. |
| Robustness/error handling | Missing images and unsupported family/backbone/category selections have regression coverage; no silent detector fallback. | Partially met | Corrupt-image, resource-exhaustion, concurrent-request, and long-run tests. |
| Multi-category generalization | Detector adapter artifacts are currently validated in the UI for bottle v2/PatchCore; historical artifacts are registry discovered. | Insufficient evidence | Complete comparable blind runs across all intended MVTec categories and real deployment material. |
| ONNX/OpenVINO readiness | No exported/validated ONNX or OpenVINO runtime artifacts. | Not met | Export frozen extractors, validate numerical equivalence, benchmark target hardware, and document supported operators. |

## Profiling boundary and evidence

Callback profiling used a preloaded detector, a real MVTec bottle normal image,
512x512 preprocessing, CUDA batch size 1, ten warm-ups, and 50 repetitions.
It measures local Python callback work only: image conversion, preprocessing,
host-to-device transfer, detector inference, CPU overlays, and JSON creation.
HTTP transport and browser rendering are deliberately excluded. The profile is
stored in timestamped `validation_runs/callback_latency_*` directories.

| Family/backbone | Mean ms | Median ms | p95 ms | p99 ms | Peak CUDA allocation |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical / ResNet-18 | 85.23 | 86.34 | 89.81 | 91.78 | 337,443,840 bytes |
| Corrected v2 / ResNet-50 | 112.76 | 109.66 | 136.79 | 138.82 | 1,239,848,448 bytes |
| PatchCore / ResNet-18 | 54.90 | 54.69 | 60.22 | 62.61 | 93,518,848 bytes |

The historical family uses ResNet-18 in this callback profile so each family
could be measured in an isolated process on the shared 6 GB GPU. It is not a
like-for-like ResNet-50 comparison. Overlay rendering is a material portion of
the total callback time (about 31-36 ms), so GPU-only benchmarks must remain
separate from this result.

## Rendering optimization and reliability validation

The heatmap renderer previously built a three-channel temporary heatmap and a
separate blended image. It now writes the same three channel equations directly
into one float32 output array. RGB uploads also avoid a redundant PIL RGB
conversion before presentation resizing. The original renderer is retained as a
private reference function for regression comparison.

No detector score, threshold, anomaly map, or predicted mask changed. A
randomized 512x512 pixel comparison found exact uint8 equality between the
reference and optimized heatmap overlays (maximum absolute pixel difference
zero). The following values compare the identical 50-iteration callback
protocol before and after this change; ordinary shared-GPU variation still
applies.

| Family/backbone | Before mean ms | After mean ms | Observed reduction | Heatmap render before/after mean ms |
| --- | ---: | ---: | ---: | ---: |
| Historical / ResNet-18 | 115.59 | 85.23 | 26.3% | 28.38 / 15.66 |
| Corrected v2 / ResNet-50 | 125.06 | 112.76 | 9.8% | 26.65 / 17.64 |
| PatchCore / ResNet-18 | 64.48 | 54.90 | 14.9% | 25.94 / 17.79 |

Regression tests cover exact visualization output, rapid mocked detector
switching, repeated callback outputs, cache initialization serialization, cache
eviction, corrupted/incompatible artifact rejection, missing-image handling,
and explicit CPU/CUDA selection validation. They do **not** prove production
thread safety, GPU-memory stability under long-lived concurrent traffic, or
browser reliability; soak and load tests remain required.

## Publication and deployment blockers

- Validate FPR/FNR with a sufficiently large, blinded production-like set.
- Do not claim generalization from the exploratory bottle test results.
- Add automated browser and concurrency/soak coverage.
- Validate ONNX/OpenVINO only after exports are created; none currently exist.
- Establish artifact signing/access control and a deployment-specific latency SLO.

## Complete 15-category exploratory benchmark

Read-only validation found completed, provenance-consistent artifacts and
evaluation records for all 15 MVTec AD categories for corrected v2 ResNet-50
and PatchCore ResNet-18. The complete per-category table, artifact digests,
confusion matrices, and charts are in
[`MVTEC_15_CATEGORY_BENCHMARK.md`](MVTEC_15_CATEGORY_BENCHMARK.md) and
`artifacts/benchmark_summary/`.

| Method | Macro image AUROC | Macro pixel AUROC | Macro FPR | Macro FNR | Pooled FPR | Pooled FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Corrected v2 / ResNet-50 | 0.948426 | 0.919948 | 0.053135 | 0.290615 | 0.053533 | 0.329094 |
| PatchCore / ResNet-18 | 0.896620 | 0.889385 | 0.041316 | 0.392555 | 0.034261 | 0.426073 |

Neither method meets the all-category image/pixel AUROC target of 0.95, nor
the operational FPR/FNR targets. The most missed defects were v2 zipper (78)
and PatchCore screw (109); the most false alarms were v2 leather (5) and a
five-way PatchCore tie at three. Generic runs preserve normal-only threshold
provenance but do not retain individual calibration-score distributions, so
post-hoc score-distribution claims would require read-only rescoring and must
not be used to select new thresholds. These MVTec results remain exploratory
test-set evidence, not production-inspection reliability validation.

## Original-brief closure update

`ORIGINAL_BRIEF_AUDIT.md` is the requirements-to-evidence matrix for the
original brief. The optional mixed-validation calibration protocol is now
implemented separately from one-class fitting. It selects a threshold on a
distinct labelled development set by maximizing sensitivity subject to a
prespecified FPR cap; it does not consume a final evaluation set. There is no
fresh independent labelled development dataset in this repository, so the
protocol has synthetic-fixture coverage only and no real-data threshold change
has been made.

Deployment export remains incomplete. `tools/export_feature_extractor_onnx.py`
implements an extractor-only ONNX export path and fixed-input ONNX Runtime
agreement check. It deliberately excludes feature-statistics scoring,
PatchCore nearest-neighbour scoring, preprocessing, and visualization. The
current environment has no `onnx`, `onnxruntime`, or OpenVINO package, so this
turn records a blocked feasibility report rather than claiming an exported
detector. Even after a successful extractor export, complete deployment needs
separate scorer, pre/post-processing, numerical-agreement, and target-hardware
latency validation.

One prespecified, isolated PatchCore representation diagnostic was executed
without reading MVTec test data: ResNet-18 layers 2/3, 512-bank, 8,192
candidates, 167 fitting-good and 42 disjoint calibration-good bottle images.
It took 7.337 s, used 147,520,512 peak CUDA bytes, produced a 788,811-byte
artifact, and had a diagnostic warm map mean of 6.910 ms over 10 repetitions.
The normal-only calibration-score q98 was 0.911798. These are resource and
normal-score diagnostics only, not an accuracy comparison or a selected model.

## Docker-first deployment status

Native PyTorch/CUDA Docker is the primary deployment path. The pinned CUDA
PyTorch image, local-only Compose port binding, read-only artifact mounts,
container health check, and startup preflight are implemented. The application
continues to use all three validated adapter families; completed corrected-v2
and PatchCore category artifacts are dynamically registered from their
read-only mount. ONNX/OpenVINO remains optional tooling and is not part of the
container image.

Container execution has now been verified on the recorded Docker
Desktop/WSL2 host. `forgesight:local` built offline, passed its unsuppressed
`pip check` and core-import gate, and `forgesight-local` was healthy with an
HTTP 200 response at the loopback-only endpoint `127.0.0.1:7860`. The startup
preflight selected `cuda:0` and registered 105 detector configurations. Docker
inspection confirmed both artifact mounts are read-only.

Formal native-Windows versus container CUDA parity covers normal and defective
bottle and screw images for historical ResNet-18, corrected v2 ResNet-50, and
PatchCore ResNet-18. All six comparisons had exact captured score, anomaly-map,
mask, decision, and dimension agreement (the documented future-compatible
tolerances are `1e-4` for scores and `2e-4` for maps). The retained validation
run is `validation_runs/docker_deployment_20261031T010042Z/`.

Warm container batch-1 Python-callback means on bottle were 68.13 ms
(historical R18), 79.63 ms (v2 R50), and 41.88 ms (PatchCore R18); corresponding
peak CUDA allocations were 337,443,840, 1,239,848,448, and 93,518,848 bytes.
These boundaries include preprocessing, inference, visualization, and JSON but
exclude HTTP transport and browser rendering. They do not by themselves prove
a complete browser or production SLO.

## Reliability-improvement evidence boundary

The PatchCore ResNet-18 512-memory-bank representation artifact was assessed
only through a prespecified synthetic-normal diagnostic on ten reserved,
disjoint bottle calibration images. All 50 synthetic edits increased their
paired image scores, but known-region peak placement was only 1/10 for thin
scratches and 3/10 for small local edits. This is useful localization diagnosis,
not real-defect performance evidence, and it did not change an artifact or
threshold. No independent labelled development data or untouched final
real-defect data exists in this repository; consequently no improved AUROC,
FPR, FNR, sensitivity, specificity, or defect-subtype confidence interval is
claimed. See `RELIABILITY_IMPROVEMENT_PROTOCOL.md` for the frozen future
selection/evaluation protocol.

Independent-data ingestion and final-evaluation tooling are implemented with
strict four-way manifests, image/mask SHA-256 records, split-content duplicate
rejection, final-partition immutability, MVTec-test reuse rejection, frozen
threshold metrics, subtype breakdowns, Wilson intervals, and warmed latency.
No external dataset has been ingested or evaluated. Consequently the original
AUROC/FPR/FNR requirements remain unsupported by independent real-data
evidence. See `INDEPENDENT_DATASET_PROTOCOL.md`.

The verified VisA capsules external-transfer experiment demonstrates that the
current prespecified PatchCore representation does not transfer reliably at the
frozen image-level operating point: sealed-final image AUROC 0.635550 and FNR
80.39%. Its 0/46 observed FPR has a 95% Wilson upper bound of 7.71%, so it also
does not substantiate the 2% operational FPR target. Its warmed CUDA callback
boundary was 34.43 ms/image. This is a frozen-protocol negative result; no
post-final tuning is permitted. A post-hoc **development-only** analysis found
40/49 defects below the q98 threshold and image AUROC 0.758277. It also found
that initial non-square VisA masks had been square-resized, so the historical
sealed-final pixel metrics are alignment-limited and are not evidence for or
against the pixel-AUROC target. The final set was not rerun. See
`VISA_CAPSULES_DEVELOPMENT_FAILURE_ANALYSIS.md`. Docker remains unchanged and
still registers its existing 105 configurations.

Because public VisA raw files and the sealed manifest coexist locally, the
final partition is technically sealed but not access-controlled blind. It is
therefore external transfer evidence, not an unbiased production acceptance
test.

The Docker dependency path is now offline-first: an audited Linux CPython 3.11
wheelhouse and transitive lock are copied into the build context, and the
default image build uses `pip --no-index --find-links`. The CUDA PyTorch base
image remains the sole source of Torch/TorchVision/CUDA packages. This removes
the observed Docker-side PyPI transfer dependency, but a live container build
and runtime verification are still required.

The Dockerfile also includes a conditional repair for the inherited
`ninja 1.11.1.1` wheel-tag failure reported by `pip check` in the pinned
PyTorch base image. It inspects the base distribution and removes Ninja only
when its wheel is incompatible with the image interpreter and no active
installed dependency requires it. The final unsuppressed `pip check` remains
the build gate. This repair and a live base-image reproduction still require a
working Docker engine on the target host.
