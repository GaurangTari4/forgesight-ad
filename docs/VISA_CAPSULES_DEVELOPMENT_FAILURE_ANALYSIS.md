# VisA capsules: development-only failure analysis

## Scope and safeguards

This is a read-only, post-hoc analysis of the **development-only** manifest
view for the frozen VisA capsules PatchCore artifact. The analysis tool refuses
a manifest view that exposes any split other than `development`. It did not
load sealed-final images or masks, did not rerun the final evaluator, and did
not fit, recalibrate, or alter an artifact.

Configuration: ImageNet-pretrained frozen ResNet-18; PatchCore layers 2/3;
512-vector bank; 8,192 candidates; seed `20261031`; 512-square letterboxed
input. Artifact: `visa_20220922_capsules_independent_v1_patchcore_r18_bank512_20261030T211101Z/model.pt`.
The analysis evidence is
`artifacts/independent_analysis/visa_capsules_development_failure_analysis_20261031.json`.

## Threshold provenance

The stored normal-only q98-higher image threshold is `0.9470425248146057`.
Recomputing q98-higher from all 90 frozen calibration-normal scores reproduced
exactly the same value. The implementation uses this *same* scalar for both
the image decision and `predicted_mask = anomaly_map > threshold`; there is no
separately calibrated pixel threshold.

That provenance is internally consistent, but an image-level upper-tail
threshold is not automatically an appropriate segmentation operating point.
This observation is a design limitation, not permission to retune against this
already examined development set.

## Image-score evidence

| Population | n | mean | median | q95 | maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibration normal | 90 | 0.897283 | 0.895483 | 0.924531 | 0.948306 |
| Development normal | 45 | 0.895760 | 0.891346 | 0.929450 | 0.936375 |
| Development defect | 49 | 0.919347 | 0.921383 | 0.963882 | 0.972067 |

The defect distribution overlaps heavily with both normal distributions. Image
AUROC is `0.758277` and image AP is `0.807287`: there is meaningful ranking
signal, but not strong separation. Forty of 49 development defects (81.63%)
are at or below the frozen threshold. The median defect is 0.025660 below that
threshold. In contrast, the development-normal maximum is below it, yielding
zero observed development false alarms; this small normal sample cannot prove
low operational FPR.

These facts support a combination of conservative normal-tail calibration and
limited representation separation. They do **not** establish that changing the
threshold would solve the problem: many defects overlap normal scores even
before a binary decision is made.

## Development false negatives and localization

There were 40 development false negatives. The official VisA `label` field is
composite (for example `bubble,discolor,scratch`), so these counts are
descriptive label groups rather than mutually exclusive causal subtypes.

| Official composite label | Development false negatives |
| --- | ---: |
| `bubble` | 15 |
| `bubble,discolor` | 6 |
| `bubble,discolor,scratch` | 7 |
| `bubble,discolor,scratch,leak` | 4 |
| `bubble,discolor,scratch,leak,misshape` | 8 |

Using the *same padded 512x512 mask transform as inference*, mean anomaly
score inside the defect masks was 0.824170 versus 0.696793 outside, a mean
foreground/background ratio of 1.18284. Thus maps often rank defect pixels
above background, but the absolute map levels are far below the global 0.9470
threshold. This is consistent with weakly localized signal and an unsuitable
shared high operating threshold, not proof of either single cause.

Five representative development false-negative heatmaps and ground-truth-mask
overlays are stored alongside the JSON evidence under
`artifacts/independent_analysis/false_negative_*`. They show the letterboxed
input and mask canvas rather than a geometrically distorted square resize.

## Preprocessing and metric alignment audit

The detector uses aspect-ratio-preserving `resize_with_padding` to 512x512.
The original independent evaluator directly resized VisA's 1500x1000 masks to
512x512. That mismatch distorted historical spatial metrics. The evaluator is
now fixed for future/development-only use: masks use the same padded transform
as detector inputs. A development-only rerun with aligned masks measured pixel
AUROC `0.950111` and pixel AP `0.136621` (IoU `0.001627`, Dice `0.003248`).

The previously published sealed-final pixel AUROC/AP/IoU/Dice are retained as
immutable historical outputs but must be treated as **spatially
alignment-limited**, not validated localization measurements. They were not
rerun or replaced. Image scores, decisions, calibration, artifact weights, and
final image-level metrics are unaffected by this mask alignment defect.

## Established observations and uncertainty

- Established: calibration provenance is exact; a shared image/pixel threshold
  is used; development score overlap is substantial; most development defects
  are missed at the frozen threshold; maps have higher average response inside
  masks than outside.
- Unresolved: whether ImageNet ResNet-18 features, bank capacity, candidate
  coverage, image resolution, or the shared threshold contributes most. The
  current development set has been examined, so it is no longer suitable for
  choosing among proposed changes.
- No causal claim is made from the composite subtype labels or five examples.

## At-most-three prespecified future experiments

Each must use new, untouched labelled development data for selection and a new
final set for one-time evaluation; no current VisA development or final labels
may choose a winner.

1. **Representation hypothesis:** frozen ResNet-50 PatchCore layers 2/3,
   bank 512, candidate 8,192, same normal-only fitting/calibration protocol.
   Measure score separation and localization on fresh data.
2. **Coverage hypothesis:** frozen ResNet-18 layers 2/3, bank 1,024,
   candidate 16,384, deterministic seed fixed in advance. Report memory,
   latency, and calibration stability before final evaluation.
3. **Operating-point hypothesis:** retain a frozen detector but prespecify
   separate normal-only image and pixel calibration methods on a *new* normal
   calibration split, then assess the frozen pair on fresh development data.
   This addresses the shared-threshold limitation without using defect labels
   for one-class fitting.
