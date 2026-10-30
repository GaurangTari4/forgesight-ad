# Detection reliability improvement protocol

## Data boundary

This protocol preserves every historical, corrected-v2, and PatchCore artifact,
threshold, benchmark, and Docker deployment record. Existing MVTec test labels
are retrospective evidence only; they must not select a representation,
threshold, or deployment configuration.

The repository currently contains MVTec `train/good` and the already-examined
MVTec `test` partition. It contains **no genuinely independent labelled
development set** and no untouched final real-defect evaluation set. Therefore
no real-data sensitivity, specificity, image/pixel AUROC, confidence interval,
or defect-subtype claim can be made for a newly proposed detector or threshold.

## Required future protocol

1. Fit feature statistics or a PatchCore memory bank using normal-only fitting
   data.
2. Reserve a disjoint normal calibration partition for a prespecified
   normal-only operating rule.
3. Supply a separate labelled development set with known-good and known-defect
   samples. It must be independent of fitting, calibration, and the final set.
4. Before looking at the final set, prespecify representation candidates and
   select at most one configuration/threshold on development data using the
   stated objective: maximize sensitivity subject to an FPR cap of 2%.
5. Publish the selected threshold as a new immutable, versioned calibration
   sidecar; do not modify historical thresholds or artifacts.
6. Evaluate once on an untouched, subtype-stratified final set. Report image
   and pixel AUROC/AP, sensitivity, specificity, FPR/FNR, Wilson confidence
   intervals, and subtype counts/uncertainty.

For planning, zero observed false positives need about 184 independent normal
images to support an upper 95% FPR bound of 2%; zero misses need about 598
independent defects to support an upper 95% FNR bound of 0.5%. These are
binomial approximations and do not replace subtype coverage requirements.

## Executed synthetic diagnostic (not a performance evaluation)

The prespecified immutable PatchCore ResNet-18 layer2/layer3, 512-memory-bank
variant was reloaded from
`artifacts/representation_experiments/patchcore_bottle_r18_l23_bank512_20261029T223031Z/`.
Its original split contains 167 fitting-good images and 42 disjoint
calibration-good images. The diagnostic used the first ten calibration images,
never accessed MVTec test data, and did not alter its normal q98 diagnostic
threshold, model, memory bank, or artifact.

| Synthetic proxy | Paired score delta mean | Positive deltas | Mean region/max-map ratio | Map argmax in known changed region |
| --- | ---: | ---: | ---: | ---: |
| Color shift | 0.272338 | 10/10 | 1.0000 | 10/10 |
| Localized change | 0.127486 | 10/10 | 0.9905 | 6/10 |
| Low contrast | 0.255107 | 10/10 | 0.9985 | 6/10 |
| Small local change | 0.225889 | 10/10 | 0.9905 | 3/10 |
| Thin scratch | 0.235235 | 10/10 | 0.9898 | 1/10 |

## Extended descriptive localization metrics

The immutable normal-only q98 diagnostic threshold was `0.911798`; it was read
from the existing resources sidecar and not changed. Pixel ranking was high for
these simple synthetic masks (AUROC 0.990665–0.998915), but average precision
and frozen-mask overlap reveal the small/elongated-edit boundary: thin scratch
AP `0.211372`, IoU `0.164788`, Dice `0.281413`; small local edit AP `0.235093`,
IoU `0.164279`, Dice `0.280663`. Background false activation averaged
0.000166–0.008393 by proxy type. These values are synthetic-mask diagnostics,
not MVTec or production pixel metrics.

This establishes only that these deterministic image edits changed scores and
often concentrated high anomaly responses near the known synthetic region. It
does not establish real-defect localization, sensitivity, specificity, AUROC,
or FPR/FNR. The lower peak-placement rates for the thin/small proxies motivate
future controlled representation work, but must not be used to choose a winner
without independent labelled development data.

Evidence: `artifacts/reliability_experiments/patchcore_r18_bank512_synthetic_20261030T200421Z/synthetic_reliability.json`.
