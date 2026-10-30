# Calibration stability study

This is a read-only bootstrap analysis of normal-only calibration scores. Bootstrap intervals quantify sampling variability, not operational FPR guarantees. Alternative estimators are not selected using MVTec defect-test outcomes.

Bootstrap seed: `42`; replicates: `5000`.

| method | unstable upper-tail categories | median calibration n |
| --- | --- | ---: |
| v2 | bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper | 48 |
| patchcore | bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, transistor, wood, zipper | 48 |

## Interpretation

The empirical q98 is determined by one or a few upper-tail normal images at these sample sizes. Tukey fences, MAD-based thresholds, and normal-model thresholds make different distributional assumptions; they are future prespecified candidates, not replacements for stored thresholds. Any operational comparison requires fresh independent calibration/development data and an untouched final evaluation set.

## Data requirements

To demonstrate an FPR below 2% with 95% confidence and zero observed false positives, an independent normal set requires approximately 184 samples (`1 - 0.05^(1/n) <= 0.02`). Demonstrating an FNR below 0.5% with zero misses similarly needs approximately 598 independent defective samples. These are binomial planning approximations; subtype coverage can require substantially more data.

## Research roadmap

1. Collect independent normal calibration pools per category with sufficient tail coverage.
2. Prespecify one empirical and one robust threshold estimator before a development comparison.
3. Use a separate development set to choose methodology, then lock it.
4. Evaluate once on an untouched, subtype-stratified test set with Wilson intervals.
5. Persist score arrays and manifests as immutable diagnostic sidecars.
