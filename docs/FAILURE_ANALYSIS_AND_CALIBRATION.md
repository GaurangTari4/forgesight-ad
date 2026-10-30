# Failure analysis and calibration diagnostics

## Scope

This is a post-hoc analysis of immutable MVTec evaluation runs. Fitted statistics, memory banks, and stored thresholds were not changed. Calibration diagnostics use original disjoint normal-only splits; test labels and masks are used only to describe existing errors.

## Completed direct diagnostics

PatchCore ResNet-18 / bottle was safely reloaded and rescored on 42 calibration-normal images and 83 test images. Evidence is under `artifacts/failure_analysis/patchcore/bottle/`.

| Item | Finding |
| --- | --- |
| Stored image threshold | 0.9641508460 |
| Recomputed normal-only q98 | 0.9640944600 |
| Absolute difference | 0.0000563860 |
| Calibration mean / std | 0.857743 / 0.036677 |
| Calibration maximum | 0.964094 |
| Normal / defect test mean | 0.858727 / 1.000413 |
| False negatives | 9/63; Wilson 95% interval 7.70%–24.97% |
| False positives | 0/20; Wilson upper bound 16.11% |

The threshold delta is within a documented relative floating-point tolerance and does not represent a methodology change. False negatives were `contamination` (7) and `broken_small` (2). The saved heatmap and ground-truth mask are post-hoc evidence, not tuning examples.

Corrected v2 ResNet-50 / bottle also completed direct rescoring. Its stored threshold of `12383.2177734375` reproduced exactly from the 42 calibration-normal scores. Its calibration distribution is strongly upper-tailed (mean `1550.44`, median `796.79`, p95 `5836.52`, maximum `12383.22`), with three values above the p95. It produced 8/63 false negatives (Wilson 95% interval 6.58%–23.11%) and 0/20 false positives; false negatives were `contamination` (6) and `broken_small` (2). Evidence is under `artifacts/failure_analysis/v2/bottle/`.

## Cross-category patterns

The benchmark aggregate contains category metrics, not persistent per-image score distributions. Corrected v2 had the largest missed-defect counts on zipper (78), capsule (62), pill (62), cable (52), and screw (45). PatchCore had the largest missed-defect counts on screw (109), capsule (87), pill (82), cable (42), zipper (39), and transistor (36).

| Category | V2 FNR | PatchCore FNR |
| --- | ---: | ---: |
| cable | 56.52% | 45.65% |
| capsule | 56.88% | 79.82% |
| screw | 37.82% | 91.60% |
| transistor | 20.00% | 90.00% |
| zipper | 65.55% | 32.77% |

These stored thresholds derive from normal-only splits. Generic records preserve threshold and manifest provenance but not individual calibration score arrays; direct rescoring is required for full distribution diagnostics and must never be used to retrospectively select thresholds.

## Localization and uncertainty

Low pixel AP may reflect poor pixel ranking, coarse feature resolution, resize/padding alignment, or maps highlighting normal texture. A binary mask threshold cannot separate these causes. The diagnostic runner writes a representative false-negative input, heatmap, and ground-truth mask when one exists.

The normal-only q98 is an unstable upper-tail estimate with typical calibration pools of roughly 42–78 images. Small test sets also yield wide FPR/FNR confidence intervals. Low observed FPR is not evidence of low production false-positive risk.

## Controlled next experiments

1. Prespecify calibration quantiles and test on new independent data.
2. Increase normal-only calibration pools and bootstrap threshold stability.
3. Use blinded subtype and defect-size evaluation sets with confidence intervals.
4. Test resize/padding and feature alignment with controlled synthetic defects.
5. Compare backbones under a prespecified protocol; v2 ResNet-50 and PatchCore ResNet-18 are not an algorithm-only comparison.
6. Persist immutable calibration-score and per-image diagnostic sidecars in future runs.

## Completion status

The 15-category benchmark is complete. Full direct rescoring is resumable but not complete for every method/category because this host limits foreground commands. Run:

```powershell
.\.venv\Scripts\python.exe -u .\tools\analyze_failure_modes.py --method both --all-categories --resume --device cuda:0
```

This writes only new `artifacts/failure_analysis/` records and never changes models or thresholds.
