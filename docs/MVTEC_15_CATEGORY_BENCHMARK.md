# MVTec AD 15-category benchmark

All metrics are read-only aggregations of completed normal-only calibration runs. Test labels were used only for evaluation.

## v2

Macro image/pixel AUROC: 0.948426 / 0.919948. Pooled FPR/FNR: 0.053533 / 0.329094.

| category | image AUROC | pixel AUROC | image AP | pixel AP | FPR | FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 1.000000 | 0.976032 | 1.000000 | 0.693423 | 0.000000 | 0.126984 |
| cable | 0.936844 | 0.926526 | 0.961746 | 0.352088 | 0.017241 | 0.565217 |
| capsule | 0.935381 | 0.977819 | 0.983778 | 0.336204 | 0.043478 | 0.568807 |
| carpet | 0.977528 | 0.957109 | 0.993275 | 0.412547 | 0.071429 | 0.078652 |
| grid | 0.949875 | 0.883525 | 0.983944 | 0.254570 | 0.142857 | 0.122807 |
| hazelnut | 0.958571 | 0.940141 | 0.966274 | 0.334126 | 0.050000 | 0.171429 |
| leather | 0.988451 | 0.948932 | 0.996582 | 0.297695 | 0.156250 | 0.021739 |
| metal_nut | 0.986315 | 0.940139 | 0.996319 | 0.713780 | 0.045455 | 0.139785 |
| pill | 0.894708 | 0.958518 | 0.976297 | 0.442703 | 0.076923 | 0.439716 |
| screw | 0.846895 | 0.943754 | 0.931598 | 0.101083 | 0.097561 | 0.378151 |
| tile | 0.975108 | 0.830699 | 0.989756 | 0.342966 | 0.000000 | 0.440476 |
| toothbrush | 1.000000 | 0.956223 | 1.000000 | 0.416128 | 0.000000 | 0.200000 |
| transistor | 0.972917 | 0.831145 | 0.964483 | 0.485644 | 0.033333 | 0.200000 |
| wood | 0.981579 | 0.811338 | 0.994309 | 0.270778 | 0.000000 | 0.250000 |
| zipper | 0.822216 | 0.917327 | 0.937320 | 0.439580 | 0.062500 | 0.655462 |
## patchcore

Macro image/pixel AUROC: 0.896620 / 0.889385. Pooled FPR/FNR: 0.034261 / 0.426073.

| category | image AUROC | pixel AUROC | image AP | pixel AP | FPR | FNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 0.987302 | 0.951566 | 0.996113 | 0.661277 | 0.000000 | 0.142857 |
| cable | 0.825337 | 0.797546 | 0.902952 | 0.270831 | 0.051724 | 0.456522 |
| capsule | 0.761069 | 0.812847 | 0.943744 | 0.230974 | 0.000000 | 0.798165 |
| carpet | 0.911316 | 0.916553 | 0.973692 | 0.490687 | 0.035714 | 0.247191 |
| grid | 0.967419 | 0.964571 | 0.990173 | 0.411528 | 0.000000 | 0.122807 |
| hazelnut | 0.998929 | 0.977434 | 0.999413 | 0.531990 | 0.000000 | 0.114286 |
| leather | 0.979959 | 0.978793 | 0.994239 | 0.554486 | 0.000000 | 0.097826 |
| metal_nut | 0.948680 | 0.762332 | 0.988184 | 0.354513 | 0.000000 | 0.354839 |
| pill | 0.831151 | 0.883599 | 0.954842 | 0.403817 | 0.115385 | 0.581560 |
| screw | 0.691330 | 0.939425 | 0.833682 | 0.147923 | 0.073171 | 0.915966 |
| tile | 0.938312 | 0.824634 | 0.976321 | 0.296235 | 0.000000 | 0.428571 |
| toothbrush | 0.900000 | 0.980462 | 0.953993 | 0.424794 | 0.250000 | 0.233333 |
| transistor | 0.779583 | 0.677512 | 0.783124 | 0.217670 | 0.000000 | 0.900000 |
| wood | 0.995614 | 0.945102 | 0.998615 | 0.503318 | 0.000000 | 0.166667 |
| zipper | 0.933298 | 0.928401 | 0.978497 | 0.442338 | 0.093750 | 0.327731 |

## Methodology limitations

v2 uses a frozen ResNet-50 and unweighted fitted layer statistics combined by configured score weights. PatchCore uses a frozen ResNet-18, normalized layer2/layer3 patch embeddings, and nearest-neighbor distances to a deterministic coreset. Backbone and scoring differ, so this is not an isolated algorithm-only comparison. Results are exploratory MVTec test-set evidence, not production reliability. Thresholds were calibrated from disjoint normal-only training splits; no threshold was selected from test outcomes. Generic per-run records do not persist individual calibration score distributions, so post-hoc distribution quantiles cannot be reconstructed without rescoring; thresholds and provenance are retained.
