# Contributing to ForgeSight

ForgeSight is a research prototype. Contributions should improve
reproducibility, safety, testability, or documentation without overstating
detector performance.

## Before opening a pull request

1. Do not add datasets, trained weights, memory banks, experiment outputs,
   wheelhouses, virtual environments, or secrets to Git.
2. Preserve existing artifacts and historical benchmark records. New
   experiments require a separate versioned output directory.
3. Do not select models or thresholds using already examined final-test labels.
4. Add synthetic or mocked regression coverage for code changes where feasible.
5. Run `python -m pytest`, `python -m pip check`, and compilation checks.

## Development conventions

- Keep device choice explicit (`auto`, `cpu`, or `cuda:0`).
- Fail clearly when trusted pretrained weights or compatible artifacts are
  unavailable; never silently use random initialization.
- Preserve artifact methodology, backbone, category, preprocessing, and
  threshold validation.
- Keep Docker local-only and do not enable Gradio sharing by default.

## Reporting research results

Label exploratory and post-hoc findings accurately. Report the dataset split,
fitting/calibration separation, artifact identity, metric definition, and
whether labels were used only for final scoring. Do not claim production
performance without independent, prespecified validation.
