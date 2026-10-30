# Independent industrial dataset protocol: screw or bottle

No new collection exists in this repository. This document and the adjacent
manifest template define the collection and ingestion contract for a future
independent study.

For the current VisA v1 attempt, see `VISA_INDEPENDENT_VALIDATION_V1.md`.
VisA is independent of the prior MVTec experiments but represents a different
object/imaging distribution; no MVTec capsule checkpoint may be used as a VisA
detector.

## Collection plan

Start with one stable object category—screw is recommended because the existing
retrospective analysis found missed defects; bottle is acceptable where its
optics and handling better match the intended line. Give every acquired image a
stable sample ID and retain the source file unchanged.

| Partition | Intended content | Permitted use |
| --- | --- | --- |
| `fitting` | Normal images only, spanning approved normal lots/finishes | Fit one-class feature statistics or memory bank only |
| `calibration` | Disjoint normal images only | Prespecified normal-only calibration only |
| `development` | Independent known-good and known-defect images with masks | Choose one prespecified representation/threshold under the declared FPR constraint |
| `final_evaluation` | Independently acquired, sealed good/defect data with masks | One frozen-model/threshold evaluation only |

For every defect image collect an image-level label, a binary pixel mask in the
native image frame, and a controlled defect subtype (for example: scratch,
missing thread, burr, deformation, contamination, crack). Normal samples have
an explicit `normal` label and no mask. Do not mix frames from one physical
capture burst across partitions.

Record camera serial/lens, working distance, pose/orientation tolerance,
lighting fixture and lux/color-temperature range, exposure/focus settings,
material/finish/lot, and a written definition of acceptable normal variation.
The final partition must be access-controlled and have
`final_evaluation_immutable: true` plus a `final_evaluation_sealed_at`
timestamp before any model-selection work starts. The validator records its
content hashes; storage ACLs/WORM retention remain an operational control.

## Ingestion contract

Copy [the template](../configs/independent_dataset_manifest.template.json) next
to the newly collected files, replace every placeholder, then run:

```powershell
.\.venv\Scripts\python.exe -u .\tools\validate_independent_dataset.py `
  C:\data\screw-study\manifest.json `
  --report C:\data\screw-study\validation_report.json
```

The validator requires all four explicit splits, checks image decoding,
defect-mask presence and dimensions, computes SHA-256 hashes for images and
masks, detects duplicate image bytes across every split, records provenance,
and rejects paths from the previously examined MVTec test partition. It does
not copy, relabel, or mutate inputs. Preserve the manifest and validation
report with detector-artifact SHA-256 values.

## Independent final evaluation

Only after development selects and freezes a *new versioned* threshold sidecar
may the sealed final partition be evaluated. For example:

```powershell
.\.venv\Scripts\python.exe -u .\tools\evaluate_independent_dataset.py `
  --manifest C:\data\screw-study\manifest.json `
  --family patchcore --artifact C:\artifacts\selected\model.pt `
  --calibration C:\artifacts\selected\frozen_threshold.json `
  --backbone resnet18 --category screw --device cuda:0 `
  --output C:\data\screw-study\final_evaluation.json
```

The evaluator records artifact and manifest hashes; image/pixel AUROC and AP;
frozen-threshold FPR/FNR, sensitivity, specificity, precision/recall and
confusion matrix; pixel IoU/Dice; subtype breakdowns; Wilson intervals for
FPR/FNR; and warmed per-image call latency. It refuses to overwrite output and
does not fit, recalibrate, or select a model.
