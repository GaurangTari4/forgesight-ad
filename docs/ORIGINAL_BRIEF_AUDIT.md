# Original one-class visual anomaly-detection brief: evidence matrix

Assessment updated 2026-10-30. Historical artifacts and exploratory MVTec
results are immutable. “Implemented” means code and an evidenced workflow
exist; it does **not** mean a numerical target has been met in deployment.

| Original requirement | Evidence | Status | Gap / next evidence needed |
| --- | --- | --- | --- |
| One-class, normal-only fitting | Corrected v2 and PatchCore fit only `train/good`. Every category artifact has a disjoint fit/calibration manifest; `tools/run_category_detectors.py` enforces this. | Implemented | Use production-good material and audit its variation coverage. |
| Image-level classification | All three detector families return image score, frozen threshold, and PASS/FAIL through standalone APIs and localhost Gradio. | Implemented | Validate operating point on independent production-like data. |
| Pixel-level localization | V2/PatchCore generate 512×512 maps and masks; MVTec pixel AUROC/AP and visual checks were recorded. | Implemented | Defect-review protocol and independent localization validation. |
| Data quantity and acceptable normal variation | MVTec has been structurally audited and all 15 categories were evaluated. Calibration subsets are small (for example bottle 42 normals), and no production variation specification exists. | Insufficient evidence | Define material, lighting, pose, camera, and process tolerances; collect representative normal data. |
| Optional mixed good/defect validation | `forgesight_ad.semi_supervised_calibration.select_threshold` and `tools/run_mixed_validation_calibration.py` implement a prespecified labelled-development rule: maximize TPR subject to an FPR cap. The command requires separate normal/defect score files and provenance labels, publishes with exclusive create, and records counts/objective. | Protocol implemented; real-data validation blocked | Obtain a distinct labelled development set (the already examined MVTec test set must not be repurposed), freeze the selected threshold, then evaluate once on untouched data. |
| Image AUROC > 0.95 | Exploratory 15-category macro: v2 0.948426; PatchCore 0.896620. Bottle-only values exceed 0.95, but do not establish generalization. | Not met overall | Prespecified representation/development experiments followed by independent evaluation. |
| Pixel AUROC > 0.95 | Exploratory 15-category macro: v2 0.919948; PatchCore 0.889385. Bottle-only values exceed 0.95. | Not met overall | Improve representations under a clean protocol, then independently evaluate. |
| FPR < 2% | Pooled FPR: v2 5.35%, PatchCore 3.43%. Bottle had 0/20 normal errors but that sample cannot substantiate a 2% guarantee. | Not met / insufficient operational evidence | Large independent normal set with confidence intervals and a frozen threshold. |
| FNR < 0.5% | Pooled FNR: v2 32.91%, PatchCore 42.61%; bottle v2 8/63 misses and PatchCore 9/63. | Not met | Fresh labelled development and final evaluation data; no threshold tuning on already examined MVTec tests. |
| End-to-end latency < 100 ms | Warm local Python callback (not HTTP/browser): historical ResNet-18 mean 85.23 ms, PatchCore ResNet-18 54.90 ms, v2 ResNet-50 112.76 ms. | Partially met by stated boundary | Measure HTTP and rendering on target hardware; v2 misses this callback target. |
| ONNX/OpenVINO deployment | `tools/export_feature_extractor_onnx.py` scopes an extractor-only export and numerical ONNX Runtime comparison. Current environment lacks `onnx`, `onnxruntime`, and OpenVINO, so no export was claimed. | Not met | Install validated deployment packages in a dedicated environment, export the frozen extractor, verify fixed-input error, then separately design scorer/pre/post-processing deployment. |

## Calibration protocol boundary

The optional mixed-validation path is intentionally separate from one-class
fitting: feature statistics and PatchCore memory banks remain fitted on normal
images only. A future study must predeclare its maximum FPR (default 2%),
provide distinct known-good and known-defect development samples, apply the
selection objective once, record the score provenance, and freeze the result
before independent final evaluation. Synthetic fixtures exercise the protocol;
they are not evidence of real-world calibration.

## Reliability-improvement update

`RELIABILITY_IMPROVEMENT_PROTOCOL.md` now records the required four-way split
protocol and an executed, synthetic-only PatchCore 512-bank diagnostic. The
repository has no independent labelled development or untouched final
real-defect partition, so the optional mixed-validation protocol remains
implemented but unexecuted on real data. The synthetic diagnostic confirms
proxy score response without changing a detector, calibration threshold, or
historical result; it is not evidence that the AUROC, FPR, or FNR targets have
improved.

`INDEPENDENT_DATASET_PROTOCOL.md`,
`configs/independent_dataset_manifest.template.json`, and the independent-data
validator/evaluator now implement collection provenance, split isolation,
duplicate/leakage detection, mask validation, and frozen-artifact final
evaluation reporting. They are executable infrastructure only: no independent
real dataset has been ingested, so all operational target rows above remain
unmet or unsupported by independent evidence.

The first genuinely external VisA capsules experiment is now complete, but
does not meet the brief targets at the frozen image operating point:
sealed-final image AUROC 0.635550, FPR 0/46 (with an upper 95% Wilson bound of
7.71%), and FNR 41/51 (80.39%). A later development-only audit also found
substantial image-score overlap and 40/49 missed development defects. The
historical sealed-final pixel metrics are alignment-limited because masks were
initially square-resized rather than letterboxed; they are retained but not
used to assess the pixel-AUROC requirement. The sealed final was not rerun.
This is evidence of poor cross-dataset transfer for the frozen configuration,
not a reason to modify historical MVTec results.
