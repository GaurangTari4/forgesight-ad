# Controlled detector-representation experiments

The prespecified configurations are in `configs/representation_experiments.json`. Existing benchmark artifacts are immutable exploratory baselines. Training-good images are the only permitted fitting source; calibration normals are disjoint; previously examined MVTec test labels are prohibited for configuration selection.

PatchCore variants vary one representation factor at a time: bank size (256, 512, 1024, 2048), candidate budget, backbone (ResNet-18 or ResNet-50), and intermediate layers. Each future run must save a seed, split manifest, memory bank size, artifact bytes, fitting time, CUDA peak memory, warm inference latency, and normal-score stability. Corrected-v2 diagnostics isolate single-layer and prespecified multi-layer score contributions without modifying existing v2 artifacts.

## Completed synthetic proxy diagnostic

V2 ResNet-50 / bottle was applied to ten held-out normal calibration images with five deterministic synthetic changes each (50 records). Mean score deltas were: color shift `121934.068`, localized change `28388.039`, low contrast `87936.285`, small local change `47486.717`, and thin scratch `93208.030`. Mean ratio of anomaly-map maximum inside the known synthetic region to global maximum was 0.988–1.000 across types.

These are synthetic diagnostic proxies only. They neither establish real defect detection performance nor select a threshold or winning representation. A valid improvement claim requires independent fitting, calibration, development/model-selection, and final untouched evaluation data with all hyperparameters locked before final evaluation.
