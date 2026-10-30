# VisA independent-validation v1: acquisition and execution record

## Official provenance

VisA is released by Amazon Science with the SPot-the-Difference project:
`amazon-science/spot-diff`. The official AWS S3 archive is
`https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`.
The AWS Open Data Registry lists CC BY 4.0 and the expected SHA-256:
`2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`.
The upstream documentation describes 12 categories, including `capsules`, with
image-level annotations and anomaly masks; normal images have no masks.

## Initial acquisition incident — resolved

On 2026-10-31 a resumable official download was attempted at
`<workspace>/external_datasets/visa/VisA_20220922.tar`.
The execution host terminated the initial foreground transfer. A subsequent
Windows background-launch quoting defect created two concurrent curl writers,
producing a full-length but invalid archive with SHA-256
`30e207ca1bb7fc0af94363f5939d093fe1ad9108f883978f61ed0e071e8b85d0`.
It was preserved under `external_datasets/visa/quarantine/` and never extracted.
The repaired downloader uses one `.partial` writer, records the official ETag
and prefix SHA-256, accepts a resume only after matching `206 Content-Range`,
and never appends an HTTP 200 response. This incident was resolved before any
extraction; the verified archive and experiment are recorded below.

For a clean recovery, retain or move the untrusted partial aside, then use a
single foreground process or one monitored downloader—not concurrent writers:

```powershell
Set-Location '<workspace>/forgesight-ad-clean'
.\.venv\Scripts\python.exe -u .\tools\download_visa.py `
  --root '..\external_datasets\visa' --category capsules --download --extract
```

The utility uses curl continuation, refuses extraction before the official
digest matches, selectively extracts only `capsules`, avoids archive traversal,
and writes non-overwriting provenance. If the existing partial is suspected
corrupt, do not resume it; archive it separately and start a clean download.

## Frozen protocol

Before a detector is fitted, run
`prepare_visa_capsules_manifest.py` with seed `20261031`. Its prespecified
policy allocates normal images 70% fitting / 15% calibration / 7.5% development
/ 7.5% final, then stratifies each official defect subtype 50/50 between
development and final. It emits a sealed full manifest plus training-only and
development-only views, so fitting cannot read development/final records.

The new VisA-specific PatchCore configuration is fixed before outcomes:
ImageNet-pretrained frozen ResNet-18, layers 2/3, L2 patch embeddings,
8,192 uniformly selected candidates, deterministic greedy 512-bank coreset,
seed `20261031`, 512×512 preprocessing, and normal-only q98-higher calibration.
It is a new versioned artifact under `artifacts/independent_experiments/`; no
MVTec artifact is reused or overwritten. Labelled development would assess the
frozen configuration once; final evaluation occurs only after that model and
calibration sidecar are sealed.

## Completed VisA capsules experiment

The archive subsequently passed exact verification at 1,929,840,640 bytes and
SHA-256 `2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`.
Safe selective extraction produced 803 capsules files. The official annotation
CSV SHA-256 is `8fced5b5b02528695d0724395d685770adcc971f514816d654177ee885f4f409`.
The frozen manifest SHA-256 is
`9c61a836a3de2e03cbc83ba8133d8034ef70540caf5202dbd73e8266539adc2a`.

Split counts: 421 normal fitting, 90 normal calibration, 94 labelled
development, and 97 sealed final-evaluation images. The official annotations
do not provide physical-object/session groups; exact-content hashes were used
as source groups, and all image/mask dimensions and cross-split content hashes
were validated.

The new artifact is
`artifacts/independent_experiments/visa_20220922_capsules_independent_v1_patchcore_r18_bank512_20261030T211101Z/model.pt`
(SHA-256 `8b01ce45418cee545b778e59081525b41bc86fdc25d999acee75a5c7a52f1443`).
Normal-only q98-higher calibration on the 90 disjoint normals produced frozen
threshold `0.9470425248146057`. Fitting took 22.588 s with 114,621,440 peak
CUDA bytes on the RTX 3060.

| Split | Image AUROC | Pixel AUROC | Image AP | Pixel AP | FPR | FNR | IoU | Dice | Mean warm call |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 0.758277 | 0.712883 | 0.807287 | 0.058299 | 0.000000 | 0.816327 | 0.001592 | 0.003179 | 34.54 ms |
| Sealed final | 0.635550 | 0.694444 | 0.725722 | 0.040457 | 0.000000 | 0.803922 | 0.005133 | 0.010215 | 34.43 ms |

### Spatial-metric qualification

The table preserves the immutable JSON outputs written at the time. A later
read-only audit found that the initial evaluator square-resized VisA's 1500x1000
ground-truth masks while the detector letterboxed inputs. Therefore the pixel
AUROC/AP/IoU/Dice values in the table, including the sealed-final values, are
**alignment-limited historical metrics**, not validated spatial measurements.
The final set was not reloaded or rerun. Image-level scores, decisions,
threshold, artifact, image AUROC/AP, FPR, FNR, and latency are unaffected.
The evaluator is corrected for future data, and the development-only aligned
audit is documented in `VISA_CAPSULES_DEVELOPMENT_FAILURE_ANALYSIS.md`.

The final result was evaluated once after the development result and frozen
configuration. Its confusion matrix is TN=46, FP=0, FN=41, TP=10. The FPR 95%
Wilson interval is [0, 0.077076] and FNR interval is [0.675417, 0.889848].
This detector does not meet the accuracy or operational sensitivity targets.
No retuning, refitting, threshold change, or subsequent final evaluation was
performed. Full machine-readable evidence is in `development_evaluation.json`
and `sealed_final_evaluation.json` beside the artifact.

### Blinding limitation

The final partition was fixed before fitting and was not used by detector,
calibration, or development-evaluation code. However, VisA is public and its
raw archive/manifest reside in this local workspace; this is a technical sealed
partition, not an independently access-controlled blind holdout. Treat the
final metrics as a prespecified external-dataset transfer measurement, not an
unbiased production-validation claim.
