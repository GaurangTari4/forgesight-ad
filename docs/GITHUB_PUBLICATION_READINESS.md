# GitHub publication readiness — research-prototype v1.0.0

Audit date: 2026-10-31. This document records local publication preparation;
it does not create a commit, tag, GitHub release, or public repository.

## Repository identity

- Branch: `main`
- Configured remote: `origin` → `https://github.com/GaurangTari4/forgesight-ad.git`
- Existing history at audit: one commit, `af2ac03 Initial commit: ForgeSight AD`.
- Citation metadata now uses that verified configured remote. `date-released`
  remains unset because no authorised v1.0.0 release has been created.

## Publication safety assessment

**Suitable for public publication as a research prototype, subject to the
remaining third-party and deployment caveats below.** The owner has confirmed
copyright as © 2026 Gaurang Tari and authorised MIT licensing for ForgeSight.
Do not describe the release as production ready.

### Checked

| Check | Result |
| --- | --- |
| Tracked object size scan | No tracked object exceeded 10 MiB. |
| Secret-pattern scan, publishable content | No matches for common cloud/GitHub tokens or private-key headers. |
| Git-history token scan | No matches for the same common token/private-key patterns. |
| Absolute personal-path scan | Removed the two publishable VisA Windows paths found during audit. |
| Git ignore policy | Datasets, archives, weights, checkpoints, artifacts, output runs, environments, caches, wheelhouses, and common secret-file extensions are ignored. |
| Dependency reproducibility | Native requirements are pinned; Docker uses a pinned base image and a hashed Linux CPython 3.11 lock/wheelhouse workflow. |
| Local validation | 72 pytest tests passed; `pip check`, Python compilation, `git diff --check`, Compose config, and YAML/CFF parsing passed. |

The scans are heuristic rather than a substitute for a hosted secret scanner,
SBOM, licence scanner, or dependency-vulnerability scan.

## Docker and model-artifact boundary

The Docker service mounts detector artifacts and TorchVision weights read-only,
binds the UI only to host loopback, and has `share=False`. It cannot train,
calibrate, rewrite artifacts, or silently download missing pretrained weights.
The registry discovers 105 configurations, but a hermetic deployment requires
the matching cached pretrained backbone weights. The recorded smoke test covers
historical ResNet-18, corrected-v2 ResNet-50, and PatchCore ResNet-18.

## Evidence disclaimer

MVTec results are exploratory. Corrected v2 macro image/pixel AUROC is
`0.948426/0.919948`; PatchCore is `0.896620/0.889385`. VisA capsules external
transfer image AUROC is `0.635550` with FNR `41/51` at the frozen normal-only
threshold `0.9470425248146057`. Historical sealed-final VisA pixel metrics are
alignment-limited and were not rerun. Industrial targets are unmet and no
production qualification is claimed.

## Files added or materially updated for publication

- `README.md`
- `.gitignore`
- `CITATION.cff`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `.github/workflows/ci.yml`
- `docs/RELEASE_NOTES_v1.0.0_DRAFT.md`
- `docs/RESEARCH_PROTOTYPE_V1_RELEASE_READINESS.md`
- `docs/RESEARCH_PROTOTYPE_V1_RELEASE_SMOKE_TEST.md`
- `docs/GITHUB_PUBLICATION_READINESS.md`

The repository also contains a pre-existing, uncommitted implementation and
documentation set that must be reviewed as part of the eventual staged diff.

## Remaining publication caveats

1. Confirm the release date before adding it to `CITATION.cff` or creating a
   tag. It remains intentionally unset in the local commit.
2. Review all staged files with a secret scanner, licence scanner, and
   dependency/container vulnerability scan.
3. Decide whether to ship a separate, verified dependency-wheelhouse release
   asset; it is intentionally not committed to source control.
4. Keep production claims out of release copy until independent industrial
   validation, operational FPR/FNR evidence, valid final localization evidence,
   and operational security testing exist.

## Proposed publication metadata

- Suggested commit message: `Replace legacy ForgeSight with research prototype v1.0.0`
- Suggested repository description: `Research prototype for one-class industrial visual anomaly detection with PyTorch, PatchCore, and local CUDA Docker inference.`
- Suggested topics: `anomaly-detection`, `computer-vision`, `industrial-inspection`, `pytorch`, `patchcore`, `mvtec-ad`, `gradio`, `docker`, `research-prototype`

## Commands requiring explicit maintainer authorisation

```powershell
git add --all
git diff --cached --check
git commit -m "Prepare ForgeSight research-prototype v1.0.0 release"
git tag -a v1.0.0 -m "ForgeSight research-prototype v1.0.0"
git push origin main --follow-tags
```

Create the GitHub release only after the pushed tag, using the reviewed
`docs/RELEASE_NOTES_v1.0.0_DRAFT.md`. None of these commands has been run.
