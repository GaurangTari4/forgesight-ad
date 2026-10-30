# Docker-first ForgeSight deployment

ForgeSight’s primary deployment path is native PyTorch/TorchVision CUDA
inference in Docker. ONNX and OpenVINO utilities are optional future
optimizations; they are not required by this deployment and are not copied into
the image.

## Image and security boundary

`Dockerfile` uses `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`, compatible
with the project’s PyTorch 2.5.1 / CUDA 12.4 validation. The build context
contains application code and pinned Python runtime dependencies only.
`.dockerignore` excludes Git history, datasets, experiment outputs, model
artifacts, virtual environments, ONNX dependency tooling, tests, and local
logs.

## Offline wheelhouse (default build path)

`wheelhouse/linux-cp311/` contains audited `manylinux x86_64` or universal
CPython 3.11 wheels for every non-PyTorch runtime dependency and transitive
dependency. `requirements-docker.lock.txt` pins that closure, while
`WHEELHOUSE_MANIFEST.json` records every filename and SHA-256. It deliberately
contains no Torch, TorchVision, or NVIDIA CUDA package: those are supplied by
the pinned CUDA PyTorch base image.

The default Docker build runs:

```text
python -m pip install --require-hashes --no-index --find-links=/opt/wheelhouse -r requirements-docker.lock.txt
```

so it does not contact PyPI. To regenerate/audit the wheelhouse from Windows
without modifying the project virtual environment, run:

```powershell
.\.venv\Scripts\python.exe .\tools\build_linux_wheelhouse.py --download
```

This command explicitly requests `manylinux_2_17_x86_64`, CPython 3.11,
`cp311`, and binary-only wheels. Review the generated manifest before a build.

The original online install remains an intentional recovery path—not the
default—for a trusted network that is refreshing dependencies:

```powershell
docker compose build --build-arg DEPENDENCY_MODE=online
```

## Inherited base-image Ninja metadata repair

The pinned PyTorch image may contain `ninja 1.11.1.1` whose installed `WHEEL`
tags do not support the image interpreter. `pip check` reports this as
`ninja 1.11.1.1 is not supported on this platform`; it is unrelated to the
ForgeSight wheelhouse, which does not include Ninja.

Before application dependencies are installed, the Dockerfile runs
`tools/repair_base_ninja.py --remove-unsupported`. It records the inherited
wheel tags, checks the active installed dependency graph, and removes Ninja
**only** when it is incompatible and has no active dependent. ForgeSight uses
native PyTorch inference and does not build C++/CUDA extensions at runtime.
The normal final `pip check` remains mandatory and is not suppressed.

To inspect an unmodified base on a Docker host, run:

```powershell
docker run --rm --entrypoint python pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime -m pip check
```

Models are never built into the image. `compose.yaml` mounts canonical
`./artifacts` and sibling `../artifacts_resnet_compare` **read-only**. This
provides the existing historical feature-statistics, corrected v2, and
PatchCore adapters. The dynamic registry exposes the 15 completed v2 and
PatchCore category artifacts only when their model and completed metrics sidecar
are present.

The Git-ignored host directory `./runtime_cache` is mounted **read-only** at
`/tmp/forgesight_torch_cache`. Docker sets `FORGESIGHT_OFFLINE_WEIGHTS=1`, so
missing TorchVision weights are a clear startup/request error; ForgeSight never
downloads weights, trains, calibrates, or writes a checkpoint at container
startup. Populate the host cache deliberately before `docker compose up`:

```powershell
New-Item -ItemType Directory -Force .\runtime_cache | Out-Null
$env:TORCH_HOME = (Resolve-Path .\runtime_cache)
Remove-Item Env:FORGESIGHT_OFFLINE_WEIGHTS -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -c "from torchvision import models; [ctor(weights=w) for ctor,w in [(models.resnet18,models.ResNet18_Weights.DEFAULT),(models.resnet34,models.ResNet34_Weights.DEFAULT),(models.resnet50,models.ResNet50_Weights.DEFAULT),(models.resnet101,models.ResNet101_Weights.DEFAULT),(models.resnet152,models.ResNet152_Weights.DEFAULT)]]; print('TorchVision cache populated')"
```

This is an explicit trusted-download preparation step, not part of deployment.
Verify the expected files and their provenance before mounting the cache. The
current local cache contains ResNet-18 and ResNet-50 only; populate ResNet-34,
ResNet-101, and ResNet-152 before serving those historical selections in a
hermetic container.

The process listens on `0.0.0.0` *inside* the container only so Docker can
route it. Compose publishes it exclusively as `127.0.0.1:7860:7860`, and the
application always uses `share=False`; it is not publicly exposed.

## Windows GPU prerequisites

1. Use Docker Desktop with the WSL2 backend enabled.
2. Install a current Windows NVIDIA driver supporting WSL CUDA. Verify from a
   WSL terminal with `nvidia-smi`.
3. In Docker Desktop, enable WSL integration for the distribution that will run
   Compose.
4. From this repository, verify the resolved configuration:

   ```powershell
   docker compose -f compose.yaml config
   ```

5. Build and launch the local-only GPU service:

   ```powershell
   docker compose up --build
   ```

6. Open `http://127.0.0.1:7860`. Stop it with `docker compose down`.

The container startup preflight fails before Gradio starts when the requested
GPU is unavailable, either artifact mount is missing, or one of the three
detector families cannot be discovered. The health check makes a loopback HTTP
request to `/` after startup.

## Local container verification

After `docker compose up --build` reports a healthy service, run:

```powershell
docker compose exec forgesight python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
docker compose exec forgesight python -u tools/container_startup_check.py
```

For a normal and defective bottle callback comparison plus 50-repeat warm
callback latency and peak CUDA allocation, mount the MVTec dataset and a
writable host validation folder only for this temporary command. Neither is a
default runtime mount:

```powershell
docker compose run --rm `
  -v '..\mvtec-ad\versions\2:/data/mvtec:ro' `
  -v '.\validation_runs:/validation' `
  forgesight python -u tools/validate_container_runtime.py `
    --normal /data/mvtec/bottle/test/good/000.png `
    --defective /data/mvtec/bottle/test/broken_large/000.png `
    --family patchcore --backbone resnet18 --category bottle --device cuda:0 `
    --output /validation/docker_patchcore_bottle.json
```

Compare the returned JSON score, threshold, decision, and output dimensions
with the saved PyTorch reference validation records. This command is an
inference-only check: it does not calibrate, fit, or modify artifacts.

## Verified runtime evidence (2026-10-31)

`forgesight:local` was built with the offline Linux CPython 3.11 wheelhouse;
the final unsuppressed `pip check` and core imports passed. The live
`forgesight-local` service was healthy, returned HTTP 200 at
`http://127.0.0.1:7860/`, and Docker published only `127.0.0.1:7860`.
Inspection confirmed that `/opt/forgesight/artifacts` and
`/opt/forgesight/historical_artifacts` have `RW=false`; the only writable
runtime mount is the separate TorchVision cache.

Formal CUDA parity was captured for normal and defective bottle and screw
inputs for historical ResNet-18, corrected-v2 ResNet-50, and PatchCore
ResNet-18. All six detector/category comparisons had exact recorded score,
map, mask, decision, and output-dimension agreement. The comparison utility
allows an absolute score tolerance of `1e-4` and map tolerance of `2e-4` to
remain suitable for future cross-platform floating-point builds; the measured
errors in this run were zero. Evidence is retained under
`validation_runs/docker_deployment_20261031T010042Z/`.

The verifier measures preloaded Python callback time (preprocessing,
inference, rendering, and JSON) and separately reports the adapters' existing
model/scoring interval. It excludes HTTP request scheduling and browser
rendering; those must be measured separately for a browser SLO. A heatmap is
contrast-normalized independently for visualization. Therefore a bright image
does not imply a `FAIL`: the decision is made only from the unnormalized image
score compared to the immutable image threshold, as shown in the returned
JSON.

For container latency and memory, use the existing callback profiler after
mounting a test-only dataset and record its output in a host-mounted
`validation_runs` directory. Do not compare a container result with prior host
measurements without recording GPU, driver, image tag, and timing boundary.

## CPU fallback

CPU is intentionally a separate Compose configuration so it cannot accidentally
mask a missing GPU reservation:

```powershell
docker compose -f compose.cpu.yaml up --build
```

It publishes only `http://127.0.0.1:7861`, sets `FORGESIGHT_DEVICE=cpu`, and
uses the same read-only artifacts. It is a functional fallback, not a latency
or throughput substitute for CUDA.

## Known limits

- HTTP/browser timing and long-running concurrency/soak testing remain to be
  measured on the target deployment host. CUDA startup, inference parity, and
  warmed callback latency have been measured on the recorded Docker
  Desktop/WSL2 host.
- The image intentionally does not contain datasets or checkpoints.
- A clean source checkout also needs a locally prepared Linux wheelhouse and
  the external read-only artifact and TorchVision-weight mounts. The lockfile
  and wheelhouse builder make this reproducible; the binary wheelhouse itself
  is Git-ignored to avoid publishing third-party binaries in the source tree.
- Artifact mounts are trusted local inputs; use filesystem permissions and
  checksum/provenance controls in production.
- ONNX/OpenVINO work remains optional and outside the Docker deployment path.
