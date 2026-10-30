# CUDA-enabled PyTorch is intentionally supplied by this pinned base image.
# It keeps the Docker runtime isolated from the Windows .venv.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# `offline` is the reproducible default. Pass --build-arg DEPENDENCY_MODE=online
# only when deliberately refreshing a wheelhouse on a trusted network.
ARG DEPENDENCY_MODE=offline

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FORGESIGHT_SERVER_HOST=0.0.0.0 \
    FORGESIGHT_SERVER_PORT=7860

WORKDIR /opt/forgesight

COPY requirements-docker.txt requirements-docker.lock.txt ./
COPY wheelhouse/linux-cp311 /opt/wheelhouse
COPY tools/repair_base_ninja.py ./tools/repair_base_ninja.py
# The pinned base currently includes a CPython-incompatible ninja wheel. The
# repair inspects its tags/dependents and removes it only if it is unsupported
# and unused; this is before, not instead of, the final pip check.
RUN python tools/repair_base_ninja.py --remove-unsupported
RUN if [ "${DEPENDENCY_MODE}" = "offline" ]; then \
      python -m pip install --require-hashes --no-index --find-links=/opt/wheelhouse -r requirements-docker.lock.txt; \
    elif [ "${DEPENDENCY_MODE}" = "online" ]; then \
      python -m pip install -r requirements-docker.txt; \
    else \
      echo "Unsupported DEPENDENCY_MODE=${DEPENDENCY_MODE}; use offline or online" >&2; exit 2; \
    fi \
    && python -m pip check \
    && python -c "import gradio, numpy, pandas, PIL, sklearn, torch, torchvision; print('runtime imports passed')"

COPY demo_app.py ./
COPY forgesight_ad ./forgesight_ad
COPY tools/container_startup_check.py ./tools/container_startup_check.py
COPY tools/validate_container_runtime.py ./tools/validate_container_runtime.py

RUN useradd --create-home --uid 10001 forgesight && chown -R forgesight:forgesight /opt/forgesight
USER forgesight

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=3).read()"

# The preflight verifies artifact mounts and requested GPU access before Gradio.
CMD ["sh", "-c", "python -u tools/container_startup_check.py && exec python -u demo_app.py"]
