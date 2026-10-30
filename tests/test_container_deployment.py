from pathlib import Path

from tools.container_startup_check import startup_issues


ROOT = Path(__file__).resolve().parents[1]


def test_docker_configuration_is_local_only_and_mounts_artifacts_read_only():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "gpus: all" in compose
    assert '"127.0.0.1:7860:7860"' in compose
    assert ":ro" in compose
    assert "FORGESIGHT_SERVER_HOST: 0.0.0.0" in compose
    assert 'FORGESIGHT_OFFLINE_WEIGHTS: "1"' in compose
    assert "./runtime_cache:/tmp/forgesight_torch_cache:ro" in compose


def test_dockerignore_excludes_weights_datasets_and_local_environments():
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for expected in (".git/", ".venv/", ".onnx-deps/", "artifacts/", "validation_runs/", "wheelhouse/"):
        assert expected in ignored
    assert "!wheelhouse/linux-cp311/**" in ignored


def test_container_preflight_reports_missing_gpu_and_artifacts(tmp_path):
    issues = startup_issues("cuda:0", False, tmp_path / "artifacts", tmp_path / "historical", set())
    message = " ".join(issues)
    assert "CUDA was requested" in message
    assert "artifacts is missing" in message
    assert "historical artifact mount" in message
    assert "patchcore" in message


def test_container_preflight_accepts_complete_cpu_mounts(tmp_path):
    artifacts, historical = tmp_path / "artifacts", tmp_path / "historical"
    artifacts.mkdir()
    historical.mkdir()
    assert startup_issues("cpu", False, artifacts, historical, {"historical", "v2", "patchcore"}) == []
