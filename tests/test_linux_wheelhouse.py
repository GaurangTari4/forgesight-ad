from pathlib import Path

from tools.build_linux_wheelhouse import FORBIDDEN, build_manifest


ROOT = Path(__file__).resolve().parents[1]
WHEELHOUSE = ROOT / "wheelhouse" / "linux-cp311"


def test_linux_wheelhouse_is_auditable_and_excludes_base_image_packages():
    manifest = build_manifest(WHEELHOUSE)
    assert manifest["target"] == "manylinux_2_17_x86_64 / CPython 3.11"
    assert manifest["wheel_count"] >= 60
    assert not set(manifest["packages"]).intersection(FORBIDDEN)
    for package in ("numpy", "gradio", "pandas", "scikit-learn", "pillow"):
        assert package in manifest["packages"]


def test_offline_dockerfile_uses_wheelhouse_and_lockfile():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY wheelhouse/linux-cp311 /opt/wheelhouse" in dockerfile
    assert "--require-hashes --no-index --find-links=/opt/wheelhouse" in dockerfile
    assert "requirements-docker.lock.txt" in dockerfile
    assert "repair_base_ninja.py --remove-unsupported" in dockerfile
