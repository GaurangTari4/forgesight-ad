from pathlib import Path

import pytest

from forgesight_ad.networking import ensure_loopback_no_proxy


def test_demo_uses_canonical_checkpoint_contract():
    source = (Path(__file__).parents[1] / "demo_app.py").read_text(encoding="utf-8")
    assert "validate_feature_statistics_checkpoint" in source
    assert "weights_only=False" not in source
    assert "forgesight_ad.backbone_pipeline" in source


def test_demo_is_explicitly_local_only_and_bypasses_proxies_for_loopback():
    source = (Path(__file__).parents[1] / "demo_app.py").read_text(encoding="utf-8")
    assert "ensure_loopback_no_proxy()" in source
    assert 'server_name="127.0.0.1"' in source
    assert "share=False" in source


def test_loopback_hosts_are_added_to_no_proxy(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "intranet.example")
    ensure_loopback_no_proxy()
    hosts = set(__import__("os").environ["NO_PROXY"].split(","))
    assert {"intranet.example", "127.0.0.1", "localhost", "::1"}.issubset(hosts)


def test_installed_gradio_file_schema_is_parseable_by_gradio_client():
    file_data = pytest.importorskip("gradio.data_classes").FileData
    client_utils = pytest.importorskip("gradio_client.utils")
    assert client_utils.json_schema_to_python_type(file_data.model_json_schema()) == "filepath"
