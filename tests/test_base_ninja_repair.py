import importlib.metadata as metadata

from tools import repair_base_ninja


def test_missing_ninja_is_a_safe_noop(monkeypatch):
    def missing(_name):
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(repair_base_ninja.metadata, "distribution", missing)
    assert repair_base_ninja.inspect_ninja() == {"installed": False, "action": "none"}


def test_repair_script_documents_conditional_safety_boundary():
    source = repair_base_ninja.__doc__
    assert "only removes" in source
    assert "dependency" in source
