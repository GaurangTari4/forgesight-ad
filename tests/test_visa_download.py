import io
import tarfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[1] / "tools" / "download_visa.py"
SPEC = spec_from_file_location("visa_download", PATH)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selective_category_extraction_is_non_overwriting_and_safe(tmp_path):
    archive = tmp_path / "visa.tar"
    with tarfile.open(archive, "w") as bundle:
        data = b"annotation"
        member = tarfile.TarInfo("capsules/image_anno.csv"); member.size = len(data)
        bundle.addfile(member, io.BytesIO(data))
        ignored = tarfile.TarInfo("candle/image_anno.csv"); ignored.size = len(data)
        bundle.addfile(ignored, io.BytesIO(data))
    files = MODULE.safe_extract_category(archive, tmp_path / "raw", "capsules")
    assert files == ["capsules/image_anno.csv"]
    assert (tmp_path / "raw" / "capsules" / "image_anno.csv").read_bytes() == b"annotation"
    with pytest.raises(FileExistsError):
        MODULE.safe_extract_category(archive, tmp_path / "raw", "capsules")


def test_resume_requires_matching_remote_identity_and_saved_prefix_digest(tmp_path):
    partial = tmp_path / "VisA_20220922.tar.partial"; partial.write_bytes(b"safe-prefix")
    metadata = {"url": "https://example.invalid/archive", "size": 100, "etag": "version-a"}
    MODULE._write_state(MODULE._state_path(partial), metadata, partial.stat().st_size, MODULE.sha256(partial))
    assert MODULE._trusted_offset(partial, metadata) == len(b"safe-prefix")
    changed = {**metadata, "etag": "version-b"}
    assert MODULE._trusted_offset(partial, changed) == 0
    assert not partial.exists()
    assert list((tmp_path / "quarantine").iterdir())
