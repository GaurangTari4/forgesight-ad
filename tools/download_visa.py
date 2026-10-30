"""Single-writer, ETag-validated download and safe selective extraction of VisA."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
EXPECTED_SHA256 = "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"
EXPECTED_SIZE = 1929840640


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_metadata(url: str = URL) -> dict:
    with urlopen(Request(url, method="HEAD"), timeout=60) as response:
        metadata = {"url": url, "size": int(response.headers.get("Content-Length", "-1")), "etag": response.headers.get("ETag", "").strip('"'), "accept_ranges": response.headers.get("Accept-Ranges", ""), "status": response.status}
    if metadata["status"] != 200 or metadata["size"] != EXPECTED_SIZE or not metadata["etag"] or metadata["accept_ranges"].lower() != "bytes":
        raise RuntimeError(f"Official object lacks expected identity/range metadata: {metadata}")
    return metadata


def _state_path(partial: Path) -> Path:
    return partial.with_name(partial.name + ".resume.json")


def _write_state(path: Path, metadata: dict, size: int, prefix_digest: str) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps({key: metadata[key] for key in ("url", "size", "etag")} | {"bytes_written": size, "prefix_sha256": prefix_digest}, indent=2), encoding="utf-8")
    os.replace(temp, path)


def _quarantine(path: Path, reason: str) -> Path:
    target_dir = path.parent / "quarantine"; target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{path.name}.{reason}.{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if target.exists(): raise FileExistsError(f"Refusing to overwrite quarantine item: {target}")
    os.replace(path, target); return target


def _trusted_offset(partial: Path, metadata: dict) -> int:
    state_path = _state_path(partial)
    if not partial.exists():
        if state_path.exists(): _quarantine(state_path, "orphaned_state")
        return 0
    try:
        state = json.loads(state_path.read_text(encoding="utf-8")); size = partial.stat().st_size
        matches = all(state.get(key) == metadata[key] for key in ("url", "size", "etag"))
        if not matches or state.get("bytes_written") != size or state.get("prefix_sha256") != sha256(partial) or not 0 < size < metadata["size"]:
            raise ValueError("unproven partial prefix")
        return size
    except Exception:
        _quarantine(partial, "untrusted_prefix")
        if state_path.exists(): _quarantine(state_path, "untrusted_state")
        return 0


def download_verified(root: Path, url: str = URL) -> Path:
    """Append only a matching HTTP 206 response; publish only after digest check."""
    root.mkdir(parents=True, exist_ok=True); archive = root / "VisA_20220922.tar"
    if archive.exists():
        if archive.stat().st_size == EXPECTED_SIZE and sha256(archive) == EXPECTED_SHA256: return archive
        raise FileExistsError(f"Existing archive is invalid; quarantine it before downloading: {archive}")
    metadata = remote_metadata(url); partial = archive.with_name(archive.name + ".partial"); offset = _trusted_offset(partial, metadata)
    headers = {"If-Match": f'"{metadata["etag"]}"'}
    if offset: headers["Range"] = f"bytes={offset}-"
    try:
        response = urlopen(Request(url, headers=headers), timeout=60)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}; no prefix was appended") from exc
    with response:
        expected_status = 206 if offset else 200
        if response.status != expected_status or response.headers.get("ETag", "").strip('"') != metadata["etag"]:
            raise RuntimeError(f"Unsafe HTTP {response.status}/ETag response; no prefix was appended")
        if offset:
            received = response.headers.get("Content-Range", "")
            if not received.startswith(f"bytes {offset}-") or not received.endswith(f"/{metadata['size']}"):
                raise RuntimeError(f"Unsafe Content-Range {received!r}; no prefix was appended")
        elif int(response.headers.get("Content-Length", "-1")) != metadata["size"]:
            raise RuntimeError("Unsafe HTTP 200 Content-Length; response was not written")
        digest = hashlib.sha256(); written = offset
        if offset:
            with partial.open("rb") as prefix:
                for block in iter(lambda: prefix.read(1024 * 1024), b""): digest.update(block)
        with partial.open("ab" if offset else "xb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block); digest.update(block); written += len(block)
                if written % (16 * 1024 * 1024) < len(block):
                    output.flush(); os.fsync(output.fileno()); _write_state(_state_path(partial), metadata, written, digest.hexdigest())
            output.flush(); os.fsync(output.fileno())
    _write_state(_state_path(partial), metadata, written, digest.hexdigest())
    if written != metadata["size"] or digest.hexdigest() != EXPECTED_SHA256:
        failed = _quarantine(partial, "size_or_sha256_mismatch")
        if _state_path(partial).exists(): _quarantine(_state_path(partial), "failed_state")
        raise RuntimeError(f"Downloaded archive failed exact integrity checks and was quarantined: {failed}")
    try:
        os.link(partial, archive)
    except OSError:
        if archive.exists(): raise FileExistsError(f"Refusing to overwrite archive: {archive}")
        os.replace(partial, archive)
    else:
        partial.unlink()
    _state_path(partial).unlink(missing_ok=True)
    return archive


def safe_extract_category(archive: Path, destination: Path, category: str) -> list[str]:
    with tarfile.open(archive, "r") as bundle:
        names = [member.name for member in bundle.getmembers()]
        prefix = next((item for item in (f"{category}/", f"VisA/{category}/") if any(name.startswith(item) for name in names)), None)
        if prefix is None: raise ValueError(f"Category '{category}' was not found in archive")
        selected = [member for member in bundle.getmembers() if member.name.startswith(prefix) and member.isfile()]
        category_root = destination / category
        if category_root.exists(): raise FileExistsError(f"Refusing to merge into existing extraction: {category_root}")
        for member in selected:
            target = (category_root / Path(member.name).relative_to(prefix)).resolve()
            if category_root.resolve() not in target.parents: raise ValueError(f"Unsafe archive member: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True); source = bundle.extractfile(member)
            if source is None: raise ValueError(f"Could not read archive member: {member.name}")
            with source, target.open("xb") as output: shutil.copyfileobj(source, output)
    return [member.name for member in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True); parser.add_argument("--category", default="capsules")
    parser.add_argument("--download", action="store_true"); parser.add_argument("--extract", action="store_true")
    args = parser.parse_args(); root = args.root.resolve(); archive = download_verified(root) if args.download else root / "VisA_20220922.tar"
    if not archive.is_file() or archive.stat().st_size != EXPECTED_SIZE or sha256(archive) != EXPECTED_SHA256:
        raise RuntimeError("Exact official archive size/SHA-256 verification is required before extraction")
    extracted = safe_extract_category(archive, root / "raw", args.category) if args.extract else []
    report = {"dataset": "VisA", "official_url": URL, "archive": str(archive), "archive_sha256": EXPECTED_SHA256, "archive_size": EXPECTED_SIZE, "etag": remote_metadata()["etag"], "category": args.category, "extracted_files": len(extracted), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    report_path = root / "download_provenance.json"
    if report_path.exists(): raise FileExistsError(f"Refusing to overwrite provenance: {report_path}")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
