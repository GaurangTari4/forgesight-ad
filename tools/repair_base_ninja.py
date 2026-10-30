"""Diagnose and narrowly remove an incompatible inherited ``ninja`` wheel.

PyTorch runtime inference does not compile extensions. This repair only removes
``ninja`` when its installed wheel tags are incompatible with the active Python
and no installed distribution declares an active dependency on it. It leaves a
compatible ninja untouched and refuses removal when a dependency requires it.
"""
from __future__ import annotations

import argparse
from email.parser import Parser
import importlib.metadata as metadata
from pathlib import Path
import subprocess
import sys

try:  # The base image normally has packaging; pip always vendors it.
    from packaging.requirements import Requirement
    from packaging.tags import sys_tags
    from packaging.utils import canonicalize_name, parse_wheel_filename
except ImportError:  # pragma: no cover - exercised only on minimal base images.
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.tags import sys_tags
    from pip._vendor.packaging.utils import canonicalize_name, parse_wheel_filename


def active_ninja_dependents() -> list[str]:
    dependents = []
    for distribution in metadata.distributions():
        for raw in distribution.requires or []:
            requirement = Requirement(raw)
            if canonicalize_name(requirement.name) != "ninja":
                continue
            if requirement.marker is None or requirement.marker.evaluate():
                dependents.append(f"{distribution.metadata['Name']} ({raw})")
    return sorted(dependents)


def inspect_ninja() -> dict:
    try:
        distribution = metadata.distribution("ninja")
    except metadata.PackageNotFoundError:
        return {"installed": False, "action": "none"}
    wheel_path = Path(distribution.locate_file("ninja-1.11.1.1.dist-info/WHEEL"))
    # Distribution paths vary by release, so discover WHEEL from the dist-info
    # metadata rather than rely on the version-specific name above.
    files = distribution.files or ()
    wheel_files = [path for path in files if str(path).endswith(".dist-info/WHEEL")]
    if len(wheel_files) != 1:
        raise RuntimeError("Cannot determine the ninja wheel metadata path")
    wheel_path = Path(distribution.locate_file(wheel_files[0]))
    tags = set()
    for raw in Parser().parsestr(wheel_path.read_text(encoding="utf-8")).get_all("Tag", []):
        _, _, parsed = parse_wheel_filename(f"ninja-0-{raw}.whl")
        tags.update(parsed)
    supported = set(sys_tags())
    return {
        "installed": True,
        "version": distribution.version,
        "wheel_path": str(wheel_path),
        "wheel_tags": sorted(map(str, tags)),
        "compatible": bool(tags.intersection(supported)),
        "dependents": active_ninja_dependents(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remove-unsupported", action="store_true")
    args = parser.parse_args()
    result = inspect_ninja()
    print(result, flush=True)
    if not args.remove_unsupported or not result["installed"] or result.get("compatible"):
        return
    if result["dependents"]:
        raise RuntimeError(f"Refusing to remove incompatible ninja required by: {result['dependents']}")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "ninja"], check=True)
    if metadata.packages_distributions().get("ninja"):
        raise RuntimeError("ninja remains installed after requested removal")
    print("Removed unsupported inherited ninja; no installed runtime dependency required it.", flush=True)


if __name__ == "__main__":
    main()
