"""Local-network safeguards for the optional Gradio interface."""

from __future__ import annotations

import os


LOOPBACK_NO_PROXY = ("127.0.0.1", "localhost", "::1")


def ensure_loopback_no_proxy() -> None:
    """Ensure local Gradio health checks never route through a proxy."""
    configured = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    entries = {entry.strip() for entry in configured.split(",") if entry.strip()}
    entries.update(LOOPBACK_NO_PROXY)
    value = ",".join(sorted(entries))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value
