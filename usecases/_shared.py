"""Shared helpers for running examples directly from the repo."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_local_import() -> None:
    """Make the editable checkout importable without installation."""
    try:
        import llama_insight  # noqa: F401
    except ImportError:
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.append(str(repo_root / "wrapper_src"))


def default_image_path() -> str:
    """Return a stable default image path used across examples."""
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / "test-images" / "debug.jpg"
    if candidate.exists():
        return str(candidate)
    return str((Path(__file__).with_name("debug.jpg")).resolve())
