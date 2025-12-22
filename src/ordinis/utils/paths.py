"""
Path resolution utilities for Ordinis.

Centralizes project-root resolution so runtime artifacts land in a predictable location
regardless of current working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT: Path | None = None


def get_project_root() -> Path:
    """Resolve the project root, honoring ORDINIS_PROJECT_ROOT if set."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is not None:
        return _PROJECT_ROOT

    env_root = os.environ.get("ORDINIS_PROJECT_ROOT")
    if env_root:
        _PROJECT_ROOT = Path(env_root).expanduser().resolve()
        return _PROJECT_ROOT

    # src/ordinis/utils/paths.py -> src/ordinis -> src -> repo root
    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    return _PROJECT_ROOT


def resolve_project_path(path: Path | str) -> Path:
    """Resolve a path relative to the project root when not absolute."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (get_project_root() / candidate).resolve()


def reset_project_root_cache() -> None:
    """Reset cached project root (primarily for tests)."""
    global _PROJECT_ROOT
    _PROJECT_ROOT = None
