"""Tests for path resolution utilities."""

from pathlib import Path

from ordinis.adapters.storage.database import DatabaseManager
from ordinis.utils.paths import resolve_project_path, reset_project_root_cache


def test_resolve_project_path_relative(monkeypatch, tmp_path: Path) -> None:
    """Relative paths should resolve under ORDINIS_PROJECT_ROOT."""
    reset_project_root_cache()
    monkeypatch.setenv("ORDINIS_PROJECT_ROOT", str(tmp_path))

    resolved = resolve_project_path("data/ordinis.db")

    assert resolved == (tmp_path / "data" / "ordinis.db").resolve()


def test_resolve_project_path_absolute(monkeypatch, tmp_path: Path) -> None:
    """Absolute paths should remain unchanged."""
    reset_project_root_cache()
    monkeypatch.setenv("ORDINIS_PROJECT_ROOT", str(tmp_path))

    absolute = (tmp_path / "absolute.db").resolve()
    resolved = resolve_project_path(absolute)

    assert resolved == absolute


def test_database_manager_resolves_relative(monkeypatch, tmp_path: Path) -> None:
    """DatabaseManager should anchor relative paths to the project root."""
    reset_project_root_cache()
    monkeypatch.setenv("ORDINIS_PROJECT_ROOT", str(tmp_path))

    manager = DatabaseManager(
        db_path="data/test.db",
        backup_dir="data/backups",
        auto_backup=False,
    )

    assert manager.db_path == (tmp_path / "data" / "test.db").resolve()
    assert manager.backup_dir == (tmp_path / "data" / "backups").resolve()
