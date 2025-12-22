"""Tests for ChromaClient path resolution."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ordinis.utils.paths import reset_project_root_cache


@pytest.mark.unit
@pytest.mark.skip(reason="Passes in isolation but has test pollution issue in full suite - tracked for fix")
def test_chroma_client_resolves_relative_path(monkeypatch, tmp_path: Path) -> None:
    """ChromaClient should anchor relative paths to ORDINIS_PROJECT_ROOT."""
    reset_project_root_cache()
    monkeypatch.setenv("ORDINIS_PROJECT_ROOT", str(tmp_path))

    # Mock chromadb.PersistentClient to avoid actual DB initialization
    mock_client = MagicMock()

    with (
        patch("chromadb.PersistentClient", return_value=mock_client),
        patch("chromadb.config.Settings"),
    ):
        from ordinis.rag.vectordb.chroma_client import ChromaClient

        client = ChromaClient(persist_directory=Path("data/chromadb"))

        assert client.persist_directory == (tmp_path / "data" / "chromadb").resolve()
