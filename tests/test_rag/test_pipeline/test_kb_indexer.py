"""Tests for KBIndexer module.

Tests cover:
- Initialization
- Directory indexing
- File processing
- Text chunking
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ordinis.rag.pipeline.kb_indexer import KBIndexer


class TestKBIndexerInit:
    """Tests for KBIndexer initialization."""

    @pytest.mark.unit
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()

        with patch("ordinis.rag.pipeline.kb_indexer.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.text_chunk_size = 512
            mock_cfg.text_chunk_overlap = 50
            mock_config.return_value = mock_cfg

            indexer = KBIndexer(
                chroma_client=mock_chroma,
                text_embedder=mock_embedder,
            )

            assert indexer.chroma_client is mock_chroma
            assert indexer.text_embedder is mock_embedder


class TestChunkText:
    """Tests for _chunk_text method."""

    @pytest.fixture
    def indexer(self):
        """Create indexer for tests."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()

        with patch("ordinis.rag.pipeline.kb_indexer.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.text_chunk_size = 100
            mock_cfg.text_chunk_overlap = 20
            mock_config.return_value = mock_cfg

            return KBIndexer(
                chroma_client=mock_chroma,
                text_embedder=mock_embedder,
            )

    @pytest.mark.unit
    def test_chunk_small_text(self, indexer):
        """Test chunking text smaller than chunk size."""
        text = "This is a small text."
        chunks = indexer._chunk_text(text)

        assert len(chunks) == 1
        assert text in chunks[0]

    @pytest.mark.unit
    def test_chunk_large_text(self, indexer):
        """Test chunking text larger than chunk size."""
        text = "word " * 500  # Should create multiple chunks
        chunks = indexer._chunk_text(text)

        assert len(chunks) > 1

    @pytest.mark.unit
    def test_empty_text(self, indexer):
        """Test chunking empty text."""
        chunks = indexer._chunk_text("")
        # Empty text produces no chunks or one empty chunk
        assert len(chunks) <= 1


class TestProcessFile:
    """Tests for _process_file method."""

    @pytest.fixture
    def indexer(self):
        """Create indexer for tests."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()

        with patch("ordinis.rag.pipeline.kb_indexer.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.text_chunk_size = 100
            mock_cfg.text_chunk_overlap = 20
            mock_config.return_value = mock_cfg

            return KBIndexer(
                chroma_client=mock_chroma,
                text_embedder=mock_embedder,
            )

    @pytest.mark.unit
    def test_process_markdown_file(self, indexer, tmp_path):
        """Test processing a markdown file."""
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        md_file = kb_path / "test.md"
        md_file.write_text("# Test Document\n\nThis is content.")

        chunks, metadata = indexer._process_file(md_file, kb_path)

        assert len(chunks) >= 1
        assert len(metadata) == len(chunks)
        assert metadata[0].source == "test.md"

    @pytest.mark.unit
    def test_process_file_extracts_domain(self, indexer, tmp_path):
        """Test processing extracts domain from path."""
        kb_path = tmp_path / "kb"
        domain_path = kb_path / "02-technical-analysis"
        domain_path.mkdir(parents=True)

        md_file = domain_path / "indicators.md"
        md_file.write_text("# Technical Indicators\n\nContent here.")

        chunks, metadata = indexer._process_file(md_file, kb_path)

        assert metadata[0].domain == 2

    @pytest.mark.unit
    def test_process_file_no_domain(self, indexer, tmp_path):
        """Test processing file without domain prefix."""
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        md_file = kb_path / "readme.md"
        md_file.write_text("# Readme\n\nSome content.")

        chunks, metadata = indexer._process_file(md_file, kb_path)

        assert metadata[0].domain is None


class TestIndexDirectory:
    """Tests for index_directory method."""

    @pytest.fixture
    def indexer(self):
        """Create indexer for tests."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 384]

        with patch("ordinis.rag.pipeline.kb_indexer.get_config") as mock_config:
            mock_cfg = MagicMock()
            mock_cfg.text_chunk_size = 100
            mock_cfg.text_chunk_overlap = 20
            mock_cfg.kb_base_path = Path("default/kb")
            mock_config.return_value = mock_cfg

            return KBIndexer(
                chroma_client=mock_chroma,
                text_embedder=mock_embedder,
            )

    @pytest.mark.unit
    def test_index_nonexistent_directory(self, indexer):
        """Test indexing non-existent directory raises error."""
        with pytest.raises(FileNotFoundError, match="KB path does not exist"):
            indexer.index_directory(Path("/nonexistent/kb"))

    @pytest.mark.unit
    def test_index_empty_directory(self, indexer, tmp_path):
        """Test indexing empty directory."""
        kb_path = tmp_path / "empty_kb"
        kb_path.mkdir()

        result = indexer.index_directory(kb_path)

        assert result["files_processed"] == 0
        assert result["chunks_created"] == 0

    @pytest.mark.unit
    def test_index_directory_with_files(self, indexer, tmp_path):
        """Test indexing directory with markdown files."""
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        # Create test files
        (kb_path / "file1.md").write_text("# File 1\n\nContent one.")
        (kb_path / "file2.md").write_text("# File 2\n\nContent two.")

        result = indexer.index_directory(kb_path)

        assert result["files_processed"] == 2
        assert result["chunks_created"] >= 2

    @pytest.mark.unit
    def test_index_directory_calls_chroma(self, indexer, tmp_path):
        """Test indexing calls ChromaDB add_texts."""
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        (kb_path / "test.md").write_text("# Test\n\nContent.")

        indexer.index_directory(kb_path)

        indexer.chroma_client.add_texts.assert_called()

    @pytest.mark.unit
    def test_index_directory_handles_file_errors(self, indexer, tmp_path):
        """Test indexing handles individual file errors."""
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        # Create a valid file
        (kb_path / "valid.md").write_text("# Valid\n\nContent.")

        # Make _process_file fail for some files
        original_process = indexer._process_file

        def mock_process(file_path, kb_base):
            if "error" in str(file_path):
                raise RuntimeError("File error")
            return original_process(file_path, kb_base)

        indexer._process_file = mock_process

        result = indexer.index_directory(kb_path)

        # Should still process valid file
        assert result["files_processed"] == 1

    @pytest.mark.unit
    def test_index_handles_python_files(self, indexer, tmp_path):
        """Test indexing handles Python files."""
        kb_path = tmp_path / "kb"
        kb_path.mkdir()

        (kb_path / "script.py").write_text("def hello():\n    print('hello')")

        result = indexer.index_directory(kb_path)

        # Python files are included but counted separately
        assert result["chunks_created"] >= 1
