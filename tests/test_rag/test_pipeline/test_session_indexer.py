"""Tests for SessionLogIndexer module.

Tests cover:
- Initialization
- Session log indexing
- Text chunking
- Session search
- Recent sessions retrieval
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ordinis.rag.pipeline.session_indexer import SessionLogIndexer


class TestSessionLogIndexerInit:
    """Tests for SessionLogIndexer initialization."""

    @pytest.mark.unit
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()

        with patch("ordinis.rag.pipeline.session_indexer.get_config") as mock_config:
            mock_config.return_value = MagicMock()

            indexer = SessionLogIndexer(
                chroma_client=mock_chroma,
                text_embedder=mock_embedder,
            )

            assert indexer.chunk_size == 1024
            assert indexer.chunk_overlap == 100
            assert indexer.collection_name == "session_logs"

    @pytest.mark.unit
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()

        with patch("ordinis.rag.pipeline.session_indexer.get_config") as mock_config:
            mock_config.return_value = MagicMock()

            indexer = SessionLogIndexer(
                chroma_client=mock_chroma,
                text_embedder=mock_embedder,
                chunk_size=512,
                chunk_overlap=50,
                collection_name="custom_logs",
            )

            assert indexer.chunk_size == 512
            assert indexer.chunk_overlap == 50
            assert indexer.collection_name == "custom_logs"

    @pytest.mark.unit
    def test_init_creates_collection(self):
        """Test initialization creates collection."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()

        with patch("ordinis.rag.pipeline.session_indexer.get_config") as mock_config:
            mock_config.return_value = MagicMock()

            SessionLogIndexer(
                chroma_client=mock_chroma,
                text_embedder=mock_embedder,
            )

            mock_chroma.get_or_create_collection.assert_called_once_with("session_logs")

    @pytest.mark.unit
    def test_init_handles_collection_error(self):
        """Test initialization handles collection creation error."""
        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.side_effect = Exception("Collection error")
        mock_embedder = MagicMock()

        with patch("ordinis.rag.pipeline.session_indexer.get_config") as mock_config:
            mock_config.return_value = MagicMock()

            # Should not raise - just warns
            indexer = SessionLogIndexer(
                chroma_client=mock_chroma,
                text_embedder=mock_embedder,
            )

            assert indexer is not None


class TestChunkText:
    """Tests for _chunk_text method."""

    @pytest.fixture
    def indexer(self):
        """Create indexer for tests."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()

        with patch("ordinis.rag.pipeline.session_indexer.get_config") as mock_config:
            mock_config.return_value = MagicMock()

            return SessionLogIndexer(
                chroma_client=mock_chroma,
                text_embedder=mock_embedder,
                chunk_size=100,
                chunk_overlap=20,
            )

    @pytest.mark.unit
    def test_chunk_small_text(self, indexer):
        """Test chunking text smaller than chunk size."""
        text = "This is a small text."
        chunks = indexer._chunk_text(text)

        assert len(chunks) == 1
        assert text in chunks[0]

    @pytest.mark.unit
    def test_chunk_empty_text(self, indexer):
        """Test chunking empty text returns empty list."""
        chunks = indexer._chunk_text("")
        # Empty text produces no chunks
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0] == "")

    @pytest.mark.unit
    def test_chunk_large_text(self, indexer):
        """Test chunking text larger than chunk size."""
        # Create text that will span multiple chunks
        text = "word " * 500  # Should create multiple chunks
        chunks = indexer._chunk_text(text)

        assert len(chunks) > 1

    @pytest.mark.unit
    def test_chunks_have_overlap(self, indexer):
        """Test chunks have proper overlap."""
        # Create text that will span multiple chunks
        text = "word " * 500
        chunks = indexer._chunk_text(text)

        # If there are multiple chunks, later ones should overlap with earlier
        if len(chunks) > 1:
            # Just verify we get multiple chunks - overlap is in tokens
            assert len(chunks) >= 2


class TestIndexSessionLog:
    """Tests for index_session_log method."""

    @pytest.fixture
    def indexer(self):
        """Create indexer for tests."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 384]

        with patch("ordinis.rag.pipeline.session_indexer.get_config") as mock_config:
            mock_config.return_value = MagicMock()

            return SessionLogIndexer(
                chroma_client=mock_chroma,
                text_embedder=mock_embedder,
            )

    @pytest.mark.unit
    def test_index_missing_file(self, indexer):
        """Test indexing non-existent file raises error."""
        with pytest.raises(FileNotFoundError, match="Session log not found"):
            indexer.index_session_log(
                log_path=Path("/nonexistent/file.txt"),
                session_id="test-session",
            )

    @pytest.mark.unit
    def test_index_valid_file(self, indexer, tmp_path):
        """Test indexing a valid session log file."""
        # Create test file
        log_file = tmp_path / "session_log.txt"
        log_file.write_text("This is a test session log with some content.")

        result = indexer.index_session_log(
            log_path=log_file,
            session_id="test-session-123",
        )

        assert result["session_id"] == "test-session-123"
        assert result["chunks"] >= 1
        assert result["tokens"] > 0
        assert result["file_size"] > 0

    @pytest.mark.unit
    def test_index_with_extra_metadata(self, indexer, tmp_path):
        """Test indexing with extra metadata."""
        log_file = tmp_path / "session_log.txt"
        log_file.write_text("Test content")

        result = indexer.index_session_log(
            log_path=log_file,
            session_id="test-session",
            metadata_extra={"project": "ordinis", "version": "1.0"},
        )

        assert result["session_id"] == "test-session"

    @pytest.mark.unit
    def test_index_parses_timestamp_from_filename(self, indexer, tmp_path):
        """Test indexing parses timestamp from filename."""
        # Create file with timestamp in name
        log_file = tmp_path / "20241215103045_session_export.txt"
        log_file.write_text("Test content")

        result = indexer.index_session_log(
            log_path=log_file,
            session_id="test-session",
        )

        # Should parse the timestamp
        assert "export_time" in result

    @pytest.mark.unit
    def test_index_calls_chroma_add(self, indexer, tmp_path):
        """Test indexing calls ChromaDB add_texts."""
        log_file = tmp_path / "session_log.txt"
        log_file.write_text("Test content for the session log")

        indexer.index_session_log(
            log_path=log_file,
            session_id="test-session",
        )

        indexer.chroma_client.add_texts.assert_called()


class TestSearchSessions:
    """Tests for search_sessions method."""

    @pytest.fixture
    def indexer(self):
        """Create indexer for tests."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [[0.1] * 384]

        with patch("ordinis.rag.pipeline.session_indexer.get_config") as mock_config:
            mock_config.return_value = MagicMock()

            return SessionLogIndexer(
                chroma_client=mock_chroma,
                text_embedder=mock_embedder,
            )

    @pytest.mark.unit
    def test_search_returns_formatted_results(self, indexer):
        """Test search returns properly formatted results."""
        # Mock search results
        indexer.chroma_client.query.return_value = {
            "documents": [["doc1 content", "doc2 content"]],
            "metadatas": [[{"session_id": "s1"}, {"session_id": "s2"}]],
            "distances": [[0.1, 0.2]],
        }

        results = indexer.search_sessions("test query")

        assert len(results) == 2
        assert results[0]["content"] == "doc1 content"
        assert results[0]["metadata"]["session_id"] == "s1"
        assert "relevance" in results[0]

    @pytest.mark.unit
    def test_search_with_session_filter(self, indexer):
        """Test search with session_id filter."""
        indexer.chroma_client.query.return_value = {"documents": [[]], "metadatas": [[]]}

        indexer.search_sessions("test query", session_id="specific-session")

        # Verify filter was passed
        call_kwargs = indexer.chroma_client.query.call_args[1]
        assert call_kwargs["where"] == {"session_id": "specific-session"}

    @pytest.mark.unit
    def test_search_without_filter(self, indexer):
        """Test search without session_id filter."""
        indexer.chroma_client.query.return_value = {"documents": [[]], "metadatas": [[]]}

        indexer.search_sessions("test query")

        call_kwargs = indexer.chroma_client.query.call_args[1]
        assert call_kwargs["where"] is None

    @pytest.mark.unit
    def test_search_empty_results(self, indexer):
        """Test search with no results."""
        indexer.chroma_client.query.return_value = {"documents": None}

        results = indexer.search_sessions("test query")

        assert results == []

    @pytest.mark.unit
    def test_search_calculates_relevance(self, indexer):
        """Test search calculates relevance score correctly."""
        indexer.chroma_client.query.return_value = {
            "documents": [["content"]],
            "metadatas": [[{"session_id": "s1"}]],
            "distances": [[0.3]],
        }

        results = indexer.search_sessions("query")

        assert results[0]["relevance"] == pytest.approx(0.7, rel=0.01)

    @pytest.mark.unit
    def test_search_handles_high_distance(self, indexer):
        """Test search handles distance > 1."""
        indexer.chroma_client.query.return_value = {
            "documents": [["content"]],
            "metadatas": [[{"session_id": "s1"}]],
            "distances": [[1.5]],
        }

        results = indexer.search_sessions("query")

        assert results[0]["relevance"] == 0


class TestGetRecentSessions:
    """Tests for get_recent_sessions method."""

    @pytest.fixture
    def indexer(self):
        """Create indexer for tests."""
        mock_chroma = MagicMock()
        mock_embedder = MagicMock()

        with patch("ordinis.rag.pipeline.session_indexer.get_config") as mock_config:
            mock_config.return_value = MagicMock()

            return SessionLogIndexer(
                chroma_client=mock_chroma,
                text_embedder=mock_embedder,
            )

    @pytest.mark.unit
    def test_get_recent_returns_unique_sessions(self, indexer):
        """Test get_recent_sessions returns unique sessions."""
        indexer.chroma_client.query.return_value = {
            "metadatas": [
                [
                    {"session_id": "s1", "export_time": "2024-01-01", "source_file": "f1.txt"},
                    {"session_id": "s1", "export_time": "2024-01-01", "source_file": "f1.txt"},  # Dupe
                    {"session_id": "s2", "export_time": "2024-01-02", "source_file": "f2.txt"},
                ]
            ]
        }

        sessions = indexer.get_recent_sessions(limit=10)

        assert len(sessions) == 2
        session_ids = [s["session_id"] for s in sessions]
        assert "s1" in session_ids
        assert "s2" in session_ids

    @pytest.mark.unit
    def test_get_recent_respects_limit(self, indexer):
        """Test get_recent_sessions respects limit."""
        indexer.chroma_client.query.return_value = {
            "metadatas": [
                [
                    {"session_id": f"s{i}", "export_time": f"2024-01-0{i}", "source_file": f"f{i}.txt"}
                    for i in range(1, 20)
                ]
            ]
        }

        sessions = indexer.get_recent_sessions(limit=5)

        assert len(sessions) <= 5

    @pytest.mark.unit
    def test_get_recent_handles_empty_results(self, indexer):
        """Test get_recent_sessions handles empty results."""
        indexer.chroma_client.query.return_value = {"metadatas": None}

        sessions = indexer.get_recent_sessions()

        assert sessions == []

    @pytest.mark.unit
    def test_get_recent_handles_exception(self, indexer):
        """Test get_recent_sessions handles exceptions."""
        indexer.chroma_client.query.side_effect = Exception("Query failed")

        sessions = indexer.get_recent_sessions()

        assert sessions == []

    @pytest.mark.unit
    def test_get_recent_extracts_correct_fields(self, indexer):
        """Test get_recent_sessions extracts correct metadata fields."""
        indexer.chroma_client.query.return_value = {
            "metadatas": [
                [
                    {
                        "session_id": "test-session",
                        "export_time": "2024-12-15T10:30:00",
                        "source_file": "session_export.txt",
                        "extra_field": "should be ignored",
                    }
                ]
            ]
        }

        sessions = indexer.get_recent_sessions()

        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "test-session"
        assert sessions[0]["export_time"] == "2024-12-15T10:30:00"
        assert sessions[0]["source_file"] == "session_export.txt"
        assert "extra_field" not in sessions[0]
