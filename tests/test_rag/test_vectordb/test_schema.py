"""Tests for vectordb schema module.

Tests cover:
- TextChunkMetadata model
- CodeChunkMetadata model
- RetrievalResult model
- QueryRequest model
- QueryResponse model
"""

import pytest
from pydantic import ValidationError

from ordinis.rag.vectordb.schema import (
    CodeChunkMetadata,
    QueryRequest,
    QueryResponse,
    RetrievalResult,
    TextChunkMetadata,
)


class TestTextChunkMetadata:
    """Tests for TextChunkMetadata model."""

    @pytest.mark.unit
    def test_minimal_creation(self):
        """Test creating with required fields only."""
        meta = TextChunkMetadata(source="doc.md", chunk_index=0)

        assert meta.source == "doc.md"
        assert meta.chunk_index == 0
        assert meta.domain is None
        assert meta.publication_id is None
        assert meta.section is None

    @pytest.mark.unit
    def test_full_creation(self):
        """Test creating with all fields."""
        meta = TextChunkMetadata(
            domain=5,
            source="trading-strategies.md",
            chunk_index=3,
            publication_id="pub-123",
            section="Introduction",
        )

        assert meta.domain == 5
        assert meta.source == "trading-strategies.md"
        assert meta.chunk_index == 3
        assert meta.publication_id == "pub-123"
        assert meta.section == "Introduction"

    @pytest.mark.unit
    def test_source_required(self):
        """Test source is required."""
        with pytest.raises(ValidationError):
            TextChunkMetadata(chunk_index=0)  # type: ignore

    @pytest.mark.unit
    def test_chunk_index_required(self):
        """Test chunk_index is required."""
        with pytest.raises(ValidationError):
            TextChunkMetadata(source="doc.md")  # type: ignore

    @pytest.mark.unit
    def test_model_dump(self):
        """Test model_dump excludes None values."""
        meta = TextChunkMetadata(source="test.md", chunk_index=0)
        dumped = meta.model_dump(exclude_none=True)

        assert "source" in dumped
        assert "chunk_index" in dumped
        assert "domain" not in dumped


class TestCodeChunkMetadata:
    """Tests for CodeChunkMetadata model."""

    @pytest.mark.unit
    def test_minimal_creation(self):
        """Test creating with required fields only."""
        meta = CodeChunkMetadata(file_path="src/main.py")

        assert meta.file_path == "src/main.py"
        assert meta.function_name is None
        assert meta.class_name is None
        assert meta.engine is None
        assert meta.line_start is None
        assert meta.line_end is None

    @pytest.mark.unit
    def test_full_creation(self):
        """Test creating with all fields."""
        meta = CodeChunkMetadata(
            file_path="src/ordinis/engines/cortex/processor.py",
            function_name="process_signal",
            class_name="SignalProcessor",
            engine="cortex",
            line_start=45,
            line_end=78,
        )

        assert meta.file_path == "src/ordinis/engines/cortex/processor.py"
        assert meta.function_name == "process_signal"
        assert meta.class_name == "SignalProcessor"
        assert meta.engine == "cortex"
        assert meta.line_start == 45
        assert meta.line_end == 78

    @pytest.mark.unit
    def test_file_path_required(self):
        """Test file_path is required."""
        with pytest.raises(ValidationError):
            CodeChunkMetadata()  # type: ignore

    @pytest.mark.unit
    def test_model_dump(self):
        """Test model_dump excludes None values."""
        meta = CodeChunkMetadata(file_path="test.py", function_name="test_func")
        dumped = meta.model_dump(exclude_none=True)

        assert "file_path" in dumped
        assert "function_name" in dumped
        assert "class_name" not in dumped


class TestRetrievalResult:
    """Tests for RetrievalResult model."""

    @pytest.mark.unit
    def test_minimal_creation(self):
        """Test creating with required fields only."""
        result = RetrievalResult(
            id="doc-123",
            text="Some retrieved text",
            score=0.95,
        )

        assert result.id == "doc-123"
        assert result.text == "Some retrieved text"
        assert result.score == 0.95
        assert result.metadata == {}

    @pytest.mark.unit
    def test_with_metadata(self):
        """Test creating with metadata."""
        result = RetrievalResult(
            id="doc-456",
            text="Retrieved text with metadata",
            score=0.87,
            metadata={"source": "doc.md", "chunk_index": 2},
        )

        assert result.metadata["source"] == "doc.md"
        assert result.metadata["chunk_index"] == 2

    @pytest.mark.unit
    def test_all_required_fields(self):
        """Test all required fields must be provided."""
        with pytest.raises(ValidationError):
            RetrievalResult(id="123")  # type: ignore


class TestQueryRequest:
    """Tests for QueryRequest model."""

    @pytest.mark.unit
    def test_minimal_creation(self):
        """Test creating with required fields only."""
        request = QueryRequest(query="What is trading?")

        assert request.query == "What is trading?"
        assert request.query_type is None
        assert request.top_k == 5
        assert request.filters is None
        assert request.min_score is None

    @pytest.mark.unit
    def test_full_creation(self):
        """Test creating with all fields."""
        request = QueryRequest(
            query="Show me the trading function",
            query_type="code",
            top_k=10,
            filters={"engine": "signalcore"},
            min_score=0.7,
        )

        assert request.query == "Show me the trading function"
        assert request.query_type == "code"
        assert request.top_k == 10
        assert request.filters == {"engine": "signalcore"}
        assert request.min_score == 0.7

    @pytest.mark.unit
    def test_query_required(self):
        """Test query is required."""
        with pytest.raises(ValidationError):
            QueryRequest()  # type: ignore

    @pytest.mark.unit
    def test_query_min_length(self):
        """Test query must have minimum length."""
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    @pytest.mark.unit
    def test_top_k_min(self):
        """Test top_k minimum validation."""
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=0)

    @pytest.mark.unit
    def test_top_k_max(self):
        """Test top_k maximum validation."""
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=51)

    @pytest.mark.unit
    def test_min_score_range(self):
        """Test min_score must be in range 0-1."""
        with pytest.raises(ValidationError):
            QueryRequest(query="test", min_score=1.5)

        with pytest.raises(ValidationError):
            QueryRequest(query="test", min_score=-0.1)


class TestQueryResponse:
    """Tests for QueryResponse model."""

    @pytest.mark.unit
    def test_minimal_creation(self):
        """Test creating with required fields only."""
        response = QueryResponse(
            query="test query",
            query_type="text",
            execution_time_ms=50.5,
            total_candidates=100,
        )

        assert response.query == "test query"
        assert response.query_type == "text"
        assert response.results == []
        assert response.execution_time_ms == 50.5
        assert response.total_candidates == 100

    @pytest.mark.unit
    def test_with_results(self):
        """Test creating with results."""
        result = RetrievalResult(id="1", text="content", score=0.9)
        response = QueryResponse(
            query="test",
            query_type="hybrid",
            results=[result],
            execution_time_ms=100.0,
            total_candidates=50,
        )

        assert len(response.results) == 1
        assert response.results[0].score == 0.9

    @pytest.mark.unit
    def test_all_required_fields(self):
        """Test all required fields must be provided."""
        with pytest.raises(ValidationError):
            QueryResponse(query="test")  # type: ignore
