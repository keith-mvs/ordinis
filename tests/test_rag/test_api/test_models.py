"""Tests for RAG API Pydantic models.

Tests cover:
- QueryRequest validation
- QueryResult construction
- QueryResponse construction
- HealthResponse, StatsResponse, ConfigResponse
"""

import pytest
from pydantic import ValidationError

from ordinis.rag.api.models import (
    ConfigResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
    StatsResponse,
)
from ordinis.rag.retrieval.query_classifier import QueryType


class TestQueryRequest:
    """Tests for QueryRequest model."""

    @pytest.mark.unit
    def test_create_minimal(self):
        """Test creating QueryRequest with minimal fields."""
        req = QueryRequest(query="test query")

        assert req.query == "test query"
        assert req.query_type is None
        assert req.top_k == 5
        assert req.filters is None

    @pytest.mark.unit
    def test_create_with_all_fields(self):
        """Test creating QueryRequest with all fields."""
        req = QueryRequest(
            query="test query",
            query_type=QueryType.TEXT,
            top_k=10,
            filters={"domain": "trading"},
        )

        assert req.query == "test query"
        assert req.query_type == QueryType.TEXT
        assert req.top_k == 10
        assert req.filters == {"domain": "trading"}

    @pytest.mark.unit
    def test_top_k_min_validation(self):
        """Test top_k minimum validation."""
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=0)

    @pytest.mark.unit
    def test_top_k_max_validation(self):
        """Test top_k maximum validation."""
        with pytest.raises(ValidationError):
            QueryRequest(query="test", top_k=21)

    @pytest.mark.unit
    def test_query_required(self):
        """Test query is required."""
        with pytest.raises(ValidationError):
            QueryRequest()  # type: ignore


class TestQueryResult:
    """Tests for QueryResult model."""

    @pytest.mark.unit
    def test_create_result(self):
        """Test creating QueryResult."""
        result = QueryResult(
            content="test content",
            score=0.95,
            metadata={"source": "doc.md"},
        )

        assert result.content == "test content"
        assert result.score == 0.95
        assert result.metadata["source"] == "doc.md"

    @pytest.mark.unit
    def test_all_fields_required(self):
        """Test all fields are required."""
        with pytest.raises(ValidationError):
            QueryResult(content="test")  # type: ignore


class TestQueryResponse:
    """Tests for QueryResponse model."""

    @pytest.mark.unit
    def test_create_response(self):
        """Test creating QueryResponse."""
        result = QueryResult(
            content="test",
            score=0.9,
            metadata={},
        )
        response = QueryResponse(
            query="test query",
            query_type=QueryType.TEXT,
            results=[result],
            total_candidates=10,
            execution_time_ms=50.5,
        )

        assert response.query == "test query"
        assert response.query_type == QueryType.TEXT
        assert len(response.results) == 1
        assert response.total_candidates == 10
        assert response.execution_time_ms == 50.5

    @pytest.mark.unit
    def test_empty_results(self):
        """Test creating QueryResponse with empty results."""
        response = QueryResponse(
            query="no results",
            query_type=QueryType.CODE,
            results=[],
            total_candidates=0,
            execution_time_ms=10.0,
        )

        assert len(response.results) == 0


class TestHealthResponse:
    """Tests for HealthResponse model."""

    @pytest.mark.unit
    def test_create_health_response(self):
        """Test creating HealthResponse."""
        health = HealthResponse(
            status="healthy",
            text_embedder_available=True,
            code_embedder_available=True,
            chroma_persist_directory="/data/chroma",
        )

        assert health.status == "healthy"
        assert health.text_embedder_available is True
        assert health.code_embedder_available is True
        assert health.chroma_persist_directory == "/data/chroma"

    @pytest.mark.unit
    def test_unhealthy_status(self):
        """Test HealthResponse with unhealthy status."""
        health = HealthResponse(
            status="unhealthy",
            text_embedder_available=False,
            code_embedder_available=False,
            chroma_persist_directory="/data/chroma",
        )

        assert health.status == "unhealthy"
        assert health.text_embedder_available is False


class TestStatsResponse:
    """Tests for StatsResponse model."""

    @pytest.mark.unit
    def test_create_stats_response(self):
        """Test creating StatsResponse."""
        stats = StatsResponse(
            text_chunks_count=1000,
            code_chunks_count=500,
            total_chunks=1500,
            text_embedder_model="nvidia/model-a",
            code_embedder_model="nvidia/model-b",
        )

        assert stats.text_chunks_count == 1000
        assert stats.code_chunks_count == 500
        assert stats.total_chunks == 1500
        assert stats.text_embedder_model == "nvidia/model-a"
        assert stats.code_embedder_model == "nvidia/model-b"


class TestConfigResponse:
    """Tests for ConfigResponse model."""

    @pytest.mark.unit
    def test_create_config_response(self):
        """Test creating ConfigResponse."""
        config = ConfigResponse(
            text_embedding_model="nvidia/text-model",
            code_embedding_model="nvidia/code-model",
            rerank_model="nvidia/rerank-model",
            use_local_embeddings=True,
            top_k_retrieval=5,
            similarity_threshold=0.7,
        )

        assert config.text_embedding_model == "nvidia/text-model"
        assert config.code_embedding_model == "nvidia/code-model"
        assert config.rerank_model == "nvidia/rerank-model"
        assert config.use_local_embeddings is True
        assert config.top_k_retrieval == 5
        assert config.similarity_threshold == 0.7
