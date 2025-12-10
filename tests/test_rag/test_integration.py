"""
Integration tests for RAG system.

These tests verify end-to-end functionality of the RAG pipeline.
Tests marked with @pytest.mark.integration require ChromaDB to be populated.
"""

import pytest

from rag.config import RAGConfig
from rag.retrieval.engine import RetrievalEngine
from rag.retrieval.query_classifier import QueryType, classify_query


@pytest.fixture
def rag_config() -> RAGConfig:
    """Create RAG configuration for testing."""
    return RAGConfig()


@pytest.fixture
def retrieval_engine(rag_config: RAGConfig) -> RetrievalEngine:
    """Create retrieval engine for testing."""
    return RetrievalEngine()


# =============================================================================
# Query Classification Tests
# =============================================================================


@pytest.mark.integration
def test_query_classification() -> None:
    """Test query type classification."""
    # Text queries
    assert classify_query("What is RSI mean reversion?") == QueryType.TEXT
    assert classify_query("How does momentum trading work?") == QueryType.TEXT
    assert classify_query("Explain risk management") == QueryType.TEXT

    # Code queries
    assert classify_query("Show me the code for strategy validation") == QueryType.CODE
    assert classify_query("def calculate_rsi function") == QueryType.CODE

    # Hybrid queries
    assert classify_query("class BaseStrategy implementation") == QueryType.HYBRID
    assert classify_query("strategy architecture design patterns") == QueryType.HYBRID
    assert classify_query("how to implement backtesting framework") == QueryType.HYBRID


# =============================================================================
# Retrieval Engine Tests
# =============================================================================


@pytest.mark.integration
def test_retrieval_engine_initialization(retrieval_engine: RetrievalEngine) -> None:
    """Test retrieval engine initializes successfully."""
    assert retrieval_engine is not None
    assert retrieval_engine.config is not None
    assert retrieval_engine.text_embedder.is_available()
    assert retrieval_engine.code_embedder.is_available()


@pytest.mark.integration
@pytest.mark.skip(reason="Requires ChromaDB to be populated with data")
def test_text_query(retrieval_engine: RetrievalEngine) -> None:
    """Test text query against knowledge base."""
    response = retrieval_engine.query(
        query="What is RSI mean reversion?",
        query_type=QueryType.TEXT,
        top_k=5,
    )

    assert response is not None
    assert response.query_type == QueryType.TEXT
    assert isinstance(response.results, list)
    assert len(response.results) <= 5

    # Verify result structure
    if response.results:
        result = response.results[0]
        assert hasattr(result, "content")
        assert hasattr(result, "score")
        assert hasattr(result, "metadata")
        assert result.score >= 0.0


@pytest.mark.integration
@pytest.mark.skip(reason="Requires ChromaDB to be populated with data")
def test_code_query(retrieval_engine: RetrievalEngine) -> None:
    """Test code query against codebase."""
    response = retrieval_engine.query(
        query="strategy validation implementation",
        query_type=QueryType.CODE,
        top_k=3,
    )

    assert response is not None
    assert response.query_type == QueryType.CODE
    assert isinstance(response.results, list)
    assert len(response.results) <= 3

    # Verify code result metadata
    if response.results:
        result = response.results[0]
        assert "file_path" in result.metadata or "source" in result.metadata


@pytest.mark.integration
@pytest.mark.skip(reason="Requires ChromaDB to be populated with data")
def test_hybrid_query(retrieval_engine: RetrievalEngine) -> None:
    """Test hybrid query against both collections."""
    response = retrieval_engine.query(
        query="strategy design patterns and implementation",
        query_type=QueryType.HYBRID,
        top_k=10,
    )

    assert response is not None
    assert response.query_type == QueryType.HYBRID
    assert isinstance(response.results, list)

    # Should have results from both collections
    collections = {r.metadata.get("collection") for r in response.results}
    # Note: May not have both if one collection is empty
    assert len(collections) > 0


@pytest.mark.integration
@pytest.mark.skip(reason="Requires ChromaDB to be populated with data")
def test_domain_filtering(retrieval_engine: RetrievalEngine) -> None:
    """Test domain-based filtering."""
    # Query with domain filter (3 = Risk Management)
    response = retrieval_engine.query(
        query="risk limits and position sizing",
        query_type=QueryType.TEXT,
        top_k=5,
        filters={"domain": 3},
    )

    assert response is not None
    assert isinstance(response.results, list)

    # Verify all results match domain filter
    for result in response.results:
        assert result.metadata.get("domain") == 3


@pytest.mark.integration
@pytest.mark.skip(reason="Requires ChromaDB to be populated with data")
def test_auto_query_detection(retrieval_engine: RetrievalEngine) -> None:
    """Test automatic query type detection."""
    # Text query without explicit type
    response = retrieval_engine.query(
        query="What is momentum trading?",
        query_type=None,  # Auto-detect
        top_k=5,
    )

    assert response is not None
    assert response.query_type == QueryType.TEXT

    # Code query without explicit type
    response = retrieval_engine.query(
        query="def generate_signals implementation",
        query_type=None,  # Auto-detect
        top_k=3,
    )

    assert response is not None
    assert response.query_type == QueryType.CODE


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.skip(reason="Requires ChromaDB to be populated with data")
def test_query_performance(retrieval_engine: RetrievalEngine) -> None:
    """Test query execution time is reasonable."""
    import time

    start = time.time()
    retrieval_engine.query(
        query="trading strategy best practices",
        query_type=QueryType.TEXT,
        top_k=10,
    )
    elapsed = time.time() - start

    # Should complete in under 2 seconds (generous for CI)
    assert elapsed < 2.0
