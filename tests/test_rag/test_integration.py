"""Integration tests for RAG system."""

import pytest
from rag.config import get_config
from rag.retrieval.engine import RetrievalEngine
from rag.vectordb.schema import QueryType


@pytest.fixture
def retrieval_engine():
    """Create retrieval engine for testing."""
    config = get_config()
    # Use API embeddings for tests to avoid VRAM issues
    config.use_local_embeddings = False
    return RetrievalEngine()


@pytest.mark.integration
def test_retrieval_engine_initialization(retrieval_engine):
    """Test that retrieval engine initializes correctly."""
    assert retrieval_engine is not None
    assert retrieval_engine.text_embedder.is_available()
    # Code embedder may not be available without GPU/API key
    # assert retrieval_engine.code_embedder.is_available()


@pytest.mark.integration
@pytest.mark.skip(reason="Requires ChromaDB to be populated with data")
def test_text_query(retrieval_engine):
    """Test text query retrieval."""
    response = retrieval_engine.query(
        query="What is RSI mean reversion?",
        query_type=QueryType.TEXT,
        top_k=5,
    )

    assert response is not None
    assert response.query_type == QueryType.TEXT
    assert response.latency_ms > 0
    # May have 0 results if DB is empty
    assert isinstance(response.results, list)


@pytest.mark.integration
@pytest.mark.skip(reason="Requires ChromaDB to be populated with code")
def test_code_query(retrieval_engine):
    """Test code query retrieval."""
    response = retrieval_engine.query(
        query="strategy implementation example",
        query_type=QueryType.CODE,
        top_k=3,
    )

    assert response is not None
    assert response.query_type == QueryType.CODE
    assert response.latency_ms > 0
    assert isinstance(response.results, list)


@pytest.mark.integration
def test_query_classification():
    """Test automatic query type classification."""
    from rag.retrieval.query_classifier import classify_query

    # Text queries
    assert classify_query("What is momentum trading?") == QueryType.TEXT
    assert classify_query("Explain RSI indicator") == QueryType.TEXT

    # Code queries
    assert classify_query("Show me the code for strategy") == QueryType.CODE
    assert classify_query("function implementation") == QueryType.CODE

    # Hybrid queries
    assert classify_query("architecture design patterns") == QueryType.HYBRID
    assert classify_query("implementation of trading system") == QueryType.HYBRID


@pytest.mark.integration
def test_retrieval_engine_stats(retrieval_engine):
    """Test retrieval engine statistics."""
    stats = retrieval_engine.get_stats()

    assert "chroma" in stats
    assert "config" in stats
    assert stats["text_embedder_available"] is not None
    assert stats["code_embedder_available"] is not None


@pytest.mark.integration
@pytest.mark.skip(reason="Requires Cortex engine and NVIDIA API key")
def test_cortex_rag_integration():
    """Test RAG integration with Cortex engine."""
    from engines.cortex.core.engine import CortexEngine

    # Create Cortex with RAG enabled
    cortex = CortexEngine(rag_enabled=True)

    # Generate hypothesis with RAG context
    hypothesis = cortex.generate_hypothesis(
        market_context={"regime": "trending", "volatility": "low"},
        constraints={"max_position_pct": 0.10},
    )

    assert hypothesis is not None
    assert hypothesis.hypothesis_id.startswith("hyp-")

    # Check that metadata includes RAG context availability
    outputs = cortex.get_outputs()
    assert len(outputs) > 0
    last_output = outputs[-1]
    assert "rag_context_available" in last_output.metadata


@pytest.mark.integration
def test_config_management():
    """Test RAG configuration management."""
    from rag.config import RAGConfig, set_config

    # Create custom config
    custom_config = RAGConfig(
        use_local_embeddings=False,
        top_k_retrieval=10,
        similarity_threshold=0.8,
    )

    # Set and retrieve
    set_config(custom_config)
    current_config = get_config()

    assert current_config.use_local_embeddings is False
    assert current_config.top_k_retrieval == 10
    assert current_config.similarity_threshold == 0.8
