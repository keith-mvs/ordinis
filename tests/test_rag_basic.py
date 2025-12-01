"""Basic RAG system tests."""

from loguru import logger

from rag.config import RAGConfig, get_config, set_config
from rag.retrieval.query_classifier import QueryType, classify_query


def test_query_classifier():
    """Test query classification."""
    # Text queries
    assert classify_query("What is RSI?") == QueryType.TEXT
    assert classify_query("Explain risk management") == QueryType.TEXT

    # Code queries
    assert classify_query("Show me the code") == QueryType.CODE
    assert classify_query("How does the function work?") == QueryType.CODE

    # Hybrid queries
    assert classify_query("What is the architecture?") == QueryType.HYBRID
    assert classify_query("Design pattern for strategies") == QueryType.HYBRID

    logger.info("Query classifier tests passed")


def test_config():
    """Test RAG configuration."""
    config = get_config()

    assert config.text_collection_name == "kb_text"
    assert config.code_collection_name == "codebase"
    assert config.top_k_retrieval == 20
    assert config.top_k_rerank == 5
    assert config.similarity_threshold == 0.7

    logger.info("Config tests passed")


def test_config_custom():
    """Test custom configuration."""
    custom_config = RAGConfig(
        top_k_retrieval=10,
        similarity_threshold=0.8,
    )

    set_config(custom_config)
    assert get_config().top_k_retrieval == 10
    assert get_config().similarity_threshold == 0.8

    # Reset to default
    set_config(RAGConfig())
    logger.info("Custom config tests passed")


# Note: Integration tests for embeddings and indexing require:
# 1. NVIDIA API key or local models
# 2. Knowledge base and code to index
# These should be run separately with proper setup

if __name__ == "__main__":
    test_query_classifier()
    test_config()
    test_config_custom()
    logger.success("All basic RAG tests passed!")
