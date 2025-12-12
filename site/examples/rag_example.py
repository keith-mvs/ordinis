"""Example: Using the RAG system for knowledge retrieval.

This example demonstrates:
1. Setting up the RAG configuration
2. Indexing the knowledge base
3. Indexing the codebase
4. Querying the system
"""

import os
from pathlib import Path

from loguru import logger

from rag.config import RAGConfig, set_config
from rag.pipeline.code_indexer import CodeIndexer
from rag.pipeline.kb_indexer import KBIndexer
from rag.retrieval.engine import RetrievalEngine


def setup_rag_system():
    """Set up the RAG system with custom configuration."""
    # Configure RAG system
    config = RAGConfig(
        # Use API for embeddings (no NVIDIA API key needed for this example)
        use_local_embeddings=False,
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
        # Adjust retrieval parameters
        top_k_retrieval=10,
        top_k_rerank=3,
        similarity_threshold=0.75,
    )

    set_config(config)
    logger.info("RAG configuration set")


def index_knowledge_base():
    """Index the knowledge base."""
    logger.info("=" * 60)
    logger.info("STEP 1: Indexing Knowledge Base")
    logger.info("=" * 60)

    kb_indexer = KBIndexer()

    # Check if KB exists
    kb_path = Path("docs/knowledge-base")
    if not kb_path.exists():
        logger.warning(f"Knowledge base not found at {kb_path}")
        logger.info("Skipping KB indexing")
        return

    # Index KB
    logger.info(f"Indexing KB from: {kb_path}")
    stats = kb_indexer.index_directory(batch_size=16)

    logger.success("KB indexing complete!")
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Chunks created: {stats['chunks_created']}")
    logger.info(f"Average chunk size: {stats['average_chunk_size']} tokens")


def index_codebase():
    """Index the codebase."""
    logger.info("=" * 60)
    logger.info("STEP 2: Indexing Codebase")
    logger.info("=" * 60)

    code_indexer = CodeIndexer()

    # Index codebase
    code_path = Path("src")
    logger.info(f"Indexing code from: {code_path}")
    stats = code_indexer.index_directory(batch_size=8)

    logger.success("Code indexing complete!")
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Chunks created: {stats['chunks_created']}")


def query_rag_system():
    """Query the RAG system."""
    logger.info("=" * 60)
    logger.info("STEP 3: Querying RAG System")
    logger.info("=" * 60)

    # Initialize retrieval engine
    engine = RetrievalEngine()

    # Example queries
    queries = [
        "What is RSI mean reversion?",  # Text query
        "Show me the RiskGuard trade evaluation code",  # Code query
        "What is the architecture of the trading system?",  # Hybrid query
    ]

    for query in queries:
        logger.info(f"\nQuery: {query}")

        # Execute query
        response = engine.query(query)

        logger.info(f"Query type: {response.query_type}")
        logger.info(f"Latency: {response.latency_ms:.0f}ms")
        logger.info(f"Results found: {len(response.results)}")

        # Display top result
        if response.results:
            top_result = response.results[0]
            logger.info(f"\nTop result (score: {top_result.score:.3f}):")
            logger.info(f"Source: {top_result.metadata.get('source', 'N/A')}")
            logger.info(f"Preview: {top_result.text[:200]}...")


def show_stats():
    """Show RAG system statistics."""
    logger.info("=" * 60)
    logger.info("RAG System Statistics")
    logger.info("=" * 60)

    engine = RetrievalEngine()
    stats = engine.get_stats()

    logger.info("\nChromaDB Statistics:")
    logger.info(f"  Text documents: {stats['chroma']['text_documents']}")
    logger.info(f"  Code documents: {stats['chroma']['code_documents']}")
    logger.info(f"  Persist directory: {stats['chroma']['persist_directory']}")

    logger.info("\nConfiguration:")
    logger.info(f"  Top-k retrieval: {stats['config']['top_k_retrieval']}")
    logger.info(f"  Top-k rerank: {stats['config']['top_k_rerank']}")
    logger.info(f"  Similarity threshold: {stats['config']['similarity_threshold']}")


def main():
    """Run the complete RAG example."""
    logger.info("=" * 60)
    logger.info("RAG System Example")
    logger.info("=" * 60)

    # Setup
    setup_rag_system()

    # Index (comment out after first run to avoid re-indexing)
    # index_knowledge_base()
    # index_codebase()

    # Query
    query_rag_system()

    # Stats
    show_stats()

    logger.success("RAG example complete!")


if __name__ == "__main__":
    # Note: This example requires:
    # 1. NVIDIA_API_KEY environment variable (or use local models)
    # 2. Knowledge base at docs/knowledge-base/
    # 3. Code to index at src/

    # For initial testing without indexing, you can run:
    # python docs/examples/rag_example.py

    main()
