#!/usr/bin/env python
"""
Index knowledge base markdown files into ChromaDB.

This script processes all markdown files in docs/knowledge-base/ and indexes
them into ChromaDB for RAG retrieval.

Usage:
    python scripts/index_knowledge_base.py
"""

from pathlib import Path
import sys

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from ordinis.rag.pipeline.kb_indexer import KBIndexer  # noqa: E402


def main() -> None:
    """Index knowledge base into ChromaDB."""
    logger.info("Starting knowledge base indexing...")

    # Create indexer
    indexer = KBIndexer()

    # Index knowledge base directory
    try:
        stats = indexer.index_directory()

        # Display results
        logger.success("Knowledge base indexing complete!")
        logger.info(f"Files processed: {stats['files_processed']}")
        logger.info(f"Total chunks: {stats['total_chunks']}")
        logger.info(f"Total tokens: {stats['total_tokens']:,}")
        logger.info(f"Average chunk size: {stats['avg_chunk_size']} tokens")

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise


if __name__ == "__main__":
    main()
