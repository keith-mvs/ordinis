"""Test RAG retrieval functionality."""

from pathlib import Path
import sys

from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ordinis.rag.embedders.text_embedder import TextEmbedder
from ordinis.rag.vectordb.chroma_client import ChromaClient


def test_retrieval():
    """Test RAG retrieval with various queries."""

    try:
        logger.info("Initializing RAG components...")
        client = ChromaClient()
        embedder = TextEmbedder(use_local=False)

        # Check collection counts
        text_col = client.get_text_collection()
        code_col = client.get_code_collection()
        logger.info(f"Text documents: {text_col.count()}")
        logger.info(f"Code documents: {code_col.count()}")

        # Test queries
        test_queries = [
            ("circuit breaker", "text"),
            ("black scholes", "code"),
            ("calculate historical var", "code"),
            ("risk management", "text"),
            ("kelly criterion", "text"),
            ("AlertSeverity", "code"),
        ]

        logger.info("\n" + "=" * 60)
        logger.info("Testing RAG Retrieval")
        logger.info("=" * 60)

        for query, collection_type in test_queries:
            logger.info(f"\nQuery: '{query}' (Collection: {collection_type})")

            # Generate embedding
            embedding = embedder.embed(query)

            # Query appropriate collection
            if collection_type == "text":
                results = client.query_texts(query_embedding=embedding, top_k=3)
            else:
                results = client.query_code(query_embedding=embedding, top_k=3)

            if results:
                logger.success(f"Found {len(results)} results")
                for i, result in enumerate(results, 1):
                    logger.info(f"  [{i}] Score: {result.score:.3f}")
                    if collection_type == "text":
                        logger.info(f"      Source: {result.metadata.get('source', 'N/A')}")
                        logger.info(f"      Preview: {result.text[:100]}...")
                    else:
                        logger.info(f"      File: {result.metadata.get('file_path', 'N/A')}")
                        logger.info(f"      Class: {result.metadata.get('class_name', 'N/A')}")
                        logger.info(
                            f"      Lines: {result.metadata.get('line_start', 'N/A')}-{result.metadata.get('line_end', 'N/A')}"
                        )
                        logger.info(f"      Preview: {result.text[:100]}...")
            else:
                logger.warning(f"No results found for '{query}'")

        logger.info("\n" + "=" * 60)
        logger.success("RAG retrieval test complete!")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"Test failed: {e}")


if __name__ == "__main__":
    test_retrieval()
