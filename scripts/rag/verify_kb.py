from pathlib import Path
import sys

from loguru import logger

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from ordinis.rag.embedders.text_embedder import TextEmbedder
from ordinis.rag.vectordb.chroma_client import ChromaClient


def verify_kb():
    try:
        client = ChromaClient()

        # Check Text Collection
        text_col = client.get_text_collection()
        text_count = text_col.count()
        logger.info(f"Text Collection Count: {text_count}")

        # Check Code Collection
        code_col = client.get_code_collection()
        code_count = code_col.count()
        logger.info(f"Code Collection Count: {code_count}")

        if code_count > 0:
            # Try a query
            embedder = TextEmbedder(use_local=False)
            query = "calculate_historical_var"
            embedding = embedder.embed(query)

            results = client.query_code(query_embedding=embedding, top_k=1)
            if results:
                logger.success(f"Query '{query}' returned: {results[0].metadata['source']}")
            else:
                logger.warning("Query returned no results")
        else:
            logger.warning("Code collection is empty!")

    except Exception as e:
        logger.exception(f"Verification failed: {e}")


if __name__ == "__main__":
    verify_kb()
