import logging
import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ordinis.rag.embedders.text_embedder import TextEmbedder
from ordinis.rag.vectordb.chroma_client import ChromaClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("populate_chroma")


def populate_db():
    logger.info("Populating ChromaDB with test data...")

    try:
        # Initialize components
        client = ChromaClient()
        embedder = TextEmbedder()

        documents = [
            "The ValuationModel generates trading signals based on fundamental valuation metrics such as P/E ratio, P/B ratio, and EV/EBITDA.",
            "The GrowthModel generates trading signals based on fundamental growth metrics such as Revenue Growth, EPS Growth, and Margin Expansion.",
            "SignalCore is the core engine for generating trading signals in the Ordinis platform.",
            "Helix is the unified LLM provider engine for Ordinis, orchestrating providers like Mistral, NVIDIA, and OpenAI.",
            "Synapse is the RAG retrieval engine that provides context for the AI models.",
        ]

        metadatas = [
            {"source": "kb", "file_path": "docs/valuation.md", "section": "overview"},
            {"source": "kb", "file_path": "docs/growth.md", "section": "overview"},
            {"source": "kb", "file_path": "docs/signalcore.md", "section": "overview"},
            {"source": "kb", "file_path": "docs/helix.md", "section": "overview"},
            {"source": "kb", "file_path": "docs/synapse.md", "section": "overview"},
        ]

        ids = [f"doc_{i}" for i in range(len(documents))]

        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = embedder.embed(documents)

        logger.info(f"Adding {len(documents)} documents with embeddings...")
        client.add_texts(texts=documents, embeddings=embeddings, metadata=metadatas, ids=ids)

        logger.info("Successfully populated ChromaDB.")

    except Exception as e:
        logger.error(f"Failed to populate ChromaDB: {e}")


if __name__ == "__main__":
    populate_db()
