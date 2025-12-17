from pathlib import Path
import sys

from loguru import logger

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

from ordinis.rag.embedders.text_embedder import TextEmbedder


def test_embedding():
    try:
        embedder = TextEmbedder(use_local=False)
        text = "This is a test sentence."
        embedding = embedder.embed(text)

        print(f"Embedding type: {type(embedding)}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"First 10 values: {embedding[:10]}")

        texts = ["Sentence 1", "Sentence 2"]
        embeddings = embedder.embed(texts)
        print(f"Batch embedding shape: {embeddings.shape}")

    except Exception as e:
        logger.exception(f"Embedding failed: {e}")


if __name__ == "__main__":
    test_embedding()
