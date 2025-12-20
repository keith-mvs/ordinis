"""Test RAG retrieval from publications collection."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ordinis.rag.embedders.text_embedder import TextEmbedder
from ordinis.rag.vectordb.chroma_client import ChromaClient


def main():
    # Initialize
    client = ChromaClient()
    embedder = TextEmbedder()

    # Check collections
    print("=== Collections ===")
    for coll in client.list_collections():
        print(f"  - {coll}")

    # Get publications collection
    try:
        pub_coll = client.client.get_collection("publications")
        count = pub_coll.count()
        print(f"\n=== Publications Collection ===")
        print(f"Document count: {count}")

        # Query test - use topics from indexed papers
        queries = [
            "alpha decay institutional trading",
            "volatility risk premium selling options",
            "equity trading strategies momentum",
        ]

        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")

            query_emb = embedder.embed([query])
            results = pub_coll.query(
                query_embeddings=query_emb.tolist(),
                n_results=2,
                include=["documents", "metadatas", "distances"],
            )

            for i, (doc, meta, dist) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                    strict=False,
                )
            ):
                score = 1.0 - (dist / 2.0)  # Convert distance to similarity
                print(f"\nResult {i + 1} (similarity: {score:.4f}):")
                print(f"Source: {meta.get('source_file', 'unknown')}")
                print(f"Preview: {doc[:250]}...")

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
