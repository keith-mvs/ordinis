#!/usr/bin/env python3
"""Test RAG retrieval after indexing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ordinis.rag.vectordb.chroma_client import ChromaClient
from ordinis.rag.embedders.text_embedder import TextEmbedder

print("Initializing RAG components...")
embedder = TextEmbedder()
chroma = ChromaClient()

collection = chroma.get_text_collection()
doc_count = collection.count()
print(f"ChromaDB documents: {doc_count}")

query = "position sizing Kelly criterion"
print(f"\nTesting query: '{query}'")
query_embedding = embedder.embed([query])[0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

print("\n=== Top 3 Results ===")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    source = meta.get('source', 'unknown')
    print(f"\n{i+1}. Source: ...{source[-60:]}")
    print(f"   Content: {doc[:150]}...")

print("\n✅ RAG retrieval test complete!")
