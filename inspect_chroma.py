"""
Inspect ChromaDB collections.
"""

import logging
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ordinis.rag.vectordb.chroma_client import ChromaClient

# Configure logging
logging.basicConfig(level=logging.INFO)


def main():
    print("--- Inspecting ChromaDB ---")

    try:
        client = ChromaClient()
        stats = client.get_stats()
        print("\nStats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # Check text collection
        text_col = client.get_text_collection()
        count = text_col.count()
        print(f"\nText Collection '{text_col.name}' has {count} documents.")

        if count > 0:
            print("Peeking at first item...")
            peek = text_col.peek(limit=1)
            print(f"ID: {peek['ids'][0]}")
            print(f"Metadata: {peek['metadatas'][0]}")
        else:
            print("Text collection is empty.")

        # Check code collection
        code_col = client.get_code_collection()
        count = code_col.count()
        print(f"\nCode Collection '{code_col.name}' has {count} documents.")

        if count > 0:
            print("Peeking at first item...")
            peek = code_col.peek(limit=1)
            print(f"ID: {peek['ids'][0]}")
            print(f"Metadata: {peek['metadatas'][0]}")
        else:
            print("Code collection is empty.")

    except Exception as e:
        print(f"Error inspecting ChromaDB: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
