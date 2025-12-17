"""
Reset ChromaDB collections.
"""

import logging
import os
import shutil
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ordinis.rag.vectordb.chroma_client import ChromaClient

# Configure logging
logging.basicConfig(level=logging.INFO)


def main():
    print("--- Resetting ChromaDB ---")

    db_path = os.path.join(os.getcwd(), "data", "chromadb")
    print(f"Target DB Path: {db_path}")

    if os.path.exists(db_path):
        print("Removing existing database files...")
        try:
            shutil.rmtree(db_path)
            print("Database files removed.")
        except Exception as e:
            print(f"Error removing files: {e}")
            return
    else:
        print("No database found at path.")

    print("Re-initializing client (will create new empty DB)...")
    try:
        client = ChromaClient()
        stats = client.get_stats()
        print("Reset complete. New stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error re-initializing: {e}")


if __name__ == "__main__":
    main()
