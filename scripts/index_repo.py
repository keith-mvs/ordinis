#!/usr/bin/env python
"""Index the entire repository for RAG."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ordinis.rag.pipeline.kb_indexer import KBIndexer


def main():
    """Run full repository indexing."""
    print("=== Full Repository Indexing ===")
    print("Initializing KBIndexer...")
    indexer = KBIndexer()
    print("KBIndexer initialized successfully\n")

    base = Path(__file__).parent.parent
    directories = [
        base / "docs",
        base / "src",
        base / "scripts",
        base / "configs",
        base / "examples",
    ]
    
    # Conservative batch size to avoid GPU OOM
    BATCH_SIZE = 32

    total_indexed = 0
    for directory in directories:
        try:
            if directory.exists():
                print(f"Indexing {directory.name}...")
                result = indexer.index_directory(directory, batch_size=BATCH_SIZE)
                chunks = result.get("total_chunks", 0)
                total_indexed += chunks
                print(f"  Completed {directory.name}: {chunks} chunks\n")
            else:
                print(f"  Skipping {directory.name} - does not exist\n")
        except Exception as e:
            print(f"  Error indexing {directory.name}: {e}\n")
            import traceback
            traceback.print_exc()

    print(f"\n=== Full repo indexing complete! Total: {total_indexed} chunks ===")


if __name__ == "__main__":
    main()
