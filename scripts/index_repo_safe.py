#!/usr/bin/env python3
"""Safe full-repo indexer with memory management for Synapse RAG.

Indexes in smaller batches with explicit garbage collection to avoid
segmentation faults from memory pressure.
"""

import gc
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")

def main():
    """Run safe full-repo indexing."""
    from ordinis.rag.pipeline.kb_indexer import KBIndexer
    from ordinis.rag.vectordb.chroma_client import ChromaClient
    from ordinis.rag.embedders.text_embedder import TextEmbedder
    
    # Initialize components
    logger.info("Initializing embedder (this may take a moment)...")
    embedder = TextEmbedder()
    
    logger.info("Initializing ChromaDB client...")
    chroma = ChromaClient()
    
    # Clear existing collection for fresh start
    logger.info("Clearing existing KB collection...")
    try:
        chroma.client.delete_collection("kb_text")
    except Exception:
        pass
    gc.collect()
    
    logger.info("Creating fresh KB collection...")
    chroma.get_text_collection()
    
    # Initialize indexer
    indexer = KBIndexer(
        text_embedder=embedder,
        chroma_client=chroma,
    )
    
    # Define directories to index
    repo_root = Path(__file__).parent.parent
    directories = [
        repo_root / "docs",
        repo_root / "src",
        repo_root / "scripts",
        repo_root / "configs",
        repo_root / "examples",
    ]
    
    total_stats = {
        "files_processed": 0,
        "chunks_created": 0,
        "total_tokens": 0,
    }
    
    # Process each directory with small batch size
    BATCH_SIZE = 16  # Very small batches to avoid memory issues
    
    for directory in directories:
        if directory.exists():
            logger.info(f"Indexing {directory.name}...")
            try:
                stats = indexer.index_directory(directory, batch_size=BATCH_SIZE)
                total_stats["files_processed"] += stats.get("files_processed", 0)
                total_stats["chunks_created"] += stats.get("chunks_created", 0)
                total_stats["total_tokens"] += stats.get("total_tokens", 0)
                
                # Force garbage collection between directories
                gc.collect()
                
            except Exception as e:
                logger.error(f"Failed to index {directory}: {e}")
                gc.collect()
        else:
            logger.warning(f"Directory not found: {directory}")
    
    logger.success(f"Full repo indexing complete!")
    logger.info(f"  Files: {total_stats['files_processed']}")
    logger.info(f"  Chunks: {total_stats['chunks_created']}")
    logger.info(f"  Tokens: {total_stats['total_tokens']}")
    
    # Verify collection
    collection = chroma.get_text_collection()
    count = collection.count()
    logger.info(f"  ChromaDB documents: {count}")

if __name__ == "__main__":
    main()
