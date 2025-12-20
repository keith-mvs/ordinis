#!/usr/bin/env python
"""
Minimal KB indexing with lightweight model for testing.

Uses sentence-transformers/all-MiniLM-L6-v2 (22MB) instead of
NVIDIA 300M model to avoid memory issues.

Indexes only 2-3 small files as proof of concept.

Usage:
    python scripts/index_kb_minimal.py
"""

from pathlib import Path
import sys

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def main() -> None:
    """Index minimal KB with lightweight model."""
    from sentence_transformers import SentenceTransformer

    from rag.vectordb.chroma_client import ChromaClient
    from rag.vectordb.schema import TextChunkMetadata

    logger.info("Starting minimal KB indexing with lightweight model...")

    # Use lightweight model (already cached from previous attempt)
    logger.info("Loading lightweight embedding model: all-MiniLM-L6-v2 (22MB)")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.success("Model loaded")

    # Initialize ChromaDB
    chroma = ChromaClient()

    # Find a couple small files to index
    kb_path = Path("docs/knowledge-base")
    md_files = list(kb_path.rglob("*.md"))[:3]  # Only first 3 files

    logger.info(f"Indexing {len(md_files)} files: {[f.name for f in md_files]}")

    total_chunks = 0

    for idx, md_file in enumerate(md_files, 1):
        logger.info(f"[{idx}/{len(md_files)}] Processing: {md_file.name}")

        # Read file
        content = md_file.read_text(encoding="utf-8")

        # Simple chunking (split into ~500 char chunks)
        chunks = []
        chunk_size = 500
        for i in range(0, len(content), chunk_size):
            chunk = content[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        if not chunks:
            continue

        logger.info(f"  Created {len(chunks)} chunks")

        # Generate embeddings (process in small batches)
        batch_size = 4
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]

            # Embed
            embeddings = model.encode(batch_chunks, show_progress_bar=False)

            # Create metadata and IDs (filter out None values for ChromaDB)
            batch_metadata = [
                {
                    k: v
                    for k, v in TextChunkMetadata(
                        domain=None,
                        source=str(md_file.relative_to(kb_path)),
                        chunk_index=i + j,
                        section=None,
                    )
                    .model_dump()
                    .items()
                    if v is not None
                }
                for j in range(len(batch_chunks))
            ]

            base_id = str(md_file.relative_to(kb_path)).replace("\\", "_").replace("/", "_")
            batch_ids = [f"{base_id}_chunk{i + j}" for j in range(len(batch_chunks))]

            # Store
            chroma.add_texts(
                texts=batch_chunks,
                embeddings=embeddings if isinstance(embeddings, list) else embeddings.tolist(),
                metadata=batch_metadata,
                ids=batch_ids,
            )

            total_chunks += len(batch_chunks)
            logger.debug(f"  Indexed {total_chunks} total chunks")

        logger.success(f"[{idx}/{len(md_files)}] Completed: {md_file.name}")

    logger.success(
        f"Minimal KB indexing complete! Indexed {total_chunks} chunks from {len(md_files)} files"
    )
    logger.info(
        "Note: This is a minimal test. For production, use GPU or cloud-based embedding service"
    )


if __name__ == "__main__":
    main()
