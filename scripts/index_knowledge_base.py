#!/usr/bin/env python
"""
Knowledge Base Indexer Script

Indexes the docs/knowledge-base directory into ChromaDB for RAG retrieval.
Includes proper error handling for:
- Non-vectored/empty documents
- Missing embeddings
- Invalid metadata
- Collection initialization failures

Usage:
    python scripts/index_knowledge_base.py
    python scripts/index_knowledge_base.py --domain options
    python scripts/index_knowledge_base.py --dry-run
    python scripts/index_knowledge_base.py --clear
"""

import argparse
from pathlib import Path
import sys
from typing import Any

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<level>[{level}]</level> {message}")


class IndexingError(Exception):
    """Base exception for indexing errors."""


class NonVectoredDataError(IndexingError):
    """Raised when data cannot be vectorized (empty, binary, too short)."""


class EmbeddingError(IndexingError):
    """Raised when embedding generation fails."""


class ChromaConnectionError(IndexingError):
    """Raised when ChromaDB connection or operation fails."""


def validate_document(content: str, file_path: Path, min_chars: int = 50) -> tuple[bool, str]:
    """
    Validate document content is suitable for vectorization.

    Args:
        content: Document text content
        file_path: Path to document (for error messages)
        min_chars: Minimum character count for valid document

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for empty content
    if not content or not content.strip():
        return False, f"Empty document: {file_path}"

    # Check minimum length
    stripped = content.strip()
    if len(stripped) < min_chars:
        return False, f"Document too short ({len(stripped)} chars): {file_path}"

    # Check for binary content markers
    binary_markers = ["\x00", "\xff\xfe", "\xfe\xff"]
    for marker in binary_markers:
        if marker in content[:100]:
            return False, f"Binary content detected: {file_path}"

    # Check for mostly non-text content
    text_chars = sum(1 for c in stripped if c.isalnum() or c.isspace())
    text_ratio = text_chars / len(stripped) if stripped else 0
    if text_ratio < 0.3:
        return False, f"Low text content ratio ({text_ratio:.1%}): {file_path}"

    return True, ""


def chunk_document(
    content: str, chunk_size: int = 512, overlap: int = 64, min_chunk_size: int = 50
) -> list[str]:
    """
    Split document into overlapping chunks.

    Args:
        content: Document content
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
        min_chunk_size: Minimum chunk size (skip smaller chunks)

    Returns:
        List of text chunks
    """
    if not content or len(content) < min_chunk_size:
        return []

    chunks = []
    start = 0

    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]

        # Try to break at sentence/paragraph boundary
        if end < len(content):
            # Look for paragraph break
            last_para = chunk.rfind("\n\n")
            if last_para > chunk_size // 2:
                chunk = chunk[:last_para]
                end = start + last_para
            else:
                # Look for sentence break
                last_sentence = max(
                    chunk.rfind(". "), chunk.rfind(".\n"), chunk.rfind("? "), chunk.rfind("! ")
                )
                if last_sentence > chunk_size // 2:
                    chunk = chunk[: last_sentence + 1]
                    end = start + last_sentence + 1

        chunk = chunk.strip()
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)

        start = end - overlap
        if start <= 0:
            start = end  # Prevent infinite loop

    return chunks


def index_knowledge_base(
    kb_path: Path,
    domain: str | None = None,
    dry_run: bool = False,
    clear: bool = False,
    batch_size: int = 32,
) -> dict[str, Any]:
    """
    Index knowledge base documents into ChromaDB.

    Args:
        kb_path: Path to knowledge-base directory
        domain: Optional domain filter (e.g., 'options', 'signals')
        dry_run: If True, only count files without indexing
        clear: If True, clear existing collection before indexing
        batch_size: Batch size for embedding

    Returns:
        Statistics dictionary

    Raises:
        ChromaConnectionError: If ChromaDB operations fail
        NonVectoredDataError: If no valid documents found
    """
    stats = {
        "files_found": 0,
        "files_processed": 0,
        "files_skipped": 0,
        "chunks_created": 0,
        "errors": [],
        "skipped_reasons": {},
    }

    # Find markdown files
    if domain:
        search_path = kb_path / "domains" / domain
        if not search_path.exists():
            search_path = kb_path / domain
    else:
        search_path = kb_path

    if not search_path.exists():
        raise FileNotFoundError(f"KB path does not exist: {search_path}")

    md_files = list(search_path.rglob("*.md"))
    py_files = list(search_path.rglob("*.py"))
    all_files = md_files + py_files

    stats["files_found"] = len(all_files)
    logger.info(f"Found {len(md_files)} markdown + {len(py_files)} python files")

    if dry_run:
        logger.info("Dry run - counting only")
        for f in all_files[:10]:
            logger.info(f"  Would index: {f.relative_to(kb_path)}")
        if len(all_files) > 10:
            logger.info(f"  ... and {len(all_files) - 10} more")
        return stats

    # Initialize ChromaDB with error handling
    try:
        from ordinis.rag.embedders.text_embedder import TextEmbedder
        from ordinis.rag.vectordb.chroma_client import ChromaClient

        client = ChromaClient()
        embedder = TextEmbedder()

        if clear:
            logger.warning("Clearing existing text collection...")
            try:
                client.client.delete_collection(client.text_collection_name)
                logger.info("Collection cleared")
            except Exception as e:
                logger.warning(f"Could not clear collection (may not exist): {e}")

        collection = client.get_text_collection()
        existing_count = collection.count()
        logger.info(f"ChromaDB collection has {existing_count} existing documents")

    except ImportError as e:
        raise ChromaConnectionError(f"Missing dependencies: {e}")
    except Exception as e:
        raise ChromaConnectionError(f"ChromaDB initialization failed: {e}")

    # Process files
    all_texts = []
    all_metadata = []
    all_ids = []

    for file_path in all_files:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")

            # Validate content
            is_valid, error_msg = validate_document(content, file_path)
            if not is_valid:
                stats["files_skipped"] += 1
                reason = error_msg.split(":")[0]
                stats["skipped_reasons"][reason] = stats["skipped_reasons"].get(reason, 0) + 1
                logger.debug(error_msg)
                continue

            # Chunk document
            chunks = chunk_document(content)
            if not chunks:
                stats["files_skipped"] += 1
                stats["skipped_reasons"]["No valid chunks"] = (
                    stats["skipped_reasons"].get("No valid chunks", 0) + 1
                )
                continue

            # Build metadata
            relative_path = str(file_path.relative_to(kb_path))
            file_type = "code" if file_path.suffix == ".py" else "text"

            # Extract domain from path
            path_parts = Path(relative_path).parts
            doc_domain = (
                path_parts[1] if len(path_parts) > 1 and path_parts[0] == "domains" else None
            )

            for i, chunk in enumerate(chunks):
                # Python 3.11 doesn't allow backslash in f-strings
                safe_path = relative_path.replace("/", "_").replace("\\", "_")
                chunk_id = f"{safe_path}_chunk{i}"
                metadata = {
                    "source": relative_path,
                    "file_path": str(file_path),
                    "doc_type": file_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
                if doc_domain:
                    metadata["domain"] = doc_domain

                all_texts.append(chunk)
                all_metadata.append(metadata)
                all_ids.append(chunk_id)

            stats["files_processed"] += 1
            stats["chunks_created"] += len(chunks)

        except UnicodeDecodeError as e:
            stats["files_skipped"] += 1
            stats["errors"].append(f"Unicode error in {file_path}: {e}")
            stats["skipped_reasons"]["Unicode error"] = (
                stats["skipped_reasons"].get("Unicode error", 0) + 1
            )
        except Exception as e:
            stats["files_skipped"] += 1
            stats["errors"].append(f"Error processing {file_path}: {e}")

    if not all_texts:
        raise NonVectoredDataError("No valid documents to index")

    # Embed and store in batches
    logger.info(f"Embedding {len(all_texts)} chunks in batches of {batch_size}")

    for i in range(0, len(all_texts), batch_size):
        batch_end = min(i + batch_size, len(all_texts))
        batch_texts = all_texts[i:batch_end]
        batch_metadata = all_metadata[i:batch_end]
        batch_ids = all_ids[i:batch_end]

        try:
            # Generate embeddings
            embeddings = embedder.embed(batch_texts)

            # Validate embeddings
            if embeddings is None or len(embeddings) == 0:
                raise EmbeddingError(f"Empty embeddings for batch {i // batch_size + 1}")

            if len(embeddings) != len(batch_texts):
                raise EmbeddingError(
                    f"Embedding count mismatch: {len(embeddings)} vs {len(batch_texts)}"
                )

            # Store in ChromaDB
            client.add_texts(
                texts=batch_texts,
                embeddings=embeddings,
                metadata=batch_metadata,
                ids=batch_ids,
            )

            logger.info(
                f"Indexed batch {i // batch_size + 1}/"
                f"{(len(all_texts) - 1) // batch_size + 1} "
                f"({batch_end}/{len(all_texts)} chunks)"
            )

        except EmbeddingError:
            raise
        except Exception as e:
            logger.error(f"Failed to index batch {i // batch_size + 1}: {e}")
            stats["errors"].append(f"Batch {i // batch_size + 1} failed: {e}")

    # Final stats
    final_count = collection.count()
    stats["total_documents"] = final_count

    logger.success(f"Indexing complete!")
    logger.info(f"  Files processed: {stats['files_processed']}")
    logger.info(f"  Files skipped: {stats['files_skipped']}")
    logger.info(f"  Chunks created: {stats['chunks_created']}")
    logger.info(f"  Total in collection: {final_count}")

    if stats["skipped_reasons"]:
        logger.info("Skip reasons:")
        for reason, count in stats["skipped_reasons"].items():
            logger.info(f"  {reason}: {count}")

    if stats["errors"]:
        logger.warning(f"Errors encountered: {len(stats['errors'])}")
        for error in stats["errors"][:5]:
            logger.warning(f"  {error}")

    return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Index knowledge base into ChromaDB for RAG retrieval"
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Index specific domain only (e.g., 'options', 'signals')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count files without indexing",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing collection before indexing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding (default: 32)",
    )
    parser.add_argument(
        "--kb-path",
        type=str,
        default="docs/knowledge-base",
        help="Path to knowledge base (default: docs/knowledge-base)",
    )

    args = parser.parse_args()

    # Resolve KB path
    kb_path = Path(args.kb_path)
    if not kb_path.is_absolute():
        # Try relative to script location, then cwd
        script_dir = Path(__file__).parent.parent
        if (script_dir / kb_path).exists():
            kb_path = script_dir / kb_path
        elif not kb_path.exists():
            logger.error(f"KB path not found: {kb_path}")
            return 1

    try:
        stats = index_knowledge_base(
            kb_path=kb_path,
            domain=args.domain,
            dry_run=args.dry_run,
            clear=args.clear,
            batch_size=args.batch_size,
        )
        return 0

    except NonVectoredDataError as e:
        logger.error(f"No valid data to index: {e}")
        return 1
    except ChromaConnectionError as e:
        logger.error(f"ChromaDB error: {e}")
        return 1
    except EmbeddingError as e:
        logger.error(f"Embedding error: {e}")
        return 1
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
