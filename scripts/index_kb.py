from pathlib import Path
import sys

from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ordinis.rag.pipeline.kb_indexer import KBIndexer


def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    kb_path = project_root / "docs" / "knowledge-base"

    if not kb_path.exists():
        logger.error(f"Knowledge base path not found: {kb_path}")
        return

    logger.info(f"Starting indexing for: {kb_path}")

    # Initialize indexer
    # Note: Ensure NVIDIA_API_KEY is set in environment if using API embeddings
    try:
        indexer = KBIndexer()

        # Run indexing
        stats = indexer.index_directory(kb_path=kb_path)

        logger.info("Indexing complete!")
        logger.info(f"Stats: {stats}")

    except Exception as e:
        logger.exception(f"Indexing failed: {e}")


if __name__ == "__main__":
    main()
