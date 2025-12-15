from pathlib import Path
import sys

from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ordinis.rag.pipeline.code_indexer import CodeIndexer


def main():
    # Define paths
    project_root = Path(__file__).parent.parent
    code_paths = [
        project_root / "src",
        project_root / "scripts",
        project_root / "docs" / "knowledge-base" / "code",
    ]

    # Filter to paths that exist
    existing_paths = [p for p in code_paths if p.exists()]

    if not existing_paths:
        logger.error("No code paths found")
        return

    logger.info(f"Starting indexing for: {[str(p) for p in existing_paths]}")

    # Initialize indexer
    try:
        indexer = CodeIndexer()

        # Run indexing with all paths
        stats = indexer.index_directory(code_paths=existing_paths)

        logger.info("Code indexing complete!")
        logger.info(f"Stats: {stats}")

    except Exception as e:
        logger.exception(f"Indexing failed: {e}")


if __name__ == "__main__":
    main()
