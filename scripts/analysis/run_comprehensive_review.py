"""
Comprehensive Code Review Script using CodeGen Engine.

This script walks through the source code, uses the CodeGen engine (powered by Helix/Codestral)
to review the code, and generates a report.
"""

import asyncio
import logging
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ordinis.ai.codegen.engine import CodeGenEngine
from ordinis.ai.helix import Helix
from ordinis.ai.helix.config import HelixConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("code_review.log"), logging.StreamHandler()],
)

logger = logging.getLogger("CodeReview")


async def review_file(codegen: CodeGenEngine, file_path: Path) -> str:
    """Review a single file."""
    try:
        with open(file_path, encoding="utf-8") as f:
            code = f.read()

        if not code.strip():
            return f"## {file_path}\n\n*Skipped: Empty file*\n"

        logger.info(f"Reviewing {file_path}...")
        review = await codegen.review_code(
            code=code, focus="security,performance,style,idiomatic_python,error_handling"
        )

        return f"## {file_path}\n\n{review}\n\n---\n"
    except Exception as e:
        logger.error(f"Error reviewing {file_path}: {e}")
        return f"## {file_path}\n\n*Error during review: {e}*\n"


async def main():
    logger.info("Starting Comprehensive Code Review...")

    # 1. Initialize Engines
    logger.info("Initializing Helix...")
    helix_config = HelixConfig()
    helix = Helix(config=helix_config)
    await helix.initialize()

    logger.info("Initializing CodeGenEngine...")
    # Passing None for synapse to avoid RAG complexity for this batch job
    codegen = CodeGenEngine(helix=helix, synapse=None)
    await codegen.initialize()

    # 2. Identify files to review
    # For this run, we'll focus on 'core' and 'ai' modules to keep it manageable
    # but you can expand this list.
    target_dirs = [
        "src/ordinis/core",
        "src/ordinis/ai/codegen",
    ]

    files_to_review: list[Path] = []
    base_path = Path(os.getcwd())

    for target in target_dirs:
        path = base_path / target
        if path.exists():
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith(".py") and not file.startswith("__"):
                        files_to_review.append(Path(root) / file)

    logger.info(f"Found {len(files_to_review)} files to review.")

    # 3. Run Reviews
    report_path = "CODE_REVIEW_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Code Review Report\n\n")
        f.write(f"**Date:** {os.getenv('DATE', '2025-12-14')}\n")
        f.write(f"**Files Reviewed:** {len(files_to_review)}\n\n")

    for i, file_path in enumerate(files_to_review):
        logger.info(f"Processing {i+1}/{len(files_to_review)}: {file_path}")

        # Calculate relative path for display
        rel_path = file_path.relative_to(base_path)

        review_content = await review_file(codegen, file_path)

        # Append to report immediately
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(f"\n### File: {rel_path}\n")
            f.write(review_content)

    logger.info(f"Review complete. Report saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
