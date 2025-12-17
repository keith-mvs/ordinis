import asyncio
import logging
import os
from pathlib import Path

from ordinis.ai.helix.engine import Helix
from ordinis.ai.helix.models import ChatMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("expand_kb")


async def generate_readme(helix, folder_path, files):
    folder_name = folder_path.name
    # Get relative path for context
    try:
        rel_path = folder_path.relative_to(Path("docs/knowledge-base"))
    except ValueError:
        rel_path = folder_path.name

    prompt = f"""
    You are an expert technical writer for a quantitative trading system documentation.
    The folder 'docs/knowledge-base/{rel_path}' contains the following files/directories:
    {', '.join(files)}

    Please generate a comprehensive README.md for this folder.
    It should include:
    1. An overview of the '{folder_name}' domain.
    2. A brief description of what each file/subdirectory likely contains (infer from names).
    3. A structure that helps a developer navigate this section of the knowledge base.

    Output ONLY the markdown content. Do not include ```markdown``` fences.
    """

    try:
        response = await helix.generate(
            messages=[ChatMessage(role="user", content=prompt)],
            model="nemotron-super",
            temperature=0.3,
        )
        return response.content
    except Exception as e:
        logger.error(f"Failed to generate README for {folder_path}: {e}")
        return None


async def generate_expansion_plan(helix, kb_root, structure):
    prompt = f"""
    You are an expert architect for a quantitative trading system.
    The current knowledge base structure is:
    {structure}

    Please analyze this structure and identify gaps or missing topics that are essential for a complete trading system documentation.
    Consider domains like:
    - Execution (Order management, connectivity)
    - Strategy (Research, backtesting, live trading)
    - Risk (Pre-trade, post-trade, portfolio)
    - Signals (Alpha generation, data processing)
    - Infrastructure (Deployment, monitoring, database)
    - Compliance (Regulatory, reporting)

    Generate a "Knowledge Base Expansion Plan" in Markdown format.
    List specific missing topics and suggest file names for them.

    Output ONLY the markdown content.
    """

    try:
        response = await helix.generate(
            messages=[ChatMessage(role="user", content=prompt)],
            model="nemotron-super",
            temperature=0.3,
        )
        return response.content
    except Exception as e:
        logger.error(f"Failed to generate expansion plan: {e}")
        return None


def get_structure(root_path):
    structure = ""
    for root, dirs, files in os.walk(root_path):
        level = root.replace(str(root_path), "").count(os.sep)
        indent = " " * 4 * (level)
        structure += f"{indent}{os.path.basename(root)}/\n"
        subindent = " " * 4 * (level + 1)
        for f in files:
            structure += f"{subindent}{f}\n"
    return structure


async def main():
    # Ensure API key is present (HelixConfig will look for NVIDIA_API_KEY)
    if not os.getenv("NVIDIA_API_KEY"):
        logger.warning(
            "NVIDIA_API_KEY not found in environment. Helix might fail if not configured otherwise."
        )

    helix = Helix()
    await helix.initialize()

    kb_root = Path("docs/knowledge-base")

    # 1. Generate READMEs for directories
    for root, dirs, files in os.walk(kb_root):
        if "README.md" not in files and "index.md" not in files:
            # Skip hidden folders
            if any(part.startswith(".") for part in Path(root).parts):
                continue

            logger.info(f"Generating README for {root}...")
            content = await generate_readme(helix, Path(root), files + dirs)

            if content:
                # Strip markdown fences if present
                if content.startswith("```markdown"):
                    content = content[11:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]

                with open(Path(root) / "README.md", "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"Created {Path(root) / 'README.md'}")

    # 2. Generate Expansion Plan
    logger.info("Generating Expansion Plan...")
    structure = get_structure(kb_root)
    plan_content = await generate_expansion_plan(helix, kb_root, structure)

    if plan_content:
        # Strip markdown fences if present
        if plan_content.startswith("```markdown"):
            plan_content = plan_content[11:]
        if plan_content.startswith("```"):
            plan_content = plan_content[3:]
        if plan_content.endswith("```"):
            plan_content = plan_content[:-3]

        with open(kb_root / "EXPANSION_PLAN.md", "w", encoding="utf-8") as f:
            f.write(plan_content)
        logger.info(f"Created {kb_root / 'EXPANSION_PLAN.md'}")

    await helix.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
