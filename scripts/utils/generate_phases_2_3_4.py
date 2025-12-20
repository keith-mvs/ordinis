"""
Generate Phase 2, 3, and 4 Models using CodeGenEngine.

Phase 2: Sentiment Signals
Phase 3: Algorithmic Signals
Phase 4: Advanced Ensemble Logic
"""

import asyncio
import logging
import os
from pathlib import Path
import sys

sys.path.append(os.path.join(os.getcwd(), "src"))

from ordinis.ai.codegen.engine import CodeGenEngine
from ordinis.ai.helix import Helix
from ordinis.ai.helix.config import HelixConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_phases_2_3_4")


async def generate_model(codegen, model_name, doc_path, output_path, base_context):
    """Generate a model from documentation."""
    logger.info(f"Generating {model_name} from {doc_path}...")

    with open(doc_path, encoding="utf-8") as f:
        doc_content = f.read()[:8000]  # Limit to avoid token issues

    prompt = f"""
    Create a Python class named `{model_name}` that inherits from `Model`.

    Based on this documentation:
    {doc_content}

    Requirements:
    1. Inherit from `Model` (from `ordinis.engines.signalcore.core.model`)
    2. Implement `async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal`
    3. Return a Signal with:
       - symbol, timestamp, signal_type, direction, probability, score (normalized to [-1, 1])
       - expected_return=0.0, confidence_interval=(0.0, 0.0)
       - model_id, model_version
       - metadata dict with intermediate scores
    4. Handle missing data gracefully
    5. Use correct enum values: Direction.LONG/SHORT/NEUTRAL, SignalType.ENTRY/EXIT/HOLD
    6. Score must be normalized to [-1, 1] range (where 0 is neutral)
    7. Only output raw Python code, no markdown blocks
    """

    try:
        code = await codegen.generate_code(prompt, context=base_context, language="python")

        # Clean markdown blocks
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.replace("```", "")

        code = code.strip()

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)

        logger.info(f"✓ Generated {model_name}")

    except Exception as e:
        logger.error(f"✗ Failed to generate {model_name}: {e}")


async def main():
    workspace_root = Path(os.getcwd())
    docs_root = workspace_root / "docs" / "knowledge-base" / "domains" / "signals"
    src_root = workspace_root / "src" / "ordinis" / "engines" / "signalcore" / "models"

    # Initialize Engines
    logger.info("Initializing Helix and CodeGenEngine...")
    helix = Helix(config=HelixConfig())
    await helix.initialize()

    codegen = CodeGenEngine(helix=helix)
    await codegen.initialize()

    # Read base model context
    base_model_path = (
        workspace_root / "src" / "ordinis" / "engines" / "signalcore" / "core" / "model.py"
    )
    with open(base_model_path, encoding="utf-8") as f:
        base_context = f.read()

    # Phase 2: Sentiment Models
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: SENTIMENT SIGNALS")
    logger.info("=" * 70)

    sentiment_dir = src_root / "sentiment"
    sentiment_dir.mkdir(parents=True, exist_ok=True)

    await generate_model(
        codegen,
        "NewsSentimentModel",
        docs_root / "sentiment" / "news-sentiment.md",
        sentiment_dir / "news_sentiment.py",
        base_context,
    )

    # Phase 3: Algorithmic Models
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: ALGORITHMIC SIGNALS")
    logger.info("=" * 70)

    algorithmic_dir = src_root / "algorithmic"
    algorithmic_dir.mkdir(parents=True, exist_ok=True)

    await generate_model(
        codegen,
        "PairsTradingModel",
        docs_root / "quantitative" / "algorithmic-strategies.md",
        algorithmic_dir / "pairs_trading.py",
        base_context + "\n\n# Focus on the Pairs Trading section (cointegration-based strategy)",
    )

    await generate_model(
        codegen,
        "IndexRebalanceModel",
        docs_root / "quantitative" / "algorithmic-strategies.md",
        algorithmic_dir / "index_rebalance.py",
        base_context + "\n\n# Focus on the Index Rebalancing section (event-driven strategy)",
    )

    logger.info("\n" + "=" * 70)
    logger.info("ALL PHASES COMPLETE")
    logger.info("=" * 70)
    logger.info("✓ Phase 2: Sentiment Models")
    logger.info("✓ Phase 3: Algorithmic Models")
    logger.info("\nNote: Phase 4 (Advanced Ensembles) will be implemented manually")


if __name__ == "__main__":
    asyncio.run(main())
