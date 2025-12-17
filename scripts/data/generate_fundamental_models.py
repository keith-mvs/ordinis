import asyncio
import logging
import os
from pathlib import Path
import re
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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_fundamental_models")


async def generate_model(codegen, model_name, doc_path, output_path, base_model_context):
    logger.info(f"Generating {model_name} from {doc_path}...")

    # Read documentation
    with open(doc_path, encoding="utf-8") as f:
        doc_content = f.read()

    prompt = f"""
    Create a Python class named `{model_name}` that inherits from `Model` (from `ordinis.engines.signalcore.core.model`).

    The model should implement the logic described in the following documentation:

    {doc_content}

    Requirements:
    1. Inherit from `Model`.
    2. Implement `async def generate(self, data: pd.DataFrame, timestamp: datetime) -> Signal`.
    3. Use `pandas` for calculations.
    4. Handle missing data gracefully.
    5. Return a `Signal` object with appropriate `SignalType`, `Direction`, `probability`, and `score`.
    6. Include a `ModelConfig` in the `__init__`.
    7. Ensure all imports are correct (relative to `ordinis.engines.signalcore`).
    8. Do not include markdown code blocks (```python ... ```) in the output, just the raw code.
    """

    context = f"""
    Base Model Definition:
    {base_model_context}

    Existing Imports in other models:
    from datetime import datetime
    import numpy as np
    import pandas as pd
    from ordinis.engines.signalcore.core.model import Model, ModelConfig
    from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType
    """

    try:
        code = await codegen.generate_code(prompt, context=context, language="python")

        # Clean up code
        # Remove <think> tags and content
        code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL)

        # Remove markdown blocks
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]

        code = code.strip()

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)

        logger.info(f"Successfully generated {model_name} at {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate {model_name}: {e}")


async def main():
    # Paths
    workspace_root = Path(os.getcwd())
    docs_root = workspace_root / "docs" / "knowledge-base" / "domains" / "signals" / "fundamental"
    models_root = (
        workspace_root / "src" / "ordinis" / "engines" / "signalcore" / "models" / "fundamental"
    )

    # Ensure output directory exists
    models_root.mkdir(parents=True, exist_ok=True)

    # Read Base Model Context
    base_model_path = (
        workspace_root / "src" / "ordinis" / "engines" / "signalcore" / "core" / "model.py"
    )
    with open(base_model_path, encoding="utf-8") as f:
        base_model_context = f.read()

    # Initialize Engines
    logger.info("Initializing Helix...")
    helix_config = HelixConfig()
    helix = Helix(config=helix_config)
    await helix.initialize()

    logger.info("Initializing CodeGenEngine...")
    codegen = CodeGenEngine(helix=helix)  # Synapse optional
    await codegen.initialize()

    # Generate Valuation Model
    await generate_model(
        codegen,
        "ValuationModel",
        docs_root / "valuation" / "value-signals.md",
        models_root / "valuation.py",
        base_model_context,
    )

    # Generate Growth Model
    await generate_model(
        codegen,
        "GrowthModel",
        docs_root / "growth" / "growth-signals.md",
        models_root / "growth.py",
        base_model_context,
    )

    logger.info("All models generated successfully.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Generation cancelled by user.")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
    finally:
        logger.info("Exiting...")
        sys.exit(0)
