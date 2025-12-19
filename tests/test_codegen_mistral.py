"""
Test CodeGenEngine with Mistral Codestral 25.01.
"""

import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ordinis.ai.codegen.engine import CodeGenEngine
from ordinis.ai.helix import Helix
from ordinis.ai.helix.config import HelixConfig
from ordinis.ai.synapse.engine import Synapse

# Configure logging
logging.basicConfig(level=logging.INFO)


async def main():
    print("--- Testing CodeGenEngine with Mistral ---")

    # 1. Initialize Helix (will use MISTRAL_API_KEY from env)
    print("Initializing Helix...")
    helix_config = HelixConfig()
    helix = Helix(config=helix_config)
    await helix.initialize()

    # 2. Initialize Synapse (Optional RAG)
    print("Initializing Synapse...")
    synapse = Synapse(helix=helix)
    await synapse.initialize()

    # 3. Initialize CodeGenEngine
    print("Initializing CodeGenEngine (WITH Synapse)...")
    codegen = CodeGenEngine(helix=helix, synapse=synapse)
    await codegen.initialize()

    # 4. Test Code Generation
    prompt = "Create a Python class for a 'MovingAverageCrossover' trading strategy with configurable window sizes."
    print(f"\nGenerating code for: '{prompt}'...\n")

    try:
        code = await codegen.generate_code(prompt, language="python")
        print("\n--- Generated Code ---\n")
        print(code)
        print("\n----------------------\n")
    except Exception as e:
        print(f"Error generating code: {e}")

    # 5. Test Explanation
    print("\nTesting Code Explanation...\n")
    try:
        explanation = await codegen.explain_code(code, detail_level="brief")
        print("\n--- Explanation ---\n")
        print(explanation)
        print("\n-------------------\n")
    except Exception as e:
        print(f"Error explaining code: {e}")

    await codegen.shutdown()
    await helix.shutdown()
    print("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        sys.exit(0)
