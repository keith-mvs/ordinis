"""
Debug RAG retrieval issues.
"""

import asyncio
import logging
import os
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ordinis.ai.helix import Helix
from ordinis.ai.helix.config import HelixConfig
from ordinis.ai.synapse.engine import Synapse

# Configure logging
logging.basicConfig(level=logging.DEBUG)


async def main():
    print("--- Debugging RAG Retrieval ---")

    # 1. Initialize Helix
    print("Initializing Helix...")
    helix = Helix(config=HelixConfig())
    await helix.initialize()

    # 2. Initialize Synapse
    print("Initializing Synapse...")
    synapse = Synapse(helix=helix)
    await synapse.initialize()

    query = "Create a Python class for a 'MovingAverageCrossover' trading strategy with configurable window sizes."

    print(f"\nTesting retrieve_for_codegen with query: '{query}'\n")

    try:
        start = time.time()
        # Call directly without thread pool first to see if it works
        context = synapse.retrieve_for_codegen(query, language="python")
        duration = time.time() - start
        print(f"\n--- Retrieval Successful ({duration:.2f}s) ---\n")
        print(context[:500] + "..." if len(context) > 500 else context)
        print("\n--------------------------------\n")
    except Exception as e:
        print(f"Error during retrieval: {e}")
        import traceback

        traceback.print_exc()

    await synapse.shutdown()
    await helix.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
