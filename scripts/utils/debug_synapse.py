import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ordinis.ai.helix import Helix
from ordinis.ai.synapse.engine import Synapse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_synapse")


async def main():
    logger.info("Starting Synapse Debug...")

    logger.info("Initializing Helix...")
    helix = Helix()
    await helix.initialize()

    logger.info("Initializing Synapse...")
    synapse = Synapse(helix=helix)
    await synapse.initialize()

    logger.info("Synapse Initialized. Attempting Retrieval...")

    try:
        # Simple retrieval test
        results = synapse.retrieve(query="What is the purpose of the ValuationModel?", context=None)
        logger.info(f"Retrieval successful. Found {len(results.snippets)} snippets.")
        for snippet in results.snippets:
            logger.info(f"- {snippet.text[:50]}...")

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")

    logger.info("Shutting down...")
    await synapse.shutdown()
    await helix.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        sys.exit(0)
