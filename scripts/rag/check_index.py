from pathlib import Path
import sys

from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ordinis.rag.vectordb.chroma_client import ChromaClient


def main():
    client = ChromaClient()

    # Check text collection
    try:
        text_coll = client.get_collection("kb_text")
        logger.info(f"Text collection count: {text_coll.count()}")
        if text_coll.count() > 0:
            logger.info(f"Sample: {text_coll.peek(1)}")
    except Exception as e:
        logger.warning(f"Could not get text collection: {e}")

    # Check code collection
    try:
        code_coll = client.get_collection("codebase")
        logger.info(f"Code collection count: {code_coll.count()}")
        if code_coll.count() > 0:
            logger.info(f"Sample: {code_coll.peek(1)}")
    except Exception as e:
        logger.warning(f"Could not get code collection: {e}")


if __name__ == "__main__":
    main()
