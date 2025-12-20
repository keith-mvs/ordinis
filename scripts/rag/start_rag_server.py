#!/usr/bin/env python
"""
Start the RAG API server.

Usage:
    python scripts/start_rag_server.py [--host HOST] [--port PORT] [--reload]

Examples:
    python scripts/start_rag_server.py
    python scripts/start_rag_server.py --port 8080
    python scripts/start_rag_server.py --reload  # Auto-reload on code changes
"""

import argparse
from pathlib import Path
import sys

# Add project root to path FIRST
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Now import dependencies
from loguru import logger  # noqa: E402
import uvicorn  # noqa: E402


def main() -> None:
    """Start the RAG API server."""
    parser = argparse.ArgumentParser(description="Start RAG API server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development only)",
    )

    args = parser.parse_args()

    logger.info(f"Starting RAG API server on {args.host}:{args.port}")
    logger.info("API docs: http://localhost:{}/docs", args.port)
    logger.info("Web UI: http://localhost:{}/ui", args.port)

    uvicorn.run(
        "rag.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
