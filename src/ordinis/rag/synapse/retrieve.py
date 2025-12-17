"""
Synapse Retrieval CLI.

This module provides a CLI for testing the RAG (Retrieval-Augmented Generation) pipeline.
It simulates retrieving documents relevant to a query.

Usage:
    python -m ordinis.rag.synapse.retrieve --query "How does the risk engine work?"
"""

import argparse
import asyncio


async def run_retrieval(query: str):
    print(f"[INFO] Synapse Retrieval: '{query}'")

    # Mock Retrieval Logic
    print("[INFO] Embedding query...")
    await asyncio.sleep(0.5)

    print("[INFO] Searching vector database...")
    await asyncio.sleep(0.5)

    # Mock Results
    results = [
        {
            "id": "doc_1",
            "score": 0.92,
            "content": "The RiskGuard engine evaluates all trading decisions against deterministic rules before execution.",
            "source": "docs/risk_engine.md",
        },
        {
            "id": "doc_2",
            "score": 0.85,
            "content": "Risk rules can be configured in the `configs/risk/` directory. Common rules include max position size and daily loss limits.",
            "source": "docs/configuration.md",
        },
        {
            "id": "doc_3",
            "score": 0.78,
            "content": "The OrchestrationEngine coordinates the flow between Signal, Risk, and Execution engines.",
            "source": "docs/architecture.md",
        },
    ]

    print("\n" + "=" * 40)
    print("RETRIEVAL RESULTS")
    print("=" * 40)

    for i, res in enumerate(results):
        print(f"\nResult {i+1} (Score: {res['score']:.2f}):")
        print(f"  Source: {res['source']}")
        print(f"  Content: {res['content']}")


def main():
    parser = argparse.ArgumentParser(description="Synapse Retrieval Tool")
    parser.add_argument("--query", type=str, required=True, help="Query string")

    args = parser.parse_args()

    asyncio.run(run_retrieval(args.query))


if __name__ == "__main__":
    main()
