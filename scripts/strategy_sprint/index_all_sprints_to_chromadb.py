#!/usr/bin/env python
"""
Index all sprint results to ChromaDB.

Loads Sprint 1, 2, and 3 results and indexes them to ChromaDB
for unified querying and analysis.

Usage:
    python scripts/strategy_sprint/index_all_sprints_to_chromadb.py
"""

from pathlib import Path

import chromadb
from chromadb.config import Settings
from loguru import logger
import pandas as pd

# Paths - use relative from cwd (project root)
SPRINT_DIR = Path("artifacts/reports/strategy_sprint")
SPRINT3_DIR = Path("artifacts/sprint/sprint3_smallcap")
CHROMA_PATH = Path("data/chromadb")


def load_sprint_results(sprint_dir: Path) -> list[dict]:
    """Load all CSV results from a sprint directory."""
    results = []

    # Find all result CSVs
    strategies = ["garch", "kalman", "hmm", "ou_pairs", "evt", "mtf", "mi", "network"]

    for strategy in strategies:
        csv_files = list(sprint_dir.glob(f"{strategy}_results_*.csv"))
        if not csv_files:
            csv_files = list(sprint_dir.glob(f"{strategy}_test_*.csv"))

        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                timestamp = csv_file.stem.split("_")[-1]

                for _, row in df.iterrows():
                    result = {
                        "symbol": row.get("symbol", "UNKNOWN"),
                        "strategy": strategy,
                        "total_trades": int(row.get("trades", row.get("total_trades", 0))),
                        "win_rate": float(row.get("win_rate", 0)),
                        "total_pnl": float(row.get("total_return", row.get("total_pnl", 0))),
                        "sharpe": float(row.get("sharpe", 0)),
                        "max_drawdown": float(row.get("max_drawdown", 0)),
                        "profit_factor": float(row.get("profit_factor", 0)),
                        "annual_return": float(row.get("annual_return", 0)),
                        "timestamp": timestamp,
                    }
                    results.append(result)

            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}")

    return results


def load_sprint3_results(sprint3_dir: Path) -> list[dict]:
    """Load Sprint 3 results from CSV."""
    results = []

    csv_files = list(sprint3_dir.glob("sprint3_details_*.csv"))

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            timestamp = csv_file.stem.split("_")[-1]

            for _, row in df.iterrows():
                result = {
                    "symbol": row.get("symbol", "UNKNOWN"),
                    "strategy": row.get("strategy", "UNKNOWN"),
                    "total_trades": int(row.get("total_trades", 0)),
                    "win_rate": float(row.get("win_rate", 0)),
                    "total_pnl": float(row.get("total_pnl", 0)),
                    "sharpe": float(row.get("sharpe", 0)),
                    "max_drawdown": float(row.get("max_drawdown", 0)),
                    "profit_factor": float(row.get("profit_factor", 0)),
                    "timestamp": timestamp,
                }
                results.append(result)

        except Exception as e:
            logger.warning(f"Error loading {csv_file}: {e}")

    return results


def index_to_chromadb(results: list[dict], sprint_name: str, collection_name: str) -> int:
    """Index results to ChromaDB collection."""

    # Initialize ChromaDB
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )

    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"description": "All sprint backtest results"}
    )

    documents = []
    metadatas = []
    ids = []

    for idx, result in enumerate(results):
        # Create document text
        doc_text = (
            f"Sprint: {sprint_name}, Strategy: {result['strategy']}, Symbol: {result['symbol']}, "
            f"Trades: {result['total_trades']}, Win Rate: {result['win_rate']:.1f}%, "
            f"Total PnL: {result['total_pnl']:.2f}%, Sharpe: {result['sharpe']:.3f}, "
            f"Max DD: {result['max_drawdown']:.2f}%, Profit Factor: {result['profit_factor']:.3f}"
        )
        documents.append(doc_text)

        # Metadata
        metadatas.append(
            {
                "symbol": str(result["symbol"]),
                "strategy": str(result["strategy"]),
                "total_trades": int(result["total_trades"]),
                "win_rate": float(result["win_rate"]),
                "total_pnl": float(result["total_pnl"]),
                "sharpe": float(result["sharpe"]),
                "max_drawdown": float(result["max_drawdown"]),
                "profit_factor": float(result["profit_factor"]),
                "timestamp": str(result["timestamp"]),
                "sprint": sprint_name,
            }
        )

        ids.append(
            f"{sprint_name}_{result['strategy']}_{result['symbol']}_{result['timestamp']}_{idx}"
        )

    # Add to collection
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    return len(documents)


def main():
    """Index all sprint results to ChromaDB."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )

    logger.info("=" * 60)
    logger.info("INDEXING ALL SPRINT RESULTS TO CHROMADB")
    logger.info("=" * 60)

    # Collection for all results
    collection_name = "all_sprint_results"

    # Initialize ChromaDB and reset collection
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH), settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )

    # Delete existing collection to reindex
    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass

    total_indexed = 0

    # Load and index Sprint 1 & 2 results (from strategy_sprint dir)
    logger.info(f"\nLoading Sprint 1 & 2 results from: {SPRINT_DIR}")

    # Group by timestamp to identify sprints
    json_files = sorted(SPRINT_DIR.glob("sprint_summary_*.json"))

    for i, json_file in enumerate(json_files, 1):
        # Extract full timestamp from filename (e.g., sprint_summary_20251217_164557.json)
        filename_parts = json_file.stem.split("_")
        timestamp = "_".join(filename_parts[-2:])  # Gets "20251217_164557"
        sprint_name = f"sprint{i}_volatile"

        logger.info(f"\n  Processing {sprint_name} (timestamp: {timestamp})...")

        # Load results for this timestamp
        results = []
        strategies = ["garch", "kalman", "hmm", "ou_pairs", "evt", "mtf", "mi", "network"]

        for strategy in strategies:
            csv_file = SPRINT_DIR / f"{strategy}_results_{timestamp}.csv"
            if csv_file.exists():
                try:
                    df = pd.read_csv(csv_file)
                    logger.info(f"    Loading {csv_file.name} ({len(df)} rows)")

                    for _, row in df.iterrows():
                        result = {
                            "symbol": str(row.get("symbol", "UNKNOWN")),
                            "strategy": strategy,
                            "total_trades": int(row.get("trades", row.get("total_trades", 0))),
                            "win_rate": float(row.get("win_rate", 0)),
                            "total_pnl": float(row.get("total_return", row.get("total_pnl", 0))),
                            "sharpe": float(row.get("sharpe", 0)),
                            "max_drawdown": float(row.get("max_drawdown", 0)),
                            "profit_factor": float(row.get("profit_factor", 0)),
                            "timestamp": timestamp,
                        }
                        results.append(result)

                except Exception as e:
                    logger.warning(f"    Error loading {csv_file.name}: {e}")

        if results:
            count = index_to_chromadb(results, sprint_name, collection_name)
            logger.info(f"    Indexed {count} results")
            total_indexed += count

    # Load and index Sprint 3 results
    logger.info(f"\nLoading Sprint 3 results from: {SPRINT3_DIR}")

    csv_files = list(SPRINT3_DIR.glob("sprint3_details_*.csv"))

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            timestamp = csv_file.stem.split("_")[-1]
            sprint_name = "sprint3_smallcap"

            results = []
            for _, row in df.iterrows():
                result = {
                    "symbol": row.get("symbol", "UNKNOWN"),
                    "strategy": row.get("strategy", "UNKNOWN"),
                    "total_trades": int(row.get("total_trades", 0)),
                    "win_rate": float(row.get("win_rate", 0)),
                    "total_pnl": float(row.get("total_pnl", 0)),
                    "sharpe": float(row.get("sharpe", 0)),
                    "max_drawdown": float(row.get("max_drawdown", 0)),
                    "profit_factor": float(row.get("profit_factor", 0)),
                    "timestamp": timestamp,
                }
                results.append(result)

            if results:
                count = index_to_chromadb(results, sprint_name, collection_name)
                logger.info(f"  Indexed {count} Sprint 3 results")
                total_indexed += count

        except Exception as e:
            logger.warning(f"  Error loading {csv_file.name}: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("INDEXING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total documents indexed: {total_indexed}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"ChromaDB path: {CHROMA_PATH}")

    # Query summary
    collection = client.get_collection(collection_name)
    logger.info(f"Collection count: {collection.count()}")

    # Show sample query
    logger.info("\nSample query - Top performers by Sharpe:")
    results = collection.get(
        include=["metadatas"],
        limit=1000,
    )

    if results["metadatas"]:
        # Sort by sharpe
        sorted_results = sorted(
            results["metadatas"], key=lambda x: x.get("sharpe", 0), reverse=True
        )[:10]

        for r in sorted_results:
            logger.info(
                f"  {r['sprint']}/{r['strategy']}/{r['symbol']}: "
                f"Sharpe={r['sharpe']:.2f}, PnL={r['total_pnl']:.2f}%"
            )

    return total_indexed


if __name__ == "__main__":
    main()
