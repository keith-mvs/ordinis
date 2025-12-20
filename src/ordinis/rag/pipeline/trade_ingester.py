"""Trade vector ingestion pipeline for SQLite → ChromaDB synchronization.

This module provides the TradeVectorIngester class that handles:
1. Querying unsynced trades from SQLite
2. Generating embeddings for trade data
3. Upserting vectors to ChromaDB with deterministic IDs
4. Updating SQLite sync status

Recommendation R2 from SYNAPSE_RAG_DATABASE_REVIEW.md:
- Batch processing with configurable size
- Incremental sync using chroma_synced flag
- Full reindex capability for schema migrations
- Progress tracking and error handling

Example:
    ingester = TradeVectorIngester()
    result = await ingester.sync_unsynced_trades(batch_size=100)
    print(f"Synced {result['synced']} trades")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from ordinis.rag.embedders.text_embedder import TextEmbedder
    from ordinis.rag.vectordb.chroma_client import ChromaDBClient

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class SyncMode(Enum):
    """Modes for trade sync operation."""
    
    INCREMENTAL = "incremental"  # Only unsynced trades
    FULL = "full"  # All trades (reindex)
    SESSION = "session"  # Trades from specific session


@dataclass
class SyncResult:
    """Result of a sync operation."""
    
    mode: SyncMode
    total_found: int = 0
    synced: int = 0
    failed: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_found == 0:
            return 100.0
        return (self.synced / self.total_found) * 100
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mode": self.mode.value,
            "total_found": self.total_found,
            "synced": self.synced,
            "failed": self.failed,
            "skipped": self.skipped,
            "duration_seconds": self.duration_seconds,
            "success_rate": self.success_rate,
            "errors": self.errors[:10],  # Limit error list
        }


class TradeVectorIngester:
    """Ingests trade data from SQLite and syncs to ChromaDB.
    
    This class handles the ETL pipeline from relational trade storage
    to vector storage, enabling semantic search over trading history.
    
    Attributes:
        db_path: Path to SQLite database
        chroma_client: ChromaDB client instance
        text_embedder: Text embedding model
        batch_size: Default batch size for processing
    """
    
    def __init__(
        self,
        db_path: Path | str | None = None,
        chroma_client: "ChromaDBClient | None" = None,
        text_embedder: "TextEmbedder | None" = None,
        batch_size: int = 100,
    ):
        """Initialize the trade vector ingester.
        
        Args:
            db_path: Path to SQLite database
            chroma_client: ChromaDB client (created if None)
            text_embedder: Text embedder (created if None)
            batch_size: Default batch size for processing
        """
        self.db_path = Path(db_path) if db_path else self._get_default_db_path()
        self.batch_size = batch_size
        
        # Lazy initialization of clients
        self._chroma_client = chroma_client
        self._text_embedder = text_embedder
        
        logger.info(f"TradeVectorIngester initialized with db: {self.db_path}")
    
    @staticmethod
    def _get_default_db_path() -> Path:
        """Get default database path from config or environment."""
        import os
        db_path = os.environ.get("ORDINIS_DB_PATH", "data/ordinis.db")
        return Path(db_path)
    
    @property
    def chroma_client(self) -> "ChromaDBClient":
        """Get or create ChromaDB client."""
        if self._chroma_client is None:
            from ordinis.rag.vectordb.chroma_client import ChromaDBClient
            self._chroma_client = ChromaDBClient()
        return self._chroma_client
    
    @property
    def text_embedder(self) -> "TextEmbedder":
        """Get or create text embedder."""
        if self._text_embedder is None:
            from ordinis.rag.embedders.text_embedder import TextEmbedder
            self._text_embedder = TextEmbedder()
        return self._text_embedder
    
    async def sync_unsynced_trades(
        self,
        batch_size: int | None = None,
        max_trades: int | None = None,
    ) -> SyncResult:
        """Sync all unsynced trades from SQLite to ChromaDB.
        
        Args:
            batch_size: Override default batch size
            max_trades: Maximum number of trades to sync (None = all)
            
        Returns:
            SyncResult with statistics
        """
        return await self._sync_trades(
            mode=SyncMode.INCREMENTAL,
            batch_size=batch_size or self.batch_size,
            max_trades=max_trades,
        )
    
    async def full_reindex(
        self,
        batch_size: int | None = None,
        clear_existing: bool = True,
    ) -> SyncResult:
        """Full reindex of all trades.
        
        Use this after schema changes or embedding model updates.
        
        Args:
            batch_size: Override default batch size
            clear_existing: Whether to clear existing vectors first
            
        Returns:
            SyncResult with statistics
        """
        if clear_existing:
            logger.info("Clearing existing trade vectors before reindex")
            await self._clear_trade_vectors()
        
        return await self._sync_trades(
            mode=SyncMode.FULL,
            batch_size=batch_size or self.batch_size,
        )
    
    async def sync_session_trades(
        self,
        session_id: str,
        batch_size: int | None = None,
    ) -> SyncResult:
        """Sync all trades from a specific session.
        
        Args:
            session_id: Session ID to sync
            batch_size: Override default batch size
            
        Returns:
            SyncResult with statistics
        """
        return await self._sync_trades(
            mode=SyncMode.SESSION,
            batch_size=batch_size or self.batch_size,
            session_id=session_id,
        )
    
    async def _sync_trades(
        self,
        mode: SyncMode,
        batch_size: int,
        max_trades: int | None = None,
        session_id: str | None = None,
    ) -> SyncResult:
        """Internal method to sync trades based on mode."""
        import time
        import aiosqlite
        
        start_time = time.perf_counter()
        result = SyncResult(mode=mode)
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Build query based on mode
                query, params = self._build_query(mode, max_trades, session_id)
                
                cursor = await db.execute(query, params)
                rows = await cursor.fetchall()
                result.total_found = len(rows)
                
                if result.total_found == 0:
                    logger.info(f"No trades to sync in {mode.value} mode")
                    result.duration_seconds = time.perf_counter() - start_time
                    return result
                
                logger.info(f"Found {result.total_found} trades to sync in {mode.value} mode")
                
                # Process in batches
                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i + batch_size]
                    batch_result = await self._process_batch(db, batch)
                    
                    result.synced += batch_result["synced"]
                    result.failed += batch_result["failed"]
                    result.errors.extend(batch_result["errors"])
                    
                    logger.debug(
                        f"Batch {i // batch_size + 1}: "
                        f"synced={batch_result['synced']}, failed={batch_result['failed']}"
                    )
        
        except Exception as e:
            logger.error(f"Sync operation failed: {e}")
            result.errors.append(str(e))
        
        finally:
            result.duration_seconds = time.perf_counter() - start_time
        
        logger.info(
            f"Sync complete: mode={mode.value}, synced={result.synced}/{result.total_found}, "
            f"duration={result.duration_seconds:.2f}s"
        )
        
        return result
    
    def _build_query(
        self,
        mode: SyncMode,
        max_trades: int | None,
        session_id: str | None,
    ) -> tuple[str, tuple]:
        """Build SQL query based on sync mode."""
        base_query = """
            SELECT id, symbol, side, quantity, price, timestamp, strategy_id,
                   session_id, metadata, notes, pnl, chroma_synced, chroma_id
            FROM trades
        """
        
        if mode == SyncMode.INCREMENTAL:
            query = base_query + " WHERE chroma_synced = 0 OR chroma_synced IS NULL"
            params: tuple = ()
        elif mode == SyncMode.SESSION:
            query = base_query + " WHERE session_id = ?"
            params = (session_id,)
        else:  # FULL
            query = base_query
            params = ()
        
        query += " ORDER BY timestamp ASC"
        
        if max_trades:
            query += f" LIMIT {max_trades}"
        
        return query, params
    
    async def _process_batch(
        self,
        db: Any,  # aiosqlite.Connection
        batch: list,
    ) -> dict[str, Any]:
        """Process a batch of trades."""
        from ordinis.rag.vectordb.id_generator import generate_trade_vector_id
        
        result = {"synced": 0, "failed": 0, "errors": []}
        
        try:
            # Prepare documents for embedding
            documents = []
            ids = []
            metadatas = []
            trade_ids = []
            
            for row in batch:
                trade_id = row["id"]
                trade_ids.append(trade_id)
                
                # Create searchable document
                doc = self._trade_to_document(dict(row))
                documents.append(doc)
                
                # Generate deterministic ID
                vector_id = generate_trade_vector_id(
                    trade_id=trade_id,
                    content=doc,
                )
                ids.append(vector_id)
                
                # Build metadata
                meta = {
                    "trade_id": trade_id,
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "quantity": float(row["quantity"]) if row["quantity"] else 0,
                    "price": float(row["price"]) if row["price"] else 0,
                    "timestamp": row["timestamp"],
                    "strategy_id": row["strategy_id"] or "unknown",
                    "session_id": row["session_id"] or "unknown",
                    "pnl": float(row["pnl"]) if row["pnl"] else 0,
                    "indexed_at": _utcnow().isoformat(),
                    "entity_type": "trade",
                }
                metadatas.append(meta)
            
            # Generate embeddings
            embeddings = self.text_embedder.embed_texts(documents)
            
            # Upsert to ChromaDB
            self.chroma_client.add_texts(
                texts=documents,
                embeddings=embeddings,
                metadata=metadatas,
                ids=ids,
            )
            
            # Update SQLite sync status
            for trade_id, vector_id in zip(trade_ids, ids):
                await db.execute(
                    "UPDATE trades SET chroma_synced = 1, chroma_id = ? WHERE id = ?",
                    (vector_id, trade_id),
                )
            await db.commit()
            
            result["synced"] = len(batch)
            
        except Exception as e:
            result["failed"] = len(batch)
            result["errors"].append(str(e))
            logger.error(f"Batch processing failed: {e}")
        
        return result
    
    def _trade_to_document(self, trade: dict[str, Any]) -> str:
        """Convert a trade record to a searchable text document.
        
        Creates a structured text representation optimized for
        semantic search and retrieval.
        """
        parts = [
            f"Trade: {trade['side'].upper()} {trade['quantity']} shares of {trade['symbol']}",
            f"at ${trade['price']:.2f}" if trade.get('price') else "",
            f"Strategy: {trade.get('strategy_id', 'unknown')}",
            f"Time: {trade.get('timestamp', 'unknown')}",
        ]
        
        if trade.get('pnl') is not None:
            pnl = float(trade['pnl'])
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            parts.append(f"P&L: {pnl_str}")
        
        if trade.get('notes'):
            parts.append(f"Notes: {trade['notes']}")
        
        if trade.get('metadata'):
            # Try to extract relevant metadata fields
            try:
                import json
                meta = json.loads(trade['metadata']) if isinstance(trade['metadata'], str) else trade['metadata']
                if meta.get('reason'):
                    parts.append(f"Reason: {meta['reason']}")
                if meta.get('signal_strength'):
                    parts.append(f"Signal strength: {meta['signal_strength']}")
            except (json.JSONDecodeError, TypeError):
                pass
        
        return " | ".join(filter(None, parts))
    
    async def _clear_trade_vectors(self) -> None:
        """Clear all trade vectors from ChromaDB."""
        try:
            collection = self.chroma_client.get_text_collection()
            
            # Get all trade IDs (those starting with "trade:")
            results = collection.get(
                where={"entity_type": "trade"},
                include=[],  # Only need IDs
            )
            
            if results and results.get("ids"):
                collection.delete(ids=results["ids"])
                logger.info(f"Cleared {len(results['ids'])} trade vectors")
            
        except Exception as e:
            logger.warning(f"Failed to clear trade vectors: {e}")
    
    async def get_sync_status(self) -> dict[str, Any]:
        """Get current sync status.
        
        Returns:
            Dictionary with sync statistics
        """
        import aiosqlite
        
        async with aiosqlite.connect(self.db_path) as db:
            # Count total trades
            cursor = await db.execute("SELECT COUNT(*) FROM trades")
            row = await cursor.fetchone()
            total = row[0] if row else 0
            
            # Count synced trades
            cursor = await db.execute(
                "SELECT COUNT(*) FROM trades WHERE chroma_synced = 1"
            )
            row = await cursor.fetchone()
            synced = row[0] if row else 0
            
            # Count unsynced trades
            cursor = await db.execute(
                "SELECT COUNT(*) FROM trades WHERE chroma_synced = 0 OR chroma_synced IS NULL"
            )
            row = await cursor.fetchone()
            unsynced = row[0] if row else 0
        
        # Get ChromaDB count
        try:
            collection = self.chroma_client.get_text_collection()
            chroma_count = collection.count()
        except Exception:
            chroma_count = -1
        
        return {
            "sqlite_total": total,
            "sqlite_synced": synced,
            "sqlite_unsynced": unsynced,
            "chroma_count": chroma_count,
            "sync_percentage": (synced / total * 100) if total > 0 else 100.0,
        }


# CLI interface for running sync operations
async def main():
    """CLI entry point for trade ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trade Vector Ingester")
    parser.add_argument(
        "--mode",
        choices=["incremental", "full", "status"],
        default="incremental",
        help="Sync mode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        help="Session ID for session sync mode",
    )
    
    args = parser.parse_args()
    
    ingester = TradeVectorIngester(
        db_path=args.db_path,
        batch_size=args.batch_size,
    )
    
    if args.mode == "status":
        status = await ingester.get_sync_status()
        print("Trade Sync Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    elif args.mode == "full":
        result = await ingester.full_reindex(batch_size=args.batch_size)
        print(f"Full reindex complete: {result.to_dict()}")
    else:
        result = await ingester.sync_unsynced_trades(batch_size=args.batch_size)
        print(f"Incremental sync complete: {result.to_dict()}")


if __name__ == "__main__":
    asyncio.run(main())
