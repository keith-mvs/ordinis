"""Dual-Write Manager for synchronized SQLite and ChromaDB persistence.

This module implements the saga pattern for atomic writes across SQLite and
ChromaDB, ensuring consistency between the relational and vector stores.

Recommendation R3 from SYNAPSE_RAG_DATABASE_REVIEW.md:
- Saga pattern for cross-store transactions
- Compensation actions for rollback on failure
- Sync queue for retry of failed ChromaDB writes
- Health monitoring and circuit breaker integration

Example:
    async with DualWriteManager(sqlite_repo, chroma_client) as manager:
        result = await manager.write_trade_with_vector(trade, embedding)
        if not result.success:
            logger.error(f"Dual write failed: {result.error}")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Callable
    import numpy as np
    from ordinis.rag.vectordb.chroma_client import ChromaDBClient

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class WritePhase(Enum):
    """Phases in the dual-write saga."""
    
    PENDING = "pending"
    SQLITE_WRITE = "sqlite_write"
    SQLITE_COMMITTED = "sqlite_committed"
    CHROMA_WRITE = "chroma_write"
    COMPLETED = "completed"
    COMPENSATING = "compensating"
    FAILED = "failed"


@dataclass
class DualWriteResult:
    """Result of a dual-write operation."""
    
    success: bool
    transaction_id: str
    sqlite_id: str | None = None
    chroma_id: str | None = None
    phase: WritePhase = WritePhase.PENDING
    error: str | None = None
    compensated: bool = False
    duration_ms: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "success": self.success,
            "transaction_id": self.transaction_id,
            "sqlite_id": self.sqlite_id,
            "chroma_id": self.chroma_id,
            "phase": self.phase.value,
            "error": self.error,
            "compensated": self.compensated,
            "duration_ms": self.duration_ms,
        }


@dataclass
class SyncQueueEntry:
    """Entry in the sync queue for retry of failed writes."""
    
    id: str
    entity_type: str
    entity_id: str
    operation: str  # "insert", "update", "delete"
    payload: dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=_utcnow)
    last_attempt_at: datetime | None = None
    error_message: str | None = None
    
    @property
    def should_retry(self) -> bool:
        """Check if this entry should be retried."""
        return self.retry_count < self.max_retries


class AsyncRepository(Protocol):
    """Protocol for async SQLite repository operations."""
    
    async def begin_transaction(self) -> None: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...
    async def insert(self, table: str, data: dict[str, Any]) -> str: ...
    async def update(self, table: str, id: str, data: dict[str, Any]) -> bool: ...
    async def delete(self, table: str, id: str) -> bool: ...
    async def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None: ...
    async def fetchone(self, sql: str, params: tuple[Any, ...] | None = None) -> dict[str, Any] | None: ...


class DualWriteManager:
    """Manager for atomic writes across SQLite and ChromaDB.
    
    Implements the saga pattern:
    1. Write to SQLite (primary, transactional)
    2. Mark as pending Chroma sync
    3. Write to ChromaDB
    4. Mark SQLite record as synced
    5. On failure: compensate (rollback SQLite, queue for retry)
    
    Attributes:
        repository: SQLite repository with transaction support
        chroma_client: ChromaDB client for vector operations
        sync_queue: In-memory queue for failed ChromaDB writes
        circuit_breaker_threshold: Failures before circuit opens
    """
    
    def __init__(
        self,
        repository: AsyncRepository,
        chroma_client: "ChromaDBClient",
        *,
        circuit_breaker_threshold: int = 5,
        sync_interval_seconds: float = 30.0,
        enable_background_sync: bool = True,
    ):
        """Initialize the dual write manager.
        
        Args:
            repository: Async SQLite repository
            chroma_client: ChromaDB client
            circuit_breaker_threshold: Consecutive failures to trip circuit
            sync_interval_seconds: Background sync interval
            enable_background_sync: Whether to run background sync task
        """
        self.repository = repository
        self.chroma_client = chroma_client
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.sync_interval_seconds = sync_interval_seconds
        self.enable_background_sync = enable_background_sync
        
        # State tracking
        self._sync_queue: list[SyncQueueEntry] = []
        self._consecutive_failures = 0
        self._circuit_open = False
        self._background_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        self._pending_tasks: set[asyncio.Task] = set()  # Prevent GC of fire-and-forget tasks
        
        # Metrics
        self._total_writes = 0
        self._successful_writes = 0
        self._failed_writes = 0
        self._compensations = 0
        
    async def __aenter__(self) -> "DualWriteManager":
        """Start the dual write manager."""
        if self.enable_background_sync:
            self._background_task = asyncio.create_task(self._background_sync_loop())
            logger.info("DualWriteManager started with background sync")
        return self
    
    async def __aexit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        """Shutdown the dual write manager."""
        self._shutdown_event.set()
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info(
            f"DualWriteManager shutdown. Stats: "
            f"total={self._total_writes}, success={self._successful_writes}, "
            f"failed={self._failed_writes}, compensations={self._compensations}"
        )
    
    @property
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (Chroma writes disabled)."""
        return self._circuit_open
    
    @property
    def queue_size(self) -> int:
        """Number of entries pending in sync queue."""
        return len(self._sync_queue)
    
    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics."""
        return {
            "total_writes": self._total_writes,
            "successful_writes": self._successful_writes,
            "failed_writes": self._failed_writes,
            "compensations": self._compensations,
            "queue_size": self.queue_size,
            "circuit_open": self._circuit_open,
            "consecutive_failures": self._consecutive_failures,
        }
    
    async def write_trade_with_vector(
        self,
        trade_data: dict[str, Any],
        embedding: "np.ndarray",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> DualWriteResult:
        """Write a trade to both SQLite and ChromaDB atomically.
        
        Args:
            trade_data: Trade data for SQLite (must include 'id')
            embedding: Vector embedding for ChromaDB
            metadata: Additional metadata for vector storage
            
        Returns:
            DualWriteResult with success status and IDs
        """
        return await self._execute_dual_write(
            entity_type="trade",
            sqlite_data=trade_data,
            embedding=embedding,
            metadata=metadata,
            collection_getter=self.chroma_client.get_text_collection,
        )
    
    async def write_order_with_vector(
        self,
        order_data: dict[str, Any],
        embedding: "np.ndarray",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> DualWriteResult:
        """Write an order to both SQLite and ChromaDB atomically."""
        return await self._execute_dual_write(
            entity_type="order",
            sqlite_data=order_data,
            embedding=embedding,
            metadata=metadata,
            collection_getter=self.chroma_client.get_text_collection,
        )
    
    async def write_code_with_vector(
        self,
        code_data: dict[str, Any],
        embedding: "np.ndarray",
        *,
        metadata: dict[str, Any] | None = None,
    ) -> DualWriteResult:
        """Write code snippet to both SQLite and ChromaDB atomically."""
        return await self._execute_dual_write(
            entity_type="code",
            sqlite_data=code_data,
            embedding=embedding,
            metadata=metadata,
            collection_getter=self.chroma_client.get_code_collection,
        )
    
    async def _execute_dual_write(
        self,
        entity_type: str,
        sqlite_data: dict[str, Any],
        embedding: "np.ndarray",
        metadata: dict[str, Any] | None,
        collection_getter: Callable,
    ) -> DualWriteResult:
        """Execute the dual-write saga.
        
        Saga steps:
        1. Begin SQLite transaction
        2. Insert to SQLite with chroma_synced=False
        3. Commit SQLite transaction
        4. Upsert to ChromaDB (if circuit not open)
        5. Update SQLite chroma_synced=True, chroma_id
        6. On any failure: compensate
        """
        import time
        start_time = time.perf_counter()
        
        transaction_id = str(uuid4())
        entity_id = sqlite_data.get("id", str(uuid4()))
        result = DualWriteResult(
            success=False,
            transaction_id=transaction_id,
            phase=WritePhase.PENDING,
        )
        
        self._total_writes += 1
        
        try:
            # Phase 1: SQLite write
            result.phase = WritePhase.SQLITE_WRITE
            await self.repository.begin_transaction()
            
            # Add sync tracking columns
            sqlite_data_with_sync = {
                **sqlite_data,
                "id": entity_id,
                "chroma_synced": False,
                "chroma_id": None,
            }
            
            await self.repository.insert(f"{entity_type}s", sqlite_data_with_sync)
            await self.repository.commit()
            result.sqlite_id = entity_id
            result.phase = WritePhase.SQLITE_COMMITTED
            
            logger.debug(f"SQLite write successful for {entity_type} {entity_id}")
            
            # Phase 2: ChromaDB write (skip if circuit open)
            if self._circuit_open:
                logger.warning(
                    f"Circuit breaker open, queuing {entity_type} {entity_id} for later sync"
                )
                self._queue_for_sync(entity_type, entity_id, "insert", {
                    "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
                    "metadata": metadata,
                    "document": sqlite_data.get("notes", sqlite_data.get("content", "")),
                })
                result.success = True
                result.phase = WritePhase.COMPLETED
            else:
                result.phase = WritePhase.CHROMA_WRITE
                try:
                    # Generate deterministic ID
                    from ordinis.rag.vectordb.id_generator import generate_vector_id
                    chroma_id = generate_vector_id(
                        entity_type=entity_type,
                        source_id=entity_id,
                        content=str(sqlite_data),
                    )
                    
                    # Prepare metadata
                    full_metadata = {
                        "entity_type": entity_type,
                        "sqlite_id": entity_id,
                        "indexed_at": _utcnow().isoformat(),
                        **(metadata or {}),
                    }
                    
                    # Upsert to ChromaDB
                    collection = collection_getter()
                    collection.upsert(
                        ids=[chroma_id],
                        embeddings=[embedding.tolist() if hasattr(embedding, "tolist") else embedding],
                        metadatas=[full_metadata],
                        documents=[sqlite_data.get("notes", sqlite_data.get("content", str(sqlite_data)))],
                    )
                    
                    result.chroma_id = chroma_id
                    
                    # Update SQLite sync status
                    await self.repository.execute(
                        f"UPDATE {entity_type}s SET chroma_synced = ?, chroma_id = ? WHERE id = ?",
                        (True, chroma_id, entity_id),
                    )
                    
                    result.success = True
                    result.phase = WritePhase.COMPLETED
                    self._successful_writes += 1
                    self._consecutive_failures = 0
                    
                    logger.debug(f"ChromaDB write successful for {entity_type} {entity_id}")
                    
                except Exception as chroma_err:
                    logger.warning(
                        f"ChromaDB write failed for {entity_type} {entity_id}: {chroma_err}. "
                        "SQLite record preserved, queuing for retry."
                    )
                    self._handle_chroma_failure()
                    self._queue_for_sync(entity_type, entity_id, "insert", {
                        "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
                        "metadata": metadata,
                        "document": sqlite_data.get("notes", sqlite_data.get("content", "")),
                    })
                    # SQLite write succeeded, so overall operation is partially successful
                    result.success = True
                    result.phase = WritePhase.COMPLETED
                    result.error = f"ChromaDB failed (queued): {chroma_err}"
                    
        except Exception as e:
            # SQLite failure - need to compensate
            result.phase = WritePhase.COMPENSATING
            result.error = str(e)
            self._failed_writes += 1
            
            try:
                await self.repository.rollback()
                result.compensated = True
                self._compensations += 1
                logger.warning(f"Rolled back SQLite transaction for {entity_type}: {e}")
            except Exception as rollback_err:
                logger.error(f"Rollback failed for {entity_type}: {rollback_err}")
            
            result.phase = WritePhase.FAILED
        
        finally:
            result.duration_ms = (time.perf_counter() - start_time) * 1000
            
        return result
    
    def _handle_chroma_failure(self) -> None:
        """Handle a ChromaDB failure for circuit breaker logic."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.circuit_breaker_threshold:
            if not self._circuit_open:
                self._circuit_open = True
                logger.error(
                    f"Circuit breaker OPENED after {self._consecutive_failures} "
                    "consecutive ChromaDB failures"
                )
    
    def _queue_for_sync(
        self,
        entity_type: str,
        entity_id: str,
        operation: str,
        payload: dict[str, Any],
    ) -> None:
        """Add an entry to the sync queue for later retry."""
        entry = SyncQueueEntry(
            id=str(uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation,
            payload=payload,
        )
        self._sync_queue.append(entry)
        
        # Also persist to SQLite sync queue - keep reference to prevent GC
        task = asyncio.create_task(self._persist_sync_entry(entry))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)
    
    async def _persist_sync_entry(self, entry: SyncQueueEntry) -> None:
        """Persist a sync queue entry to SQLite."""
        import json
        try:
            await self.repository.execute(
                """
                INSERT INTO chroma_sync_queue (id, entity_type, entity_id, operation, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.entity_type,
                    entry.entity_id,
                    entry.operation,
                    json.dumps(entry.payload, default=str),
                    entry.created_at.isoformat(),
                ),
            )
        except Exception as e:
            logger.error(f"Failed to persist sync queue entry: {e}")
    
    async def _background_sync_loop(self) -> None:
        """Background task to process sync queue."""
        logger.info("Starting background sync loop")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.sync_interval_seconds,
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue processing
            
            if self._circuit_open:
                # Try to close circuit by testing ChromaDB
                if await self._test_chroma_health():
                    self._circuit_open = False
                    self._consecutive_failures = 0
                    logger.info("Circuit breaker CLOSED - ChromaDB healthy")
                else:
                    logger.debug("Circuit still open, skipping sync")
                    continue
            
            await self._process_sync_queue()
    
    async def _test_chroma_health(self) -> bool:
        """Test if ChromaDB is healthy."""
        try:
            # Simple health check - get collection count
            collection = self.chroma_client.get_text_collection()
            _ = collection.count()
            return True
        except Exception as e:
            logger.debug(f"ChromaDB health check failed: {e}")
            return False
    
    async def _process_sync_queue(self) -> None:
        """Process pending items in sync queue."""
        if not self._sync_queue:
            return
        
        entries_to_remove = []
        
        for entry in self._sync_queue:
            if not entry.should_retry:
                logger.warning(
                    f"Sync entry {entry.id} exceeded max retries, moving to dead letter"
                )
                entries_to_remove.append(entry)
                continue
            
            entry.retry_count += 1
            entry.last_attempt_at = _utcnow()
            
            try:
                await self._sync_entry_to_chroma(entry)
                entries_to_remove.append(entry)
                logger.debug(f"Successfully synced entry {entry.id}")
            except Exception as e:
                entry.error_message = str(e)
                logger.warning(f"Retry {entry.retry_count} failed for {entry.id}: {e}")
        
        for entry in entries_to_remove:
            self._sync_queue.remove(entry)
            # Remove from SQLite queue
            await self._remove_sync_entry(entry.id)
    
    async def _sync_entry_to_chroma(self, entry: SyncQueueEntry) -> None:
        """Sync a single queue entry to ChromaDB."""
        from ordinis.rag.vectordb.id_generator import generate_vector_id
        
        payload = entry.payload
        
        chroma_id = generate_vector_id(
            entity_type=entry.entity_type,
            source_id=entry.entity_id,
            content=str(payload),
        )
        
        # Get appropriate collection
        if entry.entity_type == "code":
            collection = self.chroma_client.get_code_collection()
        else:
            collection = self.chroma_client.get_text_collection()
        
        metadata = {
            "entity_type": entry.entity_type,
            "sqlite_id": entry.entity_id,
            "indexed_at": _utcnow().isoformat(),
            "synced_from_queue": True,
            **(payload.get("metadata") or {}),
        }
        
        collection.upsert(
            ids=[chroma_id],
            embeddings=[payload["embedding"]],
            metadatas=[metadata],
            documents=[payload.get("document", "")],
        )
        
        # Update SQLite sync status
        await self.repository.execute(
            f"UPDATE {entry.entity_type}s SET chroma_synced = ?, chroma_id = ? WHERE id = ?",
            (True, chroma_id, entry.entity_id),
        )
    
    async def _remove_sync_entry(self, entry_id: str) -> None:
        """Remove a sync entry from SQLite queue."""
        try:
            await self.repository.execute(
                "DELETE FROM chroma_sync_queue WHERE id = ?",
                (entry_id,),
            )
        except Exception as e:
            logger.error(f"Failed to remove sync entry {entry_id}: {e}")
    
    async def force_sync_all(self) -> dict[str, int]:
        """Force immediate sync of all pending items.
        
        Returns:
            Dictionary with success/failure counts
        """
        # Load entries from SQLite queue
        import json
        
        try:
            rows = await self.repository.fetchone(
                "SELECT COUNT(*) as count FROM chroma_sync_queue"
            )
            if rows and rows["count"] > 0:
                # Load all pending entries
                cursor = await self.repository.execute(
                    "SELECT * FROM chroma_sync_queue ORDER BY created_at"
                )
                # Add to in-memory queue if not already present
                # (implementation depends on fetchall availability)
        except Exception as e:
            logger.warning(f"Could not load sync queue from SQLite: {e}")
        
        # Process all queued items
        initial_count = len(self._sync_queue)
        await self._process_sync_queue()
        
        return {
            "initial": initial_count,
            "remaining": len(self._sync_queue),
            "synced": initial_count - len(self._sync_queue),
        }


# Convenience function for one-off writes
async def dual_write(
    repository: AsyncRepository,
    chroma_client: "ChromaDBClient",
    entity_type: str,
    data: dict[str, Any],
    embedding: "np.ndarray",
    metadata: dict[str, Any] | None = None,
) -> DualWriteResult:
    """Convenience function for single dual-write operation.
    
    For bulk operations, use DualWriteManager as context manager.
    """
    async with DualWriteManager(
        repository, chroma_client, enable_background_sync=False
    ) as manager:
        if entity_type == "trade":
            return await manager.write_trade_with_vector(data, embedding, metadata=metadata)
        elif entity_type == "order":
            return await manager.write_order_with_vector(data, embedding, metadata=metadata)
        elif entity_type == "code":
            return await manager.write_code_with_vector(data, embedding, metadata=metadata)
        else:
            raise ValueError(f"Unsupported entity type: {entity_type}")
