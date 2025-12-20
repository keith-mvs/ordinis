"""Retention management for data lifecycle governance.

This module provides the RetentionManager class that:
1. Enforces data retention policies for SQLite and ChromaDB
2. Archives old sessions and trades
3. Purges expired data according to policy
4. Maintains audit trail of retention actions

Memory integration from SYNAPSE_RAG_DATABASE_REVIEW.md:
- Configurable retention policies per entity type
- Coordinated cleanup across SQLite and ChromaDB
- Summarization before deletion (preserve knowledge)
- Audit logging of all retention actions

Example:
    manager = RetentionManager(db_path, chroma_client)
    result = await manager.enforce_retention_policies()
    print(f"Archived: {result['archived']}, Purged: {result['purged']}")
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from ordinis.rag.vectordb.chroma_client import ChromaDBClient

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Get current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class RetentionAction(Enum):
    """Types of retention actions."""
    
    ARCHIVE = "archive"  # Move to cold storage
    PURGE = "purge"  # Permanently delete
    SUMMARIZE = "summarize"  # Replace with summary
    RETAIN = "retain"  # Keep as-is


class EntityType(Enum):
    """Types of entities subject to retention."""
    
    SESSION = "session"
    MESSAGE = "message"
    TRADE = "trade"
    ORDER = "order"
    SUMMARY = "summary"
    VECTOR = "vector"


@dataclass
class RetentionPolicy:
    """Policy for retaining a specific entity type."""
    
    entity_type: EntityType
    archive_after_days: int | None = None  # Days before archiving
    purge_after_days: int | None = None  # Days before purging
    summarize_before_purge: bool = True  # Create summary before purging
    min_retain_count: int = 0  # Minimum records to always keep
    exempt_filter: str | None = None  # SQL WHERE to exempt records
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_type": self.entity_type.value,
            "archive_after_days": self.archive_after_days,
            "purge_after_days": self.purge_after_days,
            "summarize_before_purge": self.summarize_before_purge,
            "min_retain_count": self.min_retain_count,
            "exempt_filter": self.exempt_filter,
        }


@dataclass
class RetentionResult:
    """Result of a retention enforcement run."""
    
    start_time: datetime
    end_time: datetime | None = None
    archived: int = 0
    purged: int = 0
    summarized: int = 0
    errors: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        """Duration of the retention run."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "archived": self.archived,
            "purged": self.purged,
            "summarized": self.summarized,
            "error_count": len(self.errors),
            "action_count": len(self.actions),
        }


# Default retention policies
DEFAULT_POLICIES = [
    RetentionPolicy(
        entity_type=EntityType.MESSAGE,
        archive_after_days=30,
        purge_after_days=90,
        summarize_before_purge=True,
        min_retain_count=100,
    ),
    RetentionPolicy(
        entity_type=EntityType.SESSION,
        archive_after_days=30,
        purge_after_days=180,
        summarize_before_purge=True,
        min_retain_count=10,
    ),
    RetentionPolicy(
        entity_type=EntityType.TRADE,
        archive_after_days=365,
        purge_after_days=None,  # Never purge trades
        summarize_before_purge=False,
        min_retain_count=0,
    ),
    RetentionPolicy(
        entity_type=EntityType.ORDER,
        archive_after_days=90,
        purge_after_days=365,
        summarize_before_purge=False,
        min_retain_count=0,
    ),
    RetentionPolicy(
        entity_type=EntityType.SUMMARY,
        archive_after_days=None,  # Never archive summaries
        purge_after_days=365,
        summarize_before_purge=False,
        min_retain_count=0,
    ),
]


class RetentionManager:
    """Manages data retention and lifecycle.
    
    This class enforces configurable retention policies:
    1. Archive: Move old data to separate archive tables
    2. Summarize: Create summaries before deletion
    3. Purge: Permanently delete expired data
    
    All actions are logged to the retention_audit table.
    
    Attributes:
        db_path: Path to SQLite database
        chroma_client: ChromaDB client for vector cleanup
        policies: List of retention policies
    """
    
    def __init__(
        self,
        db_path: Path | str | None = None,
        chroma_client: "ChromaDBClient | None" = None,
        policies: list[RetentionPolicy] | None = None,
    ):
        """Initialize the retention manager.
        
        Args:
            db_path: Path to SQLite database
            chroma_client: ChromaDB client
            policies: Retention policies (uses defaults if None)
        """
        self.db_path = Path(db_path) if db_path else self._get_default_db_path()
        self.policies = policies or DEFAULT_POLICIES.copy()
        
        # Lazy initialization
        self._chroma_client = chroma_client
        
        # Build policy lookup
        self._policy_map = {p.entity_type: p for p in self.policies}
        
        logger.info(f"RetentionManager initialized with {len(self.policies)} policies")
    
    @staticmethod
    def _get_default_db_path() -> Path:
        """Get default database path."""
        import os
        return Path(os.environ.get("ORDINIS_DB_PATH", "data/ordinis.db"))
    
    @property
    def chroma_client(self) -> "ChromaDBClient":
        """Get or create ChromaDB client."""
        if self._chroma_client is None:
            from ordinis.rag.vectordb.chroma_client import ChromaDBClient
            self._chroma_client = ChromaDBClient()
        return self._chroma_client
    
    def get_policy(self, entity_type: EntityType) -> RetentionPolicy | None:
        """Get policy for an entity type."""
        return self._policy_map.get(entity_type)
    
    def set_policy(self, policy: RetentionPolicy) -> None:
        """Set or update a retention policy."""
        self._policy_map[policy.entity_type] = policy
        # Update list
        self.policies = [p for p in self.policies if p.entity_type != policy.entity_type]
        self.policies.append(policy)
    
    async def enforce_retention_policies(
        self,
        dry_run: bool = False,
    ) -> RetentionResult:
        """Enforce all retention policies.
        
        Args:
            dry_run: If True, only simulate actions without executing
            
        Returns:
            RetentionResult with statistics
        """
        result = RetentionResult(start_time=_utcnow())
        
        logger.info(f"Starting retention enforcement (dry_run={dry_run})")
        
        try:
            import aiosqlite
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Process each policy
                for policy in self.policies:
                    try:
                        policy_result = await self._enforce_policy(db, policy, dry_run)
                        result.archived += policy_result.get("archived", 0)
                        result.purged += policy_result.get("purged", 0)
                        result.summarized += policy_result.get("summarized", 0)
                        result.actions.extend(policy_result.get("actions", []))
                    except Exception as e:
                        error_msg = f"Policy {policy.entity_type.value} failed: {e}"
                        logger.error(error_msg)
                        result.errors.append(error_msg)
                
                # Cleanup ChromaDB vectors for purged records
                if not dry_run:
                    await self._cleanup_orphaned_vectors(db, result)
                
                if not dry_run:
                    await db.commit()
        
        except Exception as e:
            error_msg = f"Retention enforcement failed: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        finally:
            result.end_time = _utcnow()
        
        logger.info(
            f"Retention complete: archived={result.archived}, "
            f"purged={result.purged}, summarized={result.summarized}, "
            f"errors={len(result.errors)}"
        )
        
        return result
    
    async def _enforce_policy(
        self,
        db: Any,
        policy: RetentionPolicy,
        dry_run: bool,
    ) -> dict[str, Any]:
        """Enforce a single retention policy."""
        result = {"archived": 0, "purged": 0, "summarized": 0, "actions": []}
        
        table_name = self._get_table_name(policy.entity_type)
        if not table_name:
            return result
        
        timestamp_col = self._get_timestamp_column(policy.entity_type)
        now = _utcnow()
        
        # Handle archiving
        if policy.archive_after_days is not None:
            archive_cutoff = now - timedelta(days=policy.archive_after_days)
            archive_result = await self._archive_records(
                db, table_name, timestamp_col, archive_cutoff, policy, dry_run
            )
            result["archived"] = archive_result["count"]
            result["actions"].extend(archive_result["actions"])
        
        # Handle purging
        if policy.purge_after_days is not None:
            purge_cutoff = now - timedelta(days=policy.purge_after_days)
            
            # Summarize before purge if configured
            if policy.summarize_before_purge:
                summary_result = await self._summarize_before_purge(
                    db, table_name, timestamp_col, purge_cutoff, policy, dry_run
                )
                result["summarized"] = summary_result["count"]
                result["actions"].extend(summary_result["actions"])
            
            purge_result = await self._purge_records(
                db, table_name, timestamp_col, purge_cutoff, policy, dry_run
            )
            result["purged"] = purge_result["count"]
            result["actions"].extend(purge_result["actions"])
        
        return result
    
    async def _archive_records(
        self,
        db: Any,
        table_name: str,
        timestamp_col: str,
        cutoff: datetime,
        policy: RetentionPolicy,
        dry_run: bool,
    ) -> dict[str, Any]:
        """Archive records older than cutoff."""
        result = {"count": 0, "actions": []}
        
        # Build query
        query = f"""
            SELECT id FROM {table_name}
            WHERE {timestamp_col} < ?
            AND (archived IS NULL OR archived = 0)
        """
        params: list[Any] = [cutoff.isoformat()]
        
        if policy.exempt_filter:
            query += f" AND NOT ({policy.exempt_filter})"
        
        if policy.min_retain_count > 0:
            query += f"""
                AND id NOT IN (
                    SELECT id FROM {table_name}
                    ORDER BY {timestamp_col} DESC
                    LIMIT {policy.min_retain_count}
                )
            """
        
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        
        if not rows:
            return result
        
        ids = [row["id"] for row in rows]
        
        if dry_run:
            logger.info(f"[DRY RUN] Would archive {len(ids)} {table_name} records")
        else:
            # Mark as archived
            placeholders = ",".join("?" * len(ids))
            await db.execute(
                f"UPDATE {table_name} SET archived = 1, archived_at = ? WHERE id IN ({placeholders})",
                [_utcnow().isoformat()] + ids,
            )
            
            # Log audit entry
            await self._log_audit(
                db,
                action=RetentionAction.ARCHIVE,
                entity_type=policy.entity_type,
                affected_count=len(ids),
                details={"record_ids": ids[:100]},  # Limit stored IDs
            )
        
        result["count"] = len(ids)
        result["actions"].append({
            "action": "archive",
            "entity_type": policy.entity_type.value,
            "count": len(ids),
            "cutoff": cutoff.isoformat(),
        })
        
        return result
    
    async def _purge_records(
        self,
        db: Any,
        table_name: str,
        timestamp_col: str,
        cutoff: datetime,
        policy: RetentionPolicy,
        dry_run: bool,
    ) -> dict[str, Any]:
        """Purge records older than cutoff."""
        result = {"count": 0, "actions": []}
        
        # Build query - only purge archived records
        query = f"""
            SELECT id, chroma_id FROM {table_name}
            WHERE {timestamp_col} < ?
            AND archived = 1
        """
        params: list[Any] = [cutoff.isoformat()]
        
        if policy.exempt_filter:
            query += f" AND NOT ({policy.exempt_filter})"
        
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()
        
        if not rows:
            return result
        
        ids = [row["id"] for row in rows]
        chroma_ids = [row["chroma_id"] for row in rows if row["chroma_id"]]
        
        if dry_run:
            logger.info(f"[DRY RUN] Would purge {len(ids)} {table_name} records")
        else:
            # Delete from SQLite
            placeholders = ",".join("?" * len(ids))
            await db.execute(
                f"DELETE FROM {table_name} WHERE id IN ({placeholders})",
                ids,
            )
            
            # Delete from ChromaDB
            if chroma_ids:
                try:
                    collection = self.chroma_client.get_text_collection()
                    collection.delete(ids=chroma_ids)
                    logger.debug(f"Deleted {len(chroma_ids)} vectors from ChromaDB")
                except Exception as e:
                    logger.warning(f"Failed to delete vectors: {e}")
            
            # Log audit entry
            await self._log_audit(
                db,
                action=RetentionAction.PURGE,
                entity_type=policy.entity_type,
                affected_count=len(ids),
                details={"record_ids": ids[:100], "chroma_ids": chroma_ids[:100]},
            )
        
        result["count"] = len(ids)
        result["actions"].append({
            "action": "purge",
            "entity_type": policy.entity_type.value,
            "count": len(ids),
            "cutoff": cutoff.isoformat(),
        })
        
        return result
    
    async def _summarize_before_purge(
        self,
        db: Any,
        table_name: str,
        timestamp_col: str,
        cutoff: datetime,
        policy: RetentionPolicy,
        dry_run: bool,
    ) -> dict[str, Any]:
        """Create summaries for records about to be purged."""
        result = {"count": 0, "actions": []}
        
        # Only applicable for messages and sessions
        if policy.entity_type not in [EntityType.MESSAGE, EntityType.SESSION]:
            return result
        
        if policy.entity_type == EntityType.MESSAGE:
            # Group messages by session and create summaries
            query = f"""
                SELECT DISTINCT session_id FROM {table_name}
                WHERE {timestamp_col} < ?
                AND archived = 1
            """
            cursor = await db.execute(query, [cutoff.isoformat()])
            rows = await cursor.fetchall()
            
            session_ids = [row["session_id"] for row in rows if row["session_id"]]
            
            if not session_ids and dry_run:
                logger.info(f"[DRY RUN] Would summarize messages for {len(session_ids)} sessions")
            elif session_ids and not dry_run:
                # Trigger summarization for each session
                from ordinis.rag.memory.summarizer import SessionSummarizer
                summarizer = SessionSummarizer(
                    db_path=self.db_path,
                    chroma_client=self.chroma_client,
                )
                
                for session_id in session_ids[:10]:  # Limit to prevent long runs
                    try:
                        await summarizer.create_session_summary(session_id)
                        result["count"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to summarize session {session_id}: {e}")
        
        if result["count"] > 0:
            result["actions"].append({
                "action": "summarize",
                "entity_type": policy.entity_type.value,
                "count": result["count"],
            })
        
        return result
    
    async def _cleanup_orphaned_vectors(
        self,
        db: Any,
        result: RetentionResult,
    ) -> None:
        """Clean up vectors that have no corresponding SQLite record."""
        try:
            # Get all chroma_ids from SQLite
            cursor = await db.execute(
                "SELECT chroma_id FROM trades WHERE chroma_id IS NOT NULL "
                "UNION SELECT chroma_id FROM orders WHERE chroma_id IS NOT NULL "
                "UNION SELECT chroma_id FROM session_summaries WHERE chroma_id IS NOT NULL"
            )
            rows = await cursor.fetchall()
            valid_ids = {row[0] for row in rows}
            
            # Get all IDs from ChromaDB
            collection = self.chroma_client.get_text_collection()
            chroma_results = collection.get(include=[])
            
            if chroma_results and chroma_results.get("ids"):
                chroma_ids = set(chroma_results["ids"])
                orphaned = chroma_ids - valid_ids
                
                if orphaned:
                    # Delete orphaned vectors in batches
                    orphan_list = list(orphaned)
                    for i in range(0, len(orphan_list), 100):
                        batch = orphan_list[i:i + 100]
                        collection.delete(ids=batch)
                    
                    logger.info(f"Cleaned up {len(orphaned)} orphaned vectors")
                    result.actions.append({
                        "action": "cleanup_orphaned",
                        "count": len(orphaned),
                    })
        
        except Exception as e:
            logger.warning(f"Failed to cleanup orphaned vectors: {e}")
    
    async def _log_audit(
        self,
        db: Any,
        action: RetentionAction,
        entity_type: EntityType,
        affected_count: int,
        details: dict[str, Any],
    ) -> None:
        """Log a retention action to the audit table."""
        try:
            await db.execute(
                """
                INSERT INTO retention_audit
                (id, action, entity_type, affected_count, performed_at, details)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    action.value,
                    entity_type.value,
                    affected_count,
                    _utcnow().isoformat(),
                    json.dumps(details, default=str),
                ),
            )
        except Exception as e:
            logger.error(f"Failed to log audit entry: {e}")
    
    def _get_table_name(self, entity_type: EntityType) -> str | None:
        """Get table name for entity type."""
        mapping = {
            EntityType.SESSION: "sessions",
            EntityType.MESSAGE: "messages",
            EntityType.TRADE: "trades",
            EntityType.ORDER: "orders",
            EntityType.SUMMARY: "session_summaries",
        }
        return mapping.get(entity_type)
    
    def _get_timestamp_column(self, entity_type: EntityType) -> str:
        """Get timestamp column for entity type."""
        mapping = {
            EntityType.SESSION: "started_at",
            EntityType.MESSAGE: "created_at",
            EntityType.TRADE: "timestamp",
            EntityType.ORDER: "created_at",
            EntityType.SUMMARY: "created_at",
        }
        return mapping.get(entity_type, "created_at")
    
    async def get_retention_stats(self) -> dict[str, Any]:
        """Get statistics about data retention.
        
        Returns:
            Dictionary with retention statistics per entity type
        """
        import aiosqlite
        
        stats = {"entities": {}, "policies": [p.to_dict() for p in self.policies]}
        
        async with aiosqlite.connect(self.db_path) as db:
            for entity_type in EntityType:
                table_name = self._get_table_name(entity_type)
                if not table_name:
                    continue
                
                timestamp_col = self._get_timestamp_column(entity_type)
                
                try:
                    # Total count
                    cursor = await db.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row = await cursor.fetchone()
                    total = row[0] if row else 0
                    
                    # Archived count
                    cursor = await db.execute(
                        f"SELECT COUNT(*) FROM {table_name} WHERE archived = 1"
                    )
                    row = await cursor.fetchone()
                    archived = row[0] if row else 0
                    
                    # Oldest record
                    cursor = await db.execute(
                        f"SELECT MIN({timestamp_col}) FROM {table_name}"
                    )
                    row = await cursor.fetchone()
                    oldest = row[0] if row and row[0] else None
                    
                    stats["entities"][entity_type.value] = {
                        "total": total,
                        "archived": archived,
                        "active": total - archived,
                        "oldest_record": oldest,
                    }
                
                except Exception as e:
                    stats["entities"][entity_type.value] = {"error": str(e)}
        
        return stats
    
    async def get_audit_history(
        self,
        limit: int = 50,
        entity_type: EntityType | None = None,
    ) -> list[dict[str, Any]]:
        """Get retention audit history.
        
        Args:
            limit: Maximum records to return
            entity_type: Optional filter by entity type
            
        Returns:
            List of audit records
        """
        import aiosqlite
        
        query = "SELECT * FROM retention_audit"
        params: list[Any] = []
        
        if entity_type:
            query += " WHERE entity_type = ?"
            params.append(entity_type.value)
        
        query += " ORDER BY performed_at DESC LIMIT ?"
        params.append(limit)
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            
            return [
                {
                    "id": row["id"],
                    "action": row["action"],
                    "entity_type": row["entity_type"],
                    "affected_count": row["affected_count"],
                    "performed_at": row["performed_at"],
                    "details": json.loads(row["details"]) if row["details"] else {},
                }
                for row in rows
            ]


# CLI interface
async def main():
    """CLI entry point for retention management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Retention Manager")
    parser.add_argument(
        "--action",
        choices=["enforce", "stats", "history"],
        default="stats",
        help="Action to perform",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate without making changes",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to SQLite database",
    )
    
    args = parser.parse_args()
    
    manager = RetentionManager(db_path=args.db_path)
    
    if args.action == "enforce":
        result = await manager.enforce_retention_policies(dry_run=args.dry_run)
        print(f"Retention enforcement: {result.to_dict()}")
    elif args.action == "stats":
        stats = await manager.get_retention_stats()
        print("Retention Statistics:")
        print(json.dumps(stats, indent=2, default=str))
    elif args.action == "history":
        history = await manager.get_audit_history()
        print("Audit History:")
        for entry in history:
            print(f"  {entry['performed_at']}: {entry['action']} {entry['entity_type']} ({entry['affected_count']} records)")


if __name__ == "__main__":
    asyncio.run(main())
