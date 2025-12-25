"""
Database connection manager for Ordinis persistence layer.

Features:
- Async SQLite with aiosqlite
- WAL mode for concurrent reads
- Automatic backup before trading sessions
- Connection pooling (single connection for SQLite)
- Transaction support with rollback
- Integrity checks on startup
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import shutil
from typing import Any

import aiosqlite

from ordinis.adapters.storage.schema import (
    SCHEMA_VERSION,
    get_create_schema_sql,
    get_initial_state_sql,
)
from ordinis.utils.paths import resolve_project_path

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path("data/ordinis.sqlite3")
DEFAULT_BACKUP_DIR = Path("data/backups")


class DatabaseManager:
    """
    Manages SQLite database connection and operations.

    Uses WAL mode for better concurrent read performance.
    Provides automatic backup before trading sessions.
    """

    def __init__(
        self,
        db_path: Path | str | None = None,
        backup_dir: Path | str | None = None,
        auto_backup: bool = True,
    ):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
            backup_dir: Directory for automatic backups
            auto_backup: Whether to backup on initialize
        """
        self.db_path = resolve_project_path(db_path) if db_path else resolve_project_path(
            DEFAULT_DB_PATH
        )
        self.backup_dir = resolve_project_path(backup_dir) if backup_dir else resolve_project_path(
            DEFAULT_BACKUP_DIR
        )
        self.auto_backup = auto_backup
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()
        self._initialized = False

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connection is not None

    async def initialize(self) -> bool:
        """
        Initialize database connection and schema.

        Returns:
            True if successful, False otherwise
        """
        if self._initialized:
            return True

        try:
            # Ensure directories exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.backup_dir.mkdir(parents=True, exist_ok=True)

            # Create backup before initialization (if db exists)
            if self.auto_backup and self.db_path.exists():
                await self._create_backup("pre_init")

            # Connect to database
            self._connection = await aiosqlite.connect(
                self.db_path,
                isolation_level=None,  # Autocommit mode, we handle transactions
            )

            # Enable WAL mode for better concurrency
            await self._connection.execute("PRAGMA journal_mode=WAL")
            await self._connection.execute("PRAGMA synchronous=NORMAL")
            await self._connection.execute("PRAGMA foreign_keys=ON")
            await self._connection.execute("PRAGMA busy_timeout=5000")

            # Create schema if needed
            await self._create_schema()

            # Run integrity check
            integrity_ok = await self._check_integrity()
            if not integrity_ok:
                logger.error("Database integrity check failed")
                return False

            self._initialized = True
            logger.info(f"Database initialized: {self.db_path}")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize database: {e}")
            return False

    async def shutdown(self) -> None:
        """Close database connection gracefully."""
        if self._connection:
            try:
                # Checkpoint WAL before closing
                await self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                await self._connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.exception(f"Error closing database: {e}")
            finally:
                self._connection = None
                self._initialized = False

    async def _create_schema(self) -> None:
        """Create database schema if not exists."""
        if not self._connection:
            raise RuntimeError("Database not connected")

        # Execute schema DDL
        await self._connection.executescript(get_create_schema_sql())

        # Check if initial state needs to be inserted
        cursor = await self._connection.execute("SELECT COUNT(*) FROM system_state")
        row = await cursor.fetchone()
        count = row[0] if row else 0

        if count == 0:
            # Insert initial system state
            for key, value, value_type, description in get_initial_state_sql():
                await self._connection.execute(
                    """
                    INSERT OR IGNORE INTO system_state (key, value, value_type, description)
                    VALUES (?, ?, ?, ?)
                    """,
                    (key, value, value_type, description),
                )
            await self._connection.commit()
            logger.info("Initial system state created")

        # Record schema version
        cursor = await self._connection.execute("SELECT MAX(version) FROM schema_version")
        row = await cursor.fetchone()
        current_version = row[0] if row and row[0] else 0

        if current_version < SCHEMA_VERSION:
            await self._connection.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            await self._connection.commit()
            logger.info(f"Schema version updated to {SCHEMA_VERSION}")

    async def _check_integrity(self) -> bool:
        """Run database integrity check."""
        if not self._connection:
            return False

        try:
            cursor = await self._connection.execute("PRAGMA integrity_check")
            result = await cursor.fetchone()
            is_ok = result and result[0] == "ok"
            if not is_ok:
                logger.error(f"Integrity check result: {result}")
            return is_ok
        except Exception as e:
            logger.exception(f"Integrity check failed: {e}")
            return False

    async def _create_backup(self, suffix: str = "") -> Path | None:
        """
        Create database backup.

        Args:
            suffix: Optional suffix for backup filename

        Returns:
            Path to backup file, or None if failed
        """
        if not self.db_path.exists():
            return None

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        suffix_part = f"_{suffix}" if suffix else ""
        backup_name = f"ordinis_{timestamp}{suffix_part}.sqlite3"
        backup_path = self.backup_dir / backup_name

        try:
            # If connected, use SQLite backup API
            if self._connection:
                async with aiosqlite.connect(backup_path) as backup_conn:
                    await self._connection.backup(backup_conn)
            else:
                # Simple file copy if not connected
                shutil.copy2(self.db_path, backup_path)

            logger.info(f"Database backup created: {backup_path}")

            # Clean old backups (keep last 10)
            await self._cleanup_old_backups(keep=10)

            return backup_path
        except Exception as e:
            logger.exception(f"Failed to create backup: {e}")
            return None

    async def _cleanup_old_backups(self, keep: int = 10) -> None:
        """Remove old backup files, keeping the most recent."""
        try:
            backups = sorted(
                self.backup_dir.glob("ordinis_*.sqlite3"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for old_backup in backups[keep:]:
                old_backup.unlink()
                logger.debug(f"Removed old backup: {old_backup}")
        except Exception as e:
            logger.warning(f"Failed to cleanup old backups: {e}")

    async def create_session_backup(self) -> Path | None:
        """Create backup before trading session starts."""
        return await self._create_backup("session_start")

    async def execute(
        self,
        sql: str,
        parameters: tuple[Any, ...] | None = None,
    ) -> aiosqlite.Cursor:
        """
        Execute SQL statement.

        Args:
            sql: SQL statement
            parameters: Optional parameters

        Returns:
            Cursor with results
        """
        if not self._connection:
            raise RuntimeError("Database not connected")

        async with self._lock:
            if parameters:
                return await self._connection.execute(sql, parameters)
            return await self._connection.execute(sql)

    async def execute_many(
        self,
        sql: str,
        parameters: list[tuple[Any, ...]],
    ) -> None:
        """
        Execute SQL statement with multiple parameter sets.

        Args:
            sql: SQL statement
            parameters: List of parameter tuples
        """
        if not self._connection:
            raise RuntimeError("Database not connected")

        async with self._lock:
            await self._connection.executemany(sql, parameters)

    async def fetch_one(
        self,
        sql: str,
        parameters: tuple[Any, ...] | None = None,
    ) -> tuple[Any, ...] | None:
        """
        Fetch single row.

        Args:
            sql: SQL statement
            parameters: Optional parameters

        Returns:
            Row tuple or None
        """
        cursor = await self.execute(sql, parameters)
        return await cursor.fetchone()

    async def fetch_all(
        self,
        sql: str,
        parameters: tuple[Any, ...] | None = None,
    ) -> list[tuple[Any, ...]]:
        """
        Fetch all rows.

        Args:
            sql: SQL statement
            parameters: Optional parameters

        Returns:
            List of row tuples
        """
        cursor = await self.execute(sql, parameters)
        return await cursor.fetchall()

    async def commit(self) -> None:
        """Commit current transaction."""
        if self._connection:
            await self._connection.commit()

    async def rollback(self) -> None:
        """Rollback current transaction."""
        if self._connection:
            await self._connection.rollback()

    async def begin_transaction(self) -> None:
        """Begin explicit transaction."""
        if self._connection:
            await self._connection.execute("BEGIN TRANSACTION")

    # Whitelist of allowed tables for count queries
    _ALLOWED_TABLES = frozenset(
        {
            "positions",
            "orders",
            "fills",
            "trades",
            "system_state",
            "persistence_audit",
            "portfolio_snapshots",
        }
    )

    async def get_table_count(self, table: str) -> int:
        """Get row count for table."""
        # Use whitelist to prevent SQL injection
        if table not in self._ALLOWED_TABLES:
            raise ValueError(f"Invalid table name: {table}")
        row = await self.fetch_one(f"SELECT COUNT(*) FROM {table}")
        return row[0] if row else 0

    async def vacuum(self) -> None:
        """Vacuum database to reclaim space."""
        if self._connection:
            await self._connection.execute("VACUUM")
            logger.info("Database vacuumed")


# Global database instance
_global_db: DatabaseManager | None = None


def get_database(
    db_path: Path | str | None = None,
    backup_dir: Path | str | None = None,
) -> DatabaseManager:
    """
    Get global database instance.

    Args:
        db_path: Optional custom database path
        backup_dir: Optional custom backup directory

    Returns:
        DatabaseManager instance
    """
    global _global_db

    if _global_db is None:
        _global_db = DatabaseManager(db_path=db_path, backup_dir=backup_dir)

    return _global_db


async def reset_database() -> None:
    """Reset global database instance (for testing)."""
    global _global_db

    if _global_db:
        await _global_db.shutdown()
        _global_db = None
