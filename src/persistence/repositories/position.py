"""
Position repository for portfolio state persistence.

Provides CRUD operations for positions with:
- Upsert support (create or update)
- Batch operations for efficiency
- Snapshot creation for recovery
"""

from datetime import datetime
import json
import logging
from typing import TYPE_CHECKING

from persistence.models import PortfolioSnapshotRow, PositionRow

if TYPE_CHECKING:
    from persistence.database import DatabaseManager

logger = logging.getLogger(__name__)


class PositionRepository:
    """Repository for position CRUD operations."""

    def __init__(self, db: "DatabaseManager"):
        """
        Initialize position repository.

        Args:
            db: Database manager instance
        """
        self.db = db

    async def get_by_symbol(self, symbol: str) -> PositionRow | None:
        """
        Get position by symbol.

        Args:
            symbol: Stock symbol

        Returns:
            PositionRow or None if not found
        """
        row = await self.db.fetch_one(
            "SELECT * FROM positions WHERE symbol = ?",
            (symbol,),
        )
        return PositionRow.from_row(row) if row else None

    async def get_all(self) -> list[PositionRow]:
        """
        Get all positions.

        Returns:
            List of PositionRow
        """
        rows = await self.db.fetch_all("SELECT * FROM positions")
        return [PositionRow.from_row(row) for row in rows]

    async def get_active(self) -> list[PositionRow]:
        """
        Get all non-flat positions.

        Returns:
            List of active PositionRow
        """
        rows = await self.db.fetch_all(
            "SELECT * FROM positions WHERE side != 'FLAT' AND quantity > 0"
        )
        return [PositionRow.from_row(row) for row in rows]

    async def upsert(self, position: PositionRow) -> bool:
        """
        Insert or update position.

        Args:
            position: Position to upsert

        Returns:
            True if successful
        """
        try:
            now = datetime.utcnow().isoformat()
            position.last_update = now

            await self.db.execute(
                """
                INSERT INTO positions (
                    symbol, side, quantity, avg_cost, current_price,
                    realized_pnl, unrealized_pnl, entry_time, last_update
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    side = excluded.side,
                    quantity = excluded.quantity,
                    avg_cost = excluded.avg_cost,
                    current_price = excluded.current_price,
                    realized_pnl = excluded.realized_pnl,
                    unrealized_pnl = excluded.unrealized_pnl,
                    entry_time = excluded.entry_time,
                    last_update = excluded.last_update,
                    updated_at = datetime('now')
                """,
                position.to_insert_tuple(),
            )
            await self.db.commit()

            # Log audit event
            await self._log_audit("upsert", position.symbol, position.model_dump_json())

            return True
        except Exception as e:
            logger.exception(f"Failed to upsert position {position.symbol}: {e}")
            await self.db.rollback()
            return False

    async def upsert_batch(self, positions: list[PositionRow]) -> bool:
        """
        Batch upsert positions.

        Args:
            positions: List of positions to upsert

        Returns:
            True if all successful
        """
        if not positions:
            return True

        try:
            await self.db.begin_transaction()
            now = datetime.utcnow().isoformat()

            for position in positions:
                position.last_update = now
                await self.db.execute(
                    """
                    INSERT INTO positions (
                        symbol, side, quantity, avg_cost, current_price,
                        realized_pnl, unrealized_pnl, entry_time, last_update
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET
                        side = excluded.side,
                        quantity = excluded.quantity,
                        avg_cost = excluded.avg_cost,
                        current_price = excluded.current_price,
                        realized_pnl = excluded.realized_pnl,
                        unrealized_pnl = excluded.unrealized_pnl,
                        entry_time = excluded.entry_time,
                        last_update = excluded.last_update,
                        updated_at = datetime('now')
                    """,
                    position.to_insert_tuple(),
                )

            await self.db.commit()
            logger.info(f"Batch upserted {len(positions)} positions")
            return True
        except Exception as e:
            logger.exception(f"Failed to batch upsert positions: {e}")
            await self.db.rollback()
            return False

    async def delete(self, symbol: str) -> bool:
        """
        Delete position by symbol.

        Args:
            symbol: Stock symbol

        Returns:
            True if deleted
        """
        try:
            # Get current state for audit
            current = await self.get_by_symbol(symbol)

            await self.db.execute(
                "DELETE FROM positions WHERE symbol = ?",
                (symbol,),
            )
            await self.db.commit()

            if current:
                await self._log_audit("delete", symbol, current.model_dump_json())

            return True
        except Exception as e:
            logger.exception(f"Failed to delete position {symbol}: {e}")
            await self.db.rollback()
            return False

    async def delete_flat_positions(self) -> int:
        """
        Delete all flat positions.

        Returns:
            Number of positions deleted
        """
        try:
            cursor = await self.db.execute(
                "DELETE FROM positions WHERE side = 'FLAT' OR quantity = 0"
            )
            await self.db.commit()
            deleted = cursor.rowcount
            logger.info(f"Deleted {deleted} flat positions")
            return deleted
        except Exception as e:
            logger.exception(f"Failed to delete flat positions: {e}")
            await self.db.rollback()
            return 0

    async def update_price(
        self,
        symbol: str,
        price: float,
    ) -> bool:
        """
        Update current price for position.

        Args:
            symbol: Stock symbol
            price: Current market price

        Returns:
            True if updated
        """
        try:
            position = await self.get_by_symbol(symbol)
            if not position:
                return False

            # Calculate unrealized P&L
            if position.side == "LONG" and position.quantity > 0:
                unrealized_pnl = (price - position.avg_cost) * position.quantity
            elif position.side == "SHORT" and position.quantity > 0:
                unrealized_pnl = (position.avg_cost - price) * position.quantity
            else:
                unrealized_pnl = 0.0

            now = datetime.utcnow().isoformat()
            await self.db.execute(
                """
                UPDATE positions
                SET current_price = ?, unrealized_pnl = ?, last_update = ?, updated_at = datetime('now')
                WHERE symbol = ?
                """,
                (price, unrealized_pnl, now, symbol),
            )
            await self.db.commit()
            return True
        except Exception as e:
            logger.exception(f"Failed to update price for {symbol}: {e}")
            await self.db.rollback()
            return False

    async def create_snapshot(
        self,
        cash: float,
        total_equity: float,
    ) -> bool:
        """
        Create portfolio snapshot for recovery.

        Args:
            cash: Current cash balance
            total_equity: Total portfolio equity

        Returns:
            True if successful
        """
        try:
            positions = await self.get_all()
            total_position_value = sum(p.quantity * p.current_price for p in positions)

            positions_json = json.dumps([p.model_dump() for p in positions])
            snapshot_date = datetime.utcnow().strftime("%Y-%m-%d")

            await self.db.execute(
                """
                INSERT INTO portfolio_snapshots (
                    snapshot_date, cash, total_equity, total_position_value, positions_json
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_date) DO UPDATE SET
                    cash = excluded.cash,
                    total_equity = excluded.total_equity,
                    total_position_value = excluded.total_position_value,
                    positions_json = excluded.positions_json,
                    created_at = datetime('now')
                """,
                (snapshot_date, cash, total_equity, total_position_value, positions_json),
            )
            await self.db.commit()
            logger.info(f"Created portfolio snapshot for {snapshot_date}")
            return True
        except Exception as e:
            logger.exception(f"Failed to create portfolio snapshot: {e}")
            await self.db.rollback()
            return False

    async def get_latest_snapshot(self) -> PortfolioSnapshotRow | None:
        """
        Get most recent portfolio snapshot.

        Returns:
            PortfolioSnapshotRow or None
        """
        row = await self.db.fetch_one(
            "SELECT * FROM portfolio_snapshots ORDER BY snapshot_date DESC LIMIT 1"
        )
        return PortfolioSnapshotRow.from_row(row) if row else None

    async def get_total_realized_pnl(self) -> float:
        """Get total realized P&L across all positions."""
        row = await self.db.fetch_one("SELECT COALESCE(SUM(realized_pnl), 0) FROM positions")
        return row[0] if row else 0.0

    async def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        row = await self.db.fetch_one("SELECT COALESCE(SUM(unrealized_pnl), 0) FROM positions")
        return row[0] if row else 0.0

    async def _log_audit(
        self,
        action: str,
        entity_id: str,
        new_value: str,
        old_value: str | None = None,
    ) -> None:
        """Log audit event for position change."""
        try:
            await self.db.execute(
                """
                INSERT INTO persistence_audit (
                    event_type, entity_type, entity_id, action, old_value, new_value
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("position_change", "position", entity_id, action, old_value, new_value),
            )
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")
