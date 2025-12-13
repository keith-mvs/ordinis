"""
Trade repository for completed trade persistence.

Provides CRUD operations for completed trades with:
- Trade history queries
- P&L aggregation
- Performance metrics calculation
"""

from datetime import datetime, timedelta
import logging
from typing import TYPE_CHECKING

from persistence.models import TradeRow

if TYPE_CHECKING:
    from persistence.database import DatabaseManager

logger = logging.getLogger(__name__)


class TradeRepository:
    """Repository for trade CRUD operations."""

    def __init__(self, db: "DatabaseManager"):
        """
        Initialize trade repository.

        Args:
            db: Database manager instance
        """
        self.db = db

    async def get_by_id(self, trade_id: str) -> TradeRow | None:
        """
        Get trade by trade_id.

        Args:
            trade_id: Unique trade identifier

        Returns:
            TradeRow or None if not found
        """
        row = await self.db.fetch_one(
            "SELECT * FROM trades WHERE trade_id = ?",
            (trade_id,),
        )
        return TradeRow.from_row(row) if row else None

    async def get_all(self, limit: int = 1000) -> list[TradeRow]:
        """
        Get all trades.

        Args:
            limit: Maximum trades to return

        Returns:
            List of TradeRow
        """
        rows = await self.db.fetch_all(
            "SELECT * FROM trades ORDER BY exit_time DESC LIMIT ?",
            (limit,),
        )
        return [TradeRow.from_row(row) for row in rows]

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> list[TradeRow]:
        """
        Get trades for symbol.

        Args:
            symbol: Stock symbol
            limit: Maximum trades to return

        Returns:
            List of TradeRow
        """
        rows = await self.db.fetch_all(
            "SELECT * FROM trades WHERE symbol = ? ORDER BY exit_time DESC LIMIT ?",
            (symbol, limit),
        )
        return [TradeRow.from_row(row) for row in rows]

    async def get_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> list[TradeRow]:
        """
        Get trades within date range.

        Args:
            start_date: Start datetime
            end_date: End datetime

        Returns:
            List of TradeRow
        """
        rows = await self.db.fetch_all(
            """
            SELECT * FROM trades
            WHERE exit_time >= ? AND exit_time <= ?
            ORDER BY exit_time DESC
            """,
            (start_date.isoformat(), end_date.isoformat()),
        )
        return [TradeRow.from_row(row) for row in rows]

    async def get_today(self) -> list[TradeRow]:
        """
        Get trades closed today.

        Returns:
            List of TradeRow
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        rows = await self.db.fetch_all(
            "SELECT * FROM trades WHERE exit_time >= ? ORDER BY exit_time DESC",
            (today,),
        )
        return [TradeRow.from_row(row) for row in rows]

    async def create(self, trade: TradeRow) -> bool:
        """
        Create new trade record.

        Args:
            trade: Trade to create

        Returns:
            True if successful
        """
        try:
            await self.db.execute(
                """
                INSERT INTO trades (
                    trade_id, symbol, side, entry_time, exit_time, entry_price,
                    exit_price, quantity, pnl, pnl_pct, commission, duration_seconds,
                    entry_order_id, exit_order_id, strategy_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                trade.to_insert_tuple(),
            )
            await self.db.commit()

            await self._log_audit("create", trade.trade_id, trade.model_dump_json())
            logger.info(
                f"Trade recorded: {trade.symbol} {trade.side} "
                f"P&L=${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)"
            )
            return True
        except Exception as e:
            logger.exception(f"Failed to create trade {trade.trade_id}: {e}")
            await self.db.rollback()
            return False

    # ==================== AGGREGATION METHODS ====================

    async def get_total_pnl(self) -> float:
        """Get total P&L across all trades."""
        row = await self.db.fetch_one("SELECT COALESCE(SUM(pnl), 0) FROM trades")
        return row[0] if row else 0.0

    async def get_today_pnl(self) -> float:
        """Get today's realized P&L."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        row = await self.db.fetch_one(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE exit_time >= ?",
            (today,),
        )
        return row[0] if row else 0.0

    async def get_pnl_by_date(self, date: str) -> float:
        """
        Get P&L for specific date.

        Args:
            date: Date string (YYYY-MM-DD)

        Returns:
            Total P&L for date
        """
        next_date = (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        row = await self.db.fetch_one(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE exit_time >= ? AND exit_time < ?",
            (date, next_date),
        )
        return row[0] if row else 0.0

    async def get_win_rate(self) -> float:
        """
        Calculate win rate (percentage of profitable trades).

        Returns:
            Win rate as percentage (0-100)
        """
        row = await self.db.fetch_one(
            """
            SELECT
                COUNT(CASE WHEN pnl > 0 THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0)
            FROM trades
            """
        )
        return row[0] if row and row[0] else 0.0

    async def get_trade_count(self) -> int:
        """Get total trade count."""
        row = await self.db.fetch_one("SELECT COUNT(*) FROM trades")
        return row[0] if row else 0

    async def get_today_trade_count(self) -> int:
        """Get today's trade count."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        row = await self.db.fetch_one(
            "SELECT COUNT(*) FROM trades WHERE exit_time >= ?",
            (today,),
        )
        return row[0] if row else 0

    async def get_average_pnl(self) -> float:
        """Get average P&L per trade."""
        row = await self.db.fetch_one("SELECT COALESCE(AVG(pnl), 0) FROM trades")
        return row[0] if row else 0.0

    async def get_average_winner(self) -> float:
        """Get average P&L for winning trades."""
        row = await self.db.fetch_one("SELECT COALESCE(AVG(pnl), 0) FROM trades WHERE pnl > 0")
        return row[0] if row else 0.0

    async def get_average_loser(self) -> float:
        """Get average P&L for losing trades."""
        row = await self.db.fetch_one("SELECT COALESCE(AVG(pnl), 0) FROM trades WHERE pnl < 0")
        return row[0] if row else 0.0

    async def get_profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Returns:
            Profit factor (> 1 means profitable)
        """
        row = await self.db.fetch_one(
            """
            SELECT
                COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0) as gross_profit,
                COALESCE(ABS(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END)), 0) as gross_loss
            FROM trades
            """
        )
        if row and row[1] > 0:
            return row[0] / row[1]
        return 0.0

    async def get_largest_winner(self) -> float:
        """Get largest winning trade P&L."""
        row = await self.db.fetch_one("SELECT COALESCE(MAX(pnl), 0) FROM trades")
        return row[0] if row else 0.0

    async def get_largest_loser(self) -> float:
        """Get largest losing trade P&L."""
        row = await self.db.fetch_one("SELECT COALESCE(MIN(pnl), 0) FROM trades")
        return row[0] if row else 0.0

    async def get_performance_summary(self) -> dict:
        """
        Get comprehensive performance summary.

        Returns:
            Dictionary with performance metrics
        """
        return {
            "total_trades": await self.get_trade_count(),
            "today_trades": await self.get_today_trade_count(),
            "total_pnl": await self.get_total_pnl(),
            "today_pnl": await self.get_today_pnl(),
            "win_rate": await self.get_win_rate(),
            "average_pnl": await self.get_average_pnl(),
            "average_winner": await self.get_average_winner(),
            "average_loser": await self.get_average_loser(),
            "profit_factor": await self.get_profit_factor(),
            "largest_winner": await self.get_largest_winner(),
            "largest_loser": await self.get_largest_loser(),
        }

    async def _log_audit(
        self,
        action: str,
        entity_id: str,
        new_value: str,
        old_value: str | None = None,
    ) -> None:
        """Log audit event for trade."""
        try:
            await self.db.execute(
                """
                INSERT INTO persistence_audit (
                    event_type, entity_type, entity_id, action, old_value, new_value
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("trade_recorded", "trade", entity_id, action, old_value, new_value),
            )
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")
