"""
Order repository for order lifecycle persistence.

Provides CRUD operations for orders and fills with:
- Full order lifecycle tracking
- Fill processing and storage
- Active order queries for reconciliation
"""

from datetime import datetime
import json
import logging
from typing import TYPE_CHECKING

from ordinis.adapters.storage.models import FillRow, OrderRow

if TYPE_CHECKING:
    from ordinis.adapters.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


class OrderRepository:
    """Repository for order and fill CRUD operations."""

    def __init__(self, db: "DatabaseManager"):
        """
        Initialize order repository.

        Args:
            db: Database manager instance
        """
        self.db = db

    # ==================== ORDER OPERATIONS ====================

    async def get_by_id(self, order_id: str) -> OrderRow | None:
        """
        Get order by order_id.

        Args:
            order_id: Unique order identifier

        Returns:
            OrderRow or None if not found
        """
        row = await self.db.fetch_one(
            "SELECT * FROM orders WHERE order_id = ?",
            (order_id,),
        )
        return OrderRow.from_row(row) if row else None

    async def get_by_broker_id(self, broker_order_id: str) -> OrderRow | None:
        """
        Get order by broker order ID.

        Args:
            broker_order_id: Broker's order identifier

        Returns:
            OrderRow or None if not found
        """
        row = await self.db.fetch_one(
            "SELECT * FROM orders WHERE broker_order_id = ?",
            (broker_order_id,),
        )
        return OrderRow.from_row(row) if row else None

    async def get_active(self) -> list[OrderRow]:
        """
        Get all active (non-terminal) orders.

        Returns:
            List of active OrderRow
        """
        rows = await self.db.fetch_all(
            """
            SELECT * FROM orders
            WHERE status IN ('created', 'validated', 'pending_submit', 'submitted', 'acknowledged', 'partially_filled')
            ORDER BY created_at DESC
            """
        )
        return [OrderRow.from_row(row) for row in rows]

    async def get_by_status(self, status: str) -> list[OrderRow]:
        """
        Get orders by status.

        Args:
            status: Order status

        Returns:
            List of OrderRow
        """
        rows = await self.db.fetch_all(
            "SELECT * FROM orders WHERE status = ? ORDER BY created_at DESC",
            (status,),
        )
        return [OrderRow.from_row(row) for row in rows]

    async def get_by_symbol(self, symbol: str, limit: int = 100) -> list[OrderRow]:
        """
        Get orders for symbol.

        Args:
            symbol: Stock symbol
            limit: Maximum orders to return

        Returns:
            List of OrderRow
        """
        rows = await self.db.fetch_all(
            "SELECT * FROM orders WHERE symbol = ? ORDER BY created_at DESC LIMIT ?",
            (symbol, limit),
        )
        return [OrderRow.from_row(row) for row in rows]

    async def get_recent(self, limit: int = 100) -> list[OrderRow]:
        """
        Get recent orders.

        Args:
            limit: Maximum orders to return

        Returns:
            List of OrderRow
        """
        rows = await self.db.fetch_all(
            "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [OrderRow.from_row(row) for row in rows]

    async def create(self, order: OrderRow) -> bool:
        """
        Create new order.

        Args:
            order: Order to create

        Returns:
            True if successful
        """
        try:
            await self.db.execute(
                """
                INSERT INTO orders (
                    order_id, symbol, side, quantity, order_type, limit_price, stop_price,
                    time_in_force, status, filled_quantity, remaining_quantity, avg_fill_price,
                    created_at, submitted_at, filled_at, intent_id, signal_id, strategy_id,
                    session_id, broker_order_id, broker_response, error_message, retry_count,
                    chroma_synced, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                order.to_insert_tuple(),
            )
            await self.db.commit()

            await self._log_audit("create", order.order_id, order.model_dump_json())
            return True
        except Exception as e:
            logger.exception(f"Failed to create order {order.order_id}: {e}")
            await self.db.rollback()
            return False

    async def update_status(
        self,
        order_id: str,
        status: str,
        error_message: str | None = None,
    ) -> bool:
        """
        Update order status.

        Args:
            order_id: Order identifier
            status: New status
            error_message: Optional error message

        Returns:
            True if updated
        """
        try:
            now = datetime.utcnow().isoformat()

            # Set timestamps based on status
            submitted_at_update = ""
            filled_at_update = ""
            if status == "submitted":
                submitted_at_update = ", submitted_at = ?"
            if status == "filled":
                filled_at_update = ", filled_at = ?"

            sql = f"""
                UPDATE orders
                SET status = ?, error_message = ?, updated_at = ?{submitted_at_update}{filled_at_update}
                WHERE order_id = ?
            """

            params: list = [status, error_message, now]
            if submitted_at_update:
                params.append(now)
            if filled_at_update:
                params.append(now)
            params.append(order_id)

            await self.db.execute(sql, tuple(params))
            await self.db.commit()

            await self._log_audit("status_update", order_id, json.dumps({"status": status}))
            return True
        except Exception as e:
            logger.exception(f"Failed to update order status {order_id}: {e}")
            await self.db.rollback()
            return False

    async def update_broker_info(
        self,
        order_id: str,
        broker_order_id: str,
        broker_response: dict | None = None,
    ) -> bool:
        """
        Update broker information for order.

        Args:
            order_id: Order identifier
            broker_order_id: Broker's order ID
            broker_response: Broker response dict

        Returns:
            True if updated
        """
        try:
            response_json = json.dumps(broker_response) if broker_response else None
            await self.db.execute(
                """
                UPDATE orders
                SET broker_order_id = ?, broker_response = ?, updated_at = datetime('now')
                WHERE order_id = ?
                """,
                (broker_order_id, response_json, order_id),
            )
            await self.db.commit()
            return True
        except Exception as e:
            logger.exception(f"Failed to update broker info for {order_id}: {e}")
            await self.db.rollback()
            return False

    async def update_fill_info(
        self,
        order_id: str,
        filled_quantity: int,
        remaining_quantity: int,
        avg_fill_price: float,
        status: str | None = None,
    ) -> bool:
        """
        Update fill information for order.

        Args:
            order_id: Order identifier
            filled_quantity: Total filled quantity
            remaining_quantity: Remaining quantity
            avg_fill_price: Average fill price
            status: Optional new status

        Returns:
            True if updated
        """
        try:
            now = datetime.utcnow().isoformat()
            status_update = ", status = ?" if status else ""
            filled_at_update = ", filled_at = ?" if status == "filled" else ""

            sql = f"""
                UPDATE orders
                SET filled_quantity = ?, remaining_quantity = ?, avg_fill_price = ?,
                    updated_at = ?{status_update}{filled_at_update}
                WHERE order_id = ?
            """

            params: list = [filled_quantity, remaining_quantity, avg_fill_price, now]
            if status:
                params.append(status)
            if status == "filled":
                params.append(now)
            params.append(order_id)

            await self.db.execute(sql, tuple(params))
            await self.db.commit()
            return True
        except Exception as e:
            logger.exception(f"Failed to update fill info for {order_id}: {e}")
            await self.db.rollback()
            return False

    async def increment_retry(self, order_id: str) -> bool:
        """
        Increment retry count for order.

        Args:
            order_id: Order identifier

        Returns:
            True if updated
        """
        try:
            await self.db.execute(
                """
                UPDATE orders
                SET retry_count = retry_count + 1, updated_at = datetime('now')
                WHERE order_id = ?
                """,
                (order_id,),
            )
            await self.db.commit()
            return True
        except Exception as e:
            logger.exception(f"Failed to increment retry for {order_id}: {e}")
            await self.db.rollback()
            return False

    # ==================== FILL OPERATIONS ====================

    async def get_fill_by_id(self, fill_id: str) -> FillRow | None:
        """
        Get fill by fill_id.

        Args:
            fill_id: Unique fill identifier

        Returns:
            FillRow or None if not found
        """
        row = await self.db.fetch_one(
            "SELECT * FROM fills WHERE fill_id = ?",
            (fill_id,),
        )
        return FillRow.from_row(row) if row else None

    async def get_fills_for_order(self, order_id: str) -> list[FillRow]:
        """
        Get all fills for an order.

        Args:
            order_id: Order identifier

        Returns:
            List of FillRow
        """
        rows = await self.db.fetch_all(
            "SELECT * FROM fills WHERE order_id = ? ORDER BY timestamp",
            (order_id,),
        )
        return [FillRow.from_row(row) for row in rows]

    async def create_fill(self, fill: FillRow) -> bool:
        """
        Create new fill record.

        Args:
            fill: Fill to create

        Returns:
            True if successful
        """
        try:
            await self.db.execute(
                """
                INSERT INTO fills (
                    fill_id, order_id, symbol, side, quantity, price,
                    commission, timestamp, latency_ms, slippage_bps, vs_arrival_bps, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                fill.to_insert_tuple(),
            )
            await self.db.commit()

            await self._log_audit("create_fill", fill.fill_id, fill.model_dump_json())
            return True
        except Exception as e:
            logger.exception(f"Failed to create fill {fill.fill_id}: {e}")
            await self.db.rollback()
            return False

    # ==================== UTILITY METHODS ====================

    async def get_today_order_count(self) -> int:
        """Get count of orders created today."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        row = await self.db.fetch_one(
            "SELECT COUNT(*) FROM orders WHERE created_at >= ?",
            (today,),
        )
        return row[0] if row else 0

    async def get_today_fill_count(self) -> int:
        """Get count of fills received today."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        row = await self.db.fetch_one(
            "SELECT COUNT(*) FROM fills WHERE timestamp >= ?",
            (today,),
        )
        return row[0] if row else 0

    async def cancel_all_active(self, reason: str = "System shutdown") -> int:
        """
        Mark all active orders as cancelled.

        Args:
            reason: Cancellation reason

        Returns:
            Number of orders cancelled
        """
        try:
            now = datetime.utcnow().isoformat()
            cursor = await self.db.execute(
                """
                UPDATE orders
                SET status = 'cancelled', error_message = ?, updated_at = ?
                WHERE status IN ('created', 'validated', 'pending_submit', 'submitted', 'acknowledged', 'partially_filled')
                """,
                (reason, now),
            )
            await self.db.commit()
            cancelled = cursor.rowcount
            logger.info(f"Cancelled {cancelled} active orders: {reason}")
            return cancelled
        except Exception as e:
            logger.exception(f"Failed to cancel active orders: {e}")
            await self.db.rollback()
            return 0

    async def _log_audit(
        self,
        action: str,
        entity_id: str,
        new_value: str,
        old_value: str | None = None,
    ) -> None:
        """Log audit event for order change."""
        try:
            await self.db.execute(
                """
                INSERT INTO persistence_audit (
                    event_type, entity_type, entity_id, action, old_value, new_value
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("order_change", "order", entity_id, action, old_value, new_value),
            )
        except Exception as e:
            logger.warning(f"Failed to log audit event: {e}")
