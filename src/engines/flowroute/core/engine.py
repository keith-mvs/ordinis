"""
FlowRoute execution engine for order management.

Manages order lifecycle, broker routing, and execution quality tracking.
Integrates with persistence, kill switch, and alerting for production safety.
"""

from __future__ import annotations

from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any
import uuid

from .orders import (
    ExecutionEvent,
    Fill,
    Order,
    OrderIntent,
    OrderStatus,
)

if TYPE_CHECKING:
    from adapters.alerting import AlertManager
    from adapters.storage.repositories.order import OrderRepository
    from core.protocols import BrokerAdapter
    from safety.kill_switch import KillSwitch

logger = logging.getLogger(__name__)


class FlowRouteEngine:
    """
    FlowRoute execution engine.

    Manages order lifecycle from intent to execution.
    """

    def __init__(
        self,
        broker_adapter: BrokerAdapter | None = None,
        kill_switch: KillSwitch | None = None,
        order_repository: OrderRepository | None = None,
        alert_manager: AlertManager | None = None,
    ):
        """
        Initialize FlowRoute engine.

        Args:
            broker_adapter: Broker adapter for order execution
            kill_switch: Kill switch for emergency stop
            order_repository: Repository for order persistence
            alert_manager: Alert manager for notifications
        """
        self._broker = broker_adapter
        self._kill_switch = kill_switch
        self._order_repo = order_repository
        self._alert_manager = alert_manager
        self._orders: dict[str, Order] = {}
        self._active_orders: set[str] = set()

    def create_order_from_intent(self, intent: OrderIntent) -> Order:
        """
        Create executable order from RiskGuard intent.

        Args:
            intent: Order intent from RiskGuard

        Returns:
            Order ready for submission
        """
        order_id = str(uuid.uuid4())

        order = Order(
            order_id=order_id,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            order_type=intent.order_type,
            limit_price=intent.limit_price,
            stop_price=intent.stop_price,
            time_in_force=intent.time_in_force,
            intent_id=intent.intent_id,
            signal_id=intent.signal_id,
            strategy_id=intent.strategy_id,
            metadata=intent.metadata.copy(),
        )

        self._orders[order_id] = order

        # Log creation event
        event = ExecutionEvent(
            event_id=str(uuid.uuid4()),
            order_id=order_id,
            event_type="order_created",
            timestamp=datetime.utcnow(),
            status_before=None,
            status_after=OrderStatus.CREATED,
            details={"intent_id": intent.intent_id},
        )
        order.events.append(event)

        return order

    async def submit_order(self, order: Order) -> tuple[bool, str]:
        """
        Submit order to broker.

        Checks kill switch before submission and persists order state.

        Args:
            order: Order to submit

        Returns:
            Tuple of (success, message)
        """
        # SAFETY CHECK: Kill switch
        if self._kill_switch and self._kill_switch.is_active:
            logger.warning(f"Order {order.order_id} blocked by kill switch")
            order.status = OrderStatus.REJECTED
            order.error_message = "Kill switch active - trading halted"
            await self._persist_order(order)
            await self._send_alert(
                "Order Blocked",
                f"Order {order.symbol} blocked: kill switch active",
                severity="warning",
            )
            return False, "Kill switch active - trading halted"

        if not self._broker:
            return False, "No broker adapter configured"

        if order.status != OrderStatus.CREATED:
            return False, f"Order must be in CREATED state, got {order.status.value}"

        # Update status
        order.status = OrderStatus.PENDING_SUBMIT
        order.submitted_at = datetime.utcnow()

        # Log submission event
        event = ExecutionEvent(
            event_id=str(uuid.uuid4()),
            order_id=order.order_id,
            event_type="order_submitted",
            timestamp=datetime.utcnow(),
            status_before=OrderStatus.CREATED,
            status_after=OrderStatus.PENDING_SUBMIT,
        )
        order.events.append(event)

        # Persist before broker call
        await self._persist_order(order)

        # Submit to broker
        try:
            broker_response = await self._broker.submit_order(order)

            if broker_response.get("success"):
                order.status = OrderStatus.SUBMITTED
                order.broker_order_id = broker_response.get("broker_order_id")
                order.broker_response = broker_response

                self._active_orders.add(order.order_id)

                # Log acknowledgment
                event = ExecutionEvent(
                    event_id=str(uuid.uuid4()),
                    order_id=order.order_id,
                    event_type="order_acknowledged",
                    timestamp=datetime.utcnow(),
                    status_before=OrderStatus.PENDING_SUBMIT,
                    status_after=OrderStatus.SUBMITTED,
                    details=broker_response,
                )
                order.events.append(event)

                # Persist after successful submission
                await self._persist_order(order)
                logger.info(f"Order {order.order_id} submitted: {order.symbol}")

                return True, "Order submitted successfully"

            order.status = OrderStatus.REJECTED
            order.error_message = broker_response.get("error", "Unknown error")

            event = ExecutionEvent(
                event_id=str(uuid.uuid4()),
                order_id=order.order_id,
                event_type="order_rejected",
                timestamp=datetime.utcnow(),
                status_before=OrderStatus.PENDING_SUBMIT,
                status_after=OrderStatus.REJECTED,
                error_message=order.error_message,
            )
            order.events.append(event)

            # Persist rejection and alert
            await self._persist_order(order)
            await self._send_alert(
                "Order Rejected",
                f"Order {order.symbol} rejected: {order.error_message}",
                severity="warning",
                metadata={"order_id": order.order_id, "symbol": order.symbol},
            )

            return False, order.error_message or "Unknown error"

        except Exception as e:
            order.status = OrderStatus.ERROR
            order.error_message = str(e)

            event = ExecutionEvent(
                event_id=str(uuid.uuid4()),
                order_id=order.order_id,
                event_type="order_error",
                timestamp=datetime.utcnow(),
                status_before=OrderStatus.PENDING_SUBMIT,
                status_after=OrderStatus.ERROR,
                error_message=str(e),
            )
            order.events.append(event)

            # Persist error state
            await self._persist_order(order)
            logger.exception("Error submitting order %s", order.order_id)

            return False, f"Error submitting order: {e}"

    async def cancel_order(self, order_id: str, reason: str = "") -> tuple[bool, str]:
        """
        Cancel pending order.

        Persists cancelled state.

        Args:
            order_id: Order identifier
            reason: Cancellation reason

        Returns:
            Tuple of (success, message)
        """
        if order_id not in self._orders:
            return False, f"Order {order_id} not found"

        order = self._orders[order_id]

        if not order.is_active():
            return False, f"Order is not active (status: {order.status.value})"

        if not self._broker:
            return False, "No broker adapter configured"

        try:
            prev_status = order.status
            broker_response = await self._broker.cancel_order(order.broker_order_id or order_id)

            if broker_response.get("success"):
                order.status = OrderStatus.CANCELLED

                event = ExecutionEvent(
                    event_id=str(uuid.uuid4()),
                    order_id=order_id,
                    event_type="order_cancelled",
                    timestamp=datetime.utcnow(),
                    status_before=prev_status,
                    status_after=OrderStatus.CANCELLED,
                    details={"reason": reason},
                )
                order.events.append(event)

                if order_id in self._active_orders:
                    self._active_orders.remove(order_id)

                # Persist cancelled state
                await self._persist_order(order)
                logger.info(f"Order {order_id} cancelled: {reason}")

                return True, "Order cancelled successfully"
            return False, broker_response.get("error", "Unknown error")

        except Exception as e:
            logger.exception("Error cancelling order %s", order_id)
            return False, f"Error cancelling order: {e}"

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID."""
        return self._orders.get(order_id)

    def get_active_orders(self) -> list[Order]:
        """Get all active orders."""
        return [self._orders[oid] for oid in self._active_orders if oid in self._orders]

    def get_all_orders(self) -> list[Order]:
        """Get all orders."""
        return list(self._orders.values())

    async def process_fill(self, fill: Fill) -> None:
        """
        Process fill notification from broker.

        Persists fill and updates order state.

        Args:
            fill: Fill to process
        """
        if fill.order_id not in self._orders:
            logger.warning(f"Fill received for unknown order: {fill.order_id}")
            return

        order = self._orders[fill.order_id]
        prev_status = order.status
        order.add_fill(fill)

        # Log fill event
        event = ExecutionEvent(
            event_id=str(uuid.uuid4()),
            order_id=fill.order_id,
            event_type="fill_received",
            timestamp=datetime.utcnow(),
            status_before=prev_status,
            status_after=order.status,
            details={
                "fill_id": fill.fill_id,
                "quantity": fill.quantity,
                "price": fill.price,
            },
        )
        order.events.append(event)

        # Persist fill and updated order
        await self._persist_fill(fill, order)
        await self._persist_order(order)

        logger.info(f"Fill processed: {fill.quantity} @ {fill.price} for {order.symbol}")

        # Remove from active if fully filled
        if order.status == OrderStatus.FILLED and fill.order_id in self._active_orders:
            self._active_orders.remove(fill.order_id)

    async def _persist_order(self, order: Order) -> None:
        """Persist order state to repository if configured."""
        if not self._order_repo:
            return

        try:
            await self._order_repo.upsert(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.value if hasattr(order.side, "value") else str(order.side),
                quantity=order.quantity,
                order_type=order.order_type.value
                if hasattr(order.order_type, "value")
                else str(order.order_type),
                status=order.status.value,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force.value
                if hasattr(order.time_in_force, "value")
                else str(order.time_in_force),
                broker_order_id=order.broker_order_id,
                strategy_id=order.strategy_id,
                signal_id=order.signal_id,
                error_message=order.error_message,
            )
        except Exception:
            logger.exception("Failed to persist order %s", order.order_id)

    async def _persist_fill(self, fill: Fill, order: Order) -> None:
        """Persist fill to repository if configured."""
        if not self._order_repo:
            return

        try:
            await self._order_repo.add_fill(
                fill_id=fill.fill_id,
                order_id=fill.order_id,
                quantity=fill.quantity,
                price=fill.price,
                commission=fill.commission,
            )
        except Exception:
            logger.exception("Failed to persist fill %s", fill.fill_id)

    async def _send_alert(
        self,
        title: str,
        message: str,
        severity: str = "warning",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Send alert if manager configured."""
        if not self._alert_manager:
            return

        try:
            from adapters.alerting import AlertSeverity
            from adapters.alerting.manager import AlertType

            severity_map = {
                "info": AlertSeverity.INFO,
                "warning": AlertSeverity.WARNING,
                "critical": AlertSeverity.CRITICAL,
                "emergency": AlertSeverity.EMERGENCY,
            }

            await self._alert_manager.send(
                alert_type=AlertType.ORDER_REJECTED,
                severity=severity_map.get(severity, AlertSeverity.WARNING),
                title=title,
                message=message,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Failed to send alert")

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        all_orders = self.get_all_orders()
        filled_orders = [o for o in all_orders if o.status == OrderStatus.FILLED]
        all_fills = [f for o in filled_orders for f in o.fills]

        if not filled_orders:
            return {
                "total_orders": len(all_orders),
                "filled_orders": 0,
                "fill_rate": 0.0,
                "avg_fill_time_seconds": 0.0,
                "avg_slippage_bps": 0.0,
            }

        # Calculate fill times
        fill_times = []
        for order in filled_orders:
            if order.created_at and order.filled_at:
                fill_time = (order.filled_at - order.created_at).total_seconds()
                fill_times.append(fill_time)

        avg_fill_time = sum(fill_times) / len(fill_times) if fill_times else 0.0

        # Calculate average slippage
        avg_slippage = sum(f.slippage_bps for f in all_fills) / len(all_fills) if all_fills else 0.0

        return {
            "total_orders": len(all_orders),
            "filled_orders": len(filled_orders),
            "partially_filled": len(
                [o for o in all_orders if o.status == OrderStatus.PARTIALLY_FILLED]
            ),
            "cancelled": len([o for o in all_orders if o.status == OrderStatus.CANCELLED]),
            "rejected": len([o for o in all_orders if o.status == OrderStatus.REJECTED]),
            "fill_rate": len(filled_orders) / len(all_orders) if all_orders else 0.0,
            "avg_fill_time_seconds": avg_fill_time,
            "avg_slippage_bps": avg_slippage,
            "total_fills": len(all_fills),
        }

    def to_dict(self) -> dict[str, Any]:
        """Get engine state as dictionary."""
        return {
            "total_orders": len(self._orders),
            "active_orders": len(self._active_orders),
            "has_broker": self._broker is not None,
            "execution_stats": self.get_execution_stats(),
        }
