"""
FlowRoute execution engine for order management.

Manages order lifecycle, broker routing, and execution quality tracking.
Integrates with persistence, kill switch, and alerting for production safety.

Phase 1 enhancements (2025-12-17):
- Pre-trade buying power validation
- Position sync on startup
- Broker state reconciliation

Phase 2 enhancements (2025-12-17):
- FeedbackCollector integration for closed-loop learning
- Circuit breaker integration to halt trading on error spikes
- Execution failure feedback to LearningEngine
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
import logging
from typing import TYPE_CHECKING, Any
import uuid

from ordinis.domain.enums import OrderStatus
from ordinis.domain.orders import ExecutionEvent, Fill, Order

from .orders import OrderIntent

if TYPE_CHECKING:
    from ordinis.adapters.alerting import AlertManager
    from ordinis.adapters.storage.repositories.order import OrderRepository
    from ordinis.core.protocols import BrokerAdapter
    from ordinis.engines.learning.collectors import FeedbackCollector
    from ordinis.safety.kill_switch import KillSwitch

logger = logging.getLogger(__name__)


@dataclass
class AccountState:
    """Cached broker account state for pre-trade validation."""

    equity: Decimal = Decimal("0")
    cash: Decimal = Decimal("0")
    buying_power: Decimal = Decimal("0")
    portfolio_value: Decimal = Decimal("0")
    last_sync: datetime | None = None

    def is_stale(self, max_age_seconds: float = 5.0) -> bool:
        """Check if account state is stale and needs refresh."""
        if self.last_sync is None:
            return True
        age = (datetime.utcnow() - self.last_sync).total_seconds()
        return age > max_age_seconds


@dataclass
class PositionState:
    """Cached position state from broker."""

    symbol: str
    quantity: Decimal
    side: str  # "long" or "short"
    avg_entry_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal


@dataclass
class BrokerSyncResult:
    """Result of broker state synchronization."""

    success: bool
    positions_synced: int = 0
    account_synced: bool = False
    discrepancies: list[str] = field(default_factory=list)
    error: str | None = None


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
        feedback_collector: FeedbackCollector | None = None,
        max_positions: int = 20,
        min_buying_power: Decimal = Decimal("1000"),
    ):
        """
        Initialize FlowRoute engine.

        Args:
            broker_adapter: Broker adapter for order execution
            kill_switch: Kill switch for emergency stop
            order_repository: Repository for order persistence
            alert_manager: Alert manager for notifications
            feedback_collector: Feedback collector for closed-loop learning
            max_positions: Maximum number of concurrent positions (default: 20)
            min_buying_power: Minimum buying power to allow trading (default: $1000)
        """
        self._broker = broker_adapter
        self._kill_switch = kill_switch
        self._order_repo = order_repository
        self._alert_manager = alert_manager
        self._feedback = feedback_collector
        self._orders: dict[str, Order] = {}
        self._active_orders: set[str] = set()

        # Phase 1: Broker state tracking
        self._account_state = AccountState()
        self._positions: dict[str, PositionState] = {}
        self._max_positions = max_positions
        self._min_buying_power = min_buying_power
        self._initialized = False

    async def initialize(self) -> BrokerSyncResult:
        """
        Initialize engine by syncing with broker state.

        Must be called before submitting orders. Fetches current
        positions and account info from broker.

        Returns:
            BrokerSyncResult with sync status and any discrepancies
        """
        if not self._broker:
            return BrokerSyncResult(
                success=False,
                error="No broker adapter configured",
            )

        result = await self.sync_broker_state()

        if result.success:
            self._initialized = True
            logger.info(
                f"FlowRoute initialized: {result.positions_synced} positions, "
                f"buying_power=${self._account_state.buying_power:,.2f}"
            )

        return result

    async def sync_broker_state(self) -> BrokerSyncResult:
        """
        Synchronize internal state with broker.

        Fetches current positions and account info. Detects and logs
        any discrepancies between internal tracking and broker state.

        Returns:
            BrokerSyncResult with sync details
        """
        if not self._broker:
            return BrokerSyncResult(success=False, error="No broker configured")

        discrepancies: list[str] = []

        try:
            # Sync account info
            account = await self._broker.get_account()
            old_buying_power = self._account_state.buying_power

            self._account_state = AccountState(
                equity=Decimal(str(account.equity)),
                cash=Decimal(str(account.cash)),
                buying_power=Decimal(str(account.buying_power)),
                portfolio_value=Decimal(str(account.portfolio_value)),
                last_sync=datetime.utcnow(),
            )

            if old_buying_power != Decimal("0"):
                bp_change = self._account_state.buying_power - old_buying_power
                if abs(bp_change) > Decimal("100"):
                    discrepancies.append(
                        f"Buying power changed: ${old_buying_power:,.2f} -> "
                        f"${self._account_state.buying_power:,.2f}"
                    )

            # Sync positions
            broker_positions = await self._broker.get_positions()
            old_symbols = set(self._positions.keys())
            new_symbols = {p.symbol for p in broker_positions}

            # Detect position discrepancies
            added = new_symbols - old_symbols
            removed = old_symbols - new_symbols

            if added:
                discrepancies.append(f"New positions detected: {', '.join(added)}")
            if removed:
                discrepancies.append(f"Positions closed: {', '.join(removed)}")

            # Phase 2: Record position mismatches to FeedbackCollector
            # This detects the 12/17 issue where internal tracking diverged from broker
            if self._feedback and (added or removed):
                for symbol in added:
                    broker_pos = next((p for p in broker_positions if p.symbol == symbol), None)
                    if broker_pos:
                        try:
                            await self._feedback.record_position_mismatch(
                                symbol=symbol,
                                internal_quantity=0,
                                broker_quantity=int(broker_pos.quantity),
                                internal_cost=0.0,
                                broker_cost=float(broker_pos.avg_entry_price),
                            )
                        except Exception as e:
                            logger.error(f"Failed to record position mismatch: {e}")

                for symbol in removed:
                    old_pos = self._positions.get(symbol)
                    if old_pos:
                        try:
                            await self._feedback.record_position_mismatch(
                                symbol=symbol,
                                internal_quantity=int(old_pos.quantity),
                                broker_quantity=0,
                                internal_cost=float(old_pos.avg_entry_price),
                                broker_cost=0.0,
                            )
                        except Exception as e:
                            logger.error(f"Failed to record position mismatch: {e}")

            # Update position cache
            self._positions.clear()
            for p in broker_positions:
                self._positions[p.symbol] = PositionState(
                    symbol=p.symbol,
                    quantity=Decimal(str(p.quantity)),
                    side=p.side.value if hasattr(p.side, "value") else str(p.side),
                    avg_entry_price=Decimal(str(p.avg_entry_price)),
                    market_value=Decimal(str(p.market_value)),
                    unrealized_pnl=Decimal(str(p.unrealized_pnl)),
                )

            # Log discrepancies
            for disc in discrepancies:
                logger.warning(f"Broker sync discrepancy: {disc}")

            return BrokerSyncResult(
                success=True,
                positions_synced=len(broker_positions),
                account_synced=True,
                discrepancies=discrepancies,
            )

        except Exception as e:
            logger.exception("Failed to sync broker state")
            return BrokerSyncResult(success=False, error=str(e))

    async def _refresh_account_if_stale(self) -> bool:
        """
        Refresh account state if stale.

        Returns:
            True if refresh successful or not needed, False on error
        """
        if not self._account_state.is_stale():
            return True

        if not self._broker:
            return False

        try:
            account = await self._broker.get_account()
            self._account_state = AccountState(
                equity=Decimal(str(account.equity)),
                cash=Decimal(str(account.cash)),
                buying_power=Decimal(str(account.buying_power)),
                portfolio_value=Decimal(str(account.portfolio_value)),
                last_sync=datetime.utcnow(),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to refresh account state: {e}")
            return False

    def _validate_pre_trade(self, order: Order, estimated_cost: Decimal) -> tuple[bool, str]:
        """
        Validate order against account state before submission.

        Checks:
        - Buying power sufficient for order
        - Position count within limits
        - Minimum buying power threshold

        Args:
            order: Order to validate
            estimated_cost: Estimated cost of order

        Returns:
            Tuple of (valid, rejection_reason)
        """
        # Check minimum buying power threshold
        if self._account_state.buying_power < self._min_buying_power:
            return False, (
                f"Buying power ${self._account_state.buying_power:,.2f} "
                f"below minimum ${self._min_buying_power:,.2f}"
            )

        # Check if sufficient buying power for this order
        if estimated_cost > self._account_state.buying_power:
            return False, (
                f"Insufficient buying power: need ${estimated_cost:,.2f}, "
                f"have ${self._account_state.buying_power:,.2f}"
            )

        # Check position count limit (for new positions only)
        if order.symbol not in self._positions:
            if len(self._positions) >= self._max_positions:
                return False, (
                    f"Position limit reached: {len(self._positions)}/{self._max_positions}"
                )

        return True, ""

    def get_position(self, symbol: str) -> PositionState | None:
        """Get cached position for symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> list[PositionState]:
        """Get all cached positions."""
        return list(self._positions.values())

    def get_account_state(self) -> AccountState:
        """Get cached account state."""
        return self._account_state

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

    async def execute(self, orders: list[Any]) -> list[Any]:
        """Execute orders (Protocol implementation)."""
        results = []
        for signal in orders:
            # Convert Signal to OrderIntent
            # This is a simplification. Usually RiskGuard returns approved signals
            # and we need to convert them to OrderIntent first.
            # Or Orchestrator does it.
            # For now, let's assume 'orders' are Signals and we convert them here.

            # If it's already an Order object (from some other path), use it
            if hasattr(signal, "order_id"):
                # It's an Order
                success, msg = await self.submit_order(signal)
                if success:
                    # In paper trading, we might get immediate fills
                    # But submit_order is async.
                    # We need to return fills.
                    # For now, return a mock fill if successful
                    results.append(
                        {
                            "symbol": signal.symbol,
                            "side": signal.side,
                            "quantity": signal.quantity,
                            "price": signal.limit_price or 100.0,  # Mock
                            "status": "filled",
                        }
                    )
                continue

            # If it's a Signal, create intent
            from .orders import OrderIntent, OrderSide, OrderType

            # Map Signal direction to OrderSide
            side = OrderSide.BUY  # Default
            if hasattr(signal, "direction"):
                # Assuming signal.direction is an Enum or string
                d = str(signal.direction).lower()
                if "short" in d or "sell" in d:
                    side = OrderSide.SELL

            intent = OrderIntent(
                intent_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                side=side,
                quantity=10,  # Default quantity for demo
                order_type=OrderType.MARKET,
                signal_id=getattr(signal, "signal_id", None),
            )

            order = self.create_order_from_intent(intent)
            success, _msg = await self.submit_order(order)

            if success:
                # Mock fill for demo
                results.append(
                    {
                        "symbol": order.symbol,
                        "side": order.side,
                        "quantity": order.quantity,
                        "price": 148.0,  # Mock execution price
                        "status": "filled",
                    }
                )

        return results

    async def submit_order(self, order: Order) -> tuple[bool, str]:
        """
        Submit order to broker.

        Validates buying power and position limits before submission.
        Checks kill switch, circuit breaker, and persists order state.

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

        # Phase 2: Circuit breaker check
        # This prevents submitting orders when error rate is too high
        if self._feedback:
            allowed, reason = self._feedback.should_allow_execution()
            if not allowed:
                logger.warning(f"Order {order.order_id} blocked by circuit breaker: {reason}")
                order.status = OrderStatus.REJECTED
                order.error_message = f"Circuit breaker active: {reason}"
                await self._persist_order(order)
                return False, order.error_message

        if not self._broker:
            return False, "No broker adapter configured"

        if order.status != OrderStatus.CREATED:
            return False, f"Order must be in CREATED state, got {order.status.value}"

        # PHASE 1: Pre-trade validation
        # Refresh account state if stale
        if not await self._refresh_account_if_stale():
            logger.warning(f"Order {order.order_id} blocked: failed to refresh account state")
            order.status = OrderStatus.REJECTED
            order.error_message = "Failed to verify buying power - account state unavailable"
            await self._persist_order(order)
            return False, order.error_message

        # Estimate order cost (quantity * limit_price or use a reasonable estimate)
        price_estimate = order.limit_price or Decimal("0")
        if price_estimate == Decimal("0"):
            # For market orders without price, we need to estimate
            # Use last known price from position or skip validation
            existing_pos = self._positions.get(order.symbol)
            if existing_pos:
                price_estimate = existing_pos.avg_entry_price
            else:
                # Query current price would add latency; for now use conservative estimate
                # In production, this should fetch current market price
                price_estimate = Decimal("500")  # Conservative default

        estimated_cost = Decimal(str(order.quantity)) * price_estimate

        # Validate pre-trade conditions
        valid, rejection_reason = self._validate_pre_trade(order, estimated_cost)
        if not valid:
            logger.warning(
                f"Order {order.order_id} rejected pre-trade: {rejection_reason} "
                f"[symbol={order.symbol}, qty={order.quantity}, "
                f"estimated_cost=${estimated_cost:,.2f}]"
            )
            order.status = OrderStatus.REJECTED
            order.error_message = rejection_reason
            await self._persist_order(order)
            await self._send_alert(
                "Order Rejected (Pre-Trade)",
                f"Order {order.symbol} rejected: {rejection_reason}",
                severity="warning",
                metadata={
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "quantity": str(order.quantity),
                    "estimated_cost": str(estimated_cost),
                    "buying_power": str(self._account_state.buying_power),
                },
            )

            # Phase 2: Record execution failure to FeedbackCollector
            # This is the critical feedback loop that was missing on 12/17
            if self._feedback:
                # Determine error type for circuit breaker
                error_type = "pre_trade_rejection"
                if "buying power" in rejection_reason.lower():
                    error_type = "insufficient_buying_power"
                elif "position limit" in rejection_reason.lower():
                    error_type = "position_limit"

                try:
                    _, circuit_tripped = await self._feedback.record_execution_failure(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        error_type=error_type,
                        error_message=rejection_reason,
                        order_details={
                            "quantity": str(order.quantity),
                            "estimated_cost": str(estimated_cost),
                            "buying_power": str(self._account_state.buying_power),
                            "side": str(order.side),
                        },
                        strategy=order.strategy_id,
                    )
                    if circuit_tripped:
                        logger.critical("Circuit breaker tripped - trading will be halted")
                except Exception as e:
                    logger.error(f"Failed to record execution failure: {e}")

            return False, rejection_reason

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

            # Phase 2: Record broker rejection to FeedbackCollector
            if self._feedback:
                error_msg = order.error_message or "Unknown error"
                error_type = "broker_rejection"
                if "buying power" in error_msg.lower() or "insufficient" in error_msg.lower():
                    error_type = "insufficient_buying_power"
                elif "margin" in error_msg.lower():
                    error_type = "margin_call"

                try:
                    await self._feedback.record_order_rejected(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        rejection_reason=error_msg,
                        broker_response=broker_response,
                    )
                except Exception as feedback_err:
                    logger.error(f"Failed to record order rejection: {feedback_err}")

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

            # Phase 2: Record execution error to FeedbackCollector
            if self._feedback:
                try:
                    await self._feedback.record_execution_failure(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        error_type="broker_error",
                        error_message=str(e),
                        order_details={
                            "quantity": str(order.quantity),
                            "side": str(order.side),
                        },
                        strategy=order.strategy_id,
                    )
                except Exception as feedback_err:
                    logger.error(f"Failed to record execution error: {feedback_err}")
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
            # Check if order exists
            existing = await self._order_repo.get_by_id(order.order_id)

            if existing:
                # Update status
                await self._order_repo.update_status(
                    order_id=order.order_id,
                    status=order.status.value
                    if hasattr(order.status, "value")
                    else str(order.status),
                    error_message=order.error_message,
                )
            else:
                # Create new order
                import json

                from ordinis.adapters.storage.models import OrderRow

                row = OrderRow(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side.value.lower()
                    if hasattr(order.side, "value")
                    else str(order.side).lower(),
                    quantity=order.quantity,
                    order_type=order.order_type.value.lower()
                    if hasattr(order.order_type, "value")
                    else str(order.order_type).lower(),
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    time_in_force=order.time_in_force.value.lower()
                    if hasattr(order.time_in_force, "value")
                    else str(order.time_in_force).lower(),
                    status=order.status.value.lower()
                    if hasattr(order.status, "value")
                    else str(order.status).lower(),
                    filled_quantity=order.filled_quantity,
                    remaining_quantity=order.remaining_quantity,
                    avg_fill_price=order.avg_fill_price,
                    created_at=order.created_at.isoformat() if order.created_at else None,
                    submitted_at=order.submitted_at.isoformat() if order.submitted_at else None,
                    filled_at=order.filled_at.isoformat() if order.filled_at else None,
                    intent_id=order.intent_id,
                    signal_id=order.signal_id,
                    strategy_id=order.strategy_id,
                    broker_order_id=order.broker_order_id,
                    broker_response=json.dumps(order.broker_response)
                    if order.broker_response
                    else None,
                    error_message=order.error_message,
                    retry_count=order.retry_count,
                    metadata=json.dumps(order.metadata) if order.metadata else None,
                )
                await self._order_repo.create(row)

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
            from ordinis.adapters.alerting import AlertSeverity
            from ordinis.adapters.alerting.manager import AlertType

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
            "initialized": self._initialized,
            "account_state": {
                "equity": str(self._account_state.equity),
                "cash": str(self._account_state.cash),
                "buying_power": str(self._account_state.buying_power),
                "portfolio_value": str(self._account_state.portfolio_value),
                "last_sync": self._account_state.last_sync.isoformat()
                if self._account_state.last_sync
                else None,
            },
            "positions": {
                symbol: {
                    "quantity": str(pos.quantity),
                    "side": pos.side,
                    "avg_entry_price": str(pos.avg_entry_price),
                    "market_value": str(pos.market_value),
                    "unrealized_pnl": str(pos.unrealized_pnl),
                }
                for symbol, pos in self._positions.items()
            },
            "position_count": len(self._positions),
            "max_positions": self._max_positions,
            "execution_stats": self.get_execution_stats(),
        }
