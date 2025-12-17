"""
Order domain models.
"""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from ordinis.domain.enums import OrderSide, OrderStatus, OrderType, TimeInForce


class Fill(BaseModel):
    """Execution fill record."""

    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    latency_ms: float = 0.0

    # Quality metrics
    slippage: float = 0.0  # Slippage as fraction (e.g. 0.001 for 0.1%)
    slippage_bps: float = 0.0  # vs expected price
    vs_arrival_bps: float = 0.0  # vs price at order creation

    # Instrument details
    multiplier: float = 1.0

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def total_cost(self) -> float:
        """Get total cost including commission."""
        base_cost = self.price * self.quantity * self.multiplier
        if self.side == OrderSide.BUY:
            return base_cost + self.commission
        return base_cost - self.commission

    @property
    def net_proceeds(self) -> float:
        """Get net proceeds (for sells) or net cost (for buys)."""
        if self.side == OrderSide.SELL:
            return self.price * self.quantity * self.multiplier - self.commission
        return -(self.price * self.quantity * self.multiplier + self.commission)


class ExecutionEvent(BaseModel):
    """Audit event for order lifecycle."""

    event_id: str
    order_id: str
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Event details
    status_before: OrderStatus | None = None
    status_after: OrderStatus
    details: dict[str, Any] = Field(default_factory=dict)

    # Error handling
    error_code: str | None = None
    error_message: str | None = None
    retry_count: int = 0


class Order(BaseModel):
    """
    Executable order with full lifecycle tracking.
    """

    # Identity
    order_id: str = Field(default_factory=lambda: str(uuid4()))
    client_order_id: str | None = None
    broker_order_id: str | None = None

    # Order Details
    symbol: str
    side: OrderSide
    quantity: int = Field(gt=0, description="Order quantity must be positive")
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY

    # Price Constraints
    limit_price: float | None = Field(default=None, gt=0)
    stop_price: float | None = Field(default=None, gt=0)

    # Status Tracking
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: int = Field(default=0, ge=0)
    avg_fill_price: float = Field(default=0.0, ge=0.0)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def timestamp(self) -> datetime:
        """Alias for created_at for backward compatibility."""
        return self.created_at

    # Source tracking
    intent_id: str | None = None
    signal_id: str | None = None
    strategy_id: str | None = None

    # Execution tracking
    fills: list[Fill] = Field(default_factory=list)
    events: list[ExecutionEvent] = Field(default_factory=list)
    broker_response: dict[str, Any] = Field(default_factory=dict)

    # Error handling
    error_message: str | None = None
    retry_count: int = 0

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def remaining_quantity(self) -> int:
        """Calculate remaining quantity."""
        return self.quantity - self.filled_quantity

    @model_validator(mode="after")
    def validate_prices(self) -> "Order":
        """Validate that required prices are present for specific order types."""
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            raise ValueError("Limit orders require a limit_price")

        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("Stop orders require a stop_price")

        if self.order_type == OrderType.STOP_LIMIT:
            if self.limit_price is None:
                raise ValueError("Stop-Limit orders require a limit_price")
            if self.stop_price is None:
                raise ValueError("Stop-Limit orders require a stop_price")

        return self

    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        terminal_states = {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.ERROR,
        }
        return self.status in terminal_states

    def is_active(self) -> bool:
        """Check if order is active (can still be filled)."""
        active_states = {
            OrderStatus.SUBMITTED,
            OrderStatus.ACKNOWLEDGED,
            OrderStatus.PARTIALLY_FILLED,
        }
        return self.status in active_states

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.filled_quantity >= self.quantity

    @property
    def is_partial(self) -> bool:
        """Check if order is partially filled."""
        return 0 < self.filled_quantity < self.quantity

    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity

    def add_fill(self, fill: Fill) -> None:
        """
        Add a fill to the order.
        Updates filled quantity and average fill price.
        """
        self.fills.append(fill)
        self.filled_quantity += fill.quantity

        # Update average fill price
        if self.filled_quantity > 0:
            total_value = sum(f.quantity * f.price for f in self.fills)
            self.avg_fill_price = total_value / self.filled_quantity

        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.now(UTC)
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
