"""
Order types and lifecycle management for FlowRoute.

All order states and transitions are tracked for auditability.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OrderStatus(Enum):
    """Order lifecycle states."""

    CREATED = "created"
    VALIDATED = "validated"
    PENDING_SUBMIT = "pending_submit"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    ERROR = "error"


class OrderType(Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(Enum):
    """Time in force options."""

    DAY = "day"  # Good for day
    GTC = "gtc"  # Good till canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


@dataclass
class OrderIntent:
    """
    Trading intent from RiskGuard.

    This is what RiskGuard approves - FlowRoute translates to Order.
    """

    intent_id: str
    symbol: str
    side: str  # "buy" or "sell"
    quantity: int
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY

    # Source tracking
    signal_id: str | None = None
    strategy_id: str | None = None

    # Risk parameters (from RiskGuard)
    max_slippage_pct: float = 0.01
    max_fill_time_seconds: int = 60

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Order:
    """
    Executable order with full lifecycle tracking.

    Created from OrderIntent after RiskGuard approval.
    """

    order_id: str
    symbol: str
    side: str
    quantity: int
    order_type: OrderType
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY

    # Status tracking
    status: OrderStatus = OrderStatus.CREATED
    filled_quantity: int = 0
    remaining_quantity: int = 0
    avg_fill_price: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: datetime | None = None
    filled_at: datetime | None = None

    # Source tracking
    intent_id: str | None = None
    signal_id: str | None = None
    strategy_id: str | None = None

    # Broker integration
    broker_order_id: str | None = None
    broker_response: dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    fills: list["Fill"] = field(default_factory=list)
    events: list["ExecutionEvent"] = field(default_factory=list)

    # Error handling
    error_message: str | None = None
    retry_count: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize remaining quantity."""
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity

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

    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if self.quantity == 0:
            return 0.0
        return self.filled_quantity / self.quantity

    def add_fill(self, fill: "Fill") -> None:
        """
        Add a fill to the order.

        Updates filled quantity and average fill price.
        """
        self.fills.append(fill)
        self.filled_quantity += fill.quantity
        self.remaining_quantity = self.quantity - self.filled_quantity

        # Update average fill price
        if self.filled_quantity > 0:
            total_value = sum(f.quantity * f.price for f in self.fills)
            self.avg_fill_price = total_value / self.filled_quantity

        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.utcnow()
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED

    def to_dict(self) -> dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "avg_fill_price": self.avg_fill_price,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "broker_order_id": self.broker_order_id,
            "error_message": self.error_message,
            "fills": [f.to_dict() for f in self.fills],
            "events": [e.to_dict() for e in self.events],
        }


@dataclass
class Fill:
    """Execution fill record."""

    fill_id: str
    order_id: str
    symbol: str
    side: str
    quantity: int
    price: float
    commission: float
    timestamp: datetime
    latency_ms: float = 0.0

    # Quality metrics
    slippage_bps: float = 0.0  # vs expected price
    vs_arrival_bps: float = 0.0  # vs price at order creation

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert fill to dictionary."""
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "commission": self.commission,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "slippage_bps": self.slippage_bps,
            "vs_arrival_bps": self.vs_arrival_bps,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionEvent:
    """Audit event for order lifecycle."""

    event_id: str
    order_id: str
    event_type: str
    timestamp: datetime

    # Event details
    status_before: OrderStatus | None
    status_after: OrderStatus
    details: dict[str, Any] = field(default_factory=dict)

    # Error handling
    error_code: str | None = None
    error_message: str | None = None
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "order_id": self.order_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "status_before": self.status_before.value if self.status_before else None,
            "status_after": self.status_after.value,
            "details": self.details,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
        }
