"""
StreamingBus data models.

Defines event schema and subscription structures.
"""

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
import uuid


class EventType(Enum):
    """Standard event types."""

    # Market data
    TICK = "tick"
    BAR = "bar"
    QUOTE = "quote"

    # Trading signals
    SIGNAL = "signal"
    SIGNAL_CONFIRMED = "signal_confirmed"
    SIGNAL_REJECTED = "signal_rejected"

    # Orders and fills
    ORDER = "order"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    FILL = "fill"

    # Portfolio
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    REBALANCE = "rebalance"

    # Risk
    RISK_CHECK = "risk_check"
    RISK_ALERT = "risk_alert"
    HALT = "halt"

    # System
    HEARTBEAT = "heartbeat"
    AUDIT = "audit"
    ERROR = "error"
    METRIC = "metric"


class EventPriority(Enum):
    """Event priority levels."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BusEvent:
    """
    Standard event schema for StreamingBus.

    All events flowing through the bus follow this schema.
    """

    event_type: str | EventType
    payload: dict[str, Any]
    source: str  # Originating engine/service

    # Optional fields
    symbol: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str | None = None  # For distributed tracing
    priority: EventPriority = EventPriority.NORMAL
    metadata: dict[str, Any] = field(default_factory=dict)

    # Internal
    _published_at: datetime | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Normalize event_type to string."""
        if isinstance(self.event_type, EventType):
            self.event_type = self.event_type.value

    @property
    def type_enum(self) -> EventType | None:
        """Get event type as enum if possible."""
        try:
            return EventType(self.event_type)
        except ValueError:
            return None

    @property
    def age_ms(self) -> float | None:
        """Milliseconds since event was created."""
        if self._published_at:
            delta = datetime.now(UTC) - self._published_at
            return delta.total_seconds() * 1000
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "source": self.source,
            "payload": self.payload,
            "priority": self.priority.value,
            "trace_id": self.trace_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BusEvent":
        """Create from dictionary."""
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=data["event_type"],
            timestamp=datetime.fromisoformat(data["timestamp"])
            if isinstance(data.get("timestamp"), str)
            else data.get("timestamp", datetime.now(UTC)),
            symbol=data.get("symbol"),
            source=data["source"],
            payload=data["payload"],
            priority=EventPriority(data.get("priority", 1)),
            trace_id=data.get("trace_id"),
            metadata=data.get("metadata", {}),
        )


# Type alias for event handlers
EventHandler = Callable[[BusEvent], Coroutine[Any, Any, None]]


@dataclass
class Subscription:
    """Event subscription."""

    subscription_id: str
    topic: str  # Event type pattern (exact or wildcard)
    handler: EventHandler
    source_filter: str | None = None  # Filter by source
    symbol_filter: str | None = None  # Filter by symbol
    priority_min: EventPriority = EventPriority.LOW
    active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def matches(self, event: BusEvent) -> bool:
        """Check if event matches subscription filters."""
        if not self.active:
            return False

        # Get event_type as string (it's normalized in __post_init__)
        event_type = (
            event.event_type.value if isinstance(event.event_type, EventType) else event.event_type
        )

        # Topic matching (supports wildcards)
        if not self._topic_matches(event_type):
            return False

        # Combined filters check
        source_ok = not self.source_filter or event.source == self.source_filter
        symbol_ok = not self.symbol_filter or event.symbol == self.symbol_filter
        priority_ok = event.priority.value >= self.priority_min.value

        return source_ok and symbol_ok and priority_ok

    def _topic_matches(self, event_type: str) -> bool:
        """Check if event type matches topic pattern."""
        if self.topic == "*":
            return True
        if self.topic.endswith("*"):
            return event_type.startswith(self.topic[:-1])
        return event_type == self.topic
