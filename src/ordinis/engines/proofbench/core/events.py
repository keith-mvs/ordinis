"""Event-driven simulation core.

Provides event types, event data structures, and priority queue for backtesting.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import heapq
from typing import Any


class EventType(Enum):
    """Types of events in the simulation."""

    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    BAR_UPDATE = "bar_update"
    SIGNAL = "signal"
    ORDER_SUBMIT = "order_submit"
    ORDER_FILL = "order_fill"
    ORDER_CANCEL = "order_cancel"
    POSITION_UPDATE = "position_update"
    RISK_CHECK = "risk_check"
    EOD_SETTLEMENT = "eod_settlement"


@dataclass
class Event:
    """Simulation event with timestamp, type, and priority.

    Events are ordered first by timestamp, then by priority (lower = higher priority).
    This ensures deterministic event ordering in the simulation.

    Attributes:
        timestamp: When the event occurs
        event_type: Type of event
        data: Event-specific data payload
        priority: Event priority for same-timestamp events (default: 0)
    """

    timestamp: datetime
    event_type: EventType
    data: dict[str, Any]
    priority: int = 0

    def __lt__(self, other: "Event") -> bool:
        """Compare events for priority queue ordering.

        Events are ordered by timestamp first, then priority.
        Lower priority values have higher priority.
        """
        if self.timestamp == other.timestamp:
            return self.priority < other.priority
        return self.timestamp < other.timestamp

    def __le__(self, other: "Event") -> bool:
        """Less than or equal comparison."""
        return self < other or self == other

    def __gt__(self, other: "Event") -> bool:
        """Greater than comparison."""
        return not self <= other

    def __ge__(self, other: "Event") -> bool:
        """Greater than or equal comparison."""
        return not self < other


class EventQueue:
    """Priority queue for simulation events.

    Events are automatically ordered by timestamp and priority.
    Uses Python's heapq for efficient O(log n) push/pop operations.
    """

    def __init__(self) -> None:
        """Initialize empty event queue."""
        self._queue: list[Event] = []

    def push(self, event: Event) -> None:
        """Add event to the queue.

        Args:
            event: Event to add
        """
        heapq.heappush(self._queue, event)

    def pop(self) -> Event | None:
        """Remove and return the next event.

        Returns:
            Next event in chronological order, or None if queue is empty
        """
        if self._queue:
            return heapq.heappop(self._queue)
        return None

    def peek(self) -> Event | None:
        """View the next event without removing it.

        Returns:
            Next event in chronological order, or None if queue is empty
        """
        if self._queue:
            return self._queue[0]
        return None

    def is_empty(self) -> bool:
        """Check if queue is empty.

        Returns:
            True if queue has no events
        """
        return len(self._queue) == 0

    def size(self) -> int:
        """Get number of events in queue.

        Returns:
            Number of events
        """
        return len(self._queue)

    def clear(self) -> None:
        """Remove all events from the queue."""
        self._queue.clear()
