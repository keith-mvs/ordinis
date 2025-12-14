"""
Event types and hooks for portfolio rebalancing.

Integrates with ProofBench event system for backtesting and live trading.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ordinis.engines.portfolio.core.models import StrategyType


class RebalanceEventType(Enum):
    """Types of rebalancing events."""

    # Pre-rebalance events
    CHECK_TRIGGERED = "check_triggered"  # Threshold check initiated
    THRESHOLD_BREACHED = "threshold_breached"  # Drift threshold exceeded
    TIME_CONSTRAINT_MET = "time_constraint_met"  # Time threshold met

    # Rebalancing events
    REBALANCE_STARTED = "rebalance_started"  # Rebalancing process started
    DECISIONS_GENERATED = "decisions_generated"  # Decisions created
    EXECUTION_STARTED = "execution_started"  # Order execution started
    ORDER_EXECUTED = "order_executed"  # Single order executed
    ORDER_FAILED = "order_failed"  # Single order failed
    EXECUTION_COMPLETED = "execution_completed"  # All orders processed
    REBALANCE_COMPLETED = "rebalance_completed"  # Rebalancing finished

    # Error events
    VALIDATION_FAILED = "validation_failed"  # Pre-validation failed
    EXECUTION_FAILED = "execution_failed"  # Execution process failed
    REBALANCE_CANCELLED = "rebalance_cancelled"  # Rebalancing cancelled


@dataclass
class RebalanceEvent:
    """Event representing a rebalancing action or state change.

    Attributes:
        timestamp: Event occurrence time
        event_type: Type of rebalancing event
        strategy_type: Rebalancing strategy used
        data: Event-specific data payload
        metadata: Additional context information
    """

    timestamp: datetime
    event_type: RebalanceEventType
    strategy_type: StrategyType | None = None
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "strategy_type": self.strategy_type.value if self.strategy_type else None,
            "data": self.data,
            "metadata": self.metadata,
        }


# Type alias for event handlers
EventHandler = Callable[[RebalanceEvent], None]


class EventHooks:
    """Event hook manager for portfolio rebalancing.

    Allows subscribers to register callbacks for specific event types.
    Supports multiple handlers per event type.

    Example:
        >>> hooks = EventHooks()
        >>> def on_rebalance(event):
        ...     print(f"Rebalancing started at {event.timestamp}")
        >>> hooks.register(RebalanceEventType.REBALANCE_STARTED, on_rebalance)
        >>> hooks.emit(RebalanceEvent(
        ...     timestamp=datetime.now(tz=UTC),
        ...     event_type=RebalanceEventType.REBALANCE_STARTED,
        ...     strategy_type=StrategyType.TARGET_ALLOCATION,
        ... ))
    """

    def __init__(self) -> None:
        """Initialize event hooks manager."""
        self._handlers: dict[RebalanceEventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []

    def register(
        self,
        event_type: RebalanceEventType,
        handler: EventHandler,
    ) -> None:
        """Register event handler for specific event type.

        Args:
            event_type: Type of event to handle
            handler: Callback function to invoke

        Raises:
            ValueError: If handler is not callable
        """
        if not callable(handler):
            raise ValueError(f"Handler must be callable, got {type(handler)}")

        if event_type not in self._handlers:
            self._handlers[event_type] = []

        self._handlers[event_type].append(handler)

    def register_global(self, handler: EventHandler) -> None:
        """Register handler for all event types.

        Args:
            handler: Callback function to invoke for all events

        Raises:
            ValueError: If handler is not callable
        """
        if not callable(handler):
            raise ValueError(f"Handler must be callable, got {type(handler)}")

        self._global_handlers.append(handler)

    def unregister(
        self,
        event_type: RebalanceEventType,
        handler: EventHandler,
    ) -> None:
        """Unregister event handler.

        Args:
            event_type: Event type to unregister from
            handler: Handler to remove
        """
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def unregister_global(self, handler: EventHandler) -> None:
        """Unregister global handler.

        Args:
            handler: Handler to remove
        """
        if handler in self._global_handlers:
            self._global_handlers.remove(handler)

    def emit(self, event: RebalanceEvent) -> None:
        """Emit event to all registered handlers.

        Args:
            event: Event to emit
        """
        # Call global handlers
        for handler in self._global_handlers:
            try:
                handler(event)
            except Exception as e:
                # Log error but don't stop other handlers
                print(f"Global handler error: {e!s}")

        # Call specific handlers
        if event.event_type in self._handlers:
            for handler in self._handlers[event.event_type]:
                try:
                    handler(event)
                except Exception as e:
                    # Log error but don't stop other handlers
                    print(f"Handler error for {event.event_type.value}: {e!s}")

    def clear(self, event_type: RebalanceEventType | None = None) -> None:
        """Clear registered handlers.

        Args:
            event_type: Specific event type to clear, or None to clear all
        """
        if event_type is None:
            self._handlers.clear()
            self._global_handlers.clear()
        elif event_type in self._handlers:
            self._handlers[event_type].clear()

    def has_handlers(self, event_type: RebalanceEventType) -> bool:
        """Check if event type has registered handlers.

        Args:
            event_type: Event type to check

        Returns:
            True if handlers are registered
        """
        return (event_type in self._handlers and len(self._handlers[event_type]) > 0) or len(
            self._global_handlers
        ) > 0

    def get_handler_count(self, event_type: RebalanceEventType | None = None) -> int:
        """Get number of registered handlers.

        Args:
            event_type: Specific event type, or None for total count

        Returns:
            Number of registered handlers
        """
        if event_type is None:
            total = len(self._global_handlers)
            for handlers in self._handlers.values():
                total += len(handlers)
            return total

        count = len(self._global_handlers)
        if event_type in self._handlers:
            count += len(self._handlers[event_type])
        return count
