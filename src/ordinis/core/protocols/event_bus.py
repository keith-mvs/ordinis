"""
EventBus protocol for pub/sub messaging.

Enables event-driven architecture with decoupled components.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from datetime import datetime


class Event(Protocol):
    """
    Base event protocol.

    All events must have a type identifier and timestamp.
    """

    @property
    def event_type(self) -> str:
        """Event type identifier."""
        ...

    @property
    def timestamp(self) -> datetime:
        """Event timestamp."""
        ...


T = TypeVar("T", bound=Event)


class EventBus(Protocol):
    """
    Pub/sub event bus for decoupling components.

    Delivers events to all subscribed handlers.
    Typically non-blocking (fire-and-forget publish).

    Idempotency:
    - Adding same handler twice has no effect
    - Removing non-existent handler has no effect
    """

    def publish(self, event: Event) -> None:
        """
        Publish event to all subscribers.

        Non-blocking: returns immediately after queuing.
        Handlers may run in background or synchronously.

        Args:
            event: Event to publish
        """
        ...

    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """
        Subscribe handler to event type.

        Handler called for each published event of type (or subtype).
        Idempotent: registering same handler twice has no effect.

        Args:
            event_type: Event type to subscribe to
            handler: Callback for events
        """
        ...

    def unsubscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        """
        Unsubscribe handler from event type.

        Idempotent: no error if handler was not subscribed.

        Args:
            event_type: Event type to unsubscribe from
            handler: Callback to remove
        """
        ...
