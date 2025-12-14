"""
StreamingBus - Event Fabric for Ordinis.

Provides event distribution with standard schema:
- Publish/subscribe pattern for loose coupling
- Standard event schema (timestamp, symbol, type, payload)
- Multiple adapters (in-memory, Redis Streams)
- Event tracing and replay

Example:
    from ordinis.bus import StreamingBus, BusEvent

    bus = StreamingBus()

    # Subscribe to events
    async def handle_signal(event: BusEvent):
        print(f"Received signal: {event.payload}")

    bus.subscribe("signal", handle_signal)

    # Publish event
    await bus.publish(BusEvent(
        event_type="signal",
        symbol="AAPL",
        payload={"direction": "buy", "confidence": 0.85},
        source="signalcore",
    ))
"""

from ordinis.bus.config import BusConfig
from ordinis.bus.engine import StreamingBus
from ordinis.bus.models import (
    BusEvent,
    EventPriority,
    EventType,
    Subscription,
)

__all__ = [
    "BusConfig",
    "BusEvent",
    "EventPriority",
    "EventType",
    "StreamingBus",
    "Subscription",
]
