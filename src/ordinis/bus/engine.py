"""
StreamingBus - Event Fabric Engine.

Provides pub/sub event distribution for the trading system.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import logging
import uuid

from ordinis.bus.config import BusConfig
from ordinis.bus.models import (
    BusEvent,
    EventHandler,
    EventPriority,
    EventType,
    Subscription,
)

_logger = logging.getLogger(__name__)


@dataclass
class BusMetrics:
    """Event bus metrics."""

    events_published: int = 0
    events_delivered: int = 0
    events_dropped: int = 0
    handler_errors: int = 0
    avg_latency_ms: float = 0.0
    subscriptions_active: int = 0

    # Per-type counts
    events_by_type: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class StreamingBus:
    """
    StreamingBus event fabric.

    Provides:
    - `publish(event)` to emit events
    - `subscribe(topic, handler)` to receive events
    - Event filtering by type, source, symbol
    - Handler timeout and retry
    - Event history and replay
    """

    def __init__(self, config: BusConfig | None = None):
        """
        Initialize StreamingBus.

        Args:
            config: Configuration options. Uses defaults if None.
        """
        self.config = config or BusConfig()

        # Validate configuration
        errors = self.config.validate()
        if errors:
            msg = f"Invalid BusConfig: {'; '.join(errors)}"
            raise ValueError(msg)

        # State
        self._subscriptions: dict[str, Subscription] = {}
        self._subscriptions_by_topic: dict[str, list[str]] = defaultdict(list)
        self._history: list[BusEvent] = []
        self._metrics = BusMetrics()
        self._running = False
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_handlers)
        self._redis_adapter = None

        if self.config.adapter.value == "redis":
            from ordinis.bus.adapters.redis_adapter import RedisAdapter

            self._redis_adapter = RedisAdapter(self.config)

        _logger.info("StreamingBus initialized with adapter=%s", self.config.adapter.value)

    async def publish(self, event: BusEvent) -> None:
        """
        Publish event to the bus.

        Args:
            event: Event to publish
        """
        # Mark publish time
        event._published_at = datetime.now(UTC)

        # Optional schema validation
        if self.config.schema_validator:
            try:
                self.config.schema_validator(event)
            except Exception as exc:
                _logger.warning("Schema validation failed: %s", exc)
                self._metrics.events_dropped += 1
                return

        # Optional governance hook before publish
        if self.config.publish_governance_hook:
            try:
                allowed = self.config.publish_governance_hook(event)
            except Exception as exc:
                _logger.warning("Publish governance hook raised, dropping event: %s", exc)
                self._metrics.events_dropped += 1
                return
            if allowed is False:
                _logger.info("Publish governance denied event of type %s", event.event_type)
                self._metrics.events_dropped += 1
                return
        # Validate payload size

        payload_size = len(json.dumps(event.payload))
        if payload_size > self.config.max_payload_size:
            _logger.warning("Event payload too large (%d bytes), dropping", payload_size)
            self._metrics.events_dropped += 1
            return

        # Store in history if enabled
        if self.config.enable_history:
            self._history.append(event)
            # Trim history
            if len(self._history) > self.config.history_max_events:
                self._history = self._history[-self.config.history_max_events :]

        # Persist to Redis Streams if configured
        if self._redis_adapter:
            try:
                await self._redis_adapter.write_event(event)
            except Exception:
                _logger.warning("Redis adapter write failed; dropping event", exc_info=True)
                self._metrics.events_dropped += 1
                return

        # Update metrics
        self._metrics.events_published += 1
        self._metrics.events_by_type[event.event_type] += 1

        # Find matching subscriptions
        matching_subs = self._find_matching_subscriptions(event)

        if not matching_subs:
            _logger.debug("No subscribers for event type: %s", event.event_type)
            return

        # Dispatch to handlers
        await self._dispatch_event(event, matching_subs)

    def _find_matching_subscriptions(self, event: BusEvent) -> list[Subscription]:
        """Find subscriptions matching this event."""
        matching: list[Subscription] = []

        # Check all subscriptions
        for sub in self._subscriptions.values():
            if sub.matches(event):
                matching.append(sub)

        return matching

    async def _dispatch_event(self, event: BusEvent, subscriptions: list[Subscription]) -> None:
        """Dispatch event to matching subscriptions."""
        tasks = []

        for sub in subscriptions:
            task = asyncio.create_task(
                self._call_handler(sub, event),
                name=f"handler-{sub.subscription_id[:8]}",
            )
            tasks.append(task)

        if tasks:
            # Wait for all handlers with timeout
            _done, pending = await asyncio.wait(
                tasks,
                timeout=self.config.handler_timeout_seconds,
                return_when=asyncio.ALL_COMPLETED,
            )

            # Cancel timed out handlers
            for task in pending:
                task.cancel()
                self._metrics.handler_errors += 1
                _logger.warning("Handler timed out: %s", task.get_name())

    async def _call_handler(self, sub: Subscription, event: BusEvent) -> None:
        """Call a subscription handler with error handling."""
        async with self._semaphore:
            try:
                await sub.handler(event)
                self._metrics.events_delivered += 1
            except Exception as e:
                self._metrics.handler_errors += 1
                _logger.error(
                    "Handler error for %s: %s",
                    event.event_type,
                    e,
                    exc_info=True,
                )

                # Retry if configured
                if self.config.retry_failed_handlers:
                    # Simple retry without exponential backoff for now
                    for attempt in range(self.config.max_handler_retries):
                        try:
                            await asyncio.sleep(0.1 * (attempt + 1))
                            await sub.handler(event)
                            self._metrics.events_delivered += 1
                            _logger.info("Handler succeeded on retry %d", attempt + 1)
                            return
                        except Exception:
                            _logger.debug("Handler retry %d failed", attempt + 1)
                            continue

                    _logger.error(
                        "Handler failed after %d retries",
                        self.config.max_handler_retries,
                    )

    def subscribe(
        self,
        topic: str | EventType,
        handler: EventHandler,
        source_filter: str | None = None,
        symbol_filter: str | None = None,
        priority_min: EventPriority = EventPriority.LOW,
    ) -> str:
        """
        Subscribe to events.

        Args:
            topic: Event type to subscribe to (or "*" for all)
            handler: Async function to handle events
            source_filter: Only receive events from this source
            symbol_filter: Only receive events for this symbol
            priority_min: Minimum event priority

        Returns:
            Subscription ID
        """
        # Normalize topic
        if isinstance(topic, EventType):
            topic = topic.value

        sub_id = str(uuid.uuid4())

        subscription = Subscription(
            subscription_id=sub_id,
            topic=topic,
            handler=handler,
            source_filter=source_filter,
            symbol_filter=symbol_filter,
            priority_min=priority_min,
        )

        if self.config.subscribe_governance_hook:
            try:
                allowed = self.config.subscribe_governance_hook(topic, subscription)
            except Exception as exc:
                _logger.warning("Subscribe governance hook raised, denying: %s", exc)
                raise PermissionError("Subscription denied by governance hook") from exc
            if allowed is False:
                raise PermissionError("Subscription denied by governance hook")

        self._subscriptions[sub_id] = subscription
        self._subscriptions_by_topic[topic].append(sub_id)
        self._metrics.subscriptions_active += 1

        _logger.info(
            "Subscription created: %s -> %s (filters: source=%s, symbol=%s)",
            sub_id[:8],
            topic,
            source_filter,
            symbol_filter,
        )

        return sub_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.

        Args:
            subscription_id: ID from subscribe()

        Returns:
            True if unsubscribed, False if not found
        """
        sub = self._subscriptions.pop(subscription_id, None)
        if sub:
            # Remove from topic index
            if subscription_id in self._subscriptions_by_topic.get(sub.topic, []):
                self._subscriptions_by_topic[sub.topic].remove(subscription_id)

            self._metrics.subscriptions_active -= 1
            _logger.info("Unsubscribed: %s from %s", subscription_id[:8], sub.topic)
            return True
        return False

    def get_latest(
        self,
        event_type: str | EventType | None = None,
        symbol: str | None = None,
        count: int = 1,
    ) -> list[BusEvent]:
        """
        Get latest events from history.

        Args:
            event_type: Filter by event type
            symbol: Filter by symbol
            count: Number of events to return

        Returns:
            List of recent events
        """
        if not self.config.enable_history:
            return []

        # Normalize type
        if isinstance(event_type, EventType):
            event_type = event_type.value

        # Filter history
        filtered = self._history
        if event_type:
            filtered = [e for e in filtered if e.event_type == event_type]
        if symbol:
            filtered = [e for e in filtered if e.symbol == symbol]

        # Return most recent
        return list(reversed(filtered[-count:]))

    def replay(
        self,
        handler: EventHandler,
        event_type: str | EventType | None = None,
        since: datetime | None = None,
    ) -> asyncio.Task:
        """
        Replay historical events to a handler.

        Args:
            handler: Handler to receive events
            event_type: Filter by type
            since: Only events after this time

        Returns:
            Async task for replay
        """

        async def _replay():
            if isinstance(event_type, EventType):
                type_str = event_type.value
            else:
                type_str = event_type

            for event in self._history:
                if type_str and event.event_type != type_str:
                    continue
                if since and event.timestamp < since:
                    continue
                await handler(event)

        return asyncio.create_task(_replay())

    def get_metrics(self) -> dict:
        """Get bus metrics."""
        return {
            "events_published": self._metrics.events_published,
            "events_delivered": self._metrics.events_delivered,
            "events_dropped": self._metrics.events_dropped,
            "handler_errors": self._metrics.handler_errors,
            "subscriptions_active": self._metrics.subscriptions_active,
            "history_size": len(self._history),
            "events_by_type": dict(self._metrics.events_by_type),
        }

    def clear_history(self) -> int:
        """Clear event history, returning count cleared."""
        count = len(self._history)
        self._history.clear()
        return count

    async def shutdown(self) -> None:
        """Shutdown the bus gracefully."""
        self._running = False

        # Clear subscriptions
        self._subscriptions.clear()
        self._subscriptions_by_topic.clear()

        if self._redis_adapter:
            await self._redis_adapter.close()

        _logger.info("StreamingBus shutdown complete")
