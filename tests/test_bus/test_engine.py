"""Tests for StreamingBus engine."""

import asyncio
from datetime import UTC, datetime

import pytest

from ordinis.bus import BusConfig, BusEvent, EventPriority, EventType, StreamingBus
from ordinis.bus.config import AdapterType


class TestStreamingBusInit:
    """Tests for StreamingBus initialization."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        bus = StreamingBus()

        assert bus.config.adapter == AdapterType.MEMORY
        assert bus._subscriptions == {}
        assert bus._history == []
        assert bus._metrics.events_published == 0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = BusConfig(
            max_payload_size=512 * 1024,
            handler_timeout_seconds=60.0,
        )
        bus = StreamingBus(config)

        assert bus.config.max_payload_size == 512 * 1024
        assert bus.config.handler_timeout_seconds == 60.0

    def test_init_invalid_config_raises(self):
        """Test that invalid config raises ValueError."""
        config = BusConfig(max_payload_size=-1)

        with pytest.raises(ValueError, match="Invalid BusConfig"):
            StreamingBus(config)


@pytest.mark.asyncio
class TestStreamingBusPublish:
    """Tests for publish() method."""

    async def test_publish_simple_event(self, streaming_bus: StreamingBus, sample_event: BusEvent):
        """Test publishing a simple event."""
        await streaming_bus.publish(sample_event)

        metrics = streaming_bus.get_metrics()
        assert metrics["events_published"] == 1
        assert metrics["events_by_type"]["signal"] == 1

    async def test_publish_sets_published_at(
        self, streaming_bus: StreamingBus, sample_event: BusEvent
    ):
        """Test that publish sets _published_at timestamp."""
        assert sample_event._published_at is None

        await streaming_bus.publish(sample_event)

        assert sample_event._published_at is not None
        assert isinstance(sample_event._published_at, datetime)

    async def test_publish_stores_in_history(
        self, streaming_bus: StreamingBus, create_event: callable
    ):
        """Test that published events are stored in history."""
        event1 = create_event(event_type=EventType.SIGNAL)
        event2 = create_event(event_type=EventType.ORDER)

        await streaming_bus.publish(event1)
        await streaming_bus.publish(event2)

        assert len(streaming_bus._history) == 2
        assert streaming_bus._history[0] == event1
        assert streaming_bus._history[1] == event2

    async def test_publish_no_history_when_disabled(
        self, streaming_bus_no_history: StreamingBus, sample_event: BusEvent
    ):
        """Test that history is not stored when disabled."""
        await streaming_bus_no_history.publish(sample_event)

        assert len(streaming_bus_no_history._history) == 0

    async def test_publish_trims_history(self, create_event: callable):
        """Test that history is trimmed to max size."""
        config = BusConfig(history_max_events=5)
        bus = StreamingBus(config)

        # Publish more events than max
        for i in range(10):
            event = create_event(payload={"index": i})
            await bus.publish(event)

        # Should only keep last 5
        assert len(bus._history) == 5
        assert bus._history[0].payload["index"] == 5
        assert bus._history[4].payload["index"] == 9

        await bus.shutdown()

    async def test_publish_oversized_payload_dropped(
        self, streaming_bus: StreamingBus, create_event: callable
    ):
        """Test that oversized payloads are dropped."""
        # Create event with large payload
        large_payload = {"data": "x" * (streaming_bus.config.max_payload_size + 1)}
        event = create_event(payload=large_payload)

        await streaming_bus.publish(event)

        metrics = streaming_bus.get_metrics()
        assert metrics["events_published"] == 0  # Not published
        assert metrics["events_dropped"] == 1

    async def test_publish_updates_metrics(
        self, streaming_bus: StreamingBus, create_event: callable
    ):
        """Test that publish updates metrics correctly."""
        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL))
        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL))
        await streaming_bus.publish(create_event(event_type=EventType.ORDER))

        metrics = streaming_bus.get_metrics()
        assert metrics["events_published"] == 3
        assert metrics["events_by_type"]["signal"] == 2
        assert metrics["events_by_type"]["order"] == 1

    async def test_publish_no_subscribers(
        self, streaming_bus: StreamingBus, sample_event: BusEvent
    ):
        """Test publishing with no subscribers (should not error)."""
        await streaming_bus.publish(sample_event)

        # Should complete without error
        metrics = streaming_bus.get_metrics()
        assert metrics["events_published"] == 1
        assert metrics["events_delivered"] == 0


@pytest.mark.asyncio
class TestStreamingBusSubscribe:
    """Tests for subscribe() and unsubscribe() methods."""

    async def test_subscribe_returns_id(self, streaming_bus: StreamingBus, event_collector):
        """Test that subscribe returns a subscription ID."""
        sub_id = streaming_bus.subscribe("signal", event_collector.handler)

        assert sub_id is not None
        assert isinstance(sub_id, str)
        assert len(sub_id) > 0

    async def test_subscribe_stores_subscription(
        self, streaming_bus: StreamingBus, event_collector
    ):
        """Test that subscribe stores the subscription."""
        sub_id = streaming_bus.subscribe("signal", event_collector.handler)

        assert sub_id in streaming_bus._subscriptions
        sub = streaming_bus._subscriptions[sub_id]
        assert sub.topic == "signal"
        assert sub.handler == event_collector.handler

    async def test_subscribe_with_event_type_enum(
        self, streaming_bus: StreamingBus, event_collector
    ):
        """Test subscribing with EventType enum."""
        sub_id = streaming_bus.subscribe(EventType.SIGNAL, event_collector.handler)

        sub = streaming_bus._subscriptions[sub_id]
        assert sub.topic == "signal"

    async def test_subscribe_with_filters(self, streaming_bus: StreamingBus, event_collector):
        """Test subscribing with all filters."""
        sub_id = streaming_bus.subscribe(
            "order",
            event_collector.handler,
            source_filter="flowroute",
            symbol_filter="AAPL",
            priority_min=EventPriority.HIGH,
        )

        sub = streaming_bus._subscriptions[sub_id]
        assert sub.source_filter == "flowroute"
        assert sub.symbol_filter == "AAPL"
        assert sub.priority_min == EventPriority.HIGH

    async def test_subscribe_updates_metrics(self, streaming_bus: StreamingBus, event_collector):
        """Test that subscribe updates metrics."""
        streaming_bus.subscribe("signal", event_collector.handler)
        streaming_bus.subscribe("order", event_collector.handler)

        metrics = streaming_bus.get_metrics()
        assert metrics["subscriptions_active"] == 2

    async def test_unsubscribe_removes_subscription(
        self, streaming_bus: StreamingBus, event_collector
    ):
        """Test that unsubscribe removes the subscription."""
        sub_id = streaming_bus.subscribe("signal", event_collector.handler)

        result = streaming_bus.unsubscribe(sub_id)

        assert result is True
        assert sub_id not in streaming_bus._subscriptions

    async def test_unsubscribe_updates_metrics(self, streaming_bus: StreamingBus, event_collector):
        """Test that unsubscribe updates metrics."""
        sub_id = streaming_bus.subscribe("signal", event_collector.handler)
        streaming_bus.unsubscribe(sub_id)

        metrics = streaming_bus.get_metrics()
        assert metrics["subscriptions_active"] == 0

    async def test_unsubscribe_nonexistent_returns_false(self, streaming_bus: StreamingBus):
        """Test that unsubscribing nonexistent subscription returns False."""
        result = streaming_bus.unsubscribe("nonexistent-id")

        assert result is False

    async def test_multiple_subscriptions_same_topic(
        self, streaming_bus: StreamingBus, event_collector
    ):
        """Test multiple subscriptions to the same topic."""

        async def handler1(event: BusEvent) -> None:
            pass

        async def handler2(event: BusEvent) -> None:
            pass

        sub_id1 = streaming_bus.subscribe("signal", handler1)
        sub_id2 = streaming_bus.subscribe("signal", handler2)

        assert sub_id1 != sub_id2
        assert "signal" in streaming_bus._subscriptions_by_topic
        assert len(streaming_bus._subscriptions_by_topic["signal"]) == 2


@pytest.mark.asyncio
class TestStreamingBusEventDelivery:
    """Tests for event delivery to subscribers."""

    async def test_deliver_to_single_subscriber(
        self, streaming_bus: StreamingBus, event_collector, sample_event: BusEvent
    ):
        """Test delivering event to a single subscriber."""
        streaming_bus.subscribe("signal", event_collector.handler)

        await streaming_bus.publish(sample_event)
        await asyncio.sleep(0.1)  # Allow async handlers to complete

        assert len(event_collector.events) == 1
        assert event_collector.events[0] == sample_event

    async def test_deliver_to_multiple_subscribers(
        self, streaming_bus: StreamingBus, create_event: callable
    ):
        """Test delivering event to multiple subscribers."""
        collector1 = []
        collector2 = []

        async def handler1(event: BusEvent) -> None:
            collector1.append(event)

        async def handler2(event: BusEvent) -> None:
            collector2.append(event)

        streaming_bus.subscribe("signal", handler1)
        streaming_bus.subscribe("signal", handler2)

        event = create_event(event_type=EventType.SIGNAL)
        await streaming_bus.publish(event)
        await asyncio.sleep(0.1)

        assert len(collector1) == 1
        assert len(collector2) == 1
        assert collector1[0] == event
        assert collector2[0] == event

    async def test_deliver_only_to_matching_topic(
        self, streaming_bus: StreamingBus, event_collector, create_event: callable
    ):
        """Test that events are only delivered to matching topics."""
        streaming_bus.subscribe("signal", event_collector.handler)

        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL))
        await streaming_bus.publish(create_event(event_type=EventType.ORDER))
        await asyncio.sleep(0.1)

        # Should only receive signal event
        assert len(event_collector.events) == 1
        assert event_collector.events[0].event_type == "signal"

    async def test_deliver_with_wildcard_subscription(
        self, streaming_bus: StreamingBus, event_collector, create_event: callable
    ):
        """Test wildcard subscription receives all events."""
        streaming_bus.subscribe("*", event_collector.handler)

        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL))
        await streaming_bus.publish(create_event(event_type=EventType.ORDER))
        await streaming_bus.publish(create_event(event_type="custom_event"))
        await asyncio.sleep(0.1)

        assert len(event_collector.events) == 3

    async def test_deliver_with_wildcard_prefix(
        self, streaming_bus: StreamingBus, event_collector, create_event: callable
    ):
        """Test wildcard prefix matching (e.g., 'signal*')."""
        streaming_bus.subscribe("signal*", event_collector.handler)

        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL))
        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL_CONFIRMED))
        await streaming_bus.publish(create_event(event_type=EventType.ORDER))
        await asyncio.sleep(0.1)

        assert len(event_collector.events) == 2
        assert event_collector.events[0].event_type == "signal"
        assert event_collector.events[1].event_type == "signal_confirmed"

    async def test_deliver_with_source_filter(
        self, streaming_bus: StreamingBus, event_collector, create_event: callable
    ):
        """Test delivery with source filter."""
        streaming_bus.subscribe("*", event_collector.handler, source_filter="signalcore")

        await streaming_bus.publish(create_event(source="signalcore"))
        await streaming_bus.publish(create_event(source="flowroute"))
        await asyncio.sleep(0.1)

        assert len(event_collector.events) == 1
        assert event_collector.events[0].source == "signalcore"

    async def test_deliver_with_symbol_filter(
        self, streaming_bus: StreamingBus, event_collector, create_event: callable
    ):
        """Test delivery with symbol filter."""
        streaming_bus.subscribe("*", event_collector.handler, symbol_filter="AAPL")

        await streaming_bus.publish(create_event(symbol="AAPL"))
        await streaming_bus.publish(create_event(symbol="TSLA"))
        await asyncio.sleep(0.1)

        assert len(event_collector.events) == 1
        assert event_collector.events[0].symbol == "AAPL"

    async def test_deliver_with_priority_filter(
        self, streaming_bus: StreamingBus, event_collector, create_event: callable
    ):
        """Test delivery with priority filter."""
        streaming_bus.subscribe("*", event_collector.handler, priority_min=EventPriority.HIGH)

        await streaming_bus.publish(create_event(priority=EventPriority.CRITICAL))
        await streaming_bus.publish(create_event(priority=EventPriority.HIGH))
        await streaming_bus.publish(create_event(priority=EventPriority.NORMAL))
        await asyncio.sleep(0.1)

        assert len(event_collector.events) == 2
        assert event_collector.events[0].priority == EventPriority.CRITICAL
        assert event_collector.events[1].priority == EventPriority.HIGH

    async def test_deliver_updates_metrics(
        self, streaming_bus: StreamingBus, event_collector, sample_event: BusEvent
    ):
        """Test that delivery updates metrics."""
        streaming_bus.subscribe("signal", event_collector.handler)

        await streaming_bus.publish(sample_event)
        await asyncio.sleep(0.1)

        metrics = streaming_bus.get_metrics()
        assert metrics["events_delivered"] == 1


@pytest.mark.asyncio
class TestStreamingBusHandlerErrors:
    """Tests for handler error handling."""

    async def test_handler_error_does_not_crash_bus(
        self, streaming_bus: StreamingBus, failing_handler, sample_event: BusEvent
    ):
        """Test that handler errors don't crash the bus."""
        streaming_bus.subscribe("signal", failing_handler.handler)

        # Should not raise exception
        await streaming_bus.publish(sample_event)
        await asyncio.sleep(0.1)

        # Bus should still be operational
        metrics = streaming_bus.get_metrics()
        assert metrics["events_published"] == 1

    async def test_handler_error_updates_metrics(
        self, streaming_bus: StreamingBus, failing_handler, sample_event: BusEvent
    ):
        """Test that handler errors update metrics."""
        streaming_bus.subscribe("signal", failing_handler.handler)

        await streaming_bus.publish(sample_event)
        await asyncio.sleep(0.5)  # Wait for retries

        metrics = streaming_bus.get_metrics()
        assert metrics["handler_errors"] > 0

    async def test_handler_retry_on_failure(
        self, streaming_bus: StreamingBus, event_collector, sample_event: BusEvent
    ):
        """Test that handlers are retried on failure."""
        # Configure collector to fail on first call
        event_collector.error_on_call = 1
        streaming_bus.subscribe("signal", event_collector.handler)

        await streaming_bus.publish(sample_event)
        await asyncio.sleep(0.5)  # Wait for retries

        # Should have been retried and eventually succeeded
        assert event_collector.call_count > 1
        assert len(event_collector.events) > 0

    async def test_handler_no_retry_when_disabled(
        self, streaming_bus_no_retry: StreamingBus, failing_handler, sample_event: BusEvent
    ):
        """Test that handlers are not retried when disabled."""
        streaming_bus_no_retry.subscribe("signal", failing_handler.handler)

        await streaming_bus_no_retry.publish(sample_event)
        await asyncio.sleep(0.1)

        # Should only be called once (no retries)
        assert failing_handler.call_count == 1

    async def test_handler_partial_failure(
        self, streaming_bus: StreamingBus, event_collector, create_event: callable
    ):
        """Test that one failing handler doesn't affect others."""
        collected_events = []

        async def good_handler(event: BusEvent) -> None:
            collected_events.append(event)

        async def bad_handler(event: BusEvent) -> None:
            raise RuntimeError("Intentional failure")

        streaming_bus.subscribe("signal", good_handler)
        streaming_bus.subscribe("signal", bad_handler)

        event = create_event(event_type=EventType.SIGNAL)
        await streaming_bus.publish(event)
        await asyncio.sleep(0.5)

        # Good handler should have received the event
        assert len(collected_events) == 1
        assert collected_events[0] == event


@pytest.mark.asyncio
class TestStreamingBusHandlerTimeout:
    """Tests for handler timeout behavior."""

    async def test_handler_timeout(
        self, streaming_bus: StreamingBus, slow_handler, sample_event: BusEvent
    ):
        """Test that slow handlers time out."""
        # Create handler that takes longer than timeout
        handler = slow_handler(delay_seconds=streaming_bus.config.handler_timeout_seconds + 1)
        streaming_bus.subscribe("signal", handler.handler)

        await streaming_bus.publish(sample_event)
        await asyncio.sleep(streaming_bus.config.handler_timeout_seconds + 0.5)

        # Handler should have timed out
        metrics = streaming_bus.get_metrics()
        assert metrics["handler_errors"] > 0

    async def test_handler_completes_within_timeout(
        self, streaming_bus: StreamingBus, slow_handler, sample_event: BusEvent
    ):
        """Test that handlers completing within timeout succeed."""
        # Create handler faster than timeout
        handler = slow_handler(delay_seconds=0.1)
        streaming_bus.subscribe("signal", handler.handler)

        await streaming_bus.publish(sample_event)
        await asyncio.sleep(0.2)

        assert handler.call_count == 1
        metrics = streaming_bus.get_metrics()
        assert metrics["events_delivered"] == 1


@pytest.mark.asyncio
class TestStreamingBusHistory:
    """Tests for event history and replay."""

    async def test_get_latest_no_filter(self, streaming_bus: StreamingBus, create_event: callable):
        """Test get_latest without filters."""
        event1 = create_event(payload={"id": 1})
        event2 = create_event(payload={"id": 2})
        event3 = create_event(payload={"id": 3})

        await streaming_bus.publish(event1)
        await streaming_bus.publish(event2)
        await streaming_bus.publish(event3)

        latest = streaming_bus.get_latest(count=2)

        assert len(latest) == 2
        # Should be in reverse chronological order
        assert latest[0].payload["id"] == 3
        assert latest[1].payload["id"] == 2

    async def test_get_latest_with_event_type_filter(
        self, streaming_bus: StreamingBus, create_event: callable
    ):
        """Test get_latest with event type filter."""
        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL))
        await streaming_bus.publish(create_event(event_type=EventType.ORDER))
        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL))

        latest = streaming_bus.get_latest(event_type=EventType.SIGNAL, count=10)

        assert len(latest) == 2
        assert all(e.event_type == "signal" for e in latest)

    async def test_get_latest_with_symbol_filter(
        self, streaming_bus: StreamingBus, create_event: callable
    ):
        """Test get_latest with symbol filter."""
        await streaming_bus.publish(create_event(symbol="AAPL"))
        await streaming_bus.publish(create_event(symbol="TSLA"))
        await streaming_bus.publish(create_event(symbol="AAPL"))

        latest = streaming_bus.get_latest(symbol="AAPL", count=10)

        assert len(latest) == 2
        assert all(e.symbol == "AAPL" for e in latest)

    async def test_get_latest_combined_filters(
        self, streaming_bus: StreamingBus, create_event: callable
    ):
        """Test get_latest with multiple filters."""
        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL, symbol="AAPL"))
        await streaming_bus.publish(create_event(event_type=EventType.ORDER, symbol="AAPL"))
        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL, symbol="TSLA"))
        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL, symbol="AAPL"))

        latest = streaming_bus.get_latest(event_type=EventType.SIGNAL, symbol="AAPL", count=10)

        assert len(latest) == 2
        assert all(e.event_type == "signal" and e.symbol == "AAPL" for e in latest)

    async def test_get_latest_when_history_disabled(
        self, streaming_bus_no_history: StreamingBus, sample_event: BusEvent
    ):
        """Test get_latest returns empty when history disabled."""
        await streaming_bus_no_history.publish(sample_event)

        latest = streaming_bus_no_history.get_latest()

        assert latest == []

    async def test_replay_all_events(
        self, streaming_bus: StreamingBus, event_collector, create_event: callable
    ):
        """Test replaying all events from history."""
        event1 = create_event(payload={"id": 1})
        event2 = create_event(payload={"id": 2})

        await streaming_bus.publish(event1)
        await streaming_bus.publish(event2)

        # Replay to collector
        task = streaming_bus.replay(event_collector.handler)
        await task

        assert len(event_collector.events) == 2
        assert event_collector.events[0].payload["id"] == 1
        assert event_collector.events[1].payload["id"] == 2

    async def test_replay_with_event_type_filter(
        self, streaming_bus: StreamingBus, event_collector, create_event: callable
    ):
        """Test replay with event type filter."""
        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL))
        await streaming_bus.publish(create_event(event_type=EventType.ORDER))
        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL))

        task = streaming_bus.replay(event_collector.handler, event_type=EventType.SIGNAL)
        await task

        assert len(event_collector.events) == 2
        assert all(e.event_type == "signal" for e in event_collector.events)

    async def test_replay_with_since_filter(
        self, streaming_bus: StreamingBus, event_collector, create_event: callable
    ):
        """Test replay with since timestamp filter."""
        await streaming_bus.publish(create_event())
        await asyncio.sleep(0.1)

        cutoff_time = datetime.now(UTC)
        await asyncio.sleep(0.1)

        await streaming_bus.publish(create_event())
        await streaming_bus.publish(create_event())

        task = streaming_bus.replay(event_collector.handler, since=cutoff_time)
        await task

        # Should only get events after cutoff
        assert len(event_collector.events) == 2

    async def test_clear_history(self, streaming_bus: StreamingBus, create_event: callable):
        """Test clearing event history."""
        await streaming_bus.publish(create_event())
        await streaming_bus.publish(create_event())
        await streaming_bus.publish(create_event())

        count = streaming_bus.clear_history()

        assert count == 3
        assert len(streaming_bus._history) == 0
        assert streaming_bus.get_latest() == []


@pytest.mark.asyncio
class TestStreamingBusMetrics:
    """Tests for BusMetrics tracking."""

    async def test_get_metrics_initial_state(self, streaming_bus: StreamingBus):
        """Test metrics in initial state."""
        metrics = streaming_bus.get_metrics()

        assert metrics["events_published"] == 0
        assert metrics["events_delivered"] == 0
        assert metrics["events_dropped"] == 0
        assert metrics["handler_errors"] == 0
        assert metrics["subscriptions_active"] == 0
        assert metrics["history_size"] == 0
        assert metrics["events_by_type"] == {}

    async def test_metrics_track_publications(
        self, streaming_bus: StreamingBus, create_event: callable
    ):
        """Test that metrics track publications."""
        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL))
        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL))
        await streaming_bus.publish(create_event(event_type=EventType.ORDER))

        metrics = streaming_bus.get_metrics()
        assert metrics["events_published"] == 3
        assert metrics["events_by_type"]["signal"] == 2
        assert metrics["events_by_type"]["order"] == 1

    async def test_metrics_track_deliveries(
        self, streaming_bus: StreamingBus, event_collector, create_event: callable
    ):
        """Test that metrics track deliveries."""
        streaming_bus.subscribe("*", event_collector.handler)

        await streaming_bus.publish(create_event())
        await streaming_bus.publish(create_event())
        await asyncio.sleep(0.1)

        metrics = streaming_bus.get_metrics()
        assert metrics["events_delivered"] == 2

    async def test_metrics_track_subscriptions(self, streaming_bus: StreamingBus, event_collector):
        """Test that metrics track active subscriptions."""
        sub_id1 = streaming_bus.subscribe("signal", event_collector.handler)
        sub_id2 = streaming_bus.subscribe("order", event_collector.handler)

        metrics = streaming_bus.get_metrics()
        assert metrics["subscriptions_active"] == 2

        streaming_bus.unsubscribe(sub_id1)

        metrics = streaming_bus.get_metrics()
        assert metrics["subscriptions_active"] == 1

    async def test_metrics_track_history_size(
        self, streaming_bus: StreamingBus, create_event: callable
    ):
        """Test that metrics track history size."""
        await streaming_bus.publish(create_event())
        await streaming_bus.publish(create_event())
        await streaming_bus.publish(create_event())

        metrics = streaming_bus.get_metrics()
        assert metrics["history_size"] == 3


@pytest.mark.asyncio
class TestStreamingBusShutdown:
    """Tests for bus shutdown."""

    async def test_shutdown_clears_subscriptions(
        self, streaming_bus: StreamingBus, event_collector
    ):
        """Test that shutdown clears subscriptions."""
        streaming_bus.subscribe("signal", event_collector.handler)
        streaming_bus.subscribe("order", event_collector.handler)

        await streaming_bus.shutdown()

        assert len(streaming_bus._subscriptions) == 0
        assert len(streaming_bus._subscriptions_by_topic) == 0

    async def test_shutdown_sets_running_flag(self, streaming_bus: StreamingBus):
        """Test that shutdown sets _running flag."""
        await streaming_bus.shutdown()

        assert streaming_bus._running is False

    async def test_multiple_shutdown_calls_safe(self, streaming_bus: StreamingBus):
        """Test that multiple shutdown calls are safe."""
        await streaming_bus.shutdown()
        await streaming_bus.shutdown()  # Should not error

        assert streaming_bus._running is False


@pytest.mark.asyncio
class TestStreamingBusConcurrency:
    """Tests for concurrent operations."""

    async def test_concurrent_publishes(self, streaming_bus: StreamingBus, create_event: callable):
        """Test concurrent event publishing."""
        tasks = [streaming_bus.publish(create_event(payload={"id": i})) for i in range(100)]

        await asyncio.gather(*tasks)

        metrics = streaming_bus.get_metrics()
        assert metrics["events_published"] == 100

    async def test_concurrent_subscriptions(self, streaming_bus: StreamingBus):
        """Test concurrent subscriptions."""

        async def handler(event: BusEvent) -> None:
            pass

        # Create subscriptions (subscribe is synchronous)
        sub_ids = []
        for i in range(50):
            sub_id = streaming_bus.subscribe(f"topic_{i}", handler)
            sub_ids.append(sub_id)

        metrics = streaming_bus.get_metrics()
        assert metrics["subscriptions_active"] == 50

    async def test_max_concurrent_handlers(
        self, streaming_bus: StreamingBus, create_event: callable
    ):
        """Test that concurrent handler execution is limited."""
        active_handlers = []
        max_concurrent = 0

        async def tracking_handler(event: BusEvent) -> None:
            nonlocal max_concurrent
            active_handlers.append(1)
            max_concurrent = max(max_concurrent, len(active_handlers))
            await asyncio.sleep(0.1)
            active_handlers.pop()

        # Subscribe multiple times
        for _ in range(20):
            streaming_bus.subscribe("signal", tracking_handler)

        # Publish event
        await streaming_bus.publish(create_event(event_type=EventType.SIGNAL))
        await asyncio.sleep(0.5)

        # Should respect max_concurrent_handlers limit
        assert max_concurrent <= streaming_bus.config.max_concurrent_handlers


@pytest.mark.asyncio
class TestStreamingBusEdgeCases:
    """Tests for edge cases and error conditions."""

    async def test_empty_payload(self, streaming_bus: StreamingBus):
        """Test event with empty payload."""
        event = BusEvent(
            event_type="test",
            source="test",
            payload={},
        )

        await streaming_bus.publish(event)

        metrics = streaming_bus.get_metrics()
        assert metrics["events_published"] == 1

    async def test_none_symbol(self, streaming_bus: StreamingBus):
        """Test event with None symbol."""
        event = BusEvent(
            event_type="test",
            source="test",
            payload={"data": "value"},
            symbol=None,
        )

        await streaming_bus.publish(event)

        latest = streaming_bus.get_latest()
        assert latest[0].symbol is None

    async def test_custom_event_type(self, streaming_bus: StreamingBus, event_collector):
        """Test with custom (non-enum) event type."""
        streaming_bus.subscribe("custom_event_type", event_collector.handler)

        event = BusEvent(
            event_type="custom_event_type",
            source="test",
            payload={"custom": "data"},
        )

        await streaming_bus.publish(event)
        await asyncio.sleep(0.1)

        assert len(event_collector.events) == 1
        assert event_collector.events[0].event_type == "custom_event_type"

    async def test_unsubscribe_during_publish(
        self, streaming_bus: StreamingBus, event_collector, create_event: callable
    ):
        """Test unsubscribing while events are being published."""
        sub_id = streaming_bus.subscribe("signal", event_collector.handler)

        # Start publishing
        publish_task = asyncio.create_task(streaming_bus.publish(create_event()))

        # Unsubscribe immediately
        streaming_bus.unsubscribe(sub_id)

        await publish_task
        await asyncio.sleep(0.1)

        # Event may or may not be delivered (race condition is acceptable)
        assert len(event_collector.events) in [0, 1]
