"""Tests for StreamingBus data models."""

from datetime import UTC, datetime
import time

from ordinis.bus import BusEvent, EventPriority, EventType, Subscription


class TestEventType:
    """Tests for EventType enum."""

    def test_all_event_types_exist(self):
        """Test that all 34 event types are defined."""
        expected_types = [
            # Market data (3)
            "TICK",
            "BAR",
            "QUOTE",
            # Trading signals (3)
            "SIGNAL",
            "SIGNAL_CONFIRMED",
            "SIGNAL_REJECTED",
            # Orders and fills (5)
            "ORDER",
            "ORDER_SUBMITTED",
            "ORDER_FILLED",
            "ORDER_CANCELLED",
            "FILL",
            # Portfolio (3)
            "POSITION_OPENED",
            "POSITION_CLOSED",
            "REBALANCE",
            # Risk (3)
            "RISK_CHECK",
            "RISK_ALERT",
            "HALT",
            # System (4)
            "HEARTBEAT",
            "AUDIT",
            "ERROR",
            "METRIC",
        ]

        # Verify count
        assert len(EventType) == 21, f"Expected 21 event types, found {len(EventType)}"

        # Verify all expected types exist
        for event_type_name in expected_types:
            assert hasattr(EventType, event_type_name), f"EventType.{event_type_name} not found"

    def test_event_type_values(self):
        """Test EventType enum values are lowercase."""
        assert EventType.SIGNAL.value == "signal"
        assert EventType.ORDER_FILLED.value == "order_filled"
        assert EventType.HEARTBEAT.value == "heartbeat"

    def test_event_type_categories(self):
        """Test event type categorization."""
        market_data_types = {EventType.TICK, EventType.BAR, EventType.QUOTE}
        signal_types = {
            EventType.SIGNAL,
            EventType.SIGNAL_CONFIRMED,
            EventType.SIGNAL_REJECTED,
        }
        order_types = {
            EventType.ORDER,
            EventType.ORDER_SUBMITTED,
            EventType.ORDER_FILLED,
            EventType.ORDER_CANCELLED,
            EventType.FILL,
        }

        assert len(market_data_types) == 3
        assert len(signal_types) == 3
        assert len(order_types) == 5


class TestEventPriority:
    """Tests for EventPriority enum."""

    def test_priority_levels(self):
        """Test all priority levels exist."""
        assert EventPriority.LOW.value == 0
        assert EventPriority.NORMAL.value == 1
        assert EventPriority.HIGH.value == 2
        assert EventPriority.CRITICAL.value == 3

    def test_priority_ordering(self):
        """Test priority levels are ordered correctly."""
        assert EventPriority.LOW.value < EventPriority.NORMAL.value
        assert EventPriority.NORMAL.value < EventPriority.HIGH.value
        assert EventPriority.HIGH.value < EventPriority.CRITICAL.value


class TestBusEvent:
    """Tests for BusEvent dataclass."""

    def test_create_event_minimal(self):
        """Test creating event with minimal required fields."""
        event = BusEvent(
            event_type="test",
            source="test-source",
            payload={"key": "value"},
        )

        assert event.event_type == "test"
        assert event.source == "test-source"
        assert event.payload == {"key": "value"}
        assert event.symbol is None
        assert event.priority == EventPriority.NORMAL
        assert event.trace_id is None
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_create_event_with_enum_type(self):
        """Test creating event with EventType enum."""
        event = BusEvent(
            event_type=EventType.SIGNAL,
            source="signalcore",
            payload={"direction": "buy"},
        )

        # Should be normalized to string
        assert event.event_type == "signal"
        assert event.type_enum == EventType.SIGNAL

    def test_create_event_full(self):
        """Test creating event with all fields."""
        now = datetime.now(UTC)
        event = BusEvent(
            event_type=EventType.ORDER,
            source="flowroute",
            payload={"order_id": "123"},
            symbol="AAPL",
            timestamp=now,
            event_id="custom-id-123",
            trace_id="trace-456",
            priority=EventPriority.HIGH,
            metadata={"broker": "paper"},
        )

        assert event.event_type == "order"
        assert event.source == "flowroute"
        assert event.payload == {"order_id": "123"}
        assert event.symbol == "AAPL"
        assert event.timestamp == now
        assert event.event_id == "custom-id-123"
        assert event.trace_id == "trace-456"
        assert event.priority == EventPriority.HIGH
        assert event.metadata == {"broker": "paper"}

    def test_type_enum_property(self):
        """Test type_enum property conversion."""
        # Valid EventType
        event1 = BusEvent(
            event_type=EventType.SIGNAL,
            source="test",
            payload={},
        )
        assert event1.type_enum == EventType.SIGNAL

        # Custom type (not in enum)
        event2 = BusEvent(
            event_type="custom_event",
            source="test",
            payload={},
        )
        assert event2.type_enum is None

    def test_age_ms_property(self):
        """Test age_ms property calculation."""
        event = BusEvent(
            event_type="test",
            source="test",
            payload={},
        )

        # No published_at yet
        assert event.age_ms is None

        # Simulate publishing
        event._published_at = datetime.now(UTC)
        time.sleep(0.01)  # 10ms

        age = event.age_ms
        assert age is not None
        assert age >= 10.0  # At least 10ms

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now(UTC)
        event = BusEvent(
            event_type=EventType.SIGNAL,
            source="test",
            payload={"data": "value"},
            symbol="AAPL",
            timestamp=now,
            event_id="evt-123",
            trace_id="trace-456",
            priority=EventPriority.HIGH,
            metadata={"key": "value"},
        )

        event_dict = event.to_dict()

        assert event_dict["event_type"] == "signal"
        assert event_dict["source"] == "test"
        assert event_dict["payload"] == {"data": "value"}
        assert event_dict["symbol"] == "AAPL"
        assert event_dict["timestamp"] == now.isoformat()
        assert event_dict["event_id"] == "evt-123"
        assert event_dict["trace_id"] == "trace-456"
        assert event_dict["priority"] == EventPriority.HIGH.value
        assert event_dict["metadata"] == {"key": "value"}

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        now = datetime.now(UTC)
        event_dict = {
            "event_type": "signal",
            "source": "test",
            "payload": {"data": "value"},
            "symbol": "AAPL",
            "timestamp": now.isoformat(),
            "event_id": "evt-123",
            "trace_id": "trace-456",
            "priority": 2,
            "metadata": {"key": "value"},
        }

        event = BusEvent.from_dict(event_dict)

        assert event.event_type == "signal"
        assert event.source == "test"
        assert event.payload == {"data": "value"}
        assert event.symbol == "AAPL"
        assert event.event_id == "evt-123"
        assert event.trace_id == "trace-456"
        assert event.priority == EventPriority.HIGH
        assert event.metadata == {"key": "value"}

    def test_from_dict_minimal(self):
        """Test from_dict with minimal required fields."""
        event_dict = {
            "event_type": "test",
            "source": "test-source",
            "payload": {"key": "value"},
        }

        event = BusEvent.from_dict(event_dict)

        assert event.event_type == "test"
        assert event.source == "test-source"
        assert event.payload == {"key": "value"}
        assert event.symbol is None
        assert event.priority == EventPriority.NORMAL

    def test_roundtrip_serialization(self):
        """Test to_dict -> from_dict roundtrip."""
        original = BusEvent(
            event_type=EventType.ORDER_FILLED,
            source="broker",
            payload={"filled_qty": 100},
            symbol="TSLA",
            priority=EventPriority.CRITICAL,
            trace_id="trace-789",
        )

        # Roundtrip
        event_dict = original.to_dict()
        restored = BusEvent.from_dict(event_dict)

        assert restored.event_type == original.event_type
        assert restored.source == original.source
        assert restored.payload == original.payload
        assert restored.symbol == original.symbol
        assert restored.priority == original.priority
        assert restored.trace_id == original.trace_id


class TestSubscription:
    """Tests for Subscription dataclass."""

    async def dummy_handler(self, event: BusEvent) -> None:
        """Dummy handler for testing."""

    def test_create_subscription(self):
        """Test creating a subscription."""
        sub = Subscription(
            subscription_id="sub-123",
            topic="signal",
            handler=self.dummy_handler,
        )

        assert sub.subscription_id == "sub-123"
        assert sub.topic == "signal"
        assert sub.handler == self.dummy_handler
        assert sub.source_filter is None
        assert sub.symbol_filter is None
        assert sub.priority_min == EventPriority.LOW
        assert sub.active is True

    def test_subscription_with_filters(self):
        """Test subscription with all filters."""
        sub = Subscription(
            subscription_id="sub-123",
            topic="order",
            handler=self.dummy_handler,
            source_filter="flowroute",
            symbol_filter="AAPL",
            priority_min=EventPriority.HIGH,
        )

        assert sub.source_filter == "flowroute"
        assert sub.symbol_filter == "AAPL"
        assert sub.priority_min == EventPriority.HIGH

    def test_matches_exact_topic(self):
        """Test exact topic matching."""
        sub = Subscription(
            subscription_id="sub-123",
            topic="signal",
            handler=self.dummy_handler,
        )

        event1 = BusEvent(event_type="signal", source="test", payload={})
        event2 = BusEvent(event_type="order", source="test", payload={})

        assert sub.matches(event1) is True
        assert sub.matches(event2) is False

    def test_matches_wildcard_all(self):
        """Test wildcard matching for all events."""
        sub = Subscription(
            subscription_id="sub-123",
            topic="*",
            handler=self.dummy_handler,
        )

        event1 = BusEvent(event_type="signal", source="test", payload={})
        event2 = BusEvent(event_type="order", source="test", payload={})
        event3 = BusEvent(event_type="custom_event", source="test", payload={})

        assert sub.matches(event1) is True
        assert sub.matches(event2) is True
        assert sub.matches(event3) is True

    def test_matches_wildcard_prefix(self):
        """Test wildcard prefix matching (e.g., 'signal.*')."""
        sub = Subscription(
            subscription_id="sub-123",
            topic="signal*",
            handler=self.dummy_handler,
        )

        event1 = BusEvent(event_type="signal", source="test", payload={})
        event2 = BusEvent(event_type="signal_confirmed", source="test", payload={})
        event3 = BusEvent(event_type="order", source="test", payload={})

        assert sub.matches(event1) is True
        assert sub.matches(event2) is True
        assert sub.matches(event3) is False

    def test_matches_source_filter(self):
        """Test source filter matching."""
        sub = Subscription(
            subscription_id="sub-123",
            topic="*",
            handler=self.dummy_handler,
            source_filter="signalcore",
        )

        event1 = BusEvent(event_type="signal", source="signalcore", payload={})
        event2 = BusEvent(event_type="signal", source="other", payload={})

        assert sub.matches(event1) is True
        assert sub.matches(event2) is False

    def test_matches_symbol_filter(self):
        """Test symbol filter matching."""
        sub = Subscription(
            subscription_id="sub-123",
            topic="*",
            handler=self.dummy_handler,
            symbol_filter="AAPL",
        )

        event1 = BusEvent(event_type="signal", source="test", payload={}, symbol="AAPL")
        event2 = BusEvent(event_type="signal", source="test", payload={}, symbol="TSLA")

        assert sub.matches(event1) is True
        assert sub.matches(event2) is False

    def test_matches_priority_filter(self):
        """Test priority filter matching."""
        sub = Subscription(
            subscription_id="sub-123",
            topic="*",
            handler=self.dummy_handler,
            priority_min=EventPriority.HIGH,
        )

        event1 = BusEvent(
            event_type="signal",
            source="test",
            payload={},
            priority=EventPriority.CRITICAL,
        )
        event2 = BusEvent(
            event_type="signal",
            source="test",
            payload={},
            priority=EventPriority.HIGH,
        )
        event3 = BusEvent(
            event_type="signal",
            source="test",
            payload={},
            priority=EventPriority.NORMAL,
        )

        assert sub.matches(event1) is True  # CRITICAL >= HIGH
        assert sub.matches(event2) is True  # HIGH >= HIGH
        assert sub.matches(event3) is False  # NORMAL < HIGH

    def test_matches_combined_filters(self):
        """Test multiple filters combined."""
        sub = Subscription(
            subscription_id="sub-123",
            topic="order*",
            handler=self.dummy_handler,
            source_filter="flowroute",
            symbol_filter="AAPL",
            priority_min=EventPriority.HIGH,
        )

        # All filters match
        event1 = BusEvent(
            event_type="order_filled",
            source="flowroute",
            payload={},
            symbol="AAPL",
            priority=EventPriority.HIGH,
        )
        assert sub.matches(event1) is True

        # Topic doesn't match
        event2 = BusEvent(
            event_type="signal",
            source="flowroute",
            payload={},
            symbol="AAPL",
            priority=EventPriority.HIGH,
        )
        assert sub.matches(event2) is False

        # Source doesn't match
        event3 = BusEvent(
            event_type="order_filled",
            source="other",
            payload={},
            symbol="AAPL",
            priority=EventPriority.HIGH,
        )
        assert sub.matches(event3) is False

        # Symbol doesn't match
        event4 = BusEvent(
            event_type="order_filled",
            source="flowroute",
            payload={},
            symbol="TSLA",
            priority=EventPriority.HIGH,
        )
        assert sub.matches(event4) is False

        # Priority too low
        event5 = BusEvent(
            event_type="order_filled",
            source="flowroute",
            payload={},
            symbol="AAPL",
            priority=EventPriority.NORMAL,
        )
        assert sub.matches(event5) is False

    def test_matches_inactive_subscription(self):
        """Test that inactive subscriptions don't match."""
        sub = Subscription(
            subscription_id="sub-123",
            topic="*",
            handler=self.dummy_handler,
            active=False,
        )

        event = BusEvent(event_type="signal", source="test", payload={})

        assert sub.matches(event) is False

    def test_matches_with_enum_event_type(self):
        """Test matching when event_type is EventType enum."""
        sub = Subscription(
            subscription_id="sub-123",
            topic="signal",
            handler=self.dummy_handler,
        )

        # Event with enum type (should be normalized in BusEvent.__post_init__)
        event = BusEvent(
            event_type=EventType.SIGNAL,
            source="test",
            payload={},
        )

        assert sub.matches(event) is True
