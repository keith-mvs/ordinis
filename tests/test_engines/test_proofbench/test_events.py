"""Tests for event system."""

from datetime import datetime, timedelta

from engines.proofbench.core.events import Event, EventQueue, EventType


class TestEvent:
    """Tests for Event dataclass."""

    def test_event_creation(self):
        """Test creating an event."""
        timestamp = datetime(2024, 1, 1, 9, 30)
        event = Event(
            timestamp=timestamp,
            event_type=EventType.MARKET_OPEN,
            data={"market": "NYSE"},
            priority=0,
        )

        assert event.timestamp == timestamp
        assert event.event_type == EventType.MARKET_OPEN
        assert event.data == {"market": "NYSE"}
        assert event.priority == 0

    def test_event_comparison_by_timestamp(self):
        """Test events are ordered by timestamp."""
        event1 = Event(
            timestamp=datetime(2024, 1, 1, 9, 30),
            event_type=EventType.MARKET_OPEN,
            data={},
        )
        event2 = Event(
            timestamp=datetime(2024, 1, 1, 9, 31),
            event_type=EventType.BAR_UPDATE,
            data={},
        )

        assert event1 < event2
        assert event2 > event1
        assert event1 <= event2
        assert event2 >= event1

    def test_event_comparison_by_priority(self):
        """Test events with same timestamp are ordered by priority."""
        timestamp = datetime(2024, 1, 1, 9, 30)

        event_high_priority = Event(
            timestamp=timestamp, event_type=EventType.RISK_CHECK, data={}, priority=0
        )
        event_low_priority = Event(
            timestamp=timestamp, event_type=EventType.SIGNAL, data={}, priority=10
        )

        # Lower priority number = higher priority
        assert event_high_priority < event_low_priority
        assert event_low_priority > event_high_priority

    def test_event_equality(self):
        """Test event equality comparison."""
        timestamp = datetime(2024, 1, 1, 9, 30)

        event1 = Event(timestamp=timestamp, event_type=EventType.MARKET_OPEN, data={}, priority=0)
        event2 = Event(timestamp=timestamp, event_type=EventType.MARKET_OPEN, data={}, priority=0)

        # Events with same timestamp and priority are equal for ordering
        assert event1 <= event2
        assert event1 >= event2


class TestEventQueue:
    """Tests for EventQueue."""

    def test_queue_creation(self):
        """Test creating an empty queue."""
        queue = EventQueue()
        assert queue.is_empty()
        assert queue.size() == 0

    def test_push_and_pop(self):
        """Test pushing and popping events."""
        queue = EventQueue()

        event = Event(
            timestamp=datetime(2024, 1, 1, 9, 30),
            event_type=EventType.MARKET_OPEN,
            data={},
        )

        queue.push(event)
        assert queue.size() == 1
        assert not queue.is_empty()

        popped = queue.pop()
        assert popped == event
        assert queue.is_empty()

    def test_chronological_ordering(self):
        """Test events are popped in chronological order."""
        queue = EventQueue()

        # Push events in random order
        event3 = Event(
            timestamp=datetime(2024, 1, 1, 9, 32),
            event_type=EventType.BAR_UPDATE,
            data={},
        )
        event1 = Event(
            timestamp=datetime(2024, 1, 1, 9, 30),
            event_type=EventType.MARKET_OPEN,
            data={},
        )
        event2 = Event(
            timestamp=datetime(2024, 1, 1, 9, 31),
            event_type=EventType.BAR_UPDATE,
            data={},
        )

        queue.push(event3)
        queue.push(event1)
        queue.push(event2)

        # Should pop in chronological order
        assert queue.pop() == event1
        assert queue.pop() == event2
        assert queue.pop() == event3
        assert queue.is_empty()

    def test_priority_ordering(self):
        """Test events with same timestamp are ordered by priority."""
        queue = EventQueue()
        timestamp = datetime(2024, 1, 1, 9, 30)

        # Push events with same timestamp but different priorities
        event_low = Event(timestamp=timestamp, event_type=EventType.SIGNAL, data={}, priority=10)
        event_high = Event(
            timestamp=timestamp, event_type=EventType.RISK_CHECK, data={}, priority=0
        )
        event_med = Event(
            timestamp=timestamp,
            event_type=EventType.ORDER_SUBMIT,
            data={},
            priority=5,
        )

        queue.push(event_low)
        queue.push(event_high)
        queue.push(event_med)

        # Should pop in priority order (0, 5, 10)
        assert queue.pop() == event_high
        assert queue.pop() == event_med
        assert queue.pop() == event_low

    def test_peek(self):
        """Test peeking at next event without removing it."""
        queue = EventQueue()

        event1 = Event(
            timestamp=datetime(2024, 1, 1, 9, 30),
            event_type=EventType.MARKET_OPEN,
            data={},
        )
        event2 = Event(
            timestamp=datetime(2024, 1, 1, 9, 31),
            event_type=EventType.BAR_UPDATE,
            data={},
        )

        queue.push(event2)
        queue.push(event1)

        # Peek should return earliest event without removing
        peeked = queue.peek()
        assert peeked == event1
        assert queue.size() == 2

        # Verify it's still there
        popped = queue.pop()
        assert popped == event1

    def test_peek_empty_queue(self):
        """Test peeking at empty queue returns None."""
        queue = EventQueue()
        assert queue.peek() is None

    def test_pop_empty_queue(self):
        """Test popping from empty queue returns None."""
        queue = EventQueue()
        assert queue.pop() is None

    def test_clear(self):
        """Test clearing the queue."""
        queue = EventQueue()

        for i in range(5):
            event = Event(
                timestamp=datetime(2024, 1, 1, 9, 30) + timedelta(minutes=i),
                event_type=EventType.BAR_UPDATE,
                data={},
            )
            queue.push(event)

        assert queue.size() == 5

        queue.clear()
        assert queue.is_empty()
        assert queue.size() == 0

    def test_multiple_event_types(self):
        """Test queue handles all event types correctly."""
        queue = EventQueue()
        timestamp = datetime(2024, 1, 1, 9, 30)

        # Test all event types
        event_types = [
            EventType.MARKET_OPEN,
            EventType.MARKET_CLOSE,
            EventType.BAR_UPDATE,
            EventType.SIGNAL,
            EventType.ORDER_SUBMIT,
            EventType.ORDER_FILL,
            EventType.ORDER_CANCEL,
            EventType.POSITION_UPDATE,
            EventType.RISK_CHECK,
            EventType.EOD_SETTLEMENT,
        ]

        events = []
        for i, event_type in enumerate(event_types):
            event = Event(
                timestamp=timestamp + timedelta(seconds=i),
                event_type=event_type,
                data={"index": i},
            )
            events.append(event)
            queue.push(event)

        assert queue.size() == len(event_types)

        # Should pop in chronological order
        for i, expected_event in enumerate(events):
            popped = queue.pop()
            assert popped.event_type == expected_event.event_type
            assert popped.data["index"] == i

    def test_large_queue_performance(self):
        """Test queue can handle large number of events."""
        queue = EventQueue()
        base_time = datetime(2024, 1, 1, 9, 30)

        # Push 1000 events
        num_events = 1000
        for i in range(num_events):
            event = Event(
                timestamp=base_time + timedelta(seconds=i),
                event_type=EventType.BAR_UPDATE,
                data={"index": i},
            )
            queue.push(event)

        assert queue.size() == num_events

        # Verify they come out in order
        for i in range(num_events):
            event = queue.pop()
            assert event.data["index"] == i

        assert queue.is_empty()
