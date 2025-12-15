"""Shared fixtures for StreamingBus tests."""

import asyncio
from collections.abc import Callable

import fakeredis.aioredis as fakeredis
import pytest

from ordinis.bus import BusConfig, BusEvent, EventPriority, EventType, StreamingBus
from ordinis.bus.config import AdapterType


@pytest.fixture
def bus_config() -> BusConfig:
    """Default BusConfig for testing."""
    return BusConfig(
        adapter=AdapterType.MEMORY,
        max_payload_size=1024 * 100,  # 100KB for tests
        handler_timeout_seconds=5.0,
        retry_failed_handlers=True,
        max_handler_retries=2,
        enable_history=True,
        history_max_events=100,
        max_concurrent_handlers=5,
        emit_metrics=True,
    )


@pytest.fixture
def bus_config_no_history() -> BusConfig:
    """BusConfig with history disabled."""
    return BusConfig(
        enable_history=False,
    )


@pytest.fixture
def bus_config_no_retry() -> BusConfig:
    """BusConfig with retry disabled."""
    return BusConfig(
        retry_failed_handlers=False,
    )


@pytest.fixture
async def redis_client():
    """In-memory Redis client for tests."""
    client = fakeredis.FakeRedis()
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def bus_config_redis(redis_client) -> BusConfig:
    """BusConfig pointing to in-memory Redis."""
    return BusConfig(
        adapter=AdapterType.REDIS,
        redis_client=redis_client,
        redis_stream_prefix="test:bus:",
    )


@pytest.fixture
async def streaming_bus(bus_config: BusConfig) -> StreamingBus:
    """StreamingBus instance for testing."""
    bus = StreamingBus(bus_config)
    yield bus
    await bus.shutdown()


@pytest.fixture
async def streaming_bus_no_history(bus_config_no_history: BusConfig) -> StreamingBus:
    """StreamingBus instance with history disabled."""
    bus = StreamingBus(bus_config_no_history)
    yield bus
    await bus.shutdown()


@pytest.fixture
async def streaming_bus_no_retry(bus_config_no_retry: BusConfig) -> StreamingBus:
    """StreamingBus instance with retry disabled."""
    bus = StreamingBus(bus_config_no_retry)
    yield bus
    await bus.shutdown()


@pytest.fixture
async def streaming_bus_redis(bus_config_redis: BusConfig) -> StreamingBus:
    """StreamingBus instance using Redis adapter (fakeredis)."""
    bus = StreamingBus(bus_config_redis)
    yield bus
    await bus.shutdown()


@pytest.fixture
def sample_event() -> BusEvent:
    """Create a sample BusEvent for testing."""
    return BusEvent(
        event_type=EventType.SIGNAL,
        symbol="AAPL",
        source="test-source",
        payload={"direction": "buy", "confidence": 0.85},
        priority=EventPriority.NORMAL,
        trace_id="test-trace-123",
    )


@pytest.fixture
def create_event() -> Callable:
    """Factory fixture to create custom BusEvents."""

    def _create(
        event_type: str | EventType = EventType.SIGNAL,
        symbol: str | None = "AAPL",
        source: str = "test-source",
        payload: dict | None = None,
        priority: EventPriority = EventPriority.NORMAL,
    ) -> BusEvent:
        return BusEvent(
            event_type=event_type,
            symbol=symbol,
            source=source,
            payload=payload or {"test": "data"},
            priority=priority,
        )

    return _create


@pytest.fixture
def event_collector():
    """Fixture that returns a handler collecting events into a list."""

    class EventCollector:
        def __init__(self):
            self.events: list[BusEvent] = []
            self.call_count = 0
            self.error_on_call: int | None = None

        async def handler(self, event: BusEvent) -> None:
            """Handler that collects events."""
            self.call_count += 1
            if self.error_on_call is not None and self.call_count == self.error_on_call:
                raise ValueError("Simulated handler error")
            self.events.append(event)

        def reset(self) -> None:
            """Reset collector state."""
            self.events.clear()
            self.call_count = 0
            self.error_on_call = None

    return EventCollector()


@pytest.fixture
def slow_handler():
    """Fixture that returns a handler that sleeps."""

    class SlowHandler:
        def __init__(self, delay_seconds: float = 1.0):
            self.delay_seconds = delay_seconds
            self.call_count = 0

        async def handler(self, event: BusEvent) -> None:
            """Handler that sleeps before processing."""
            self.call_count += 1
            await asyncio.sleep(self.delay_seconds)

    return SlowHandler


@pytest.fixture
def failing_handler():
    """Fixture that returns a handler that always fails."""

    class FailingHandler:
        def __init__(self):
            self.call_count = 0

        async def handler(self, event: BusEvent) -> None:
            """Handler that always raises an exception."""
            self.call_count += 1
            raise RuntimeError(f"Handler failure #{self.call_count}")

    return FailingHandler()
