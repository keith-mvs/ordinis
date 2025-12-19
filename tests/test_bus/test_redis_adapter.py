"""Tests for Redis-backed StreamingBus adapter and governance hooks."""

import json

import pytest

from ordinis.bus import BusConfig, EventType, StreamingBus
from ordinis.bus.config import AdapterType


@pytest.mark.asyncio
async def test_publish_writes_to_redis_stream(streaming_bus_redis, sample_event, redis_client):
    """Publish should persist event to Redis Streams with prefixed key."""
    await streaming_bus_redis.publish(sample_event)

    entries = await redis_client.xrange("test:bus:signal", count=10)
    assert len(entries) == 1
    _, data = entries[0]
    payload = json.loads(data[b"event"])

    assert payload["event_type"] == EventType.SIGNAL.value
    assert payload["payload"]["direction"] == "buy"


@pytest.mark.asyncio
async def test_schema_validator_blocks_publish(redis_client, sample_event):
    """Schema validator failure should drop the event and avoid Redis writes."""

    def schema_validator(event):
        raise ValueError("invalid schema")

    config = BusConfig(
        adapter=AdapterType.REDIS,
        redis_client=redis_client,
        redis_stream_prefix="test:bus:",
        schema_validator=schema_validator,
    )
    bus = StreamingBus(config)
    try:
        await bus.publish(sample_event)

        metrics = bus.get_metrics()
        assert metrics["events_published"] == 0
        assert metrics["events_dropped"] == 1

        entries = await redis_client.xrange("test:bus:signal", count=1)
        assert entries == []
    finally:
        await bus.shutdown()


@pytest.mark.asyncio
async def test_publish_governance_denies(redis_client, sample_event):
    """Publish governance hook returning False should drop the event and skip Redis."""

    def deny_publish(event):
        return False

    config = BusConfig(
        adapter=AdapterType.REDIS,
        redis_client=redis_client,
        redis_stream_prefix="test:bus:",
        publish_governance_hook=deny_publish,
    )
    bus = StreamingBus(config)
    try:
        await bus.publish(sample_event)

        metrics = bus.get_metrics()
        assert metrics["events_published"] == 0
        assert metrics["events_dropped"] == 1

        entries = await redis_client.xrange("test:bus:signal", count=1)
        assert entries == []
    finally:
        await bus.shutdown()


def test_subscribe_governance_denies(redis_client, event_collector):
    """Subscribe governance hook returning False should raise PermissionError."""

    def deny_subscribe(topic, subscription):
        return False

    config = BusConfig(
        adapter=AdapterType.REDIS,
        redis_client=redis_client,
        redis_stream_prefix="test:bus:",
        subscribe_governance_hook=deny_subscribe,
    )
    bus = StreamingBus(config)
    try:
        with pytest.raises(PermissionError):
            bus.subscribe(EventType.SIGNAL, event_collector.handler)
    finally:
        # No async resources allocated before failure, but keep API symmetry.
        # The Redis client still needs closing via shutdown.
        import asyncio

        asyncio.get_event_loop().run_until_complete(bus.shutdown())
