"""Tests for memory cache implementation."""

import asyncio
import time

import pytest

from ordinis.adapters.caching.cache_protocol import CacheConfig
from ordinis.adapters.caching.memory_cache import CacheEntry, MemoryCache


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CacheConfig()
        assert config.quote_ttl_seconds == 5
        assert config.historical_daily_ttl_seconds == 3600
        assert config.historical_intraday_ttl_seconds == 300
        assert config.company_info_ttl_seconds == 86400
        assert config.news_ttl_seconds == 900
        assert config.max_entries == 10000
        assert config.enabled is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CacheConfig(
            quote_ttl_seconds=10,
            max_entries=100,
            enabled=False,
        )
        assert config.quote_ttl_seconds == 10
        assert config.max_entries == 100
        assert config.enabled is False

    def test_get_ttl_for_data_type(self):
        """Test TTL lookup by data type."""
        config = CacheConfig(
            quote_ttl_seconds=5,
            historical_daily_ttl_seconds=3600,
            historical_intraday_ttl_seconds=300,
        )
        assert config.get_ttl_for_data_type("quote") == 5
        assert config.get_ttl_for_data_type("historical_daily") == 3600
        assert config.get_ttl_for_data_type("historical_intraday") == 300
        # Unknown type returns quote TTL
        assert config.get_ttl_for_data_type("unknown") == 5


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_creation(self):
        """Test cache entry creation."""
        now = time.time()
        entry = CacheEntry(value={"price": 100}, expires_at=now + 60, created_at=now)
        assert entry.value == {"price": 100}
        assert entry.expires_at > now
        assert entry.created_at == now


class TestMemoryCache:
    """Tests for MemoryCache class."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache for each test."""
        return MemoryCache(CacheConfig(max_entries=100))

    @pytest.fixture
    def disabled_cache(self):
        """Create a disabled cache."""
        return MemoryCache(CacheConfig(enabled=False))

    @pytest.mark.asyncio
    async def test_set_and_get(self, cache: MemoryCache):
        """Test basic set and get operations."""
        await cache.set("key1", "value1", ttl_seconds=60)
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_missing_key(self, cache: MemoryCache):
        """Test get on missing key returns None."""
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, cache: MemoryCache):
        """Test that entries expire after TTL."""
        await cache.set("key1", "value1", ttl_seconds=1)
        assert await cache.get("key1") == "value1"

        # Wait for expiration
        await asyncio.sleep(1.1)
        result = await cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, cache: MemoryCache):
        """Test delete operation."""
        await cache.set("key1", "value1", ttl_seconds=60)
        assert await cache.delete("key1") is True
        assert await cache.get("key1") is None
        assert await cache.delete("key1") is False

    @pytest.mark.asyncio
    async def test_exists(self, cache: MemoryCache):
        """Test exists check."""
        assert await cache.exists("key1") is False
        await cache.set("key1", "value1", ttl_seconds=60)
        assert await cache.exists("key1") is True

    @pytest.mark.asyncio
    async def test_exists_expired(self, cache: MemoryCache):
        """Test exists returns False for expired entries."""
        await cache.set("key1", "value1", ttl_seconds=1)
        await asyncio.sleep(1.1)
        assert await cache.exists("key1") is False

    @pytest.mark.asyncio
    async def test_clear(self, cache: MemoryCache):
        """Test clear all entries."""
        await cache.set("key1", "value1", ttl_seconds=60)
        await cache.set("key2", "value2", ttl_seconds=60)
        await cache.set("key3", "value3", ttl_seconds=60)

        count = await cache.clear()
        assert count == 3
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
        assert await cache.get("key3") is None

    @pytest.mark.asyncio
    async def test_clear_pattern(self, cache: MemoryCache):
        """Test clear entries by pattern."""
        await cache.set("quote:AAPL", "data1", ttl_seconds=60)
        await cache.set("quote:MSFT", "data2", ttl_seconds=60)
        await cache.set("historical:AAPL", "data3", ttl_seconds=60)

        count = await cache.clear_pattern("quote:*")
        assert count == 2
        assert await cache.get("quote:AAPL") is None
        assert await cache.get("quote:MSFT") is None
        assert await cache.get("historical:AAPL") == "data3"

    @pytest.mark.asyncio
    async def test_clear_pattern_symbol(self, cache: MemoryCache):
        """Test clear entries by symbol pattern."""
        await cache.set("quote:AAPL", "data1", ttl_seconds=60)
        await cache.set("historical:AAPL:1d", "data2", ttl_seconds=60)
        await cache.set("quote:MSFT", "data3", ttl_seconds=60)

        count = await cache.clear_pattern("*:AAPL*")
        assert count == 2
        assert await cache.get("quote:MSFT") == "data3"

    @pytest.mark.asyncio
    async def test_clear_expired(self, cache: MemoryCache):
        """Test removing expired entries."""
        await cache.set("key1", "value1", ttl_seconds=1)
        await cache.set("key2", "value2", ttl_seconds=60)

        await asyncio.sleep(1.1)
        count = await cache.clear_expired()
        assert count == 1
        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_max_entries_eviction(self):
        """Test LRU eviction when at capacity."""
        cache = MemoryCache(CacheConfig(max_entries=3))

        await cache.set("key1", "value1", ttl_seconds=60)
        await asyncio.sleep(0.01)
        await cache.set("key2", "value2", ttl_seconds=60)
        await asyncio.sleep(0.01)
        await cache.set("key3", "value3", ttl_seconds=60)
        await asyncio.sleep(0.01)

        # Adding 4th key should evict oldest (key1)
        await cache.set("key4", "value4", ttl_seconds=60)

        assert await cache.get("key1") is None  # Evicted
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

    @pytest.mark.asyncio
    async def test_update_existing_key(self, cache: MemoryCache):
        """Test updating an existing key doesn't trigger eviction."""
        await cache.set("key1", "value1", ttl_seconds=60)
        await cache.set("key1", "value1_updated", ttl_seconds=60)

        result = await cache.get("key1")
        assert result == "value1_updated"

    @pytest.mark.asyncio
    async def test_stats(self, cache: MemoryCache):
        """Test statistics tracking."""
        await cache.set("key1", "value1", ttl_seconds=60)

        # Hit
        await cache.get("key1")
        # Miss
        await cache.get("key2")

        stats = cache.stats()
        assert stats["size"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["enabled"] is True

    @pytest.mark.asyncio
    async def test_disabled_cache_set(self, disabled_cache: MemoryCache):
        """Test that disabled cache doesn't store values."""
        await disabled_cache.set("key1", "value1", ttl_seconds=60)
        result = await disabled_cache.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_disabled_cache_exists(self, disabled_cache: MemoryCache):
        """Test that disabled cache always returns False for exists."""
        await disabled_cache.set("key1", "value1", ttl_seconds=60)
        assert await disabled_cache.exists("key1") is False

    @pytest.mark.asyncio
    async def test_get_many(self, cache: MemoryCache):
        """Test bulk get operation."""
        await cache.set("key1", "value1", ttl_seconds=60)
        await cache.set("key2", "value2", ttl_seconds=60)
        await cache.set("key3", "value3", ttl_seconds=60)

        results = await cache.get_many(["key1", "key2", "key4"])
        assert results == {"key1": "value1", "key2": "value2"}

    @pytest.mark.asyncio
    async def test_set_many(self, cache: MemoryCache):
        """Test bulk set operation."""
        items = {"key1": "value1", "key2": "value2", "key3": "value3"}
        await cache.set_many(items, ttl_seconds=60)

        assert await cache.get("key1") == "value1"
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"

    @pytest.mark.asyncio
    async def test_complex_values(self, cache: MemoryCache):
        """Test caching complex data structures."""
        complex_value = {
            "symbol": "AAPL",
            "price": 150.25,
            "nested": {"bid": 150.20, "ask": 150.30},
            "array": [1, 2, 3],
        }

        await cache.set("quote:AAPL", complex_value, ttl_seconds=60)
        result = await cache.get("quote:AAPL")
        assert result == complex_value

    @pytest.mark.asyncio
    async def test_concurrent_access(self, cache: MemoryCache):
        """Test thread-safe concurrent access."""

        async def writer(key: str, value: str):
            await cache.set(key, value, ttl_seconds=60)

        async def reader(key: str):
            return await cache.get(key)

        # Run concurrent writes
        tasks = [writer(f"key{i}", f"value{i}") for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify all writes succeeded
        for i in range(10):
            result = await reader(f"key{i}")
            assert result == f"value{i}"
