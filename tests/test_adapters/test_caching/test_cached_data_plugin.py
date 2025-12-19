"""Tests for cached data plugin wrapper."""

from datetime import datetime

import pytest

from ordinis.adapters.caching.cache_protocol import CacheConfig
from ordinis.adapters.caching.cached_data_plugin import CachedDataPlugin
from ordinis.adapters.caching.memory_cache import MemoryCache
from ordinis.plugins.base import DataPlugin, PluginConfig, PluginHealth, PluginStatus


class MockDataPlugin(DataPlugin):
    """Mock data plugin for testing."""

    name = "mock_provider"
    version = "1.0.0"

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self.get_quote_calls = 0
        self.get_historical_calls = 0

    async def initialize(self) -> bool:
        await self._set_status(PluginStatus.READY)
        return True

    async def shutdown(self) -> None:
        await self._set_status(PluginStatus.STOPPED)

    async def health_check(self) -> PluginHealth:
        return PluginHealth(
            status=PluginStatus.READY,
            last_check=datetime.utcnow(),
            latency_ms=10.0,
        )

    async def get_quote(self, symbol: str) -> dict:
        self.get_quote_calls += 1
        return {
            "symbol": symbol.upper(),
            "last": 150.0,
            "volume": 1000000,
            "source": "mock",
        }

    async def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[dict]:
        self.get_historical_calls += 1
        return [
            {
                "symbol": symbol.upper(),
                "timestamp": start.isoformat(),
                "open": 149.0,
                "high": 151.0,
                "low": 148.0,
                "close": 150.0,
                "volume": 1000000,
            }
        ]


class TestCachedDataPlugin:
    """Tests for CachedDataPlugin class."""

    @pytest.fixture
    def mock_plugin(self):
        """Create a mock data plugin."""
        config = PluginConfig(name="mock")
        return MockDataPlugin(config)

    @pytest.fixture
    def cache(self):
        """Create a memory cache."""
        return MemoryCache(
            CacheConfig(
                quote_ttl_seconds=5,
                historical_daily_ttl_seconds=60,
                historical_intraday_ttl_seconds=10,
            )
        )

    @pytest.fixture
    def cached_plugin(self, mock_plugin: MockDataPlugin, cache: MemoryCache):
        """Create a cached data plugin."""
        return CachedDataPlugin(
            plugin=mock_plugin,
            cache=cache,
            cache_config=CacheConfig(
                quote_ttl_seconds=5,
                historical_daily_ttl_seconds=60,
                historical_intraday_ttl_seconds=10,
            ),
        )

    def test_initialization(self, cached_plugin: CachedDataPlugin, mock_plugin: MockDataPlugin):
        """Test cached plugin initialization."""
        assert cached_plugin.name == "cached_mock_provider"
        assert cached_plugin.underlying_plugin == mock_plugin
        assert cached_plugin.capabilities == mock_plugin.capabilities

    @pytest.mark.asyncio
    async def test_initialize_delegates(
        self, cached_plugin: CachedDataPlugin, mock_plugin: MockDataPlugin
    ):
        """Test that initialize delegates to underlying plugin."""
        result = await cached_plugin.initialize()
        assert result is True
        assert mock_plugin.status == PluginStatus.READY

    @pytest.mark.asyncio
    async def test_shutdown_delegates(
        self, cached_plugin: CachedDataPlugin, mock_plugin: MockDataPlugin
    ):
        """Test that shutdown delegates to underlying plugin."""
        await cached_plugin.initialize()
        await cached_plugin.shutdown()
        assert mock_plugin.status == PluginStatus.STOPPED

    @pytest.mark.asyncio
    async def test_health_check_delegates(self, cached_plugin: CachedDataPlugin):
        """Test that health_check delegates to underlying plugin."""
        health = await cached_plugin.health_check()
        assert health.status == PluginStatus.READY

    @pytest.mark.asyncio
    async def test_get_quote_caches_result(
        self, cached_plugin: CachedDataPlugin, mock_plugin: MockDataPlugin
    ):
        """Test that get_quote caches results."""
        # First call - cache miss
        result1 = await cached_plugin.get_quote("AAPL")
        assert result1["symbol"] == "AAPL"
        assert mock_plugin.get_quote_calls == 1

        # Second call - cache hit
        result2 = await cached_plugin.get_quote("AAPL")
        assert result2["symbol"] == "AAPL"
        assert mock_plugin.get_quote_calls == 1  # No additional call

    @pytest.mark.asyncio
    async def test_get_quote_different_symbols(
        self, cached_plugin: CachedDataPlugin, mock_plugin: MockDataPlugin
    ):
        """Test that different symbols are cached separately."""
        await cached_plugin.get_quote("AAPL")
        await cached_plugin.get_quote("MSFT")

        assert mock_plugin.get_quote_calls == 2

        # Both should be cached
        await cached_plugin.get_quote("AAPL")
        await cached_plugin.get_quote("MSFT")

        assert mock_plugin.get_quote_calls == 2  # Still 2

    @pytest.mark.asyncio
    async def test_get_historical_caches_result(
        self, cached_plugin: CachedDataPlugin, mock_plugin: MockDataPlugin
    ):
        """Test that get_historical caches results."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        # First call - cache miss
        result1 = await cached_plugin.get_historical("AAPL", start, end, "1d")
        assert len(result1) == 1
        assert mock_plugin.get_historical_calls == 1

        # Second call with same params - cache hit
        result2 = await cached_plugin.get_historical("AAPL", start, end, "1d")
        assert len(result2) == 1
        assert mock_plugin.get_historical_calls == 1

    @pytest.mark.asyncio
    async def test_get_historical_different_params(
        self, cached_plugin: CachedDataPlugin, mock_plugin: MockDataPlugin
    ):
        """Test that different params create different cache keys."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        await cached_plugin.get_historical("AAPL", start, end, "1d")
        await cached_plugin.get_historical("AAPL", start, end, "1h")  # Different timeframe

        assert mock_plugin.get_historical_calls == 2

    @pytest.mark.asyncio
    async def test_get_historical_intraday_ttl(
        self, cached_plugin: CachedDataPlugin, mock_plugin: MockDataPlugin
    ):
        """Test that intraday data uses shorter TTL."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 2)

        # This should use intraday TTL (10 seconds in fixture)
        await cached_plugin.get_historical("AAPL", start, end, "1m")
        assert mock_plugin.get_historical_calls == 1

        # Should still be cached
        await cached_plugin.get_historical("AAPL", start, end, "1m")
        assert mock_plugin.get_historical_calls == 1

    @pytest.mark.asyncio
    async def test_validate_symbol_caches(
        self, cached_plugin: CachedDataPlugin, mock_plugin: MockDataPlugin
    ):
        """Test that validate_symbol caches results."""
        # First call
        result1 = await cached_plugin.validate_symbol("AAPL")
        assert result1 is True
        call_count = mock_plugin.get_quote_calls

        # Second call - should be cached
        result2 = await cached_plugin.validate_symbol("AAPL")
        assert result2 is True
        assert mock_plugin.get_quote_calls == call_count  # No additional call

    @pytest.mark.asyncio
    async def test_invalidate_symbol(
        self, cached_plugin: CachedDataPlugin, mock_plugin: MockDataPlugin, cache: MemoryCache
    ):
        """Test invalidating cached data for a symbol."""
        # Cache some data
        await cached_plugin.get_quote("AAPL")
        await cached_plugin.validate_symbol("AAPL")

        # Verify cached
        assert await cache.exists("mock_provider:quote:AAPL")
        assert await cache.exists("mock_provider:validate:AAPL")

        # Invalidate
        count = await cached_plugin.invalidate_symbol("AAPL")
        assert count >= 2

        # Verify invalidated
        assert not await cache.exists("mock_provider:quote:AAPL")
        assert not await cache.exists("mock_provider:validate:AAPL")

    @pytest.mark.asyncio
    async def test_invalidate_all(
        self, cached_plugin: CachedDataPlugin, mock_plugin: MockDataPlugin
    ):
        """Test invalidating all cached data."""
        # Cache some data
        await cached_plugin.get_quote("AAPL")
        await cached_plugin.get_quote("MSFT")

        # Invalidate all
        count = await cached_plugin.invalidate_all()
        assert count >= 2

        # Verify - next calls should hit underlying plugin
        initial_calls = mock_plugin.get_quote_calls
        await cached_plugin.get_quote("AAPL")
        assert mock_plugin.get_quote_calls == initial_calls + 1

    @pytest.mark.asyncio
    async def test_get_cache_stats(
        self, cached_plugin: CachedDataPlugin, mock_plugin: MockDataPlugin
    ):
        """Test getting cache statistics."""
        # Generate some cache activity
        await cached_plugin.get_quote("AAPL")  # Miss
        await cached_plugin.get_quote("AAPL")  # Hit
        await cached_plugin.get_quote("MSFT")  # Miss

        stats = cached_plugin.get_cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["size"] == 2

    def test_make_key(self, cached_plugin: CachedDataPlugin):
        """Test cache key generation."""
        key = cached_plugin._make_key("quote", "aapl")
        assert key == "mock_provider:quote:AAPL"

    def test_make_historical_key(self, cached_plugin: CachedDataPlugin):
        """Test historical cache key generation."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)

        key1 = cached_plugin._make_historical_key("AAPL", start, end, "1d")
        key2 = cached_plugin._make_historical_key("AAPL", start, end, "1d")

        # Same params should produce same key
        assert key1 == key2

        # Different params should produce different keys
        key3 = cached_plugin._make_historical_key("MSFT", start, end, "1d")
        assert key1 != key3

    def test_get_historical_ttl_daily(self, cached_plugin: CachedDataPlugin):
        """Test TTL selection for daily timeframe."""
        ttl = cached_plugin._get_historical_ttl("1d")
        assert ttl == 60  # From fixture config

    def test_get_historical_ttl_intraday(self, cached_plugin: CachedDataPlugin):
        """Test TTL selection for intraday timeframes."""
        for tf in ["1m", "5m", "15m", "30m", "1h", "4h"]:
            ttl = cached_plugin._get_historical_ttl(tf)
            assert ttl == 10  # From fixture config

    @pytest.mark.asyncio
    async def test_symbol_case_insensitive(
        self, cached_plugin: CachedDataPlugin, mock_plugin: MockDataPlugin
    ):
        """Test that symbols are normalized to uppercase."""
        await cached_plugin.get_quote("aapl")
        await cached_plugin.get_quote("AAPL")
        await cached_plugin.get_quote("Aapl")

        # All should use same cache entry
        assert mock_plugin.get_quote_calls == 1
