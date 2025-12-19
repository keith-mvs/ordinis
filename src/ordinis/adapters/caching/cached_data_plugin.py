"""
Cached data plugin wrapper that adds caching to any DataPlugin.
"""

from datetime import datetime
import hashlib
import json
import logging
from typing import Any

from ordinis.adapters.caching.cache_protocol import CacheConfig, CacheProtocol
from ordinis.plugins.base import DataPlugin, PluginHealth

logger = logging.getLogger(__name__)


class CachedDataPlugin(DataPlugin):
    """Wrapper that adds caching to any DataPlugin.

    Uses the decorator pattern to transparently cache responses from
    the underlying data provider, reducing API calls and latency.

    Example:
        from ordinis.adapters.caching import CachedDataPlugin, MemoryCache, CacheConfig
        from ordinis.adapters.market_data import AlphaVantageDataPlugin

        # Create the underlying plugin
        av_plugin = AlphaVantageDataPlugin(PluginConfig(name="av", api_key="..."))

        # Wrap with caching
        cache = MemoryCache(CacheConfig(quote_ttl_seconds=10))
        cached_plugin = CachedDataPlugin(
            plugin=av_plugin,
            cache=cache,
            cache_config=CacheConfig()
        )

        # Use as normal - responses are cached
        quote = await cached_plugin.get_quote("AAPL")
    """

    def __init__(
        self,
        plugin: DataPlugin,
        cache: CacheProtocol,
        cache_config: CacheConfig | None = None,
    ):
        """Initialize the cached data plugin.

        Args:
            plugin: The underlying DataPlugin to wrap.
            cache: Cache implementation to use.
            cache_config: Cache configuration. Uses defaults if not provided.
        """
        # Initialize parent with the wrapped plugin's config
        super().__init__(plugin.config)

        self._plugin = plugin
        self._cache = cache
        self._cache_config = cache_config or CacheConfig()

        # Copy attributes from wrapped plugin
        self.name = f"cached_{plugin.name}"
        self.version = plugin.version
        self.description = f"Cached wrapper for {plugin.name}"
        self.capabilities = plugin.capabilities

    @property
    def underlying_plugin(self) -> DataPlugin:
        """Get the underlying unwrapped plugin."""
        return self._plugin

    async def initialize(self) -> bool:
        """Initialize the underlying plugin."""
        return await self._plugin.initialize()

    async def shutdown(self) -> None:
        """Shutdown the underlying plugin."""
        await self._plugin.shutdown()

    async def health_check(self) -> PluginHealth:
        """Check health of underlying plugin."""
        return await self._plugin.health_check()

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get quote with caching.

        Args:
            symbol: The ticker symbol.

        Returns:
            Quote data dictionary.
        """
        cache_key = self._make_key("quote", symbol)

        # Try cache first
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for quote: {symbol}")
            return cached

        # Fetch from underlying plugin
        logger.debug(f"Cache miss for quote: {symbol}")
        result = await self._plugin.get_quote(symbol)

        # Cache the result
        ttl = self._cache_config.quote_ttl_seconds
        await self._cache.set(cache_key, result, ttl)

        return result

    async def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[dict[str, Any]]:
        """Get historical data with caching.

        Args:
            symbol: The ticker symbol.
            start: Start datetime.
            end: End datetime.
            timeframe: Bar timeframe (1m, 5m, 1h, 1d, etc.).

        Returns:
            List of OHLCV bar dictionaries.
        """
        cache_key = self._make_historical_key(symbol, start, end, timeframe)

        # Try cache first
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for historical: {symbol} {timeframe}")
            return cached

        # Fetch from underlying plugin
        logger.debug(f"Cache miss for historical: {symbol} {timeframe}")
        result = await self._plugin.get_historical(symbol, start, end, timeframe)

        # Determine TTL based on timeframe
        ttl = self._get_historical_ttl(timeframe)
        await self._cache.set(cache_key, result, ttl)

        return result

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol with caching.

        Args:
            symbol: The ticker symbol.

        Returns:
            True if symbol is valid.
        """
        cache_key = self._make_key("validate", symbol)

        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = await self._plugin.validate_symbol(symbol)

        # Cache validation results for longer (company info TTL)
        ttl = self._cache_config.company_info_ttl_seconds
        await self._cache.set(cache_key, result, ttl)

        return result

    async def invalidate_symbol(self, symbol: str) -> int:
        """Invalidate all cached data for a symbol.

        Args:
            symbol: The ticker symbol.

        Returns:
            Number of cache entries cleared.
        """
        pattern = f"*:{symbol}:*"
        count = await self._cache.clear_pattern(pattern)

        # Also clear simple patterns
        for data_type in ["quote", "validate"]:
            key = self._make_key(data_type, symbol)
            if await self._cache.delete(key):
                count += 1

        logger.info(f"Invalidated {count} cache entries for {symbol}")
        return count

    async def invalidate_all(self) -> int:
        """Invalidate all cached data.

        Returns:
            Number of cache entries cleared.
        """
        count = await self._cache.clear()
        logger.info(f"Invalidated all {count} cache entries")
        return count

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, hit_rate, etc.).
        """
        return self._cache.stats()

    def _make_key(self, data_type: str, symbol: str) -> str:
        """Create a cache key for simple lookups.

        Args:
            data_type: Type of data (quote, validate, etc.).
            symbol: The ticker symbol.

        Returns:
            Cache key string.
        """
        return f"{self._plugin.name}:{data_type}:{symbol.upper()}"

    def _make_historical_key(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
    ) -> str:
        """Create a cache key for historical data.

        Args:
            symbol: The ticker symbol.
            start: Start datetime.
            end: End datetime.
            timeframe: Bar timeframe.

        Returns:
            Cache key string.
        """
        # Create a deterministic hash of the parameters
        params = {
            "symbol": symbol.upper(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timeframe": timeframe,
        }
        param_str = json.dumps(params, sort_keys=True)
        # Using sha256 for cache key (not security-sensitive, just needs to be deterministic)
        param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:12]

        return f"{self._plugin.name}:historical:{symbol.upper()}:{timeframe}:{param_hash}"

    def _get_historical_ttl(self, timeframe: str) -> int:
        """Get TTL for historical data based on timeframe.

        Args:
            timeframe: Bar timeframe (1m, 5m, 1h, 1d, etc.).

        Returns:
            TTL in seconds.
        """
        # Intraday timeframes get shorter TTL
        intraday_timeframes = {"1m", "5m", "15m", "30m", "1h", "4h"}

        if timeframe.lower() in intraday_timeframes:
            return self._cache_config.historical_intraday_ttl_seconds

        return self._cache_config.historical_daily_ttl_seconds
