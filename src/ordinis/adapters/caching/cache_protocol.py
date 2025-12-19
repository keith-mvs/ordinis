"""
Cache protocol and configuration for market data caching.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class CacheConfig:
    """Configuration for cache behavior.

    Attributes:
        quote_ttl_seconds: TTL for real-time quotes (default 5s).
        historical_daily_ttl_seconds: TTL for daily historical data (default 1h).
        historical_intraday_ttl_seconds: TTL for intraday data (default 5m).
        company_info_ttl_seconds: TTL for company info (default 24h).
        news_ttl_seconds: TTL for news data (default 15m).
        max_entries: Maximum cache entries (0 = unlimited).
        enabled: Whether caching is enabled.
    """

    quote_ttl_seconds: int = 5
    historical_daily_ttl_seconds: int = 3600
    historical_intraday_ttl_seconds: int = 300
    company_info_ttl_seconds: int = 86400
    news_ttl_seconds: int = 900
    max_entries: int = 10000
    enabled: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def get_ttl_for_data_type(self, data_type: str) -> int:
        """Get TTL in seconds for a given data type.

        Args:
            data_type: One of 'quote', 'historical_daily', 'historical_intraday',
                      'company_info', 'news'.

        Returns:
            TTL in seconds.
        """
        ttl_map = {
            "quote": self.quote_ttl_seconds,
            "historical_daily": self.historical_daily_ttl_seconds,
            "historical_intraday": self.historical_intraday_ttl_seconds,
            "company_info": self.company_info_ttl_seconds,
            "news": self.news_ttl_seconds,
        }
        return ttl_map.get(data_type, self.quote_ttl_seconds)


class CacheProtocol(Protocol):
    """Protocol for cache implementations.

    Defines the interface that all cache backends must implement.
    """

    async def get(self, key: str) -> Any | None:
        """Get a value from cache.

        Args:
            key: The cache key.

        Returns:
            The cached value or None if not found/expired.
        """
        ...

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Set a value in cache with TTL.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl_seconds: Time to live in seconds.
        """
        ...

    async def delete(self, key: str) -> bool:
        """Delete a key from cache.

        Args:
            key: The cache key.

        Returns:
            True if key was deleted, False if key didn't exist.
        """
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache.

        Args:
            key: The cache key.

        Returns:
            True if key exists and is not expired.
        """
        ...

    async def clear(self) -> int:
        """Clear all entries from cache.

        Returns:
            Number of entries cleared.
        """
        ...

    async def clear_pattern(self, pattern: str) -> int:
        """Clear entries matching a pattern.

        Args:
            pattern: Glob-style pattern (e.g., 'quote:*', '*:AAPL').

        Returns:
            Number of entries cleared.
        """
        ...

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, size, etc.
        """
        ...
