"""
In-memory cache implementation with TTL support.
"""

import asyncio
from dataclasses import dataclass
import fnmatch
import logging
import time
from typing import Any

from ordinis.adapters.caching.cache_protocol import CacheConfig

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with expiration tracking."""

    value: Any
    expires_at: float
    created_at: float


class MemoryCache:
    """Thread-safe in-memory cache with TTL and LRU eviction.

    Provides a simple but efficient caching layer for market data.
    Supports automatic expiration and pattern-based invalidation.

    Example:
        cache = MemoryCache(config=CacheConfig(max_entries=1000))
        await cache.set("quote:AAPL", {"price": 150.0}, ttl_seconds=5)
        data = await cache.get("quote:AAPL")
    """

    def __init__(self, config: CacheConfig | None = None):
        """Initialize the memory cache.

        Args:
            config: Cache configuration. Uses defaults if not provided.
        """
        self.config = config or CacheConfig()
        self._cache: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    async def get(self, key: str) -> Any | None:
        """Get a value from cache.

        Args:
            key: The cache key.

        Returns:
            The cached value or None if not found/expired.
        """
        if not self.config.enabled:
            return None

        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            # Check expiration
            if time.time() > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                return None

            self._hits += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """Set a value in cache with TTL.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl_seconds: Time to live in seconds.
        """
        if not self.config.enabled:
            return

        async with self._lock:
            # Evict if at capacity
            if (
                self.config.max_entries > 0
                and len(self._cache) >= self.config.max_entries
                and key not in self._cache
            ):
                await self._evict_oldest_unlocked()

            now = time.time()
            self._cache[key] = CacheEntry(
                value=value,
                expires_at=now + ttl_seconds,
                created_at=now,
            )

    async def delete(self, key: str) -> bool:
        """Delete a key from cache.

        Args:
            key: The cache key.

        Returns:
            True if key was deleted, False if key didn't exist.
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache.

        Args:
            key: The cache key.

        Returns:
            True if key exists and is not expired.
        """
        if not self.config.enabled:
            return False

        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            if time.time() > entry.expires_at:
                del self._cache[key]
                return False
            return True

    async def clear(self) -> int:
        """Clear all entries from cache.

        Returns:
            Number of entries cleared.
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    async def clear_pattern(self, pattern: str) -> int:
        """Clear entries matching a pattern.

        Args:
            pattern: Glob-style pattern (e.g., 'quote:*', '*:AAPL').

        Returns:
            Number of entries cleared.
        """
        async with self._lock:
            keys_to_delete = [key for key in self._cache if fnmatch.fnmatch(key, pattern)]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

    async def clear_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            now = time.time()
            keys_to_delete = [key for key, entry in self._cache.items() if now > entry.expires_at]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)

    def stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with hits, misses, size, hit_rate, etc.
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_entries": self.config.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
            "enabled": self.config.enabled,
        }

    async def _evict_oldest_unlocked(self) -> None:
        """Evict the oldest entry. Must be called with lock held."""
        if not self._cache:
            return

        # Find oldest entry (by creation time)
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]
        self._evictions += 1
        logger.debug(f"Cache evicted key: {oldest_key}")

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache.

        Args:
            keys: List of cache keys.

        Returns:
            Dictionary of key -> value for found keys.
        """
        results = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        return results

    async def set_many(self, items: dict[str, Any], ttl_seconds: int) -> None:
        """Set multiple values in cache.

        Args:
            items: Dictionary of key -> value pairs.
            ttl_seconds: TTL for all entries.
        """
        for key, value in items.items():
            await self.set(key, value, ttl_seconds)
