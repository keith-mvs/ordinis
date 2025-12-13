"""
Caching adapters for market data.

Provides caching layer to reduce API calls and improve latency.
"""

from ordinis.adapters.caching.cache_protocol import CacheConfig, CacheProtocol
from ordinis.adapters.caching.cached_data_plugin import CachedDataPlugin
from ordinis.adapters.caching.memory_cache import MemoryCache

__all__ = [
    "CacheConfig",
    "CacheProtocol",
    "CachedDataPlugin",
    "MemoryCache",
]
