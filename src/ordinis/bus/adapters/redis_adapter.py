"""
Redis Streams adapter for StreamingBus.

Writes events to Redis Streams for durability and cross-process fan-out.
Designed to be used by StreamingBus when AdapterType.REDIS is selected.
"""

from __future__ import annotations

import json
import logging
from typing import Any

try:
    import redis.asyncio as redis
except Exception as exc:  # pragma: no cover - handled at runtime
    redis = None  # type: ignore
    _import_error = exc
else:
    _import_error = None

from ordinis.bus.config import BusConfig
from ordinis.bus.models import BusEvent

_logger = logging.getLogger(__name__)


class RedisAdapter:
    """Redis Streams adapter.

    Uses one stream per event type: `<prefix><event_type>`.
    Events are serialized as JSON in field `event`.
    """

    def __init__(self, config: BusConfig):
        if redis is None:
            raise ImportError(
                "redis-py with asyncio support is required for Redis adapter."
            ) from _import_error

        self.config = config
        self._client: Any = config.redis_client or redis.from_url(config.redis_url)
        self._stream_prefix = config.redis_stream_prefix
        self._max_len = config.redis_max_len

    async def write_event(self, event: BusEvent) -> None:
        """Append event to its Redis stream."""
        key = f"{self._stream_prefix}{event.event_type}"
        payload = json.dumps(event.to_dict())
        try:
            await self._client.xadd(
                key,
                {"event": payload},
                maxlen=self._max_len,
                approximate=True,
            )
        except Exception:
            _logger.exception("Failed to write event to Redis stream: %s", key)
            raise

    async def close(self) -> None:
        """Close Redis client."""
        try:
            await self._client.aclose()
        except AttributeError:
            # fakeredis exposes close/close instead of aclose
            if hasattr(self._client, "close"):
                await self._client.close()
        except Exception:
            _logger.warning("Error closing Redis client", exc_info=True)
