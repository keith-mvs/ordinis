"""
StreamingBus configuration.

Defines configuration for event bus settings.
"""

from dataclasses import dataclass
from enum import Enum


class AdapterType(Enum):
    """Event bus adapter backend."""

    MEMORY = "memory"  # In-memory (testing, single-process)
    REDIS = "redis"  # Redis Streams (production)


@dataclass
class BusConfig:
    """Configuration for StreamingBus."""

    # Adapter selection
    adapter: AdapterType = AdapterType.MEMORY

    # Redis settings (when adapter=REDIS)
    redis_url: str = "redis://localhost:6379"
    redis_stream_prefix: str = "ordinis:"
    redis_max_len: int = 10000  # Max events per stream

    # Event settings
    max_payload_size: int = 1024 * 1024  # 1MB max payload
    default_ttl_seconds: int = 3600  # Event TTL

    # Processing
    max_concurrent_handlers: int = 10
    handler_timeout_seconds: float = 30.0
    retry_failed_handlers: bool = True
    max_handler_retries: int = 3

    # Batching
    batch_size: int = 100
    batch_timeout_ms: int = 100

    # Replay / history
    enable_history: bool = True
    history_max_events: int = 10000

    # Metrics
    emit_metrics: bool = True
    metrics_interval_seconds: int = 60

    def validate(self) -> list[str]:
        """Validate configuration, returning list of errors."""
        errors: list[str] = []

        if self.max_payload_size <= 0:
            errors.append("max_payload_size must be > 0")

        if self.handler_timeout_seconds <= 0:
            errors.append("handler_timeout_seconds must be > 0")

        if self.adapter == AdapterType.REDIS and not self.redis_url:
            errors.append("redis_url required when adapter=REDIS")

        return errors
