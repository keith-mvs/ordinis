"""Base configuration for all Ordinis engines.

This module defines the standard configuration structure that
all engines must implement for consistent configuration management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BaseEngineConfig:
    """Base configuration shared by all engines.

    All engine-specific configs should inherit from this class
    to ensure consistent configuration patterns.

    Attributes:
        enabled: Whether the engine is enabled.
        name: Engine instance name (for logging/metrics).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        metrics_enabled: Enable metrics collection.
        health_check_interval_seconds: Interval between health checks.
        timeout_seconds: Default operation timeout.
        retry_attempts: Number of retry attempts for recoverable errors.
        retry_delay_seconds: Delay between retries.
        governance_enabled: Enable governance hooks.
        audit_enabled: Enable audit logging.
    """

    enabled: bool = True
    name: str = ""
    log_level: str = "INFO"
    metrics_enabled: bool = True
    health_check_interval_seconds: float = 30.0
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    governance_enabled: bool = True
    audit_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.name:
            self.name = self.__class__.__name__.replace("Config", "")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "name": self.name,
            "log_level": self.log_level,
            "metrics_enabled": self.metrics_enabled,
            "health_check_interval_seconds": self.health_check_interval_seconds,
            "timeout_seconds": self.timeout_seconds,
            "retry_attempts": self.retry_attempts,
            "retry_delay_seconds": self.retry_delay_seconds,
            "governance_enabled": self.governance_enabled,
            "audit_enabled": self.audit_enabled,
        }


@dataclass
class AIEngineConfig(BaseEngineConfig):
    """Configuration for AI-enabled engines.

    Extends base config with AI/LLM-specific settings.

    Attributes:
        model_name: Default model to use.
        fallback_model: Fallback model if primary unavailable.
        max_tokens: Maximum tokens for generation.
        temperature: Model temperature (0.0-1.0).
        api_timeout_seconds: API call timeout.
        cache_enabled: Enable response caching.
        cache_ttl_seconds: Cache time-to-live.
    """

    model_name: str = ""
    fallback_model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    api_timeout_seconds: float = 60.0
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300


@dataclass
class DataEngineConfig(BaseEngineConfig):
    """Configuration for data-processing engines.

    Extends base config with data-specific settings.

    Attributes:
        batch_size: Default batch size for processing.
        buffer_size: Internal buffer size.
        checkpoint_enabled: Enable checkpointing.
        checkpoint_path: Path for checkpoint files.
        parallel_workers: Number of parallel workers.
    """

    batch_size: int = 1000
    buffer_size: int = 10000
    checkpoint_enabled: bool = True
    checkpoint_path: Path = field(default_factory=lambda: Path("data/checkpoints"))
    parallel_workers: int = 4


@dataclass
class TradingEngineConfig(BaseEngineConfig):
    """Configuration for trading-related engines.

    Extends base config with trading-specific settings.

    Attributes:
        paper_mode: Run in paper trading mode.
        max_position_size: Maximum position size.
        daily_loss_limit: Daily loss limit in dollars.
        max_drawdown_pct: Maximum drawdown percentage.
        symbols: List of symbols to trade.
    """

    paper_mode: bool = True
    max_position_size: float = 10000.0
    daily_loss_limit: float = 1000.0
    max_drawdown_pct: float = 5.0
    symbols: list[str] = field(default_factory=list)
