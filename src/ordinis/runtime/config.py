"""
Configuration management for Ordinis trading system.

Uses Pydantic BaseSettings for typed configuration with support for:
- YAML config files (configs/default.yaml, configs/environments/*.yaml)
- Environment variable overrides with ORDINIS_ prefix
- Validation and type coercion
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml  # type: ignore[import-untyped]


class SystemConfig(BaseModel):
    """System identification configuration."""

    name: str = "ordinis"
    version: str = "1.0.0"
    environment: Literal["dev", "test", "prod"] = "dev"


class DatabaseConfig(BaseModel):
    """Database configuration."""

    path: str = "data/ordinis.db"
    backup_dir: str = "data/backups"
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    busy_timeout_ms: int = 5000
    auto_backup_on_start: bool = True


class KillSwitchConfig(BaseModel):
    """Kill switch configuration."""

    file_path: str = "data/KILL_SWITCH"
    check_interval_seconds: float = 1.0
    daily_loss_limit: float = 1000.0
    max_drawdown_pct: float = 5.0
    consecutive_loss_limit: int = 5
    persist_state: bool = True


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""

    failure_threshold: int = 5
    success_threshold: int = 3
    recovery_timeout_seconds: float = 30.0
    half_open_max_calls: int = 3


class OrchestratorConfig(BaseModel):
    """Orchestrator configuration."""

    reconciliation_on_startup: bool = True
    cancel_stale_orders: bool = True
    shutdown_timeout_seconds: float = 30.0
    health_check_interval_seconds: float = 30.0


class ChannelConfig(BaseModel):
    """Alert channel configuration."""

    enabled: bool = False
    min_severity: str = "warning"


class AlertingConfig(BaseModel):
    """Alerting configuration."""

    enabled: bool = True
    rate_limit_seconds: int = 60
    dedup_window_seconds: int = 300
    max_history: int = 1000
    channels: dict[str, ChannelConfig] = Field(default_factory=dict)


class BrokerConfig(BaseModel):
    """Broker configuration."""

    provider: str = "alpaca"
    mode: Literal["paper", "live"] = "paper"
    rate_limit_per_minute: int = 200


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: Literal["structured", "simple"] = "structured"
    file: str = "artifacts/logs/ordinis.log"
    max_size_mb: int = 100
    backup_count: int = 5


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_position_size: int = 100
    max_portfolio_exposure_pct: float = 25.0
    max_sector_concentration_pct: float = 30.0
    require_stop_loss: bool = False


class DataConfig(BaseModel):
    """Data provider configuration."""

    primary_provider: str = "alpaca"
    cache_dir: str = "artifacts/cache"
    historical_lookback_days: int = 365


class ArtifactsConfig(BaseModel):
    """Artifacts storage configuration."""

    base_dir: str = "artifacts"
    runs_dir: str = "artifacts/runs"
    reports_dir: str = "artifacts/reports"
    logs_dir: str = "artifacts/logs"
    cache_dir: str = "artifacts/cache"
    retention_days: int = 30
    max_size_gb: float = 10.0


class KafkaConfig(BaseModel):
    """Kafka configuration."""

    bootstrap_servers: str = "localhost:9092"


class BusConfig(BaseModel):
    """Event bus configuration."""

    type: Literal["in_memory", "kafka", "nats"] = "in_memory"
    kafka: KafkaConfig = Field(default_factory=KafkaConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, override takes precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml_config(environment: str | None = None) -> dict[str, Any]:
    """Load configuration from YAML files."""
    config_dir = Path("configs")

    # Load default config
    default_path = config_dir / "default.yaml"
    config: dict[str, Any] = {}

    if default_path.exists():
        with default_path.open() as f:
            config = yaml.safe_load(f) or {}

    # Load environment-specific override
    if environment:
        env_path = config_dir / "environments" / f"{environment}.yaml"
        if env_path.exists():
            with env_path.open() as f:
                env_config = yaml.safe_load(f) or {}
                config = _deep_merge(config, env_config)

    return config


class Settings(BaseSettings):
    """
    Main application settings.

    Configuration is loaded in order of precedence (highest first):
    1. Environment variables (ORDINIS_ prefix)
    2. Environment-specific YAML (configs/environments/{env}.yaml)
    3. Default YAML (configs/default.yaml)
    4. Default values defined here
    """

    model_config = SettingsConfigDict(
        env_prefix="ORDINIS_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    system: SystemConfig = Field(default_factory=SystemConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    kill_switch: KillSwitchConfig = Field(default_factory=KillSwitchConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    bus: BusConfig = Field(default_factory=BusConfig)
    broker: BrokerConfig = Field(default_factory=BrokerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)

    @classmethod
    def from_yaml(cls, environment: str | None = None) -> Settings:
        """
        Load settings from YAML configuration files.

        Args:
            environment: Environment name (dev, test, prod). If None,
                         uses default.yaml only.

        Returns:
            Settings instance with merged configuration.
        """
        yaml_config = _load_yaml_config(environment)
        return cls.model_validate(yaml_config)


@lru_cache(maxsize=1)
def get_settings(environment: str | None = None) -> Settings:
    """
    Get cached settings instance.

    Args:
        environment: Environment name for loading config.

    Returns:
        Cached Settings instance.
    """
    import os

    env = environment or os.getenv("ORDINIS_ENVIRONMENT", "dev")
    return Settings.from_yaml(env)


def reset_settings() -> None:
    """Clear cached settings (useful for testing)."""
    get_settings.cache_clear()
