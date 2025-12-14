"""Tests for engine configuration classes.

This module tests the configuration dataclasses and their validation logic.
"""

from pathlib import Path

from ordinis.engines.base import (
    AIEngineConfig,
    BaseEngineConfig,
    DataEngineConfig,
    TradingEngineConfig,
)


class TestBaseEngineConfig:
    """Test BaseEngineConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = BaseEngineConfig()

        assert config.enabled is True
        assert config.name == "BaseEngine"
        assert config.log_level == "INFO"
        assert config.metrics_enabled is True
        assert config.health_check_interval_seconds == 30.0
        assert config.timeout_seconds == 30.0
        assert config.retry_attempts == 3
        assert config.retry_delay_seconds == 1.0
        assert config.governance_enabled is True
        assert config.audit_enabled is True

    def test_custom_name(self) -> None:
        """Test setting custom name."""
        config = BaseEngineConfig(name="CustomEngine")

        assert config.name == "CustomEngine"

    def test_name_auto_generation(self) -> None:
        """Test automatic name generation from class name."""
        config = BaseEngineConfig()

        assert config.name == "BaseEngine"

    def test_post_init_name_generation(self) -> None:
        """Test __post_init__ generates name if empty."""
        config = BaseEngineConfig()
        config.name = ""
        config.__post_init__()

        assert config.name == "BaseEngine"

    def test_disable_features(self) -> None:
        """Test disabling features."""
        config = BaseEngineConfig(
            enabled=False,
            metrics_enabled=False,
            governance_enabled=False,
            audit_enabled=False,
        )

        assert config.enabled is False
        assert config.metrics_enabled is False
        assert config.governance_enabled is False
        assert config.audit_enabled is False

    def test_custom_timeouts(self) -> None:
        """Test custom timeout values."""
        config = BaseEngineConfig(
            timeout_seconds=60.0,
            health_check_interval_seconds=10.0,
        )

        assert config.timeout_seconds == 60.0
        assert config.health_check_interval_seconds == 10.0

    def test_custom_retry_settings(self) -> None:
        """Test custom retry settings."""
        config = BaseEngineConfig(
            retry_attempts=5,
            retry_delay_seconds=2.5,
        )

        assert config.retry_attempts == 5
        assert config.retry_delay_seconds == 2.5

    def test_to_dict(self) -> None:
        """Test converting config to dictionary."""
        config = BaseEngineConfig(name="TestEngine")
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "TestEngine"
        assert config_dict["enabled"] is True
        assert config_dict["log_level"] == "INFO"
        assert config_dict["metrics_enabled"] is True
        assert config_dict["governance_enabled"] is True
        assert config_dict["audit_enabled"] is True

    def test_to_dict_all_fields(self) -> None:
        """Test to_dict includes all expected fields."""
        config = BaseEngineConfig()
        config_dict = config.to_dict()

        expected_fields = [
            "enabled",
            "name",
            "log_level",
            "metrics_enabled",
            "health_check_interval_seconds",
            "timeout_seconds",
            "retry_attempts",
            "retry_delay_seconds",
            "governance_enabled",
            "audit_enabled",
        ]

        for field in expected_fields:
            assert field in config_dict


class TestAIEngineConfig:
    """Test AIEngineConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default AI-specific values."""
        config = AIEngineConfig()

        assert config.model_name == ""
        assert config.fallback_model == ""
        assert config.max_tokens == 4096
        assert config.temperature == 0.7
        assert config.api_timeout_seconds == 60.0
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 300

    def test_inherits_base_config(self) -> None:
        """Test AIEngineConfig inherits from BaseEngineConfig."""
        config = AIEngineConfig(name="TestAI")

        assert config.enabled is True
        assert config.name == "TestAI"
        assert config.log_level == "INFO"
        assert config.governance_enabled is True

    def test_custom_model_settings(self) -> None:
        """Test custom model settings."""
        config = AIEngineConfig(
            model_name="gpt-4",
            fallback_model="gpt-3.5-turbo",
            max_tokens=8192,
            temperature=0.5,
        )

        assert config.model_name == "gpt-4"
        assert config.fallback_model == "gpt-3.5-turbo"
        assert config.max_tokens == 8192
        assert config.temperature == 0.5

    def test_cache_settings(self) -> None:
        """Test cache configuration."""
        config = AIEngineConfig(
            cache_enabled=False,
            cache_ttl_seconds=600,
        )

        assert config.cache_enabled is False
        assert config.cache_ttl_seconds == 600

    def test_api_timeout(self) -> None:
        """Test API timeout configuration."""
        config = AIEngineConfig(api_timeout_seconds=120.0)

        assert config.api_timeout_seconds == 120.0

    def test_temperature_range(self) -> None:
        """Test temperature values."""
        config_low = AIEngineConfig(temperature=0.0)
        config_high = AIEngineConfig(temperature=1.0)
        config_mid = AIEngineConfig(temperature=0.7)

        assert config_low.temperature == 0.0
        assert config_high.temperature == 1.0
        assert config_mid.temperature == 0.7


class TestDataEngineConfig:
    """Test DataEngineConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default data-specific values."""
        config = DataEngineConfig()

        assert config.batch_size == 1000
        assert config.buffer_size == 10000
        assert config.checkpoint_enabled is True
        assert isinstance(config.checkpoint_path, Path)
        assert config.parallel_workers == 4

    def test_inherits_base_config(self) -> None:
        """Test DataEngineConfig inherits from BaseEngineConfig."""
        config = DataEngineConfig(name="TestData")

        assert config.enabled is True
        assert config.name == "TestData"
        assert config.log_level == "INFO"

    def test_custom_batch_settings(self) -> None:
        """Test custom batch and buffer settings."""
        config = DataEngineConfig(
            batch_size=5000,
            buffer_size=50000,
        )

        assert config.batch_size == 5000
        assert config.buffer_size == 50000

    def test_custom_checkpoint_path(self) -> None:
        """Test custom checkpoint path."""
        custom_path = Path("custom/checkpoints")
        config = DataEngineConfig(checkpoint_path=custom_path)

        assert config.checkpoint_path == custom_path

    def test_checkpoint_disabled(self) -> None:
        """Test disabling checkpoints."""
        config = DataEngineConfig(checkpoint_enabled=False)

        assert config.checkpoint_enabled is False

    def test_parallel_workers(self) -> None:
        """Test parallel workers configuration."""
        config = DataEngineConfig(parallel_workers=8)

        assert config.parallel_workers == 8

    def test_checkpoint_path_default_factory(self) -> None:
        """Test checkpoint path uses default_factory."""
        config1 = DataEngineConfig()
        config2 = DataEngineConfig()

        assert config1.checkpoint_path is not config2.checkpoint_path


class TestTradingEngineConfig:
    """Test TradingEngineConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default trading-specific values."""
        config = TradingEngineConfig()

        assert config.paper_mode is True
        assert config.max_position_size == 10000.0
        assert config.daily_loss_limit == 1000.0
        assert config.max_drawdown_pct == 5.0
        assert config.symbols == []

    def test_inherits_base_config(self) -> None:
        """Test TradingEngineConfig inherits from BaseEngineConfig."""
        config = TradingEngineConfig(name="TestTrading")

        assert config.enabled is True
        assert config.name == "TestTrading"
        assert config.log_level == "INFO"

    def test_live_trading_mode(self) -> None:
        """Test live trading mode configuration."""
        config = TradingEngineConfig(paper_mode=False)

        assert config.paper_mode is False

    def test_risk_limits(self) -> None:
        """Test risk limit configuration."""
        config = TradingEngineConfig(
            max_position_size=50000.0,
            daily_loss_limit=5000.0,
            max_drawdown_pct=10.0,
        )

        assert config.max_position_size == 50000.0
        assert config.daily_loss_limit == 5000.0
        assert config.max_drawdown_pct == 10.0

    def test_symbols_list(self) -> None:
        """Test symbols list configuration."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        config = TradingEngineConfig(symbols=symbols)

        assert config.symbols == symbols
        assert len(config.symbols) == 3

    def test_symbols_default_factory(self) -> None:
        """Test symbols list uses default_factory."""
        config1 = TradingEngineConfig()
        config2 = TradingEngineConfig()

        assert config1.symbols is not config2.symbols

        config1.symbols.append("AAPL")
        assert len(config1.symbols) == 1
        assert len(config2.symbols) == 0


class TestConfigInheritance:
    """Test configuration class inheritance."""

    def test_ai_config_is_base_config(self) -> None:
        """Test AIEngineConfig is a BaseEngineConfig."""
        config = AIEngineConfig()

        assert isinstance(config, BaseEngineConfig)

    def test_data_config_is_base_config(self) -> None:
        """Test DataEngineConfig is a BaseEngineConfig."""
        config = DataEngineConfig()

        assert isinstance(config, BaseEngineConfig)

    def test_trading_config_is_base_config(self) -> None:
        """Test TradingEngineConfig is a BaseEngineConfig."""
        config = TradingEngineConfig()

        assert isinstance(config, BaseEngineConfig)

    def test_config_override_base_values(self) -> None:
        """Test derived configs can override base values."""
        config = AIEngineConfig(
            name="CustomAI",
            enabled=False,
            log_level="DEBUG",
            model_name="gpt-4",
        )

        assert config.name == "CustomAI"
        assert config.enabled is False
        assert config.log_level == "DEBUG"
        assert config.model_name == "gpt-4"


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_empty_name_auto_populated(self) -> None:
        """Test empty name is auto-populated in __post_init__."""
        config = BaseEngineConfig()
        config.name = ""
        config.__post_init__()

        assert config.name != ""
        assert "BaseEngine" in config.name

    def test_custom_subclass_name_generation(self) -> None:
        """Test name generation for custom subclass."""

        class CustomEngineConfig(BaseEngineConfig):
            pass

        config = CustomEngineConfig()

        assert config.name == "CustomEngine"

    def test_explicit_name_preserved(self) -> None:
        """Test explicit name is preserved after __post_init__."""
        config = BaseEngineConfig(name="ExplicitName")

        assert config.name == "ExplicitName"

        config.__post_init__()

        assert config.name == "ExplicitName"
