"""
Configuration Module Unit Tests

Tests for OptionsEngineConfig validation and serialization.
"""

import pytest

from src.engines.optionscore.core.config import OptionsEngineConfig


def test_config_defaults():
    """Test default configuration values."""
    config = OptionsEngineConfig(engine_id="test")

    assert config.engine_id == "test"
    assert config.enabled is True
    assert config.cache_ttl_seconds == 300
    assert config.default_risk_free_rate == 0.05
    assert config.default_dividend_yield == 0.0
    assert config.calculation_mode == "european"
    assert config.enable_iv_calculation is False
    assert config.metadata == {}


def test_config_custom_values():
    """Test custom configuration values."""
    config = OptionsEngineConfig(
        engine_id="custom",
        enabled=False,
        cache_ttl_seconds=600,
        default_risk_free_rate=0.04,
        default_dividend_yield=0.02,
        metadata={"version": "1.0"},
    )

    assert config.engine_id == "custom"
    assert config.enabled is False
    assert config.cache_ttl_seconds == 600
    assert config.default_risk_free_rate == 0.04
    assert config.default_dividend_yield == 0.02
    assert config.metadata == {"version": "1.0"}


def test_config_empty_engine_id():
    """Test validation of empty engine_id."""
    with pytest.raises(ValueError, match="engine_id cannot be empty"):
        OptionsEngineConfig(engine_id="")


def test_config_negative_cache_ttl():
    """Test validation of negative cache TTL."""
    with pytest.raises(ValueError, match="cache_ttl_seconds must be non-negative"):
        OptionsEngineConfig(engine_id="test", cache_ttl_seconds=-1)


def test_config_negative_risk_free_rate():
    """Test validation of negative risk-free rate."""
    with pytest.raises(ValueError, match="default_risk_free_rate must be non-negative"):
        OptionsEngineConfig(engine_id="test", default_risk_free_rate=-0.01)


def test_config_negative_dividend_yield():
    """Test validation of negative dividend yield."""
    with pytest.raises(ValueError, match="default_dividend_yield must be non-negative"):
        OptionsEngineConfig(engine_id="test", default_dividend_yield=-0.01)


def test_config_invalid_calculation_mode():
    """Test validation of invalid calculation mode."""
    with pytest.raises(ValueError, match="calculation_mode must be"):
        OptionsEngineConfig(engine_id="test", calculation_mode="invalid")


def test_config_american_not_implemented():
    """Test that American mode raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="American options pricing not yet implemented"):
        OptionsEngineConfig(engine_id="test", calculation_mode="american")


def test_config_to_dict():
    """Test configuration serialization."""
    config = OptionsEngineConfig(
        engine_id="test",
        cache_ttl_seconds=600,
        metadata={"key": "value"},
    )

    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert config_dict["engine_id"] == "test"
    assert config_dict["enabled"] is True
    assert config_dict["cache_ttl_seconds"] == 600
    assert config_dict["default_risk_free_rate"] == 0.05
    assert config_dict["default_dividend_yield"] == 0.0
    assert config_dict["calculation_mode"] == "european"
    assert config_dict["enable_iv_calculation"] is False
    assert config_dict["metadata"] == {"key": "value"}
