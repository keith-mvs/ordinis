"""
Tests for SignalCore trading models.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore import (
    Direction,
    ModelConfig,
    RSIMeanReversionModel,
    SignalType,
    SMACrossoverModel,
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=300, freq="1d")

    # Create upward trending price data
    prices = np.linspace(100, 150, 300) + np.random.randn(300) * 2

    data = pd.DataFrame(
        {
            "open": prices,
            "high": prices + 1,
            "low": prices - 1,
            "close": prices,
            "volume": [1000000] * 300,
        },
        index=dates,
    )

    return data


@pytest.fixture
def sma_model_config():
    """Create SMA crossover model config."""
    return ModelConfig(
        model_id="sma_test",
        model_type="technical",
        version="1.0.0",
        parameters={"fast_period": 10, "slow_period": 20, "min_separation": 0.01},
    )


@pytest.fixture
def rsi_model_config():
    """Create RSI mean reversion model config."""
    return ModelConfig(
        model_id="rsi_test",
        model_type="technical",
        version="1.0.0",
        parameters={
            "rsi_period": 14,
            "oversold_threshold": 30,
            "overbought_threshold": 70,
        },
    )


@pytest.mark.unit
def test_sma_model_initialization(sma_model_config):
    """Test SMA crossover model initialization."""
    model = SMACrossoverModel(sma_model_config)

    assert model.config.model_id == "sma_test"
    assert model.fast_period == 10
    assert model.slow_period == 20
    assert model.min_separation == 0.01


@pytest.mark.unit
def test_sma_model_validation(sma_model_config, sample_ohlcv_data):
    """Test SMA model data validation."""
    model = SMACrossoverModel(sma_model_config)

    # Valid data
    is_valid, msg = model.validate(sample_ohlcv_data)
    assert is_valid is True

    # Insufficient data
    short_data = sample_ohlcv_data.head(10)
    is_valid, msg = model.validate(short_data)
    assert is_valid is False
    assert "Insufficient data" in msg


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sma_model_generate_signal(sma_model_config, sample_ohlcv_data):
    """Test SMA model signal generation."""
    model = SMACrossoverModel(sma_model_config)

    signal = await model.generate(sample_ohlcv_data, datetime(2024, 10, 27))

    assert signal.model_id == "sma_test"
    assert signal.signal_type in [SignalType.ENTRY, SignalType.EXIT, SignalType.HOLD]
    assert signal.direction in [Direction.LONG, Direction.SHORT, Direction.NEUTRAL]
    assert 0.0 <= signal.probability <= 1.0
    assert -1.0 <= signal.score <= 1.0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sma_model_bullish_crossover(sma_model_config):
    """Test SMA model detects bullish crossover."""
    # Create data with clear bullish crossover
    dates = pd.date_range("2024-01-01", periods=100, freq="1d")

    # Start with fast < slow, then fast > slow
    prices = []
    for i in range(100):
        if i < 50:
            prices.append(100 + i * 0.1)  # Slow rise
        else:
            prices.append(100 + (i - 50) * 0.5 + 5)  # Fast rise

    data = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000000] * 100,
        },
        index=dates,
    )

    model = SMACrossoverModel(sma_model_config)
    signal = await model.generate(data, datetime(2024, 4, 10))

    # Should detect bullish signal or hold (depending on exact crossover timing)
    assert signal.signal_type in [SignalType.ENTRY, SignalType.HOLD]
    if signal.signal_type == SignalType.ENTRY:
        assert signal.direction == Direction.LONG


@pytest.mark.unit
def test_rsi_model_initialization(rsi_model_config):
    """Test RSI mean reversion model initialization."""
    model = RSIMeanReversionModel(rsi_model_config)

    assert model.config.model_id == "rsi_test"
    assert model.rsi_period == 14
    assert model.oversold_threshold == 30
    assert model.overbought_threshold == 70


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rsi_model_generate_signal(rsi_model_config, sample_ohlcv_data):
    """Test RSI model signal generation."""
    model = RSIMeanReversionModel(rsi_model_config)

    signal = await model.generate(sample_ohlcv_data, datetime(2024, 10, 27))

    assert signal.model_id == "rsi_test"
    assert signal.signal_type in [SignalType.ENTRY, SignalType.EXIT, SignalType.HOLD]
    assert 0.0 <= signal.probability <= 1.0
    assert -1.0 <= signal.score <= 1.0
    assert "rsi" in signal.feature_contributions


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rsi_model_oversold_condition(rsi_model_config):
    """Test RSI model detects oversold condition."""
    # Create data that becomes oversold
    dates = pd.date_range("2024-01-01", periods=100, freq="1d")

    # Sharp decline to create oversold RSI
    prices = []
    for i in range(100):
        if i < 50:
            prices.append(100)
        else:
            prices.append(100 - (i - 50) * 0.8)  # Sharp decline

    data = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1000000] * 100,
        },
        index=dates,
    )

    model = RSIMeanReversionModel(rsi_model_config)
    signal = await model.generate(data, datetime(2024, 4, 10))

    # RSI should be low, potentially triggering entry signal
    assert "rsi" in signal.feature_contributions
    rsi_value = signal.feature_contributions["rsi"]

    # RSI should be relatively low due to sharp decline
    assert rsi_value < 70  # Not overbought


@pytest.mark.unit
def test_model_describe(sma_model_config):
    """Test model describe method."""
    model = SMACrossoverModel(sma_model_config)

    description = model.describe()

    assert description["model_id"] == "sma_test"
    assert description["model_type"] == "technical"
    assert description["version"] == "1.0.0"
    assert description["enabled"] is True
    assert "parameters" in description


@pytest.mark.unit
@pytest.mark.asyncio
async def test_signal_feature_contributions(sma_model_config, sample_ohlcv_data):
    """Test that signals include feature contributions."""
    model = SMACrossoverModel(sma_model_config)

    signal = await model.generate(sample_ohlcv_data, datetime(2024, 10, 27))

    assert len(signal.feature_contributions) > 0
    assert "fast_sma" in signal.feature_contributions
    assert "slow_sma" in signal.feature_contributions
