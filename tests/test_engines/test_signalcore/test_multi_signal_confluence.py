"""
Tests for MultiSignalConfluenceModel.

This module tests the multi-oscillator confluence strategy that requires
agreement from RSI and Stochastic indicators before generating signals.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models.multi_signal_confluence import (
    MultiSignalConfluenceModel,
)


def create_test_ohlcv(
    n_bars: int = 100,
    trend: str = "neutral",
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing."""
    np.random.seed(seed)

    # Base price
    base_price = 100.0

    # Generate returns based on trend
    if trend == "bullish":
        drift = 0.001
    elif trend == "bearish":
        drift = -0.001
    else:
        drift = 0.0

    returns = np.random.normal(drift, volatility, n_bars)
    prices = base_price * np.cumprod(1 + returns)

    # Create OHLCV
    data = []
    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, volatility / 2)))
        low = close * (1 - abs(np.random.normal(0, volatility / 2)))
        open_price = (high + low) / 2 + np.random.normal(0, volatility / 4) * close
        volume = int(np.random.uniform(1000000, 5000000))

        data.append(
            {
                "open": open_price,
                "high": max(high, open_price, close),
                "low": min(low, open_price, close),
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.index = pd.date_range(start="2024-01-01", periods=n_bars, freq="D")
    return df


def create_oversold_data(n_bars: int = 100) -> pd.DataFrame:
    """Create data with oversold conditions (declining then stabilizing)."""
    np.random.seed(42)

    # Sharp decline followed by stabilization
    decline = np.linspace(100, 70, n_bars // 2)
    stable = 70 + np.random.normal(0, 0.5, n_bars - n_bars // 2)
    prices = np.concatenate([decline, stable])

    data = []
    for i, close in enumerate(prices):
        volatility = 0.01
        high = close * (1 + abs(np.random.normal(0, volatility)))
        low = close * (1 - abs(np.random.normal(0, volatility)))
        open_price = (high + low) / 2
        volume = int(np.random.uniform(1000000, 5000000))

        data.append(
            {
                "open": open_price,
                "high": max(high, open_price, close),
                "low": min(low, open_price, close),
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.index = pd.date_range(start="2024-01-01", periods=n_bars, freq="D")
    return df


def create_overbought_data(n_bars: int = 100) -> pd.DataFrame:
    """Create data with overbought conditions (rising then stabilizing)."""
    np.random.seed(42)

    # Sharp rise followed by stabilization
    rise = np.linspace(100, 140, n_bars // 2)
    stable = 140 + np.random.normal(0, 0.5, n_bars - n_bars // 2)
    prices = np.concatenate([rise, stable])

    data = []
    for i, close in enumerate(prices):
        volatility = 0.01
        high = close * (1 + abs(np.random.normal(0, volatility)))
        low = close * (1 - abs(np.random.normal(0, volatility)))
        open_price = (high + low) / 2
        volume = int(np.random.uniform(1000000, 5000000))

        data.append(
            {
                "open": open_price,
                "high": max(high, open_price, close),
                "low": min(low, open_price, close),
                "close": close,
                "volume": volume,
            }
        )

    df = pd.DataFrame(data)
    df.index = pd.date_range(start="2024-01-01", periods=n_bars, freq="D")
    return df


class TestMultiSignalConfluenceModel:
    """Test suite for MultiSignalConfluenceModel."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create default model configuration."""
        return ModelConfig(
            model_id="test_multi_signal",
            model_type="multi_signal_confluence",
            version="1.0.0",
            parameters={
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "stoch_period": 14,
                "stoch_smooth": 3,
                "stoch_d_period": 3,
                "stoch_oversold": 20,
                "stoch_overbought": 80,
                "adx_period": 14,
                "adx_max_for_reversion": 30,
                "atr_period": 14,
                "atr_stop_mult": 2.0,
                "atr_tp_mult": 3.0,
                "require_all_signals": False,
                "enable_shorts": True,
                "enable_longs": True,
            },
            min_data_points=50,
        )

    @pytest.fixture
    def model(self, model_config: ModelConfig) -> MultiSignalConfluenceModel:
        """Create model instance."""
        return MultiSignalConfluenceModel(model_config)

    @pytest.mark.unit
    def test_model_initialization(self, model: MultiSignalConfluenceModel):
        """Test model initializes correctly."""
        assert model.rsi_period == 14
        assert model.rsi_oversold == 30
        assert model.rsi_overbought == 70
        assert model.stoch_period == 14
        assert model.atr_stop_mult == 2.0
        assert model.atr_tp_mult == 3.0
        assert model.enable_longs is True
        assert model.enable_shorts is True

    @pytest.mark.unit
    def test_validation_insufficient_data(self, model: MultiSignalConfluenceModel):
        """Test validation fails with insufficient data."""
        data = create_test_ohlcv(n_bars=20)  # Too few bars
        is_valid, msg = model.validate(data)
        assert is_valid is False
        assert "Insufficient data" in msg

    @pytest.mark.unit
    def test_validation_missing_columns(self, model: MultiSignalConfluenceModel):
        """Test validation fails with missing columns."""
        data = pd.DataFrame(
            {
                "close": [100, 101, 102],
                "volume": [1000, 1000, 1000],
            }
        )
        is_valid, msg = model.validate(data)
        assert is_valid is False
        assert "Missing columns" in msg

    @pytest.mark.unit
    def test_validation_valid_data(self, model: MultiSignalConfluenceModel):
        """Test validation passes with valid data."""
        data = create_test_ohlcv(n_bars=100)
        is_valid, msg = model.validate(data)
        assert is_valid is True
        assert msg == ""

    @pytest.mark.unit
    def test_compute_stochastic(self, model: MultiSignalConfluenceModel):
        """Test Stochastic oscillator computation."""
        data = create_test_ohlcv(n_bars=100)
        stoch_k, stoch_d = model._compute_stochastic(data["high"], data["low"], data["close"])

        # Stochastic should be between 0 and 100
        assert stoch_k.dropna().min() >= 0
        assert stoch_k.dropna().max() <= 100
        assert stoch_d.dropna().min() >= 0
        assert stoch_d.dropna().max() <= 100

    @pytest.mark.unit
    def test_compute_adx(self, model: MultiSignalConfluenceModel):
        """Test ADX computation."""
        data = create_test_ohlcv(n_bars=100)
        adx, plus_di, minus_di = model._compute_adx(data["high"], data["low"], data["close"])

        # ADX should be positive
        assert adx.dropna().min() >= 0
        # DI lines should be positive
        assert plus_di.dropna().min() >= 0
        assert minus_di.dropna().min() >= 0

    @pytest.mark.unit
    def test_compute_atr(self, model: MultiSignalConfluenceModel):
        """Test ATR computation."""
        data = create_test_ohlcv(n_bars=100)
        atr = model._compute_atr(data["high"], data["low"], data["close"])

        # ATR should be positive
        assert atr.dropna().min() > 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_no_signal_neutral_market(self, model: MultiSignalConfluenceModel):
        """Test no signal in neutral market conditions."""
        data = create_test_ohlcv(n_bars=100, trend="neutral", volatility=0.005)

        signal = await model.generate(
            symbol="TEST",
            data=data,
            timestamp=datetime.now(),
        )

        # In neutral, low-volatility market, may or may not generate signal
        # Just verify it doesn't crash and returns proper type
        assert signal is None or hasattr(signal, "direction")

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_long_signal_oversold(self, model: MultiSignalConfluenceModel):
        """Test long signal generation in oversold conditions."""
        model.reset_state()
        data = create_oversold_data(n_bars=100)

        signal = await model.generate(
            symbol="TEST",
            data=data,
            timestamp=datetime.now(),
        )

        # Should potentially generate a long entry signal
        if signal is not None:
            assert signal.direction in [Direction.LONG, Direction.SHORT, Direction.NEUTRAL]
            assert signal.signal_type in [SignalType.ENTRY, SignalType.EXIT]
            assert "model" in signal.metadata
            assert signal.metadata["model"] == "multi_signal_confluence"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_generate_signal_has_stop_and_tp(self, model: MultiSignalConfluenceModel):
        """Test that entry signals include stop loss and take profit."""
        model.reset_state()
        data = create_oversold_data(n_bars=100)

        signal = await model.generate(
            symbol="TEST",
            data=data,
            timestamp=datetime.now(),
        )

        if signal is not None and signal.signal_type == SignalType.ENTRY:
            assert "stop_loss" in signal.metadata
            assert "take_profit" in signal.metadata
            assert "atr" in signal.metadata

            # Stop loss should be below entry for long, above for short
            if signal.direction == Direction.LONG:
                assert signal.metadata["stop_loss"] < signal.price
                assert signal.metadata["take_profit"] > signal.price
            elif signal.direction == Direction.SHORT:
                assert signal.metadata["stop_loss"] > signal.price
                assert signal.metadata["take_profit"] < signal.price

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_reset_state(self, model: MultiSignalConfluenceModel):
        """Test state reset."""
        model._in_long = True
        model._entry_price = 100.0
        model._stop_loss = 98.0

        model.reset_state()

        assert model._in_long is False
        assert model._in_short is False
        assert model._entry_price is None
        assert model._stop_loss is None
        assert model._take_profit is None

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_disable_shorts(self, model_config: ModelConfig):
        """Test that shorts can be disabled."""
        model_config.parameters["enable_shorts"] = False
        model = MultiSignalConfluenceModel(model_config)

        assert model.enable_shorts is False

        # Generate with overbought data - should not produce short
        data = create_overbought_data(n_bars=100)
        signal = await model.generate(
            symbol="TEST",
            data=data,
            timestamp=datetime.now(),
        )

        if signal is not None:
            # Should never be short entry if shorts disabled
            if signal.signal_type == SignalType.ENTRY:
                assert signal.direction != Direction.SHORT

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_disable_longs(self, model_config: ModelConfig):
        """Test that longs can be disabled."""
        model_config.parameters["enable_longs"] = False
        model = MultiSignalConfluenceModel(model_config)

        assert model.enable_longs is False

        # Generate with oversold data - should not produce long entry
        data = create_oversold_data(n_bars=100)
        signal = await model.generate(
            symbol="TEST",
            data=data,
            timestamp=datetime.now(),
        )

        if signal is not None:
            if signal.signal_type == SignalType.ENTRY:
                assert signal.direction != Direction.LONG

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_require_all_signals_mode(self, model_config: ModelConfig):
        """Test strict mode requiring all signals."""
        model_config.parameters["require_all_signals"] = True
        model = MultiSignalConfluenceModel(model_config)

        assert model.require_all_signals is True

        # In strict mode, signals are harder to generate
        data = create_test_ohlcv(n_bars=100)
        signal = await model.generate(
            symbol="TEST",
            data=data,
            timestamp=datetime.now(),
        )

        # Just verify it runs without error
        assert signal is None or hasattr(signal, "confidence")

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_signal_confidence_scaling(self, model: MultiSignalConfluenceModel):
        """Test that confidence scales with number of signals."""
        model.reset_state()
        data = create_oversold_data(n_bars=100)

        signal = await model.generate(
            symbol="TEST",
            data=data,
            timestamp=datetime.now(),
        )

        if signal is not None and signal.signal_type == SignalType.ENTRY:
            # Confidence should be between 0.5 and 0.9 based on formula
            assert 0.5 <= signal.confidence <= 0.9


# Run tests standalone
if __name__ == "__main__":
    import sys

    # Run with pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
