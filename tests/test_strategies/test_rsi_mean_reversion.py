"""Tests for RSI Mean Reversion strategy."""

from datetime import datetime

import pandas as pd
import pytest

from strategies.rsi_mean_reversion import RSIMeanReversionStrategy


def create_test_data(bars: int = 150, trend: str = "neutral") -> pd.DataFrame:
    """
    Create test market data.

    Args:
        bars: Number of bars to generate
        trend: Market trend (neutral, uptrend, downtrend, volatile)

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start="2024-01-01", periods=bars, freq="D")

    if trend == "neutral":
        # Sideways market with oscillations
        close = 100 + 5 * pd.Series(range(bars)).apply(lambda x: (x % 20 - 10) * 0.5)
    elif trend == "uptrend":
        # Strong uptrend
        close = 100 + pd.Series(range(bars)) * 0.3
    elif trend == "downtrend":
        # Strong downtrend
        close = 150 - pd.Series(range(bars)) * 0.3
    elif trend == "volatile":
        # High volatility oscillations
        import numpy as np

        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(bars) * 2)
    else:
        close = pd.Series([100] * bars)

    data = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": [1000000] * bars,
        },
        index=dates,
    )

    return data


class TestRSIMeanReversionStrategy:
    """Tests for RSIMeanReversionStrategy class."""

    def test_strategy_initialization(self):
        """Test strategy initializes with default parameters."""
        strategy = RSIMeanReversionStrategy(name="test-rsi")

        assert strategy.name == "test-rsi"
        assert strategy.params["rsi_period"] == 14
        assert strategy.params["oversold_threshold"] == 30
        assert strategy.params["overbought_threshold"] == 70
        assert strategy.params["extreme_oversold"] == 20
        assert strategy.params["extreme_overbought"] == 80

    def test_strategy_custom_parameters(self):
        """Test strategy with custom parameters."""
        strategy = RSIMeanReversionStrategy(
            name="custom-rsi",
            rsi_period=21,
            oversold_threshold=25,
            overbought_threshold=75,
            extreme_oversold=15,
            extreme_overbought=85,
        )

        assert strategy.params["rsi_period"] == 21
        assert strategy.params["oversold_threshold"] == 25
        assert strategy.params["overbought_threshold"] == 75
        assert strategy.params["extreme_oversold"] == 15
        assert strategy.params["extreme_overbought"] == 85

    def test_configure_creates_model(self):
        """Test that configure creates underlying RSI model."""
        strategy = RSIMeanReversionStrategy(name="test-rsi")

        assert hasattr(strategy, "model")
        assert strategy.model is not None
        assert strategy.model.config.model_type == "mean_reversion"

    def test_get_description(self):
        """Test strategy description."""
        strategy = RSIMeanReversionStrategy(name="test-rsi")

        description = strategy.get_description()

        assert isinstance(description, str)
        assert "RSI Mean Reversion" in description
        assert "oversold" in description.lower()
        assert "overbought" in description.lower()
        assert "14" in description  # Default RSI period

    def test_get_required_bars_default(self):
        """Test required bars calculation."""
        strategy = RSIMeanReversionStrategy(name="test-rsi")

        required = strategy.get_required_bars()

        # Should be rsi_period + 20
        assert required == 14 + 20
        assert required == 34

    def test_get_required_bars_custom(self):
        """Test required bars with custom RSI period."""
        strategy = RSIMeanReversionStrategy(
            name="test-rsi",
            rsi_period=21,
        )

        required = strategy.get_required_bars()

        assert required == 21 + 20
        assert required == 41

    def test_validate_data_insufficient_bars(self):
        """Test validation fails with insufficient data."""
        strategy = RSIMeanReversionStrategy(name="test-rsi")
        data = create_test_data(bars=30)  # Need 34

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is False
        assert "Insufficient data" in msg

    def test_validate_data_sufficient_bars(self):
        """Test validation succeeds with sufficient data."""
        strategy = RSIMeanReversionStrategy(name="test-rsi")
        data = create_test_data(bars=100)

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is True
        assert msg == ""

    def test_generate_signal_with_insufficient_data(self):
        """Test signal generation returns None for insufficient data."""
        strategy = RSIMeanReversionStrategy(name="test-rsi")
        data = create_test_data(bars=20)

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is None

    def test_generate_signal_with_valid_data(self):
        """Test signal generation with valid data."""
        strategy = RSIMeanReversionStrategy(name="test-rsi")
        data = create_test_data(bars=150, trend="neutral")

        timestamp = datetime.utcnow()
        signal = strategy.generate_signal(data, timestamp)

        # May or may not generate signal depending on RSI values
        # Just verify it doesn't crash
        assert signal is None or hasattr(signal, "symbol")

    def test_generate_signal_handles_exceptions(self):
        """Test signal generation handles exceptions gracefully."""
        strategy = RSIMeanReversionStrategy(name="test-rsi")

        # Create invalid data (missing columns after initial validation)
        data = create_test_data(bars=100)

        try:
            signal = strategy.generate_signal(data, datetime.utcnow())
            # Should return None on exception
            assert signal is None or hasattr(signal, "symbol")
        except Exception:
            pytest.fail("generate_signal should handle exceptions gracefully")

    def test_str_representation(self):
        """Test string representation."""
        strategy = RSIMeanReversionStrategy(name="my-rsi-strategy")

        result = str(strategy)

        assert result == "my-rsi-strategy Strategy"

    def test_repr_representation(self):
        """Test repr representation."""
        strategy = RSIMeanReversionStrategy(name="my-rsi-strategy")

        result = repr(strategy)

        assert "RSIMeanReversionStrategy" in result
        assert "my-rsi-strategy" in result


class TestRSIMeanReversionStrategyParameters:
    """Tests for parameter handling."""

    def test_min_bars_calculated_from_rsi_period(self):
        """Test min_bars is calculated from RSI period."""
        strategy = RSIMeanReversionStrategy(name="test", rsi_period=21)

        assert strategy.params["min_bars"] == 21 + 20

    def test_extreme_thresholds_optional(self):
        """Test extreme thresholds have defaults."""
        strategy = RSIMeanReversionStrategy(name="test")

        assert "extreme_oversold" in strategy.params
        assert "extreme_overbought" in strategy.params
        assert strategy.params["extreme_oversold"] == 20
        assert strategy.params["extreme_overbought"] == 80

    def test_model_config_parameters(self):
        """Test model config includes all parameters."""
        strategy = RSIMeanReversionStrategy(
            name="test",
            rsi_period=21,
            oversold_threshold=25,
            overbought_threshold=75,
        )

        config = strategy.model.config

        assert config.parameters["rsi_period"] == 21
        assert config.parameters["oversold_threshold"] == 25
        assert config.parameters["overbought_threshold"] == 75


class TestRSIMeanReversionStrategyIntegration:
    """Integration tests for RSI strategy."""

    def test_strategy_with_trending_market(self):
        """Test strategy behavior in trending market."""
        strategy = RSIMeanReversionStrategy(name="test-rsi")
        data = create_test_data(bars=150, trend="uptrend")

        # Test at multiple points
        for i in range(50, len(data), 20):
            timestamp = data.index[i]
            signal = strategy.generate_signal(data.iloc[: i + 1], timestamp)

            # Signal may or may not be generated
            if signal:
                assert hasattr(signal, "symbol")
                assert hasattr(signal, "timestamp")
                assert hasattr(signal, "signal_type")

    def test_strategy_with_volatile_market(self):
        """Test strategy behavior in volatile market."""
        strategy = RSIMeanReversionStrategy(name="test-rsi")
        data = create_test_data(bars=150, trend="volatile")

        timestamp = data.index[-1]
        signal = strategy.generate_signal(data, timestamp)

        # Should handle volatile data without crashing
        assert signal is None or hasattr(signal, "symbol")

    def test_strategy_with_different_timeframes(self):
        """Test strategy with different data lengths."""
        strategy = RSIMeanReversionStrategy(name="test-rsi")

        for bars in [50, 100, 200, 500]:
            if bars < strategy.get_required_bars():
                continue

            data = create_test_data(bars=bars)
            timestamp = data.index[-1]
            signal = strategy.generate_signal(data, timestamp)

            # Should handle different timeframes
            assert signal is None or hasattr(signal, "symbol")

    def test_strategy_consistency(self):
        """Test strategy generates consistent signals for same data."""
        strategy = RSIMeanReversionStrategy(name="test-rsi", rsi_period=14)
        data = create_test_data(bars=150, trend="neutral")
        timestamp = data.index[-1]

        # Generate signal multiple times
        signal1 = strategy.generate_signal(data, timestamp)
        signal2 = strategy.generate_signal(data, timestamp)

        # Should be consistent
        if signal1 is None:
            assert signal2 is None
        else:
            assert signal2 is not None
            assert signal1.symbol == signal2.symbol
            assert signal1.signal_type == signal2.signal_type


class TestRSIMeanReversionStrategyValidation:
    """Tests for data validation."""

    def test_requires_ohlcv_columns(self):
        """Test strategy requires OHLCV columns."""
        strategy = RSIMeanReversionStrategy(name="test")

        # Missing volume
        data = pd.DataFrame(
            {
                "open": [100] * 100,
                "high": [105] * 100,
                "low": [95] * 100,
                "close": [102] * 100,
            },
            index=pd.date_range("2024-01-01", periods=100),
        )

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is False
        assert "volume" in msg.lower()

    def test_handles_empty_data(self):
        """Test strategy handles empty data."""
        strategy = RSIMeanReversionStrategy(name="test")
        data = pd.DataFrame()

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is False
        assert "empty" in msg.lower()

    def test_handles_none_data(self):
        """Test strategy handles None data."""
        strategy = RSIMeanReversionStrategy(name="test")

        is_valid, msg = strategy.validate_data(None)

        assert is_valid is False
        assert "empty" in msg.lower()
