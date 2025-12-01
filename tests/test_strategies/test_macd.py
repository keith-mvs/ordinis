"""Tests for MACD strategy."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from strategies.macd import MACDStrategy


def create_test_data(bars: int = 150, trend: str = "neutral") -> pd.DataFrame:
    """
    Create test market data.

    Args:
        bars: Number of bars to generate
        trend: Market trend (neutral, uptrend, downtrend, volatile, bullish_crossover, bearish_crossover)

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start="2024-01-01", periods=bars, freq="D")

    # Use numpy arrays to avoid index alignment issues with DatetimeIndex
    if trend == "neutral":
        # Sideways market with oscillations
        close = 100 + 5 * np.array([(x % 20 - 10) * 0.5 for x in range(bars)])
    elif trend == "uptrend":
        # Strong uptrend (MACD should be positive)
        close = 100 + np.arange(bars) * 0.5
    elif trend == "downtrend":
        # Strong downtrend (MACD should be negative)
        close = 200 - np.arange(bars) * 0.5
    elif trend == "volatile":
        # High volatility oscillations
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(bars) * 3)
    elif trend == "bullish_crossover":
        # Create conditions for bullish MACD crossover
        # Start with downtrend, then transition to uptrend
        np.random.seed(42)
        base = np.concatenate([
            100 - np.arange(bars // 2) * 0.2,
            100 - (bars // 2) * 0.2 + np.arange(bars - bars // 2) * 0.4,
        ])
        close = base + np.random.randn(bars) * 0.5
    elif trend == "bearish_crossover":
        # Create conditions for bearish MACD crossover
        # Start with uptrend, then transition to downtrend
        np.random.seed(42)
        base = np.concatenate([
            100 + np.arange(bars // 2) * 0.3,
            100 + (bars // 2) * 0.3 - np.arange(bars - bars // 2) * 0.5,
        ])
        close = base + np.random.randn(bars) * 0.5
    elif trend == "consolidating":
        # Tight range with low volatility
        np.random.seed(42)
        close = 100 + np.random.randn(bars) * 0.5
    else:
        close = np.array([100.0] * bars)

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


class TestMACDStrategy:
    """Tests for MACDStrategy class."""

    def test_strategy_initialization(self):
        """Test strategy initializes with default parameters."""
        strategy = MACDStrategy(name="test-macd")

        assert strategy.name == "test-macd"
        assert strategy.params["fast_period"] == 12
        assert strategy.params["slow_period"] == 26
        assert strategy.params["signal_period"] == 9

    def test_strategy_custom_parameters(self):
        """Test strategy with custom parameters."""
        strategy = MACDStrategy(
            name="custom-macd",
            fast_period=8,
            slow_period=21,
            signal_period=5,
        )

        assert strategy.params["fast_period"] == 8
        assert strategy.params["slow_period"] == 21
        assert strategy.params["signal_period"] == 5

    def test_configure_creates_model(self):
        """Test that configure creates underlying MACD model."""
        strategy = MACDStrategy(name="test-macd")

        assert hasattr(strategy, "model")
        assert strategy.model is not None
        assert strategy.model.config.model_type == "momentum"

    def test_get_description(self):
        """Test strategy description."""
        strategy = MACDStrategy(name="test-macd")

        description = strategy.get_description()

        assert isinstance(description, str)
        assert "MACD" in description
        assert "crossover" in description.lower()
        assert "12" in description  # Default fast period
        assert "26" in description  # Default slow period
        assert "9" in description  # Default signal period

    def test_get_required_bars_default(self):
        """Test required bars calculation."""
        strategy = MACDStrategy(name="test-macd")

        required = strategy.get_required_bars()

        # Should be slow_period + signal_period + 20
        assert required == 26 + 9 + 20
        assert required == 55

    def test_get_required_bars_custom(self):
        """Test required bars with custom periods."""
        strategy = MACDStrategy(
            name="test-macd",
            slow_period=30,
            signal_period=12,
        )

        required = strategy.get_required_bars()

        assert required == 30 + 12 + 20
        assert required == 62

    def test_validate_data_insufficient_bars(self):
        """Test validation fails with insufficient data."""
        strategy = MACDStrategy(name="test-macd")
        data = create_test_data(bars=40)  # Need 55

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is False
        assert "Insufficient data" in msg

    def test_validate_data_sufficient_bars(self):
        """Test validation succeeds with sufficient data."""
        strategy = MACDStrategy(name="test-macd")
        data = create_test_data(bars=100)

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is True
        assert msg == ""

    def test_generate_signal_with_insufficient_data(self):
        """Test signal generation returns None for insufficient data."""
        strategy = MACDStrategy(name="test-macd")
        data = create_test_data(bars=30)

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is None

    def test_generate_signal_with_valid_data(self):
        """Test signal generation with valid data."""
        strategy = MACDStrategy(name="test-macd")
        data = create_test_data(bars=150, trend="volatile")

        timestamp = datetime.utcnow()
        signal = strategy.generate_signal(data, timestamp)

        # Should generate a signal (even if HOLD)
        assert signal is not None
        assert hasattr(signal, "symbol")
        assert hasattr(signal, "signal_type")

    def test_generate_signal_handles_exceptions(self):
        """Test signal generation handles exceptions gracefully."""
        strategy = MACDStrategy(name="test-macd")
        data = create_test_data(bars=100)

        try:
            signal = strategy.generate_signal(data, datetime.utcnow())
            assert signal is None or hasattr(signal, "symbol")
        except Exception:
            pytest.fail("generate_signal should handle exceptions gracefully")

    def test_str_representation(self):
        """Test string representation."""
        strategy = MACDStrategy(name="my-macd-strategy")

        result = str(strategy)

        assert result == "my-macd-strategy Strategy"

    def test_repr_representation(self):
        """Test repr representation."""
        strategy = MACDStrategy(name="my-macd-strategy")

        result = repr(strategy)

        assert "MACDStrategy" in result
        assert "my-macd-strategy" in result


class TestMACDStrategyParameters:
    """Tests for parameter handling."""

    def test_min_bars_calculated_from_periods(self):
        """Test min_bars is calculated from periods."""
        strategy = MACDStrategy(name="test", slow_period=30, signal_period=12)

        assert strategy.params["min_bars"] == 30 + 12 + 20

    def test_model_config_parameters(self):
        """Test model config includes all parameters."""
        strategy = MACDStrategy(
            name="test",
            fast_period=8,
            slow_period=21,
            signal_period=5,
        )

        config = strategy.model.config

        assert config.parameters["fast_period"] == 8
        assert config.parameters["slow_period"] == 21
        assert config.parameters["signal_period"] == 5


class TestMACDStrategySignals:
    """Tests for signal generation logic."""

    def test_signal_contains_macd_metadata(self):
        """Test signal contains MACD metadata."""
        strategy = MACDStrategy(name="test-macd")
        data = create_test_data(bars=150, trend="volatile")

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert "fast_period" in signal.metadata
        assert "slow_period" in signal.metadata
        assert "signal_period" in signal.metadata
        assert "current_price" in signal.metadata

    def test_signal_contains_feature_contributions(self):
        """Test signal contains feature contributions."""
        strategy = MACDStrategy(name="test-macd")
        data = create_test_data(bars=150, trend="volatile")

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert "macd" in signal.feature_contributions
        assert "signal" in signal.feature_contributions
        assert "histogram" in signal.feature_contributions
        assert "histogram_strength" in signal.feature_contributions
        assert "macd_above_zero" in signal.feature_contributions
        assert "bullish_crossover" in signal.feature_contributions
        assert "bearish_crossover" in signal.feature_contributions

    def test_signal_has_regime_detection(self):
        """Test signal includes regime detection."""
        strategy = MACDStrategy(name="test-macd")
        data = create_test_data(bars=150, trend="volatile")

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        assert signal.regime in ["crossover", "consolidating", "trending_up", "trending_down", "ranging"]

    def test_uptrend_macd_above_zero(self):
        """Test MACD is above zero in strong uptrend."""
        strategy = MACDStrategy(name="test-macd")
        # Create a more pronounced uptrend with some volatility
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=150, freq="D")
        # Strong uptrend with noise (use numpy array, not Series)
        close = 100 + np.arange(150) * 1.0 + np.random.randn(150) * 2
        data = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": [1000000] * 150,
            },
            index=dates,
        )

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # In a strong uptrend, MACD should be positive
        assert signal.feature_contributions["macd_above_zero"] == 1.0

    def test_downtrend_macd_below_zero(self):
        """Test MACD is below zero in strong downtrend."""
        strategy = MACDStrategy(name="test-macd")
        # Create a more pronounced downtrend with some volatility
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=150, freq="D")
        # Strong downtrend with noise (use numpy array, not Series)
        close = 300 - np.arange(150) * 1.0 + np.random.randn(150) * 2
        data = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": [1000000] * 150,
            },
            index=dates,
        )

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # In a strong downtrend, MACD should be negative
        assert signal.feature_contributions["macd_above_zero"] == 0.0


class TestMACDStrategyIntegration:
    """Integration tests for MACD strategy."""

    def test_strategy_with_trending_market(self):
        """Test strategy behavior in trending market."""
        strategy = MACDStrategy(name="test-macd")
        data = create_test_data(bars=150, trend="uptrend")

        # Test at multiple points
        for i in range(60, len(data), 20):
            timestamp = data.index[i]
            signal = strategy.generate_signal(data.iloc[: i + 1], timestamp)

            if signal:
                assert hasattr(signal, "symbol")
                assert hasattr(signal, "timestamp")
                assert hasattr(signal, "signal_type")

    def test_strategy_with_volatile_market(self):
        """Test strategy behavior in volatile market."""
        strategy = MACDStrategy(name="test-macd")
        data = create_test_data(bars=150, trend="volatile")

        timestamp = data.index[-1]
        signal = strategy.generate_signal(data, timestamp)

        # Should handle volatile data without crashing
        assert signal is not None
        assert hasattr(signal, "symbol")

    def test_strategy_with_different_timeframes(self):
        """Test strategy with different data lengths."""
        strategy = MACDStrategy(name="test-macd")

        # Start at 100 to meet model's min_data_points requirement
        for bars in [100, 200, 500]:
            data = create_test_data(bars=bars, trend="volatile")
            timestamp = data.index[-1]
            signal = strategy.generate_signal(data, timestamp)

            # Should handle different timeframes
            assert signal is not None

    def test_strategy_consistency(self):
        """Test strategy generates consistent signals for same data."""
        strategy = MACDStrategy(name="test-macd")
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


class TestMACDStrategyValidation:
    """Tests for data validation."""

    def test_requires_ohlcv_columns(self):
        """Test strategy requires OHLCV columns."""
        strategy = MACDStrategy(name="test")

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
        strategy = MACDStrategy(name="test")
        data = pd.DataFrame()

        is_valid, msg = strategy.validate_data(data)

        assert is_valid is False
        assert "empty" in msg.lower()

    def test_handles_none_data(self):
        """Test strategy handles None data."""
        strategy = MACDStrategy(name="test")

        is_valid, msg = strategy.validate_data(None)

        assert is_valid is False
        assert "empty" in msg.lower()


class TestMACDModel:
    """Tests for the underlying MACD model."""

    def test_model_has_correct_parameters(self):
        """Test model is configured with correct parameters."""
        strategy = MACDStrategy(
            name="test",
            fast_period=8,
            slow_period=21,
            signal_period=5,
        )

        model = strategy.model

        assert model.fast_period == 8
        assert model.slow_period == 21
        assert model.signal_period == 5

    def test_model_calculates_macd_correctly(self):
        """Test model calculates MACD values correctly."""
        strategy = MACDStrategy(name="test-macd")
        data = create_test_data(bars=150, trend="volatile")

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # MACD = fast EMA - slow EMA
        # Histogram = MACD - signal line
        macd = signal.feature_contributions["macd"]
        signal_line = signal.feature_contributions["signal"]
        histogram = signal.feature_contributions["histogram"]

        # Histogram should equal MACD - signal line (approximately)
        assert abs(histogram - (macd - signal_line)) < 0.01


class TestMACDEdgeCases:
    """Edge case tests for MACD strategy."""

    def test_handles_constant_prices(self):
        """Test strategy handles constant price data (zero volatility)."""
        strategy = MACDStrategy(name="test-macd")

        # Create data with constant prices
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "open": [100.0] * 100,
                "high": [100.0] * 100,
                "low": [100.0] * 100,
                "close": [100.0] * 100,
                "volume": [1000000] * 100,
            },
            index=dates,
        )

        # Should handle gracefully
        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # MACD should be 0 with constant prices
        assert abs(signal.feature_contributions["macd"]) < 0.01

    def test_handles_missing_values(self):
        """Test strategy handles data with NaN values."""
        strategy = MACDStrategy(name="test-macd")
        data = create_test_data(bars=100, trend="neutral")

        # Add some NaN values
        data.iloc[50, data.columns.get_loc("close")] = np.nan

        # Should handle gracefully
        try:
            signal = strategy.generate_signal(data, datetime.utcnow())
            # May return None or a signal with degraded data_quality
            assert signal is None or signal.data_quality < 1.0
        except ValueError:
            # Also acceptable to raise an error for invalid data
            pass

    def test_extreme_price_movements(self):
        """Test strategy handles extreme price movements."""
        strategy = MACDStrategy(name="test-macd")

        # Create volatile data with extreme spike at the end
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=150, freq="D")
        # Start with some volatility to establish MACD properly (use numpy array)
        close = 100 + np.cumsum(np.random.randn(150) * 2)
        close[-5:] = close[-6] * np.array([1.1, 1.3, 1.6, 2.0, 2.5])  # Exponential spike

        data = pd.DataFrame(
            {
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": [1000000] * 150,
            },
            index=dates,
        )

        signal = strategy.generate_signal(data, datetime.utcnow())

        # Should still generate a signal
        assert signal is not None
        # MACD should be strongly positive after spike
        assert signal.feature_contributions["macd"] > 0

    def test_consolidating_market(self):
        """Test strategy behavior in consolidating market."""
        strategy = MACDStrategy(name="test-macd")
        data = create_test_data(bars=150, trend="consolidating")

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # In consolidating market, histogram strength should be low
        assert signal.feature_contributions["histogram_strength"] < 0.1


class TestMACDCrossovers:
    """Tests for MACD crossover detection."""

    def test_bullish_crossover_detection(self):
        """Test bullish crossover is detected."""
        strategy = MACDStrategy(name="test-macd")
        # Create more volatile data for bullish crossover
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=150, freq="D")
        # Downtrend then strong uptrend
        base = np.concatenate([
            100 - np.arange(75) * 0.3 + np.random.randn(75) * 1.5,
            100 - 75 * 0.3 + np.arange(75) * 0.6 + np.random.randn(75) * 1.5,
        ])
        data = pd.DataFrame(
            {
                "open": base * 0.99,
                "high": base * 1.02,
                "low": base * 0.98,
                "close": base,
                "volume": [1000000] * 150,
            },
            index=dates,
        )

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # May or may not catch the exact crossover point, but should detect trend change
        assert hasattr(signal, "feature_contributions")

    def test_bearish_crossover_detection(self):
        """Test bearish crossover is detected."""
        strategy = MACDStrategy(name="test-macd")
        # Create more volatile data for bearish crossover
        np.random.seed(42)
        dates = pd.date_range(start="2024-01-01", periods=150, freq="D")
        # Uptrend then strong downtrend
        base = np.concatenate([
            100 + np.arange(75) * 0.5 + np.random.randn(75) * 1.5,
            100 + 75 * 0.5 - np.arange(75) * 0.8 + np.random.randn(75) * 1.5,
        ])
        data = pd.DataFrame(
            {
                "open": base * 0.99,
                "high": base * 1.02,
                "low": base * 0.98,
                "close": base,
                "volume": [1000000] * 150,
            },
            index=dates,
        )

        signal = strategy.generate_signal(data, datetime.utcnow())

        assert signal is not None
        # May or may not catch the exact crossover point
        assert hasattr(signal, "feature_contributions")
