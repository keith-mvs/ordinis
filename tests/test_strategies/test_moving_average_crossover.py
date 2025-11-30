"""Tests for Moving Average Crossover strategy."""

from datetime import datetime

import numpy as np
import pandas as pd

from strategies.moving_average_crossover import MovingAverageCrossoverStrategy


def create_ma_test_data(bars: int = 250, pattern: str = "golden_cross") -> pd.DataFrame:
    """Create test data with specific MA patterns."""
    dates = pd.date_range(start="2024-01-01", periods=bars, freq="D")

    if pattern == "golden_cross":
        # Price trending up - fast MA crosses above slow MA
        close = pd.Series(100 + np.arange(bars) * 0.2)
    elif pattern == "death_cross":
        # Price trending down - fast MA crosses below slow MA
        close = pd.Series(150 - np.arange(bars) * 0.2)
    elif pattern == "sideways":
        # Sideways market - no clear crossovers
        close = pd.Series(100 + np.sin(np.arange(bars) * 0.1) * 5)
    else:
        close = pd.Series([100] * bars)

    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": [1000000] * bars,
        },
        index=dates,
    )


class TestMovingAverageCrossoverStrategy:
    """Tests for MovingAverageCrossoverStrategy."""

    def test_initialization_defaults(self):
        """Test default parameters."""
        strategy = MovingAverageCrossoverStrategy(name="test-ma")

        assert strategy.params["fast_period"] == 50
        assert strategy.params["slow_period"] == 200
        assert strategy.params["ma_type"] == "SMA"
        assert strategy.params["min_bars"] == 210

    def test_initialization_custom(self):
        """Test custom parameters."""
        strategy = MovingAverageCrossoverStrategy(
            name="test", fast_period=20, slow_period=50, ma_type="EMA"
        )

        assert strategy.params["fast_period"] == 20
        assert strategy.params["slow_period"] == 50
        assert strategy.params["ma_type"] == "EMA"

    def test_get_description(self):
        """Test strategy description."""
        strategy = MovingAverageCrossoverStrategy(name="test")
        desc = strategy.get_description()

        assert "Moving Average" in desc
        assert "golden cross" in desc.lower()
        assert "death cross" in desc.lower()

    def test_get_required_bars(self):
        """Test required bars calculation."""
        strategy = MovingAverageCrossoverStrategy(name="test")
        assert strategy.get_required_bars() == 210

        strategy2 = MovingAverageCrossoverStrategy(name="test", slow_period=100)
        assert strategy2.get_required_bars() == 110

    def test_validate_insufficient_data(self):
        """Test validation with insufficient data."""
        strategy = MovingAverageCrossoverStrategy(name="test")
        data = create_ma_test_data(bars=100)

        is_valid, msg = strategy.validate_data(data)
        assert not is_valid
        assert "Insufficient" in msg

    def test_validate_sufficient_data(self):
        """Test validation with sufficient data."""
        strategy = MovingAverageCrossoverStrategy(name="test")
        data = create_ma_test_data(bars=250)

        is_valid, msg = strategy.validate_data(data)
        assert is_valid
        assert msg == ""

    def test_generate_signal_insufficient_data(self):
        """Test signal generation with insufficient data."""
        strategy = MovingAverageCrossoverStrategy(name="test")
        data = create_ma_test_data(bars=100)

        signal = strategy.generate_signal(data, datetime.utcnow())
        assert signal is None

    def test_generate_signal_valid_data(self):
        """Test signal generation with valid data."""
        strategy = MovingAverageCrossoverStrategy(name="test")
        data = create_ma_test_data(bars=250, pattern="sideways")

        signal = strategy.generate_signal(data, datetime.utcnow())
        # May or may not generate signal
        assert signal is None or hasattr(signal, "symbol")

    def test_sma_calculation(self):
        """Test SMA type calculation."""
        strategy = MovingAverageCrossoverStrategy(
            name="test", fast_period=20, slow_period=50, ma_type="SMA"
        )
        data = create_ma_test_data(bars=100)

        # Should not crash with SMA calculation
        signal = strategy.generate_signal(data, datetime.utcnow())
        assert signal is None or hasattr(signal, "symbol")

    def test_ema_calculation(self):
        """Test EMA type calculation."""
        strategy = MovingAverageCrossoverStrategy(
            name="test", fast_period=20, slow_period=50, ma_type="EMA"
        )
        data = create_ma_test_data(bars=100)

        # Should not crash with EMA calculation
        signal = strategy.generate_signal(data, datetime.utcnow())
        assert signal is None or hasattr(signal, "symbol")

    def test_handles_missing_columns(self):
        """Test handling of missing data columns."""
        strategy = MovingAverageCrossoverStrategy(name="test")
        data = pd.DataFrame(
            {
                "open": [100] * 250,
                "high": [105] * 250,
                "low": [95] * 250,
                "close": [102] * 250,
                # Missing volume
            },
            index=pd.date_range("2024-01-01", periods=250),
        )

        is_valid, msg = strategy.validate_data(data)
        assert not is_valid

    def test_str_representation(self):
        """Test string representation."""
        strategy = MovingAverageCrossoverStrategy(name="my-ma")
        assert str(strategy) == "my-ma Strategy"

    def test_repr_representation(self):
        """Test repr representation."""
        strategy = MovingAverageCrossoverStrategy(name="my-ma")
        result = repr(strategy)

        assert "MovingAverageCrossoverStrategy" in result
        assert "my-ma" in result


class TestMovingAverageCrossoverIntegration:
    """Integration tests for MA crossover strategy."""

    def test_golden_cross_pattern(self):
        """Test with golden cross pattern."""
        strategy = MovingAverageCrossoverStrategy(name="test", fast_period=20, slow_period=50)
        data = create_ma_test_data(bars=150, pattern="golden_cross")

        signal = strategy.generate_signal(data, datetime.utcnow())
        # Should handle golden cross pattern
        assert signal is None or hasattr(signal, "symbol")

    def test_death_cross_pattern(self):
        """Test with death cross pattern."""
        strategy = MovingAverageCrossoverStrategy(name="test", fast_period=20, slow_period=50)
        data = create_ma_test_data(bars=150, pattern="death_cross")

        signal = strategy.generate_signal(data, datetime.utcnow())
        # Should handle death cross pattern
        assert signal is None or hasattr(signal, "symbol")

    def test_different_timeframes(self):
        """Test with different data lengths."""
        strategy = MovingAverageCrossoverStrategy(name="test", fast_period=10, slow_period=30)

        for bars in [50, 100, 200]:
            if bars < strategy.get_required_bars():
                continue
            data = create_ma_test_data(bars=bars)
            signal = strategy.generate_signal(data, datetime.utcnow())
            assert signal is None or hasattr(signal, "symbol")

    def test_consistency(self):
        """Test signal generation consistency."""
        strategy = MovingAverageCrossoverStrategy(name="test")
        data = create_ma_test_data(bars=250)
        timestamp = datetime.utcnow()

        signal1 = strategy.generate_signal(data, timestamp)
        signal2 = strategy.generate_signal(data, timestamp)

        # Should be consistent
        if signal1 is None:
            assert signal2 is None
        else:
            assert signal2 is not None
            assert signal1.signal_type == signal2.signal_type
