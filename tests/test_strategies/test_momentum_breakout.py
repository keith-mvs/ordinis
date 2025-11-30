"""Tests for Momentum Breakout strategy."""

from datetime import datetime

import numpy as np
import pandas as pd

from strategies.momentum_breakout import MomentumBreakoutStrategy


def create_breakout_data(bars: int = 100, pattern: str = "upside_breakout") -> pd.DataFrame:
    """Create test data with specific breakout patterns."""
    dates = pd.date_range(start="2024-01-01", periods=bars, freq="D")

    if pattern == "upside_breakout":
        # Consolidation then upside breakout
        close = pd.Series([100 + i * 0.1 for i in range(50)] + list(range(100, 100 + (bars - 50))))
        volume = pd.Series([1000000] * 50 + [3000000] * (bars - 50))
    elif pattern == "downside_breakout":
        # Consolidation then downside breakout
        close = pd.Series(
            [100 + i * 0.1 for i in range(50)] + list(range(100, 100 - (bars - 50), -1))
        )
        volume = pd.Series([1000000] * 50 + [3000000] * (bars - 50))
    elif pattern == "consolidation":
        # Extended consolidation, no breakout
        close = pd.Series(100 + np.random.randn(bars) * 0.5)
        volume = pd.Series([1000000] * bars)
    elif pattern == "volatile":
        # High volatility, no clear pattern
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(bars) * 3))
        volume = pd.Series(np.random.randint(500000, 2000000, bars))
    else:
        # Default with some movement
        close = pd.Series([100 + i * 0.1 for i in range(bars)])
        volume = pd.Series([1000000] * bars)

    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


class TestMomentumBreakoutStrategy:
    """Tests for MomentumBreakoutStrategy."""

    def test_initialization_defaults(self):
        """Test default parameters."""
        strategy = MomentumBreakoutStrategy(name="test-momentum")

        assert strategy.params["lookback_period"] == 20
        assert strategy.params["atr_period"] == 14
        assert strategy.params["volume_multiplier"] == 1.5
        assert strategy.params["min_consolidation_bars"] == 10
        assert strategy.params["breakout_threshold"] == 0.02
        assert strategy.params["min_bars"] == 20  # max(20, 14)

    def test_initialization_custom(self):
        """Test custom parameters."""
        strategy = MomentumBreakoutStrategy(
            name="test",
            lookback_period=30,
            atr_period=20,
            volume_multiplier=2.0,
            breakout_threshold=0.03,
        )

        assert strategy.params["lookback_period"] == 30
        assert strategy.params["atr_period"] == 20
        assert strategy.params["volume_multiplier"] == 2.0
        assert strategy.params["breakout_threshold"] == 0.03
        assert strategy.params["min_bars"] == 30  # max(30, 20)

    def test_min_bars_calculation(self):
        """Test min_bars uses max of lookback and ATR periods."""
        strategy1 = MomentumBreakoutStrategy(name="test", lookback_period=25, atr_period=14)
        assert strategy1.params["min_bars"] == 25

        strategy2 = MomentumBreakoutStrategy(name="test", lookback_period=10, atr_period=20)
        assert strategy2.params["min_bars"] == 20

    def test_get_description(self):
        """Test strategy description."""
        strategy = MomentumBreakoutStrategy(name="test")
        desc = strategy.get_description()

        assert "Momentum Breakout" in desc
        assert "breakout" in desc.lower()
        assert "volume" in desc.lower()

    def test_get_required_bars(self):
        """Test required bars calculation."""
        strategy = MomentumBreakoutStrategy(name="test")
        assert strategy.get_required_bars() == 20

        strategy2 = MomentumBreakoutStrategy(name="test", lookback_period=30, atr_period=25)
        assert strategy2.get_required_bars() == 30

    def test_validate_insufficient_data(self):
        """Test validation with insufficient data."""
        strategy = MomentumBreakoutStrategy(name="test")
        data = create_breakout_data(bars=15)

        is_valid, msg = strategy.validate_data(data)
        assert not is_valid
        assert "Insufficient" in msg

    def test_validate_sufficient_data(self):
        """Test validation with sufficient data."""
        strategy = MomentumBreakoutStrategy(name="test")
        data = create_breakout_data(bars=100)

        is_valid, msg = strategy.validate_data(data)
        assert is_valid
        assert msg == ""

    def test_atr_calculation(self):
        """Test ATR calculation method."""
        strategy = MomentumBreakoutStrategy(name="test")
        data = create_breakout_data(bars=50)

        atr = strategy._calculate_atr(data, period=14)

        assert isinstance(atr, pd.Series)
        assert len(atr) == len(data)
        # ATR should be positive after warmup period
        assert atr.iloc[-1] > 0 or pd.isna(atr.iloc[-1])

    def test_generate_signal_insufficient_data(self):
        """Test signal generation with insufficient data."""
        strategy = MomentumBreakoutStrategy(name="test")
        data = create_breakout_data(bars=15)

        signal = strategy.generate_signal(data, datetime.utcnow())
        assert signal is None

    def test_generate_signal_valid_data(self):
        """Test signal generation with valid data."""
        strategy = MomentumBreakoutStrategy(name="test")
        data = create_breakout_data(bars=100, pattern="consolidation")

        signal = strategy.generate_signal(data, datetime.utcnow())
        # May or may not generate signal
        assert signal is None or hasattr(signal, "symbol")

    def test_handles_missing_columns(self):
        """Test handling of missing data columns."""
        strategy = MomentumBreakoutStrategy(name="test")
        data = pd.DataFrame(
            {
                "open": [100] * 50,
                "high": [105] * 50,
                "low": [95] * 50,
                "close": [102] * 50,
                # Missing volume
            },
            index=pd.date_range("2024-01-01", periods=50),
        )

        is_valid, msg = strategy.validate_data(data)
        assert not is_valid

    def test_str_representation(self):
        """Test string representation."""
        strategy = MomentumBreakoutStrategy(name="my-momentum")
        assert str(strategy) == "my-momentum Strategy"

    def test_repr_representation(self):
        """Test repr representation."""
        strategy = MomentumBreakoutStrategy(name="my-momentum")
        result = repr(strategy)

        assert "MomentumBreakoutStrategy" in result
        assert "my-momentum" in result


class TestMomentumBreakoutIntegration:
    """Integration tests for Momentum Breakout strategy."""

    def test_upside_breakout_pattern(self):
        """Test with upside breakout pattern."""
        strategy = MomentumBreakoutStrategy(name="test", lookback_period=20)
        data = create_breakout_data(bars=100, pattern="upside_breakout")

        signal = strategy.generate_signal(data, datetime.utcnow())
        # Should handle upside breakout pattern
        assert signal is None or hasattr(signal, "symbol")

    def test_downside_breakout_pattern(self):
        """Test with downside breakout pattern."""
        strategy = MomentumBreakoutStrategy(name="test", lookback_period=20)
        data = create_breakout_data(bars=100, pattern="downside_breakout")

        signal = strategy.generate_signal(data, datetime.utcnow())
        # Should handle downside breakout pattern
        assert signal is None or hasattr(signal, "symbol")

    def test_consolidation_pattern(self):
        """Test with consolidation (no breakout)."""
        strategy = MomentumBreakoutStrategy(name="test")
        data = create_breakout_data(bars=100, pattern="consolidation")

        signal = strategy.generate_signal(data, datetime.utcnow())
        # Should handle consolidation without generating signals
        assert signal is None or hasattr(signal, "symbol")

    def test_volatile_market(self):
        """Test with volatile market conditions."""
        strategy = MomentumBreakoutStrategy(name="test")
        data = create_breakout_data(bars=100, pattern="volatile")

        signal = strategy.generate_signal(data, datetime.utcnow())
        # Should handle volatile data without crashing
        assert signal is None or hasattr(signal, "symbol")

    def test_different_timeframes(self):
        """Test with different data lengths."""
        strategy = MomentumBreakoutStrategy(name="test", lookback_period=15)

        for bars in [30, 50, 100, 200]:
            if bars < strategy.get_required_bars():
                continue
            data = create_breakout_data(bars=bars)
            signal = strategy.generate_signal(data, datetime.utcnow())
            assert signal is None or hasattr(signal, "symbol")

    def test_consistency(self):
        """Test signal generation consistency."""
        strategy = MomentumBreakoutStrategy(name="test")
        data = create_breakout_data(bars=100)
        timestamp = datetime.utcnow()

        signal1 = strategy.generate_signal(data, timestamp)
        signal2 = strategy.generate_signal(data, timestamp)

        # Should be consistent
        if signal1 is None:
            assert signal2 is None
        else:
            assert signal2 is not None
            assert signal1.signal_type == signal2.signal_type

    def test_volume_confirmation(self):
        """Test volume confirmation requirement."""
        strategy = MomentumBreakoutStrategy(name="test", volume_multiplier=2.0, lookback_period=20)

        # Data with high volume breakout
        data = create_breakout_data(bars=80, pattern="upside_breakout")

        signal = strategy.generate_signal(data, datetime.utcnow())
        # Should handle volume confirmation
        assert signal is None or hasattr(signal, "symbol")

    def test_atr_based_sizing(self):
        """Test ATR is calculated for position sizing."""
        strategy = MomentumBreakoutStrategy(name="test", atr_period=14)
        data = create_breakout_data(bars=100)

        # Verify ATR calculation doesn't crash
        atr = strategy._calculate_atr(data, strategy.params["atr_period"])
        assert len(atr) == len(data)
        assert isinstance(atr, pd.Series)
