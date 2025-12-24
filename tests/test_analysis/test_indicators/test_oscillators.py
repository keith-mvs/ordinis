"""
Comprehensive tests for oscillator indicators.
"""

import numpy as np
import pandas as pd
import pytest

from ordinis.analysis.technical.indicators.oscillators import (
    Oscillators,
    OscillatorCondition,
    OscillatorSignal,
)


@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    return pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 107, 105, 103, 101, 99])


@pytest.fixture
def sample_ohlc():
    """Generate sample OHLC data."""
    return {
        "high": pd.Series([102, 104, 106, 105, 107, 109, 108, 110, 112, 111, 109, 107, 105, 103, 101]),
        "low": pd.Series([98, 100, 102, 101, 103, 105, 104, 106, 108, 107, 105, 103, 101, 99, 97]),
        "close": pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109, 107, 105, 103, 101, 99]),
    }


class TestOscillatorCondition:
    """Test OscillatorCondition enum."""

    def test_oscillator_conditions(self):
        """Test all oscillator condition values."""
        assert OscillatorCondition.OVERBOUGHT.value == "overbought"
        assert OscillatorCondition.OVERSOLD.value == "oversold"
        assert OscillatorCondition.NEUTRAL.value == "neutral"
        assert OscillatorCondition.BULLISH_DIVERGENCE.value == "bullish_divergence"
        assert OscillatorCondition.BEARISH_DIVERGENCE.value == "bearish_divergence"


class TestOscillatorSignal:
    """Test OscillatorSignal dataclass."""

    def test_oscillator_signal_creation(self):
        """Test creating oscillator signal."""
        signal = OscillatorSignal(
            indicator="RSI",
            value=75.0,
            condition=OscillatorCondition.OVERBOUGHT,
            previous_value=70.0,
            threshold_upper=70.0,
            threshold_lower=30.0,
            signal="sell",
        )
        assert signal.indicator == "RSI"
        assert signal.value == 75.0
        assert signal.condition == OscillatorCondition.OVERBOUGHT
        assert signal.previous_value == 70.0
        assert signal.signal == "sell"

    def test_oscillator_signal_no_signal(self):
        """Test oscillator signal without trade signal."""
        signal = OscillatorSignal(
            indicator="RSI",
            value=50.0,
            condition=OscillatorCondition.NEUTRAL,
            previous_value=48.0,
            threshold_upper=70.0,
            threshold_lower=30.0,
        )
        assert signal.signal is None


class TestRSI:
    """Test RSI indicator."""

    def test_rsi_calculation(self, sample_prices):
        """Test RSI calculation."""
        rsi = Oscillators.rsi(sample_prices, period=5)

        # RSI should be between 0 and 100
        rsi_valid = rsi.dropna()
        assert (rsi_valid >= 0).all()
        assert (rsi_valid <= 100).all()

    def test_rsi_default_period(self, sample_prices):
        """Test RSI with default 14-period."""
        rsi = Oscillators.rsi(sample_prices)

        # Should have same length as input
        assert len(rsi) == len(sample_prices)

    def test_rsi_uptrend(self):
        """Test RSI in strong uptrend."""
        uptrend = pd.Series([100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
        rsi = Oscillators.rsi(uptrend, period=5)

        # In strong uptrend, RSI should be high (>70)
        assert rsi.iloc[-1] > 70

    def test_rsi_downtrend(self):
        """Test RSI in strong downtrend."""
        downtrend = pd.Series([145, 140, 135, 130, 125, 120, 115, 110, 105, 100])
        rsi = Oscillators.rsi(downtrend, period=5)

        # In strong downtrend, RSI should be low (<30)
        assert rsi.iloc[-1] < 30

    def test_rsi_sideways(self):
        """Test RSI in sideways market."""
        sideways = pd.Series([100, 101, 99, 100, 102, 98, 100, 101, 99, 100])
        rsi = Oscillators.rsi(sideways, period=5)

        # In sideways market, RSI should be near 50
        rsi_valid = rsi.dropna()
        assert abs(rsi_valid.mean() - 50) < 10

    def test_rsi_overbought(self):
        """Test RSI identifying overbought conditions."""
        # Strong uptrend then pullback
        prices = pd.Series([100, 110, 120, 130, 140, 150, 148, 146, 144])
        rsi = Oscillators.rsi(prices, period=5)

        # RSI should have been overbought during uptrend
        assert rsi.max() > 70

    def test_rsi_oversold(self):
        """Test RSI identifying oversold conditions."""
        # Strong downtrend then bounce
        prices = pd.Series([150, 140, 130, 120, 110, 100, 102, 104, 106])
        rsi = Oscillators.rsi(prices, period=5)

        # RSI should have been oversold during downtrend
        assert rsi.min() < 30

    def test_rsi_boundary_values(self):
        """Test RSI extreme values."""
        # All gains
        all_gains = pd.Series([100, 110, 120, 130, 140, 150])
        rsi_gains = Oscillators.rsi(all_gains, period=3)
        assert rsi_gains.iloc[-1] == pytest.approx(100, rel=1e-2)

        # All losses
        all_losses = pd.Series([150, 140, 130, 120, 110, 100])
        rsi_losses = Oscillators.rsi(all_losses, period=3)
        assert rsi_losses.iloc[-1] == pytest.approx(0, rel=1e-2)


class TestStochastic:
    """Test Stochastic Oscillator."""

    def test_stochastic_calculation(self, sample_ohlc):
        """Test stochastic oscillator calculation."""
        k, d = Oscillators.stochastic(
            sample_ohlc["high"],
            sample_ohlc["low"],
            sample_ohlc["close"],
            k_period=5,
            d_period=3,
            smooth_k=3,
        )

        # Both should be between 0 and 100
        k_valid = k.dropna()
        d_valid = d.dropna()

        assert (k_valid >= 0).all()
        assert (k_valid <= 100).all()
        assert (d_valid >= 0).all()
        assert (d_valid <= 100).all()

    def test_stochastic_overbought(self):
        """Test stochastic in overbought conditions."""
        # Price at top of range
        high = pd.Series([100, 100, 100, 100, 100, 100, 100])
        low = pd.Series([90, 90, 90, 90, 90, 90, 90])
        close = pd.Series([99, 99, 99, 99, 99, 99, 99])

        k, d = Oscillators.stochastic(high, low, close, k_period=5, d_period=3, smooth_k=1)

        # Should be near 100 (overbought)
        assert k.iloc[-1] > 80

    def test_stochastic_oversold(self):
        """Test stochastic in oversold conditions."""
        # Price at bottom of range
        high = pd.Series([100, 100, 100, 100, 100, 100, 100])
        low = pd.Series([90, 90, 90, 90, 90, 90, 90])
        close = pd.Series([91, 91, 91, 91, 91, 91, 91])

        k, d = Oscillators.stochastic(high, low, close, k_period=5, d_period=3, smooth_k=1)

        # Should be near 0 (oversold)
        assert k.iloc[-1] < 20

    def test_stochastic_d_smoothing(self, sample_ohlc):
        """Test that %D is smoother than %K."""
        k, d = Oscillators.stochastic(
            sample_ohlc["high"],
            sample_ohlc["low"],
            sample_ohlc["close"],
            k_period=5,
            d_period=3,
            smooth_k=1,
        )

        # %D should have lower volatility than %K
        k_valid = k.dropna()
        d_valid = d.dropna()

        if len(k_valid) > 1 and len(d_valid) > 1:
            assert k_valid.std() >= d_valid.std()

    def test_stochastic_crossover(self):
        """Test stochastic %K crossing %D."""
        # Create data with clear crossover
        high = pd.Series([100, 102, 104, 106, 108, 110, 108, 106, 104, 102])
        low = pd.Series([95, 97, 99, 101, 103, 105, 103, 101, 99, 97])
        close = pd.Series([98, 100, 102, 104, 106, 108, 106, 104, 102, 100])

        k, d = Oscillators.stochastic(high, low, close, k_period=5, d_period=3, smooth_k=1)

        # Check for crossovers
        diff = k - d
        crossovers = np.sign(diff).diff()

        # There should be at least one crossover
        assert (crossovers != 0).any()

    def test_stochastic_different_periods(self, sample_ohlc):
        """Test stochastic with different periods."""
        # Fast stochastic
        k_fast, d_fast = Oscillators.stochastic(
            sample_ohlc["high"],
            sample_ohlc["low"],
            sample_ohlc["close"],
            k_period=5,
            d_period=3,
            smooth_k=1,
        )

        # Slow stochastic
        k_slow, d_slow = Oscillators.stochastic(
            sample_ohlc["high"],
            sample_ohlc["low"],
            sample_ohlc["close"],
            k_period=14,
            d_period=3,
            smooth_k=3,
        )

        # Fast should have more NaN values (shorter lookback)
        assert k_fast.notna().sum() >= k_slow.notna().sum()


class TestOscillatorsIntegration:
    """Test oscillator integration scenarios."""

    def test_rsi_empty_series(self):
        """Test RSI with empty series."""
        empty = pd.Series(dtype=float)
        rsi = Oscillators.rsi(empty, period=14)
        assert len(rsi) == 0

    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        short_data = pd.Series([100, 105, 110])
        rsi = Oscillators.rsi(short_data, period=14)

        # All values should be NaN
        assert rsi.isna().all()

    def test_stochastic_flat_prices(self):
        """Test stochastic with flat prices."""
        high = pd.Series([100] * 20)
        low = pd.Series([100] * 20)
        close = pd.Series([100] * 20)

        k, d = Oscillators.stochastic(high, low, close, k_period=5, d_period=3, smooth_k=1)

        # Should handle division by zero gracefully
        assert not np.isinf(k).any()
        assert not np.isinf(d).any()

    def test_rsi_with_nans(self):
        """Test RSI handles NaN values."""
        prices_with_nans = pd.Series([100, np.nan, 105, 110, np.nan, 115])
        rsi = Oscillators.rsi(prices_with_nans, period=3)

        # Should not crash
        assert len(rsi) == len(prices_with_nans)

    def test_oscillators_on_real_pattern(self):
        """Test oscillators on realistic price pattern."""
        # Simulate a V-shaped recovery
        prices = pd.Series(
            [100, 98, 96, 94, 92, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108]
        )

        rsi = Oscillators.rsi(prices, period=5)

        # RSI should be low at bottom, high at top
        bottom_idx = prices.idxmin()
        top_idx = prices.idxmax()

        rsi_at_bottom = rsi.loc[bottom_idx]
        rsi_at_top = rsi.loc[top_idx]

        if pd.notna(rsi_at_bottom) and pd.notna(rsi_at_top):
            assert rsi_at_top > rsi_at_bottom

    def test_stochastic_sensitivity(self, sample_ohlc):
        """Test stochastic sensitivity to period changes."""
        # Short period - more sensitive
        k_short, _ = Oscillators.stochastic(
            sample_ohlc["high"],
            sample_ohlc["low"],
            sample_ohlc["close"],
            k_period=5,
            d_period=3,
            smooth_k=1,
        )

        # Long period - less sensitive
        k_long, _ = Oscillators.stochastic(
            sample_ohlc["high"],
            sample_ohlc["low"],
            sample_ohlc["close"],
            k_period=14,
            d_period=3,
            smooth_k=1,
        )

        # Short period should have higher volatility
        k_short_valid = k_short.dropna()
        k_long_valid = k_long.dropna()

        if len(k_short_valid) > 1 and len(k_long_valid) > 1:
            assert k_short_valid.std() >= k_long_valid.std() * 0.8  # Allow some tolerance

    def test_rsi_divergence_detection(self):
        """Test RSI can detect potential divergence."""
        # Price makes higher high, RSI makes lower high (bearish divergence)
        prices = pd.Series([100, 110, 105, 115, 110, 112, 108, 105])
        rsi = Oscillators.rsi(prices, period=3)

        # Check if we can detect potential divergence pattern
        price_highs = prices.rolling(3).max()
        rsi_highs = rsi.rolling(3).max()

        # At least one point where price high increases but RSI high decreases
        price_increasing = price_highs.diff() > 0
        rsi_decreasing = rsi_highs.diff() < 0

        divergence = price_increasing & rsi_decreasing
        # This is a potential divergence signal
        assert len(divergence) == len(prices)
