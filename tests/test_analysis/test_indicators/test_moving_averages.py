"""
Comprehensive tests for moving average indicators.
"""

import numpy as np
import pandas as pd
import pytest

from ordinis.analysis.technical.indicators.moving_averages import (
    MASignal,
    MovingAverages,
)


@pytest.fixture
def sample_prices():
    """Generate sample price data."""
    return pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109])


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    return {
        "high": pd.Series([102, 104, 106, 105, 107, 109, 108, 110, 112, 111]),
        "low": pd.Series([98, 100, 102, 101, 103, 105, 104, 106, 108, 107]),
        "close": pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109]),
        "volume": pd.Series([1000, 1200, 1100, 1300, 1150, 1250, 1100, 1200, 1300, 1150]),
    }


class TestMASignal:
    """Test MASignal dataclass."""

    def test_ma_signal_creation(self):
        """Test creating MA signal."""
        signal = MASignal(
            current_price=105.0,
            ma_value=100.0,
            ma_type="SMA",
            period=20,
            price_vs_ma=5.0,
            slope=0.5,
            crossover="golden",
        )
        assert signal.current_price == 105.0
        assert signal.ma_value == 100.0
        assert signal.ma_type == "SMA"
        assert signal.period == 20
        assert signal.price_vs_ma == 5.0
        assert signal.slope == 0.5
        assert signal.crossover == "golden"

    def test_ma_signal_no_crossover(self):
        """Test MA signal without crossover."""
        signal = MASignal(
            current_price=100.0,
            ma_value=100.0,
            ma_type="EMA",
            period=10,
            price_vs_ma=0.0,
            slope=0.0,
        )
        assert signal.crossover is None


class TestMovingAverages:
    """Test MovingAverages class."""

    def test_sma_calculation(self, sample_prices):
        """Test Simple Moving Average calculation."""
        sma = MovingAverages.sma(sample_prices, period=3)

        # First two values should be NaN
        assert pd.isna(sma.iloc[0])
        assert pd.isna(sma.iloc[1])

        # Third value should be mean of first 3 prices
        assert sma.iloc[2] == pytest.approx((100 + 102 + 104) / 3)

        # Fourth value
        assert sma.iloc[3] == pytest.approx((102 + 104 + 103) / 3)

    def test_sma_full_series(self):
        """Test SMA returns same length series."""
        prices = pd.Series([10, 20, 30, 40, 50])
        sma = MovingAverages.sma(prices, period=3)
        assert len(sma) == len(prices)

    def test_ema_calculation(self, sample_prices):
        """Test Exponential Moving Average calculation."""
        ema = MovingAverages.ema(sample_prices, period=3)

        # EMA gives more weight to recent prices
        assert len(ema) == len(sample_prices)
        assert not pd.isna(ema.iloc[-1])

    def test_ema_adjust_parameter(self, sample_prices):
        """Test EMA with adjust parameter."""
        ema_adjusted = MovingAverages.ema(sample_prices, period=5, adjust=True)
        ema_not_adjusted = MovingAverages.ema(sample_prices, period=5, adjust=False)

        # Both should have same length
        assert len(ema_adjusted) == len(ema_not_adjusted)

        # Values should differ slightly
        assert not np.allclose(ema_adjusted.dropna(), ema_not_adjusted.dropna())

    def test_wma_calculation(self, sample_prices):
        """Test Weighted Moving Average calculation."""
        wma = MovingAverages.wma(sample_prices, period=3)

        # First two values should be NaN
        assert pd.isna(wma.iloc[0])
        assert pd.isna(wma.iloc[1])

        # WMA gives more weight to recent prices than SMA
        sma = MovingAverages.sma(sample_prices, period=3)
        # In uptrend, WMA should be >= SMA
        assert wma.iloc[2] >= sma.iloc[2]

    def test_vwap_calculation(self, sample_ohlcv):
        """Test Volume Weighted Average Price calculation."""
        vwap = MovingAverages.vwap(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
        )

        # VWAP should exist and be a valid number
        assert not pd.isna(vwap.iloc[-1])
        assert vwap.iloc[-1] > 0

    def test_moving_averages_different_periods(self, sample_prices):
        """Test moving averages with different periods."""
        sma_5 = MovingAverages.sma(sample_prices, period=5)
        sma_10 = MovingAverages.sma(sample_prices, period=10)

        # Shorter period should have fewer NaN values
        assert sma_5.notna().sum() > sma_10.notna().sum()

    def test_moving_averages_uptrend(self):
        """Test moving averages in uptrend."""
        uptrend_prices = pd.Series([100, 105, 110, 115, 120, 125, 130])
        sma = MovingAverages.sma(uptrend_prices, period=3)

        # In uptrend, SMA should be increasing
        sma_values = sma.dropna().values
        assert all(sma_values[i] < sma_values[i + 1] for i in range(len(sma_values) - 1))

    def test_moving_averages_downtrend(self):
        """Test moving averages in downtrend."""
        downtrend_prices = pd.Series([130, 125, 120, 115, 110, 105, 100])
        sma = MovingAverages.sma(downtrend_prices, period=3)

        # In downtrend, SMA should be decreasing
        sma_values = sma.dropna().values
        assert all(sma_values[i] > sma_values[i + 1] for i in range(len(sma_values) - 1))

    def test_moving_averages_sideways(self):
        """Test moving averages in sideways market."""
        sideways_prices = pd.Series([100, 101, 99, 100, 102, 98, 100, 101, 99])
        sma = MovingAverages.sma(sideways_prices, period=3)

        # In sideways market, SMA should stay relatively flat
        sma_values = sma.dropna().values
        assert np.std(sma_values) < 2.0  # Low standard deviation

    def test_ema_responds_faster_than_sma(self):
        """Test that EMA responds faster to price changes than SMA."""
        # Price with sudden spike
        prices = pd.Series([100, 100, 100, 100, 100, 150, 150, 150])

        sma = MovingAverages.sma(prices, period=5)
        ema = MovingAverages.ema(prices, period=5)

        # After the spike, EMA should be closer to new price than SMA
        assert abs(ema.iloc[-1] - 150) < abs(sma.iloc[-1] - 150)

    def test_moving_averages_with_zeros(self):
        """Test moving averages handle zero prices."""
        prices_with_zeros = pd.Series([100, 0, 100, 0, 100])
        sma = MovingAverages.sma(prices_with_zeros, period=3)

        # Should handle zeros without error
        assert not pd.isna(sma.iloc[-1])

    def test_moving_averages_empty_series(self):
        """Test moving averages with empty series."""
        empty_prices = pd.Series(dtype=float)
        sma = MovingAverages.sma(empty_prices, period=5)

        assert len(sma) == 0

    def test_moving_averages_period_larger_than_data(self):
        """Test moving averages when period > data length."""
        short_prices = pd.Series([100, 105, 110])
        sma = MovingAverages.sma(short_prices, period=5)

        # All values should be NaN
        assert sma.isna().all()

    def test_wma_weights(self):
        """Test WMA applies correct weights."""
        prices = pd.Series([10, 20, 30])  # Simple sequence
        wma = MovingAverages.wma(prices, period=3)

        # WMA = (1*10 + 2*20 + 3*30) / (1+2+3) = (10+40+90)/6 = 140/6 = 23.33
        expected = (1 * 10 + 2 * 20 + 3 * 30) / (1 + 2 + 3)
        assert wma.iloc[-1] == pytest.approx(expected, rel=1e-2)

    def test_vwap_typical_price(self, sample_ohlcv):
        """Test VWAP uses typical price correctly."""
        vwap = MovingAverages.vwap(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
        )

        # VWAP should be cumulative
        assert len(vwap) == len(sample_ohlcv["close"])

    def test_moving_averages_with_nans(self):
        """Test moving averages handle NaN values."""
        prices_with_nans = pd.Series([100, np.nan, 105, np.nan, 110])
        sma = MovingAverages.sma(prices_with_nans, period=2)

        # Should propagate NaNs appropriately
        assert pd.isna(sma.iloc[1])

    def test_sma_period_one(self):
        """Test SMA with period=1 returns original series."""
        prices = pd.Series([100, 105, 110, 115], dtype=float)
        sma = MovingAverages.sma(prices, period=1)

        # Period=1 should return original values
        pd.testing.assert_series_equal(sma, prices)

    def test_ema_convergence(self):
        """Test EMA converges to stable value with constant prices."""
        constant_prices = pd.Series([100] * 50)
        ema = MovingAverages.ema(constant_prices, period=10)

        # Should converge to 100
        assert ema.iloc[-1] == pytest.approx(100, rel=1e-2)

    def test_crossover_detection(self, sample_prices):
        """Test detection of MA crossovers."""
        fast_ma = MovingAverages.ema(sample_prices, period=2)
        slow_ma = MovingAverages.ema(sample_prices, period=5)

        # Find crossover points
        diff = fast_ma - slow_ma
        crossovers = np.sign(diff).diff()

        # Positive crossover: fast crosses above slow (golden cross)
        golden_crosses = crossovers > 0

        # Negative crossover: fast crosses below slow (death cross)
        death_crosses = crossovers < 0

        # At least some crossovers should exist in volatile data
        assert golden_crosses.any() or death_crosses.any()
