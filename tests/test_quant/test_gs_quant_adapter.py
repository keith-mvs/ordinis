"""
Tests for the gs-quant adapter module.

This module tests the standalone quantitative finance functions adapted from
Goldman Sachs' gs-quant library.
"""

import numpy as np
import pandas as pd
import pytest

from ordinis.quant.gs_quant_adapter import (
    AnnualizationFactor,
    # Types
    Returns,
    Window,
    beta,
    bollinger_bands,
    correlation,
    generate_series,
    macd,
    max_drawdown,
    # Technical Indicators
    moving_average,
    percentiles,
    prices,
    # Returns/Prices
    returns,
    rolling_max,
    rolling_min,
    rolling_std,
    rsi,
    sharpe_ratio,
    # Risk Metrics
    volatility,
    # Statistics
    zscores,
)


class TestGenerateSeries:
    """Tests for generate_series utility."""

    def test_generate_series_length(self) -> None:
        series = generate_series(100)
        assert len(series) == 100

    def test_generate_series_reproducible(self) -> None:
        s1 = generate_series(50, seed=42)
        s2 = generate_series(50, seed=42)
        # Values should be equal (index timestamps may differ slightly)
        np.testing.assert_array_almost_equal(s1.values, s2.values)

    def test_generate_series_has_datetime_index(self) -> None:
        series = generate_series(10)
        assert isinstance(series.index, pd.DatetimeIndex)


class TestReturns:
    """Tests for returns calculation."""

    def test_simple_returns(self) -> None:
        prices_series = pd.Series([100, 110, 105, 115])
        ret = returns(prices_series, return_type=Returns.SIMPLE)

        assert np.isnan(ret.iloc[0])  # First value is NaN
        assert pytest.approx(ret.iloc[1], rel=1e-5) == 0.10  # 10% return
        assert pytest.approx(ret.iloc[2], rel=1e-5) == -0.0454545  # ~-4.5%
        assert pytest.approx(ret.iloc[3], rel=1e-5) == 0.0952381  # ~9.5%

    def test_log_returns(self) -> None:
        prices_series = pd.Series([100, 110, 105, 115])
        ret = returns(prices_series, return_type=Returns.LOGARITHMIC)

        assert np.isnan(ret.iloc[0])
        assert pytest.approx(ret.iloc[1], rel=1e-5) == np.log(110 / 100)

    def test_absolute_returns(self) -> None:
        prices_series = pd.Series([100, 110, 105, 115])
        ret = returns(prices_series, return_type=Returns.ABSOLUTE)

        assert np.isnan(ret.iloc[0])
        assert ret.iloc[1] == 10


class TestPrices:
    """Tests for prices reconstruction from returns."""

    def test_prices_from_simple_returns(self) -> None:
        ret = pd.Series([0.0, 0.10, -0.05, 0.05])
        price = prices(ret, initial=100, return_type=Returns.SIMPLE)

        assert pytest.approx(price.iloc[0], rel=1e-5) == 100
        assert pytest.approx(price.iloc[1], rel=1e-5) == 110
        assert pytest.approx(price.iloc[2], rel=1e-5) == 104.5
        assert pytest.approx(price.iloc[3], rel=1e-5) == 109.725


class TestMovingAverage:
    """Tests for moving average."""

    def test_simple_moving_average(self) -> None:
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ma = moving_average(series, w=3)

        # After ramp-up, should have valid values
        assert len(ma) == len(series) - 3  # Ramp removes first 3
        # First remaining value is index 3 (value 4), which is avg of 2,3,4 = 3.0
        assert pytest.approx(ma.iloc[0], rel=1e-5) == 3.0

    def test_moving_average_with_window_object(self) -> None:
        series = pd.Series([1, 2, 3, 4, 5])
        ma = moving_average(series, w=Window(3, 2))

        assert len(ma) == 3  # 5 - 2 = 3


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bollinger_bands_structure(self) -> None:
        series = generate_series(100, seed=42)
        bb = bollinger_bands(series, w=20, k=2)

        assert "lower" in bb.columns
        assert "middle" in bb.columns
        assert "upper" in bb.columns

    def test_bollinger_bands_order(self) -> None:
        series = generate_series(100, seed=42)
        bb = bollinger_bands(series, w=20, k=2)

        # Upper should always be >= middle >= lower
        assert (bb["upper"] >= bb["middle"]).all()
        assert (bb["middle"] >= bb["lower"]).all()


class TestRSI:
    """Tests for RSI indicator."""

    def test_rsi_bounds(self) -> None:
        series = generate_series(100, seed=42)
        rsi_values = rsi(series, w=14)

        # RSI should be between 0 and 100
        valid = rsi_values.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestMACD:
    """Tests for MACD indicator."""

    def test_macd_structure(self) -> None:
        series = generate_series(100, seed=42)
        result = macd(series)

        assert "macd" in result.columns
        assert "signal" in result.columns
        assert "histogram" in result.columns

    def test_macd_histogram_equals_difference(self) -> None:
        series = generate_series(100, seed=42)
        result = macd(series)

        expected_hist = result["macd"] - result["signal"]
        pd.testing.assert_series_equal(result["histogram"], expected_hist, check_names=False)


class TestVolatility:
    """Tests for volatility calculation."""

    def test_volatility_positive(self) -> None:
        series = generate_series(100, seed=42)
        vol = volatility(series, w=22)

        valid = vol.dropna()
        assert (valid >= 0).all()

    def test_volatility_annualized(self) -> None:
        # With known volatility, check annualization
        np.random.seed(42)
        daily_returns = np.random.normal(0, 0.01, 252)  # 1% daily vol
        prices_series = pd.Series(100 * np.cumprod(1 + daily_returns))
        prices_series.index = pd.date_range(end="2024-01-01", periods=252)

        # Use Window with ramp to get valid values
        vol = volatility(prices_series, w=Window(252, 0))
        # Should be approximately 15.8% (0.01 * sqrt(252) * 100)
        # Allow wider range due to random sampling variance
        assert 5 < vol.iloc[-1] < 30


class TestSharpeRatio:
    """Tests for Sharpe ratio."""

    def test_sharpe_ratio_scalar(self) -> None:
        series = generate_series(252, seed=42)
        sr = sharpe_ratio(series, risk_free_rate=0.0)

        assert isinstance(sr, float)

    def test_sharpe_ratio_rolling(self) -> None:
        series = generate_series(100, seed=42)
        sr = sharpe_ratio(series, w=22)

        assert isinstance(sr, pd.Series)


class TestMaxDrawdown:
    """Tests for max drawdown."""

    def test_max_drawdown_negative(self) -> None:
        series = generate_series(100, seed=42)
        dd = max_drawdown(series)

        valid = dd.dropna()
        assert (valid <= 0).all()

    def test_max_drawdown_known_values(self) -> None:
        # Simple case: peak at 100, drop to 80, recover
        series = pd.Series([100, 100, 90, 80, 85, 90, 95, 100])
        dd = max_drawdown(series)

        # Max drawdown should be -0.2 (20% from peak)
        assert pytest.approx(dd.min(), rel=1e-5) == -0.2


class TestCorrelation:
    """Tests for correlation calculation."""

    def test_correlation_bounds(self) -> None:
        s1 = generate_series(100, seed=42)
        s2 = generate_series(100, seed=123)

        corr = correlation(s1, s2, w=22)
        valid = corr.dropna()

        assert (valid >= -1).all()
        assert (valid <= 1).all()

    def test_correlation_with_self(self) -> None:
        series = generate_series(100, seed=42)
        corr = correlation(series, series, w=22)

        valid = corr.dropna()
        # Correlation with self should be 1
        assert pytest.approx(valid.iloc[-1], rel=1e-5) == 1.0


class TestBeta:
    """Tests for beta calculation."""

    def test_beta_with_self(self) -> None:
        series = generate_series(100, seed=42)
        b = beta(series, series, w=22)

        valid = b.dropna()
        # Beta with self should be 1
        assert pytest.approx(valid.iloc[-1], rel=1e-5) == 1.0


class TestZscores:
    """Tests for z-score calculation."""

    def test_zscores_full_series(self) -> None:
        series = pd.Series([1, 2, 3, 4, 5])
        z = zscores(series)

        # Mean of z-scores should be ~0
        assert pytest.approx(z.mean(), abs=1e-10) == 0

    def test_zscores_rolling(self) -> None:
        series = generate_series(100, seed=42)
        z = zscores(series, w=22)

        assert isinstance(z, pd.Series)


class TestPercentiles:
    """Tests for percentile calculation."""

    def test_percentiles_bounds(self) -> None:
        series = generate_series(100, seed=42)
        pct = percentiles(series, w=22)

        valid = pct.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()


class TestRollingStatistics:
    """Tests for rolling statistics functions."""

    def test_rolling_std(self) -> None:
        series = generate_series(100, seed=42)
        std = rolling_std(series, w=22)

        valid = std.dropna()
        assert (valid >= 0).all()

    def test_rolling_min_max(self) -> None:
        series = generate_series(100, seed=42)
        rmin = rolling_min(series, w=22)
        rmax = rolling_max(series, w=22)

        # Max should always be >= min
        assert (rmax >= rmin).all()


class TestWindowClass:
    """Tests for Window dataclass."""

    def test_window_defaults(self) -> None:
        w = Window(22)
        assert w.w == 22
        assert w.r == 22  # Defaults to w

    def test_window_custom_ramp(self) -> None:
        w = Window(22, 10)
        assert w.w == 22
        assert w.r == 10


class TestAnnualizationFactor:
    """Tests for AnnualizationFactor enum."""

    def test_annualization_values(self) -> None:
        assert AnnualizationFactor.DAILY == 252
        assert AnnualizationFactor.WEEKLY == 52
        assert AnnualizationFactor.MONTHLY == 12
