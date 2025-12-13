"""
Unit tests for Volatility Targeting position sizing.
"""

import numpy as np
import pandas as pd
import pytest

from ordinis.risk.position_sizing.volatility_targeting import (
    AdaptiveVolatilityTargeting,
    MultiAssetVolatilityTargeting,
    VolatilityCalculator,
    VolatilityEstimator,
    VolatilityTargetConfig,
    VolatilityTargeting,
    VolatilityTargetResult,
)


class TestVolatilityTargetConfig:
    """Test VolatilityTargetConfig dataclass."""

    def test_default_config(self):
        """Test default configuration."""
        config = VolatilityTargetConfig()

        assert config.target_volatility == 0.15
        assert config.lookback_period == 20
        assert config.vol_estimator == VolatilityEstimator.EWMA
        assert config.vol_floor == 0.05
        assert config.vol_cap == 0.50
        assert config.max_leverage == 2.0
        assert config.min_leverage == 0.1

    def test_custom_config(self):
        """Test custom configuration."""
        config = VolatilityTargetConfig(
            target_volatility=0.12,
            lookback_period=60,
            vol_estimator=VolatilityEstimator.YANG_ZHANG,
            max_leverage=1.5,
        )

        assert config.target_volatility == 0.12
        assert config.lookback_period == 60
        assert config.vol_estimator == VolatilityEstimator.YANG_ZHANG
        assert config.max_leverage == 1.5


class TestVolatilityCalculator:
    """Test VolatilityCalculator methods."""

    @pytest.fixture
    def returns(self):
        """Generate sample returns for testing."""
        np.random.seed(42)
        return np.random.normal(0.0005, 0.02, 100)

    @pytest.fixture
    def ohlc_data(self):
        """Generate OHLC data for testing."""
        np.random.seed(42)
        n = 100
        close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
        open_ = close * (1 + np.random.normal(0, 0.005, n))

        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})

    def test_initialization(self):
        """Test initialization with default and custom annualization."""
        calc = VolatilityCalculator()
        assert calc.annualization == 252.0
        assert calc.sqrt_annualization == pytest.approx(np.sqrt(252))

        calc_custom = VolatilityCalculator(annualization=365.0)
        assert calc_custom.annualization == 365.0

    def test_simple_volatility(self, returns):
        """Test simple standard deviation volatility."""
        calc = VolatilityCalculator(annualization=252.0)

        vol = calc.simple_volatility(returns, lookback=20)

        assert vol > 0
        # Annual vol should be daily vol * sqrt(252)
        daily_vol = np.std(returns[-20:])
        expected = daily_vol * np.sqrt(252)
        assert vol == pytest.approx(expected)

    def test_simple_volatility_short_series(self):
        """Test simple volatility with series shorter than lookback."""
        calc = VolatilityCalculator()
        short_returns = np.array([0.01, 0.02, -0.01])

        vol = calc.simple_volatility(short_returns, lookback=20)
        assert vol > 0  # Should use all available data

    def test_ewma_volatility(self, returns):
        """Test EWMA volatility calculation."""
        calc = VolatilityCalculator()

        vol = calc.ewma_volatility(returns, halflife=10)

        assert vol > 0
        # EWMA should give positive volatility

    def test_ewma_volatility_short_series(self):
        """Test EWMA with very short series."""
        calc = VolatilityCalculator()
        short_returns = np.array([0.01])

        vol = calc.ewma_volatility(short_returns, halflife=10)
        assert vol == 0.0  # Too few observations

    def test_parkinson_volatility(self, ohlc_data):
        """Test Parkinson high-low range estimator."""
        calc = VolatilityCalculator()

        vol = calc.parkinson_volatility(
            ohlc_data["High"].values, ohlc_data["Low"].values, lookback=20
        )

        assert vol > 0
        # Parkinson should give higher efficiency than simple

    def test_garman_klass_volatility(self, ohlc_data):
        """Test Garman-Klass OHLC estimator."""
        calc = VolatilityCalculator()

        vol = calc.garman_klass_volatility(
            ohlc_data["Open"].values,
            ohlc_data["High"].values,
            ohlc_data["Low"].values,
            ohlc_data["Close"].values,
            lookback=20,
        )

        assert vol > 0
        # Should handle negative variance gracefully

    def test_yang_zhang_volatility(self, ohlc_data):
        """Test Yang-Zhang volatility estimator."""
        calc = VolatilityCalculator()

        vol = calc.yang_zhang_volatility(
            ohlc_data["Open"].values,
            ohlc_data["High"].values,
            ohlc_data["Low"].values,
            ohlc_data["Close"].values,
            lookback=20,
        )

        assert vol > 0
        # Yang-Zhang is most efficient OHLC estimator

    def test_yang_zhang_short_series(self):
        """Test Yang-Zhang with series shorter than lookback."""
        calc = VolatilityCalculator()

        close = np.array([100, 101, 102, 103])
        high = np.array([101, 102, 103, 104])
        low = np.array([99, 100, 101, 102])
        open_ = np.array([100, 101, 102, 103])

        vol = calc.yang_zhang_volatility(open_, high, low, close, lookback=20)
        assert vol > 0  # Should adapt to available data


class TestVolatilityTargetResult:
    """Test VolatilityTargetResult dataclass."""

    def test_valid_result(self):
        """Test creation with valid result."""
        result = VolatilityTargetResult(
            target_leverage=1.2,
            current_volatility=0.18,
            target_volatility=0.15,
            position_scalar=1.2,
            is_capped=False,
            vol_estimator_used="ewma",
        )

        assert result.target_leverage == 1.2
        assert result.current_volatility == 0.18
        assert result.target_volatility == 0.15
        assert result.position_scalar == 1.2
        assert result.is_capped is False
        assert result.vol_estimator_used == "ewma"


class TestVolatilityTargeting:
    """Test VolatilityTargeting engine."""

    @pytest.fixture
    def returns(self):
        """Generate sample returns for testing."""
        np.random.seed(42)
        return np.random.normal(0.0005, 0.02, 100)

    @pytest.fixture
    def ohlc_data(self):
        """Generate OHLC data for testing."""
        np.random.seed(42)
        n = 100
        close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.02, n))
        high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
        open_ = close * (1 + np.random.normal(0, 0.005, n))

        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close})

    def test_initialization(self):
        """Test initialization with default and custom config."""
        vt = VolatilityTargeting()
        assert vt.config.target_volatility == 0.15

        config = VolatilityTargetConfig(target_volatility=0.12)
        vt_custom = VolatilityTargeting(config)
        assert vt_custom.config.target_volatility == 0.12

    def test_calculate_position_scalar_basic(self, returns):
        """Test basic position scalar calculation."""
        config = VolatilityTargetConfig(
            target_volatility=0.15, vol_estimator=VolatilityEstimator.SIMPLE
        )
        vt = VolatilityTargeting(config)

        result = vt.calculate_position_scalar(returns)

        assert result.target_leverage > 0
        assert result.current_volatility > 0
        assert result.target_volatility == 0.15
        # Leverage = target / current
        expected_leverage = 0.15 / result.current_volatility
        assert result.target_leverage == pytest.approx(
            min(max(expected_leverage, 0.1), 2.0), abs=0.01
        )

    def test_calculate_position_scalar_with_ohlc(self, returns, ohlc_data):
        """Test position scalar with OHLC data."""
        config = VolatilityTargetConfig(vol_estimator=VolatilityEstimator.YANG_ZHANG)
        vt = VolatilityTargeting(config)

        result = vt.calculate_position_scalar(returns, ohlc_data)

        assert result.target_leverage > 0
        assert result.vol_estimator_used == "yang_zhang"

    def test_calculate_position_scalar_applies_floor(self, returns):
        """Test that volatility floor is applied."""
        # Create very low volatility returns
        low_vol_returns = np.random.normal(0.0001, 0.001, 100)

        config = VolatilityTargetConfig(target_volatility=0.15, vol_floor=0.05, max_leverage=2.0)
        vt = VolatilityTargeting(config)

        result = vt.calculate_position_scalar(low_vol_returns)

        # Vol should be floored at 0.05
        assert result.current_volatility >= 0.05

    def test_calculate_position_scalar_applies_cap(self, returns):
        """Test that volatility cap is applied."""
        # Create very high volatility returns
        high_vol_returns = np.random.normal(0.0, 0.10, 100)

        config = VolatilityTargetConfig(target_volatility=0.15, vol_cap=0.30)
        vt = VolatilityTargeting(config)

        result = vt.calculate_position_scalar(high_vol_returns)

        # Vol should be capped at 0.30
        assert result.current_volatility <= 0.30

    def test_calculate_position_scalar_max_leverage(self, returns):
        """Test that max leverage is enforced."""
        config = VolatilityTargetConfig(target_volatility=0.30, max_leverage=1.5)
        vt = VolatilityTargeting(config)

        result = vt.calculate_position_scalar(returns)

        assert result.target_leverage <= 1.5
        if result.target_leverage == 1.5:
            assert result.is_capped is True

    def test_calculate_position_scalar_min_leverage(self, returns):
        """Test that min leverage is enforced."""
        config = VolatilityTargetConfig(target_volatility=0.05, min_leverage=0.2)
        vt = VolatilityTargeting(config)

        result = vt.calculate_position_scalar(returns)

        assert result.target_leverage >= 0.2

    def test_calculate_position_size(self, returns):
        """Test full position size calculation."""
        vt = VolatilityTargeting()

        position = vt.calculate_position_size(
            account_value=100000, asset_price=150.0, returns=returns
        )

        assert "shares" in position
        assert "notional" in position
        assert "leverage" in position
        assert "current_vol" in position
        assert "target_vol" in position
        assert "is_capped" in position
        assert position["shares"] > 0

    def test_estimate_volatility_fallback(self, returns):
        """Test OHLC estimator fallback to EWMA without OHLC data."""
        config = VolatilityTargetConfig(vol_estimator=VolatilityEstimator.YANG_ZHANG)
        vt = VolatilityTargeting(config)

        # No OHLC provided - should fallback to EWMA
        result = vt.calculate_position_scalar(returns, ohlc=None)

        assert result.target_leverage > 0  # Should work despite no OHLC

    def test_vol_history_tracking(self, returns):
        """Test that volatility history is tracked."""
        vt = VolatilityTargeting()

        assert len(vt._vol_history) == 0

        vt.calculate_position_scalar(returns)
        assert len(vt._vol_history) == 1

        vt.calculate_position_scalar(returns)
        assert len(vt._vol_history) == 2


class TestMultiAssetVolatilityTargeting:
    """Test MultiAssetVolatilityTargeting."""

    @pytest.fixture
    def returns_df(self):
        """Generate multi-asset returns DataFrame."""
        np.random.seed(42)
        n = 100
        assets = ["SPY", "TLT", "GLD"]

        data = {}
        for asset in assets:
            data[asset] = np.random.normal(0.0005, 0.015, n)

        return pd.DataFrame(data)

    def test_initialization(self):
        """Test initialization with default parameters."""
        mvt = MultiAssetVolatilityTargeting()

        assert mvt.target_volatility == 0.10
        assert mvt.correlation_lookback == 60
        assert mvt.max_concentration == 0.30

    def test_calculate_weights_equal_base(self, returns_df):
        """Test weight calculation with equal base weights."""
        mvt = MultiAssetVolatilityTargeting(target_volatility=0.10)

        weights = mvt.calculate_weights(returns_df)

        assert len(weights) == len(returns_df.columns) + 2  # +2 for meta fields
        assert "_total_leverage" in weights
        assert "_portfolio_vol" in weights

        # Check all asset weights are present
        for col in returns_df.columns:
            assert col in weights

    def test_calculate_weights_custom_base(self, returns_df):
        """Test weight calculation with custom base weights."""
        mvt = MultiAssetVolatilityTargeting()

        base_weights = np.array([0.5, 0.3, 0.2])
        weights = mvt.calculate_weights(returns_df, base_weights)

        assert len(weights) == 5  # 3 assets + 2 meta

    def test_calculate_weights_concentration_limit(self, returns_df):
        """Test that concentration limits are applied."""
        mvt = MultiAssetVolatilityTargeting(max_concentration=0.20)

        weights = mvt.calculate_weights(returns_df)

        # All individual weights should be <= max_concentration
        for col in returns_df.columns:
            assert abs(weights[col]) <= 0.20

    def test_calculate_weights_portfolio_vol(self, returns_df):
        """Test that portfolio volatility is calculated."""
        mvt = MultiAssetVolatilityTargeting(target_volatility=0.10)

        weights = mvt.calculate_weights(returns_df)

        assert weights["_portfolio_vol"] > 0


class TestAdaptiveVolatilityTargeting:
    """Test AdaptiveVolatilityTargeting."""

    def test_initialization(self):
        """Test initialization with default parameters."""
        avt = AdaptiveVolatilityTargeting()

        assert avt.base_target == 0.12
        assert avt.low_vol_target == 0.15
        assert avt.high_vol_target == 0.08

    def test_get_adaptive_target_high_vol(self):
        """Test adaptive target in high volatility regime."""
        avt = AdaptiveVolatilityTargeting(high_vol_target=0.08)

        target = avt.get_adaptive_target(current_vol=0.30, vol_percentile=0.85)

        assert target == 0.08  # High vol regime

    def test_get_adaptive_target_low_vol(self):
        """Test adaptive target in low volatility regime."""
        avt = AdaptiveVolatilityTargeting(low_vol_target=0.15)

        target = avt.get_adaptive_target(current_vol=0.10, vol_percentile=0.15)

        assert target == 0.15  # Low vol regime

    def test_get_adaptive_target_normal(self):
        """Test adaptive target in normal regime."""
        avt = AdaptiveVolatilityTargeting(base_target=0.12)

        target = avt.get_adaptive_target(current_vol=0.15, vol_percentile=0.50)

        assert target == 0.12  # Normal regime
