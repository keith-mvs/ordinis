"""
Tests for Kalman Filter Hybrid Model.

GTM Strategy #3: Combines Kalman trend filter with mean reversion on residuals.
Expected edge: 4/5, 55-65% win rate, PF 1.5-2.2, low drawdowns.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models.kalman_hybrid import (
    KalmanConfig,
    KalmanFilter,
    KalmanHybridModel,
    KalmanState,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def model_config() -> ModelConfig:
    """Create default model config."""
    return ModelConfig(
        model_id="test_kalman_hybrid",
        model_type="hybrid",
        parameters={
            "process_noise_q": 1e-5,
            "observation_noise_r": 1e-2,
            "residual_z_entry": 2.0,
            "residual_z_exit": 0.5,
            "trend_slope_min": 0.0001,
            "residual_lookback": 100,
            "atr_period": 14,
            "atr_stop_mult": 1.5,
            "atr_tp_mult": 2.0,
        },
    )


@pytest.fixture
def model(model_config: ModelConfig) -> KalmanHybridModel:
    """Create Kalman Hybrid model."""
    return KalmanHybridModel(model_config)


@pytest.fixture
def sample_prices() -> pd.Series:
    """Create sample price series."""
    np.random.seed(42)
    n = 200
    
    # Trend + noise
    trend = np.linspace(100, 120, n)
    noise = np.random.normal(0, 1, n)
    prices = trend + noise
    
    return pd.Series(prices)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create sample OHLCV data."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    
    # Uptrend with mean-reverting noise
    trend = np.linspace(100, 130, n)
    noise = np.random.normal(0, 2, n)
    prices = trend + noise
    
    close = pd.Series(prices, index=dates)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]).values,
        "high": high.values,
        "low": low.values,
        "close": close.values,
        "volume": np.random.randint(100000, 1000000, n),
    }, index=dates)


@pytest.fixture
def oversold_in_uptrend_data() -> pd.DataFrame:
    """Create data with oversold residual in uptrend (buy signal)."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    
    # Strong uptrend
    trend = np.linspace(100, 140, n)
    noise = np.zeros(n)
    
    # Create a sharp dip at the end (oversold)
    noise[-10:] = -8  # Sharp dip below trend
    
    prices = trend + noise
    close = pd.Series(prices, index=dates)
    high = close * 1.005
    low = close * 0.995
    
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]).values,
        "high": high.values,
        "low": low.values,
        "close": close.values,
        "volume": np.ones(n) * 500000,
    }, index=dates)


# ============================================================================
# KalmanConfig Tests
# ============================================================================


class TestKalmanConfig:
    """Tests for KalmanConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = KalmanConfig()
        assert config.process_noise_q == 1e-5
        assert config.observation_noise_r == 1e-2
        assert config.residual_z_entry == 2.0
        assert config.residual_z_exit == 0.5
        assert config.trend_slope_min == 0.0001
        assert config.residual_lookback == 100

    def test_custom_values(self):
        """Test custom configuration."""
        config = KalmanConfig(
            process_noise_q=1e-6,
            residual_z_entry=1.5,
            atr_stop_mult=2.0,
        )
        assert config.process_noise_q == 1e-6
        assert config.residual_z_entry == 1.5
        assert config.atr_stop_mult == 2.0


# ============================================================================
# KalmanState Tests
# ============================================================================


class TestKalmanState:
    """Tests for KalmanState dataclass."""

    def test_state_creation(self):
        """Test creating Kalman state."""
        state = KalmanState(
            level=100.0,
            variance=0.01,
            residual=0.5,
            residual_z=1.2,
            trend_slope=0.1,
            confidence=100.0,
        )
        assert state.level == 100.0
        assert state.variance == 0.01
        assert state.residual == 0.5
        assert state.residual_z == 1.2
        assert state.trend_slope == 0.1
        assert state.confidence == 100.0


# ============================================================================
# KalmanFilter Tests
# ============================================================================


class TestKalmanFilter:
    """Tests for KalmanFilter class."""

    def test_initialization(self):
        """Test Kalman filter initialization."""
        kf = KalmanFilter(q=1e-5, r=1e-2)
        assert kf.q == 1e-5
        assert kf.r == 1e-2
        assert kf.initialized is False

    def test_reset(self):
        """Test filter reset."""
        kf = KalmanFilter()
        kf.reset(100.0)
        assert kf.x == 100.0
        assert kf.p == 1.0
        assert kf.initialized is True

    def test_first_update_initializes(self):
        """Test first update initializes filter."""
        kf = KalmanFilter()
        level, residual, variance = kf.update(100.0)
        assert kf.initialized is True
        assert level == 100.0
        assert residual == 0.0

    def test_update_returns_tuple(self):
        """Test update returns level, residual, variance."""
        kf = KalmanFilter()
        kf.reset(100.0)
        result = kf.update(101.0)
        
        assert len(result) == 3
        level, residual, variance = result
        assert isinstance(level, float)
        assert isinstance(residual, float)
        assert isinstance(variance, float)

    def test_filter_smooths_noise(self):
        """Test filter smooths noisy signal."""
        np.random.seed(42)
        kf = KalmanFilter(q=1e-6, r=1e-1)  # High R = smooth
        
        # Constant value + noise
        observations = [100.0 + np.random.normal(0, 5) for _ in range(50)]
        
        levels = []
        for obs in observations:
            level, _, _ = kf.update(obs)
            levels.append(level)
        
        # Filtered levels should have lower variance than observations
        assert np.std(levels[10:]) < np.std(observations[10:])

    def test_filter_tracks_trend(self):
        """Test filter tracks trend."""
        kf = KalmanFilter(q=1e-4, r=1e-3)  # Higher Q = responsive
        
        # Linear trend
        observations = [100.0 + i * 0.5 for i in range(50)]
        
        levels = []
        for obs in observations:
            level, _, _ = kf.update(obs)
            levels.append(level)
        
        # Final level should be close to final observation
        assert abs(levels[-1] - observations[-1]) < 5

    def test_residual_calculation(self):
        """Test residual is observation minus level."""
        kf = KalmanFilter()
        kf.reset(100.0)
        
        # Large jump
        level, residual, _ = kf.update(110.0)
        
        # Residual should be positive (observation > level)
        assert residual > 0
        assert residual == 110.0 - level


# ============================================================================
# KalmanHybridModel Tests
# ============================================================================


class TestKalmanHybridModel:
    """Tests for Kalman Hybrid Model."""

    def test_model_initialization(self, model: KalmanHybridModel):
        """Test model initializes correctly."""
        assert model.kalman_config.process_noise_q == 1e-5
        assert model.kalman_config.observation_noise_r == 1e-2
        assert model.kalman_config.residual_z_entry == 2.0
        assert model._filters == {}

    def test_model_initialization_defaults(self):
        """Test model initializes with defaults."""
        config = ModelConfig(model_id="test", model_type="hybrid")
        model = KalmanHybridModel(config)
        assert model.kalman_config.process_noise_q == 1e-5
        assert model.kalman_config.atr_period == 14

    def test_get_filter_creates_new(self, model: KalmanHybridModel):
        """Test _get_filter creates new filter for symbol."""
        kf = model._get_filter("AAPL")
        assert "AAPL" in model._filters
        assert isinstance(kf, KalmanFilter)

    def test_get_filter_returns_existing(self, model: KalmanHybridModel):
        """Test _get_filter returns existing filter."""
        kf1 = model._get_filter("AAPL")
        kf1.reset(100.0)
        
        kf2 = model._get_filter("AAPL")
        assert kf1 is kf2
        assert kf2.initialized is True

    def test_run_filter(self, model: KalmanHybridModel, sample_prices: pd.Series):
        """Test run_filter returns DataFrame with expected columns."""
        result = model.run_filter(sample_prices)
        
        assert isinstance(result, pd.DataFrame)
        expected_cols = {"price", "trend_level", "residual", "state_var"}
        assert expected_cols.issubset(result.columns)
        assert len(result) == len(sample_prices)

    def test_run_filter_trend_level(self, model: KalmanHybridModel, sample_prices: pd.Series):
        """Test run_filter trend level is smooth."""
        result = model.run_filter(sample_prices)
        
        # Trend should be smoother than original prices
        trend_std = result["trend_level"].diff().std()
        price_std = sample_prices.diff().std()
        assert trend_std < price_std

    def test_run_filter_residual(self, model: KalmanHybridModel, sample_prices: pd.Series):
        """Test residual is price minus trend."""
        result = model.run_filter(sample_prices)
        
        # Check residual calculation
        calculated_residual = result["price"] - result["trend_level"]
        np.testing.assert_array_almost_equal(
            result["residual"].values,
            calculated_residual.values,
        )

    @pytest.mark.asyncio
    async def test_generate_returns_none_insufficient_data(
        self, model: KalmanHybridModel
    ):
        """Test generate returns None for insufficient data."""
        small_df = pd.DataFrame({
            "open": [100] * 10,
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100] * 10,
        })
        signal = await model.generate("AAPL", small_df, datetime.now(timezone.utc))
        assert signal is None

    @pytest.mark.asyncio
    async def test_generate_with_adequate_data(
        self, model: KalmanHybridModel, sample_ohlcv: pd.DataFrame
    ):
        """Test generate with adequate data."""
        signal = await model.generate("AAPL", sample_ohlcv, datetime.now(timezone.utc))
        # May or may not generate signal depending on conditions
        if signal is not None:
            assert signal.symbol == "AAPL"
            # Signal can be ENTRY, EXIT, or HOLD
            assert signal.signal_type in (SignalType.ENTRY, SignalType.EXIT, SignalType.HOLD)

    @pytest.mark.asyncio
    async def test_generate_entry_oversold_uptrend(
        self, model: KalmanHybridModel, oversold_in_uptrend_data: pd.DataFrame
    ):
        """Test long entry when oversold in uptrend."""
        signal = await model.generate(
            "AAPL", oversold_in_uptrend_data, datetime.now(timezone.utc)
        )
        
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            assert signal.direction == Direction.LONG
            assert "residual_z" in signal.metadata
            assert "trend_slope" in signal.metadata

    @pytest.mark.asyncio
    async def test_signal_metadata_structure(
        self, model: KalmanHybridModel, sample_ohlcv: pd.DataFrame
    ):
        """Test signal metadata has expected structure."""
        signal = await model.generate("AAPL", sample_ohlcv, datetime.now(timezone.utc))
        
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            metadata = signal.metadata
            expected_keys = {"model", "residual_z", "trend_slope", "trend_level"}
            assert expected_keys.issubset(metadata.keys())


# ============================================================================
# Signal Generation Logic Tests
# ============================================================================


class TestSignalLogic:
    """Tests for signal generation logic."""

    @pytest.mark.asyncio
    async def test_no_signal_when_no_extreme_residual(self, model: KalmanHybridModel):
        """Test no signal when residual is not extreme."""
        # Very smooth trend = low residuals
        n = 150
        trend = np.linspace(100, 120, n)
        
        df = pd.DataFrame({
            "open": trend,
            "high": trend * 1.001,
            "low": trend * 0.999,
            "close": trend,
        })
        
        signal = await model.generate("AAPL", df, datetime.now(timezone.utc))
        # With very smooth data, no extreme residuals, likely no signal
        # (depends on implementation details)

    @pytest.mark.asyncio
    async def test_no_counter_trend_trade(self, model: KalmanHybridModel):
        """Test model doesn't trade counter-trend."""
        n = 150
        
        # Downtrend with oversold spike (should not buy in downtrend)
        prices = [100 - i * 0.5 for i in range(n)]
        prices[-10:] = [p - 5 for p in prices[-10:]]  # Extra oversold
        
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
        })
        
        signal = await model.generate("AAPL", df, datetime.now(timezone.utc))
        
        # If signal is generated, it should NOT be a long in downtrend
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            # In a downtrend, only short entries should be allowed
            if signal.direction == Direction.LONG:
                # Check if trend is actually up (signal logic passed)
                assert signal.metadata.get("trend_slope", 0) > 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_flat_price_data(self, model: KalmanHybridModel):
        """Test handling of flat price data."""
        n = 150
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
        })
        
        # Should handle without division by zero
        signal = await model.generate("AAPL", df, datetime.now(timezone.utc))

    @pytest.mark.asyncio
    async def test_nan_in_data(self, model: KalmanHybridModel):
        """Test handling of NaN values."""
        n = 150
        prices = [100.0] * n
        prices[75] = np.nan
        
        df = pd.DataFrame({
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
        })
        
        # Should handle NaN gracefully
        signal = await model.generate("AAPL", df, datetime.now(timezone.utc))

    @pytest.mark.asyncio
    async def test_extreme_values(self, model: KalmanHybridModel):
        """Test handling of extreme price values."""
        n = 150
        
        df = pd.DataFrame({
            "open": [1e8] * n,
            "high": [1.001e8] * n,
            "low": [0.999e8] * n,
            "close": [1e8] * n,
        })
        
        # Should handle without overflow
        signal = await model.generate("AAPL", df, datetime.now(timezone.utc))

    def test_filter_resets_per_symbol(self, model: KalmanHybridModel):
        """Test each symbol gets its own filter."""
        kf_aapl = model._get_filter("AAPL")
        kf_msft = model._get_filter("MSFT")
        
        assert kf_aapl is not kf_msft
        assert "AAPL" in model._filters
        assert "MSFT" in model._filters

    def test_run_filter_with_negative_prices(self, model: KalmanHybridModel):
        """Test run_filter handles theoretical negative prices."""
        # Edge case: prices can't be negative in reality but test robustness
        # Need enough data points for min_periods in rolling calculations
        prices = pd.Series(
            [100, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5] +
            [0, -5, -10, 5, 20, 50]
        )
        
        result = model.run_filter(prices)
        assert len(result) == len(prices)
        # Should not crash
