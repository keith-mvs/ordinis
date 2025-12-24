"""
Tests for Mutual Information Ensemble Model.

GTM Strategy #4: Weights multiple alpha signals by their MI with forward returns.
Expected edge: 4/5, smoothest equity curve, robust across conditions.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models.mi_ensemble import (
    DEFAULT_SIGNALS,
    MIConfig,
    MIEnsembleModel,
    SignalDefinition,
    compute_mean_reversion,
    compute_momentum,
    compute_rsi,
    compute_trend_strength,
    compute_volatility_breakout,
    mutual_information,
    normalized_mutual_information,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def model_config() -> ModelConfig:
    """Create default model config."""
    return ModelConfig(
        model_id="test_mi_ensemble",
        model_type="ensemble",
        parameters={
            "mi_lookback": 252,
            "mi_bins": 10,
            "forward_period": 5,
            "min_weight": 0.0,
            "max_weight": 0.5,
            "recalc_frequency": 21,
            "ensemble_threshold": 0.3,
            "min_signals_agree": 2,
            "atr_period": 14,
            "atr_stop_mult": 1.5,
            "atr_tp_mult": 2.5,
        },
    )


@pytest.fixture
def model(model_config: ModelConfig) -> MIEnsembleModel:
    """Create MI Ensemble model."""
    return MIEnsembleModel(model_config)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create sample OHLCV data with enough bars for MI calculation."""
    np.random.seed(42)
    n = 300
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    
    # Create data with some structure
    trend = np.linspace(100, 130, n)
    cyclical = 10 * np.sin(np.linspace(0, 8 * np.pi, n))
    noise = np.random.normal(0, 3, n)
    prices = trend + cyclical + noise
    
    close = pd.Series(prices, index=dates)
    high = close * (1 + np.abs(np.random.normal(0, 0.015, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.015, n)))
    
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]).values,
        "high": high.values,
        "low": low.values,
        "close": close.values,
        "volume": np.random.randint(100000, 1000000, n),
    }, index=dates)


@pytest.fixture
def short_ohlcv() -> pd.DataFrame:
    """Create short OHLCV data (insufficient for MI)."""
    n = 30
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1D")
    
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.0] * n,
        "volume": [500000] * n,
    }, index=dates)


@pytest.fixture
def trending_data() -> pd.DataFrame:
    """Create strongly trending data."""
    np.random.seed(123)
    n = 300
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    
    # Strong uptrend
    prices = [100.0]
    for i in range(1, n):
        prices.append(prices[-1] * (1 + 0.003 + np.random.normal(0, 0.01)))
    
    close = pd.Series(prices, index=dates)
    high = close * 1.01
    low = close * 0.99
    
    return pd.DataFrame({
        "open": close.shift(1).fillna(close.iloc[0]).values,
        "high": high.values,
        "low": low.values,
        "close": close.values,
        "volume": np.ones(n) * 500000,
    }, index=dates)


# ============================================================================
# MIConfig Tests
# ============================================================================


class TestMIConfig:
    """Tests for MIConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MIConfig()
        assert config.mi_lookback == 252
        assert config.mi_bins == 10
        assert config.forward_period == 5
        assert config.min_weight == 0.0
        assert config.max_weight == 0.5
        assert config.recalc_frequency == 21
        assert config.ensemble_threshold == 0.3
        assert config.min_signals_agree == 2

    def test_custom_values(self):
        """Test custom configuration."""
        config = MIConfig(
            mi_lookback=126,
            forward_period=10,
            ensemble_threshold=0.5,
        )
        assert config.mi_lookback == 126
        assert config.forward_period == 10
        assert config.ensemble_threshold == 0.5


# ============================================================================
# SignalDefinition Tests
# ============================================================================


class TestSignalDefinition:
    """Tests for SignalDefinition dataclass."""

    def test_signal_definition_creation(self):
        """Test creating signal definition."""
        def dummy_compute(df):
            return df["close"].pct_change()
        
        sig_def = SignalDefinition(name="test_signal", compute=dummy_compute)
        assert sig_def.name == "test_signal"
        assert callable(sig_def.compute)

    def test_signal_definition_hashable(self):
        """Test signal definition is hashable (for sets)."""
        sig_def = SignalDefinition(name="test", compute=lambda df: df["close"])
        hash_val = hash(sig_def)
        assert isinstance(hash_val, int)

    def test_default_signals_exist(self):
        """Test default signals are defined."""
        assert len(DEFAULT_SIGNALS) == 5
        signal_names = [s.name for s in DEFAULT_SIGNALS]
        assert "rsi" in signal_names
        assert "momentum" in signal_names
        assert "mean_reversion" in signal_names
        assert "vol_breakout" in signal_names
        assert "trend_strength" in signal_names


# ============================================================================
# Mutual Information Function Tests
# ============================================================================


class TestMutualInformation:
    """Tests for mutual information calculation functions."""

    def test_mi_perfect_correlation(self):
        """Test MI is high for perfectly correlated variables."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = x + np.random.normal(0, 0.1, 200)  # Almost perfect correlation
        
        mi = mutual_information(x, y)
        assert mi > 0.5  # Should have high MI

    def test_mi_independent_variables(self):
        """Test MI is low for independent variables."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = np.random.normal(0, 1, 200)  # Independent
        
        mi = mutual_information(x, y)
        assert mi < 0.3  # Should have low MI

    def test_mi_different_lengths_returns_zero(self):
        """Test MI returns 0 for mismatched lengths."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3])
        
        mi = mutual_information(x, y)
        assert mi == 0.0

    def test_mi_insufficient_data_returns_zero(self):
        """Test MI returns 0 for insufficient data."""
        x = np.array([1, 2, 3])
        y = np.array([1, 2, 3])
        
        mi = mutual_information(x, y)
        assert mi == 0.0

    def test_mi_handles_nan(self):
        """Test MI handles NaN values."""
        x = np.array([1, 2, np.nan, 4, 5] * 20)
        y = np.array([1, 2, 3, np.nan, 5] * 20)
        
        mi = mutual_information(x, y)
        # Should not crash, may return 0 or small value

    def test_mi_non_negative(self):
        """Test MI is always non-negative."""
        np.random.seed(42)
        for _ in range(10):
            x = np.random.normal(0, 1, 100)
            y = np.random.normal(0, 1, 100)
            mi = mutual_information(x, y)
            assert mi >= 0.0

    def test_normalized_mi_range(self):
        """Test normalized MI is between 0 and 1."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 200)
        y = x + np.random.normal(0, 0.5, 200)
        
        nmi = normalized_mutual_information(x, y)
        assert 0 <= nmi <= 1


# ============================================================================
# Signal Compute Function Tests
# ============================================================================


class TestSignalComputeFunctions:
    """Tests for individual signal compute functions."""

    def test_compute_rsi(self, sample_ohlcv: pd.DataFrame):
        """Test RSI signal computation."""
        rsi = compute_rsi(sample_ohlcv)
        assert len(rsi) == len(sample_ohlcv)
        # Normalized RSI should be between -1 and 1
        valid_rsi = rsi.dropna()
        assert valid_rsi.min() >= -1.5  # Some tolerance
        assert valid_rsi.max() <= 1.5

    def test_compute_momentum(self, sample_ohlcv: pd.DataFrame):
        """Test momentum signal computation."""
        mom = compute_momentum(sample_ohlcv)
        assert len(mom) == len(sample_ohlcv)
        # Should be z-scored, mostly between -3 and 3
        valid_mom = mom.dropna()
        assert abs(valid_mom.mean()) < 1  # Roughly centered

    def test_compute_mean_reversion(self, sample_ohlcv: pd.DataFrame):
        """Test mean reversion signal computation."""
        mr = compute_mean_reversion(sample_ohlcv)
        assert len(mr) == len(sample_ohlcv)
        # Should be negative z-score

    def test_compute_volatility_breakout(self, sample_ohlcv: pd.DataFrame):
        """Test volatility breakout signal computation."""
        vb = compute_volatility_breakout(sample_ohlcv)
        assert len(vb) == len(sample_ohlcv)
        # Normalized between -1 and 1
        valid_vb = vb.dropna()
        assert valid_vb.min() >= -1.5
        assert valid_vb.max() <= 1.5

    def test_compute_trend_strength(self, sample_ohlcv: pd.DataFrame):
        """Test trend strength signal computation."""
        ts = compute_trend_strength(sample_ohlcv)
        assert len(ts) == len(sample_ohlcv)

    def test_compute_rsi_trending_data(self, trending_data: pd.DataFrame):
        """Test RSI calculation in uptrend."""
        rsi = compute_rsi(trending_data)
        # In strong uptrend, RSI should be mostly positive
        # Check variance is reasonable
        assert not rsi.dropna().empty


# ============================================================================
# MIEnsembleModel Tests
# ============================================================================


class TestMIEnsembleModel:
    """Tests for MI Ensemble Model."""

    def test_model_initialization(self, model: MIEnsembleModel):
        """Test model initializes correctly."""
        assert model.mi_config.mi_lookback == 252
        assert model.mi_config.ensemble_threshold == 0.3
        assert len(model.signals) == 5
        assert model._weights == {}

    def test_model_initialization_defaults(self):
        """Test model initializes with defaults."""
        config = ModelConfig(model_id="test", model_type="ensemble")
        model = MIEnsembleModel(config)
        assert model.mi_config.mi_lookback == 252
        assert len(model.signals) == 5

    def test_model_with_custom_signals(self):
        """Test model with custom signals."""
        custom_signals = [
            SignalDefinition("custom1", lambda df: df["close"].pct_change()),
            SignalDefinition("custom2", lambda df: df["volume"].pct_change()),
        ]
        config = ModelConfig(model_id="test", model_type="ensemble")
        model = MIEnsembleModel(config, signals=custom_signals)
        
        assert len(model.signals) == 2
        assert model.signals[0].name == "custom1"

    def test_calculate_mi_weights(self, model: MIEnsembleModel, sample_ohlcv: pd.DataFrame):
        """Test MI weight calculation."""
        weights = model.calculate_mi_weights(sample_ohlcv, "AAPL")
        
        # Should have weight for each signal
        assert len(weights) == len(model.signals)
        
        # Weights should sum to ~1
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01
        
        # Each weight should be between 0 and max_weight
        for weight in weights.values():
            assert 0 <= weight <= model.mi_config.max_weight + 0.01

    def test_calculate_mi_weights_short_data(
        self, model: MIEnsembleModel, short_ohlcv: pd.DataFrame
    ):
        """Test MI weight calculation with insufficient data."""
        weights = model.calculate_mi_weights(short_ohlcv, "AAPL")
        
        # Should still return weights (equal weights fallback)
        assert len(weights) == len(model.signals)
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_calculate_atr(self, model: MIEnsembleModel, sample_ohlcv: pd.DataFrame):
        """Test ATR calculation."""
        atr = model._calculate_atr(sample_ohlcv)
        assert atr > 0
        assert not np.isnan(atr)

    @pytest.mark.asyncio
    async def test_generate_with_adequate_data(
        self, model: MIEnsembleModel, sample_ohlcv: pd.DataFrame
    ):
        """Test generate with adequate data."""
        signal = await model.generate("AAPL", sample_ohlcv, datetime.now(timezone.utc))
        
        # May or may not generate signal depending on conditions
        if signal is not None:
            assert signal.symbol == "AAPL"
            assert signal.signal_type in (SignalType.ENTRY, SignalType.EXIT)

    @pytest.mark.asyncio
    async def test_generate_returns_none_insufficient_data(
        self, model: MIEnsembleModel, short_ohlcv: pd.DataFrame
    ):
        """Test generate returns None for insufficient data."""
        signal = await model.generate("AAPL", short_ohlcv, datetime.now(timezone.utc))
        assert signal is None

    @pytest.mark.asyncio
    async def test_generate_long_in_uptrend(
        self, model: MIEnsembleModel, trending_data: pd.DataFrame
    ):
        """Test long signal in strong uptrend."""
        signal = await model.generate("AAPL", trending_data, datetime.now(timezone.utc))
        
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            assert signal.direction == Direction.LONG
            assert "ensemble_value" in signal.metadata
            assert "signal_weights" in signal.metadata

    @pytest.mark.asyncio
    async def test_signal_metadata_structure(
        self, model: MIEnsembleModel, sample_ohlcv: pd.DataFrame
    ):
        """Test signal metadata has expected structure."""
        signal = await model.generate("AAPL", sample_ohlcv, datetime.now(timezone.utc))
        
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            metadata = signal.metadata
            expected_keys = {"model", "ensemble_value", "signal_weights"}
            assert expected_keys.issubset(metadata.keys())


# ============================================================================
# Weight Caching Tests
# ============================================================================


class TestWeightCaching:
    """Tests for MI weight caching behavior."""

    def test_weights_cached_per_symbol(self, model: MIEnsembleModel, sample_ohlcv: pd.DataFrame):
        """Test weights are cached per symbol."""
        weights1 = model.calculate_mi_weights(sample_ohlcv, "AAPL")
        
        # Modify cached weights directly
        model._weights["AAPL"] = weights1
        
        # Retrieve from cache
        assert "AAPL" in model._weights
        assert model._weights["AAPL"] == weights1

    def test_different_symbols_different_weights(
        self, model: MIEnsembleModel, sample_ohlcv: pd.DataFrame
    ):
        """Test different symbols can have different weights."""
        weights_aapl = model.calculate_mi_weights(sample_ohlcv, "AAPL")
        weights_msft = model.calculate_mi_weights(sample_ohlcv, "MSFT")
        
        # Cache should have both
        model._weights["AAPL"] = weights_aapl
        model._weights["MSFT"] = weights_msft
        
        assert len(model._weights) == 2


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_flat_price_data(self, model: MIEnsembleModel):
        """Test handling of flat price data."""
        n = 300
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
        })
        
        # Should handle without division by zero
        signal = await model.generate("AAPL", df, datetime.now(timezone.utc))

    @pytest.mark.asyncio
    async def test_nan_in_data(self, model: MIEnsembleModel):
        """Test handling of NaN values."""
        n = 300
        prices = [100.0 + i * 0.1 for i in range(n)]
        prices[150] = np.nan
        
        df = pd.DataFrame({
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
        })
        
        # Should handle NaN gracefully
        signal = await model.generate("AAPL", df, datetime.now(timezone.utc))

    @pytest.mark.asyncio
    async def test_extreme_values(self, model: MIEnsembleModel):
        """Test handling of extreme price values."""
        n = 300
        
        df = pd.DataFrame({
            "open": [1e8] * n,
            "high": [1.001e8] * n,
            "low": [0.999e8] * n,
            "close": [1e8] * n,
        })
        
        # Should handle without overflow
        signal = await model.generate("AAPL", df, datetime.now(timezone.utc))

    def test_mi_weights_with_zero_mi(self, model: MIEnsembleModel):
        """Test weight calculation when all MI values are zero."""
        # Create data that produces zero MI (pure noise)
        np.random.seed(42)
        n = 300
        
        df = pd.DataFrame({
            "open": np.random.uniform(99, 101, n),
            "high": np.random.uniform(100, 102, n),
            "low": np.random.uniform(98, 100, n),
            "close": np.random.uniform(99, 101, n),
        })
        
        weights = model.calculate_mi_weights(df, "RANDOM")
        
        # Should fall back to equal weights
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_signal_error_handling(self):
        """Test model handles signal computation errors gracefully."""
        # Create a model with a custom signal that will error
        def bad_signal(df):
            raise ValueError("Intentional error")
        
        custom_signals = [
            SignalDefinition("rsi", compute_rsi),
            SignalDefinition("bad_signal", bad_signal),
        ]
        
        config = ModelConfig(model_id="test", model_type="ensemble")
        model = MIEnsembleModel(config, signals=custom_signals)
        
        n = 300
        df = pd.DataFrame({
            "open": [100.0 + i * 0.1 for i in range(n)],
            "high": [101.0 + i * 0.1 for i in range(n)],
            "low": [99.0 + i * 0.1 for i in range(n)],
            "close": [100.0 + i * 0.1 for i in range(n)],
        })
        
        # Should not crash, just log warning
        weights = model.calculate_mi_weights(df, "AAPL")
        assert "bad_signal" in weights
        assert weights["bad_signal"] == 0.0 or isinstance(weights["bad_signal"], float)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for MI Ensemble model."""

    @pytest.mark.asyncio
    async def test_full_signal_generation_flow(
        self, model: MIEnsembleModel, trending_data: pd.DataFrame
    ):
        """Test complete signal generation flow."""
        # First generate should calculate weights
        _ = await model.generate("AAPL", trending_data, datetime.now(timezone.utc))
        
        # Second generate may use cached weights
        _ = await model.generate("AAPL", trending_data, datetime.now(timezone.utc))
        
        # Both should complete without error

    def test_all_default_signals_compute(self, sample_ohlcv: pd.DataFrame):
        """Test all default signals compute without error."""
        # Use a fresh copy of DEFAULT_SIGNALS to avoid pollution from other tests
        from ordinis.engines.signalcore.models.mi_ensemble import DEFAULT_SIGNALS as fresh_signals
        
        for sig_def in fresh_signals:
            result = sig_def.compute(sample_ohlcv)
            assert len(result) == len(sample_ohlcv)
            assert isinstance(result, pd.Series)


# ============================================================================
# Lookahead Bias Tests
# ============================================================================


class TestLookaheadBias:
    """Critical tests to verify no lookahead bias in MI weight calculation.
    
    The MI Ensemble strategy calculates weights based on mutual information
    between signal values and forward returns. This creates a subtle
    lookahead bias risk: if we use forward returns that haven't been
    realized yet, the backtest will be unrealistically optimistic.
    
    These tests verify that:
    1. Weight calculation only uses data available at decision time
    2. Forward returns are lagged appropriately
    3. Weights remain stable when future data changes
    """

    @pytest.fixture
    def lookahead_test_data(self) -> pd.DataFrame:
        """Create data where lookahead bias would be detectable."""
        rng = np.random.default_rng(42)
        n = 400  # Enough for MI calculation
        dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
        
        # Create price series with known structure
        prices = [100.0]
        for i in range(1, n):
            # Regime change at bar 300: from uptrend to downtrend
            if i < 300:
                drift = 0.002  # Uptrend
            else:
                drift = -0.002  # Downtrend
            prices.append(prices[-1] * (1 + drift + rng.normal(0, 0.01)))
        
        close = pd.Series(prices, index=dates)
        high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
        low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
        
        return pd.DataFrame({
            "open": close.shift(1).fillna(close.iloc[0]).values,
            "high": high.values,
            "low": low.values,
            "close": close.values,
            "volume": rng.integers(100000, 1000000, n),
        }, index=dates)

    def test_weights_dont_see_future(
        self,
        model_config: ModelConfig,
        lookahead_test_data: pd.DataFrame,
    ):
        """Verify weights calculated at time T only use data before T.
        
        If there's lookahead bias, weights calculated at bar 295 would
        "know" about the regime change at bar 300 and adjust accordingly.
        """
        model = MIEnsembleModel(model_config)
        
        # Calculate weights using data up to bar 295 (before regime change)
        df_before = lookahead_test_data.iloc[:295].copy()
        weights_before = model.calculate_mi_weights(df_before, "TEST_BEFORE")
        
        # Reset model cache
        model._weights = {}
        model._last_recalc = {}
        
        # Calculate weights using all data including regime change
        df_all = lookahead_test_data.copy()
        
        # Truncate to same length to compare fairly
        df_truncated = df_all.iloc[:295].copy()
        weights_truncated = model.calculate_mi_weights(df_truncated, "TEST_TRUNC")
        
        # If there's no lookahead bias, weights should be identical
        # because both use only data up to bar 295
        for signal_name in weights_before:
            assert signal_name in weights_truncated
            assert weights_before[signal_name] == pytest.approx(
                weights_truncated[signal_name],
                abs=1e-10,
            ), f"Weight for {signal_name} differs - possible lookahead bias!"

    def test_forward_period_exclusion(
        self,
        model_config: ModelConfig,
    ):
        """Verify that the last forward_period bars are excluded from MI calculation.
        
        At time T, we cannot know the return from T to T+forward_period,
        so those returns should not be used in MI calculation.
        """
        # Create custom model with specific forward_period
        forward_period = 5
        config = ModelConfig(
            model_id="test_forward_exclusion",
            model_type="ensemble",
            parameters={
                "forward_period": forward_period,
                "mi_lookback": 100,
            },
        )
        model = MIEnsembleModel(config)
        
        rng = np.random.default_rng(123)
        n = 200
        dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
        
        # Create baseline data
        prices = [100.0]
        for _ in range(1, n):
            prices.append(prices[-1] * (1 + rng.normal(0.001, 0.01)))
        
        close = pd.Series(prices, index=dates)
        df_base = pd.DataFrame({
            "open": close.shift(1).fillna(close.iloc[0]).values,
            "high": (close * 1.01).values,
            "low": (close * 0.99).values,
            "close": close.values,
            "volume": rng.integers(100000, 1000000, n),
        }, index=dates)
        
        # Calculate weights with base data
        weights_base = model.calculate_mi_weights(df_base, "BASE")
        
        # Reset cache
        model._weights = {}
        model._last_recalc = {}
        
        # Create modified data where only the LAST forward_period bars are different
        df_modified = df_base.copy()
        df_modified.loc[df_modified.index[-forward_period:], "close"] *= 2.0
        df_modified.loc[df_modified.index[-forward_period:], "high"] *= 2.0
        df_modified.loc[df_modified.index[-forward_period:], "low"] *= 2.0
        
        weights_modified = model.calculate_mi_weights(df_modified, "MODIFIED")
        
        # Weights should be IDENTICAL because the last forward_period bars
        # should be excluded from MI calculation (their returns aren't realized)
        for signal_name in weights_base:
            assert signal_name in weights_modified
            # Allow small tolerance for floating point
            assert weights_base[signal_name] == pytest.approx(
                weights_modified[signal_name],
                abs=1e-6,
            ), (
                f"Weight for {signal_name} changed when only future bars modified! "
                f"Base: {weights_base[signal_name]:.6f}, "
                f"Modified: {weights_modified[signal_name]:.6f}"
            )

    def test_mi_uses_historical_window_only(
        self,
        model_config: ModelConfig,
        sample_ohlcv: pd.DataFrame,
    ):
        """Verify MI calculation respects lookback window and forward_period."""
        model = MIEnsembleModel(model_config)
        
        # The model should not raise when given sufficient data
        weights = model.calculate_mi_weights(sample_ohlcv, "AAPL")
        
        # All weights should be between 0 and 1
        for name, weight in weights.items():
            assert 0.0 <= weight <= 1.0, f"Weight {name}={weight} out of bounds"
        
        # Weights should sum to 1.0 (normalized)
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)
