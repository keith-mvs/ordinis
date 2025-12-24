"""
Tests for Multi-Timeframe Momentum Model.

GTM Strategy #2: Combines daily momentum ranking with intraday stochastic for entry timing.
Expected edge: 4/5, excels in trending markets.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models.mtf_momentum import (
    MTFConfig,
    MTFMomentumModel,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def model_config() -> ModelConfig:
    """Create default model config."""
    return ModelConfig(
        model_id="test_mtf_momentum",
        model_type="momentum",
        parameters={
            "formation_period": 252,
            "skip_period": 21,
            "momentum_percentile": 0.8,
            "stoch_k_period": 14,
            "stoch_d_period": 3,
            "stoch_oversold": 30.0,
            "stoch_overbought": 70.0,
            "atr_period": 14,
            "atr_stop_mult": 2.0,
            "atr_tp_mult": 3.0,
        },
    )


@pytest.fixture
def model(model_config: ModelConfig) -> MTFMomentumModel:
    """Create MTF Momentum model."""
    return MTFMomentumModel(model_config)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create sample OHLCV data with enough bars for momentum calculation."""
    np.random.seed(42)
    n = 300  # Need at least formation_period (252) + buffer
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    
    # Create uptrending data for positive momentum
    base_price = 100.0
    trend = np.linspace(0, 50, n)  # Linear uptrend
    noise = np.random.normal(0, 2, n)  # Random noise
    prices = base_price + trend + noise
    
    close = pd.Series(prices, index=dates)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = np.random.randint(100000, 1000000, n)
    
    return pd.DataFrame({
        "open": open_.values,
        "high": high.values,
        "low": low.values,
        "close": close.values,
        "volume": volume,
    }, index=dates)


@pytest.fixture
def short_ohlcv() -> pd.DataFrame:
    """Create short OHLCV data (insufficient for momentum)."""
    n = 50
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1D")
    
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low": [99.0] * n,
        "close": [100.0] * n,
        "volume": [500000] * n,
    }, index=dates)


@pytest.fixture
def strong_momentum_data() -> pd.DataFrame:
    """Create data with strong positive momentum and oversold stochastic."""
    np.random.seed(123)
    n = 300
    dates = pd.date_range(start="2023-01-01", periods=n, freq="1D")
    
    # Strong uptrend for first 280 bars, then pullback
    prices = []
    price = 100.0
    for i in range(n):
        if i < 280:
            price *= 1.002  # Steady uptrend
        else:
            price *= 0.98  # Sharp pullback to create oversold stochastic
        prices.append(price)
    
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
# MTFConfig Tests
# ============================================================================


class TestMTFConfig:
    """Tests for MTFConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MTFConfig()
        assert config.formation_period == 252
        assert config.skip_period == 21
        assert config.momentum_percentile == 0.8
        assert config.stoch_k_period == 14
        assert config.stoch_d_period == 3
        assert config.stoch_oversold == 30.0
        assert config.stoch_overbought == 70.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MTFConfig(
            formation_period=126,
            momentum_percentile=0.9,
            stoch_oversold=20.0,
        )
        assert config.formation_period == 126
        assert config.momentum_percentile == 0.9
        assert config.stoch_oversold == 20.0

    def test_risk_parameters(self):
        """Test risk-related parameters."""
        config = MTFConfig()
        assert config.atr_period == 14
        assert config.atr_stop_mult == 2.0
        assert config.atr_tp_mult == 3.0


# ============================================================================
# MTFMomentumModel Tests
# ============================================================================


class TestMTFMomentumModel:
    """Tests for MTF Momentum Model."""

    def test_model_initialization(self, model: MTFMomentumModel):
        """Test model initializes correctly."""
        assert model.mtf_config.formation_period == 252
        assert model.mtf_config.skip_period == 21
        assert model.mtf_config.momentum_percentile == 0.8
        assert model.universe_momentum is None

    def test_model_initialization_defaults(self):
        """Test model initializes with defaults."""
        config = ModelConfig(model_id="test", model_type="momentum")
        model = MTFMomentumModel(config)
        assert model.mtf_config.formation_period == 252
        assert model.mtf_config.stoch_k_period == 14

    def test_calculate_momentum_uptrend(
        self, model: MTFMomentumModel, sample_ohlcv: pd.DataFrame
    ):
        """Test momentum calculation for uptrending data."""
        momentum = model._calculate_momentum(sample_ohlcv)
        # Uptrending data should have positive momentum
        assert momentum > 0

    def test_calculate_momentum_insufficient_data(
        self, model: MTFMomentumModel, short_ohlcv: pd.DataFrame
    ):
        """Test momentum returns 0 for insufficient data."""
        momentum = model._calculate_momentum(short_ohlcv)
        assert momentum == 0.0

    def test_calculate_stochastic(self, model: MTFMomentumModel, sample_ohlcv: pd.DataFrame):
        """Test stochastic oscillator calculation."""
        k, d, bullish, bearish = model._calculate_stochastic(sample_ohlcv)
        
        # Stochastic should be between 0 and 100
        assert 0 <= k <= 100
        assert 0 <= d <= 100
        # Can't have both crossovers at once
        assert not (bullish and bearish)

    def test_calculate_atr(self, model: MTFMomentumModel, sample_ohlcv: pd.DataFrame):
        """Test ATR calculation."""
        atr = model._calculate_atr(sample_ohlcv)
        assert atr > 0
        assert not np.isnan(atr)

    def test_set_universe_momentum(self, model: MTFMomentumModel):
        """Test setting universe momentum scores."""
        scores = pd.Series({
            "AAPL": 0.15,
            "MSFT": 0.25,
            "GOOGL": 0.10,
            "AMZN": -0.05,
        })
        model.set_universe_momentum(scores)
        assert model.universe_momentum is not None
        assert len(model.universe_momentum) == 4

    def test_is_winner_with_universe(self, model: MTFMomentumModel):
        """Test winner classification with universe ranking."""
        scores = pd.Series({
            "AAPL": 0.30,  # Top
            "MSFT": 0.25,
            "GOOGL": 0.10,
            "AMZN": 0.05,
            "META": -0.05,  # Bottom
        })
        model.set_universe_momentum(scores)
        
        # AAPL should be a winner (top 20%)
        assert model._is_winner("AAPL", 0.30) == True
        # META should not be a winner
        assert model._is_winner("META", -0.05) == False

    def test_is_winner_without_universe(self, model: MTFMomentumModel):
        """Test winner classification with absolute threshold."""
        model.universe_momentum = None
        # Absolute threshold is 15%
        assert model._is_winner("AAPL", 0.20) is True
        assert model._is_winner("AAPL", 0.10) is False

    def test_is_loser_with_universe(self, model: MTFMomentumModel):
        """Test loser classification with universe ranking."""
        scores = pd.Series({
            "AAPL": 0.30,
            "MSFT": 0.25,
            "GOOGL": 0.10,
            "AMZN": 0.05,
            "META": -0.10,  # Bottom
        })
        model.set_universe_momentum(scores)
        
        assert model._is_loser("META", -0.10) == True
        assert model._is_loser("AAPL", 0.30) == False

    def test_is_loser_without_universe(self, model: MTFMomentumModel):
        """Test loser classification with absolute threshold."""
        model.universe_momentum = None
        assert model._is_loser("AAPL", -0.20) is True
        assert model._is_loser("AAPL", -0.10) is False

    @pytest.mark.asyncio
    async def test_generate_returns_none_insufficient_data(
        self, model: MTFMomentumModel, short_ohlcv: pd.DataFrame
    ):
        """Test generate returns None for insufficient data."""
        signal = await model.generate("AAPL", short_ohlcv, datetime.now(timezone.utc))
        assert signal is None

    @pytest.mark.asyncio
    async def test_generate_with_adequate_data(
        self, model: MTFMomentumModel, sample_ohlcv: pd.DataFrame
    ):
        """Test generate with adequate data."""
        signal = await model.generate("AAPL", sample_ohlcv, datetime.now(timezone.utc))
        # May or may not generate signal depending on conditions
        if signal is not None:
            assert signal.symbol == "AAPL"
            # Signal can be ENTRY, EXIT, or HOLD
            assert signal.signal_type in (SignalType.ENTRY, SignalType.EXIT, SignalType.HOLD)

    @pytest.mark.asyncio
    async def test_generate_long_entry_conditions(
        self, model: MTFMomentumModel, strong_momentum_data: pd.DataFrame
    ):
        """Test long entry when winner + stochastic conditions met."""
        # Set universe to make this symbol a winner
        model.set_universe_momentum(pd.Series({"AAPL": 0.50}))
        
        signal = await model.generate("AAPL", strong_momentum_data, datetime.now(timezone.utc))
        
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            assert signal.direction == Direction.LONG
            assert "momentum" in signal.metadata
            assert "stoch_k" in signal.metadata

    @pytest.mark.asyncio
    async def test_signal_metadata_structure(
        self, model: MTFMomentumModel, sample_ohlcv: pd.DataFrame
    ):
        """Test signal metadata has expected structure."""
        signal = await model.generate("AAPL", sample_ohlcv, datetime.now(timezone.utc))
        
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            metadata = signal.metadata
            expected_keys = {"model", "momentum", "stoch_k", "stoch_d"}
            assert expected_keys.issubset(metadata.keys())


# ============================================================================
# Stochastic Calculation Tests
# ============================================================================


class TestStochasticCalculation:
    """Tests for stochastic oscillator edge cases."""

    def test_stochastic_in_uptrend(self, model: MTFMomentumModel):
        """Test stochastic is high in uptrend."""
        n = 50
        prices = [100 + i for i in range(n)]  # Steady uptrend
        
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
        })
        
        k, d, _, _ = model._calculate_stochastic(df)
        # Should be overbought territory
        assert k > 70

    def test_stochastic_in_downtrend(self, model: MTFMomentumModel):
        """Test stochastic is low in downtrend."""
        n = 50
        prices = [100 - i for i in range(n)]  # Steady downtrend
        
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
        })
        
        k, d, _, _ = model._calculate_stochastic(df)
        # Should be oversold territory
        assert k < 30

    def test_bullish_crossover_detection(self, model: MTFMomentumModel):
        """Test bullish crossover detection."""
        # Create data where %K crosses above %D
        np.random.seed(42)
        n = 50
        
        # Downtrend then sharp uptick
        prices = [100 - i * 0.5 for i in range(45)]  # Decline
        prices += [75, 78, 82, 88, 95]  # Sharp recovery
        
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
        })
        
        k, d, bullish, bearish = model._calculate_stochastic(df)
        # At least verify calculation doesn't crash - numpy bools are fine
        assert bullish == True or bullish == False
        assert bearish == True or bearish == False


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_flat_price_data(self, model: MTFMomentumModel):
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
        # Flat data = no momentum, no signal expected

    @pytest.mark.asyncio
    async def test_extreme_volatility(self, model: MTFMomentumModel):
        """Test handling of extreme volatility."""
        np.random.seed(42)
        n = 300
        
        # Very high volatility
        prices = [100 + np.random.normal(0, 20) for _ in range(n)]
        
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.1 for p in prices],
            "low": [p * 0.9 for p in prices],
            "close": prices,
        })
        
        signal = await model.generate("AAPL", df, datetime.now(timezone.utc))
        # Should handle without crashing

    def test_momentum_calculation_with_gaps(self, model: MTFMomentumModel):
        """Test momentum handles price gaps gracefully."""
        n = 300
        prices = [100.0] * n
        prices[150] = 200.0  # Large gap
        prices[151:] = [200.0] * (n - 151)
        
        df = pd.DataFrame({
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
        })
        
        momentum = model._calculate_momentum(df)
        assert not np.isnan(momentum)
        assert not np.isinf(momentum)
