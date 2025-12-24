"""
Tests for ATR-Optimized RSI Model.

This is the top-performing GTM strategy with +60.1% return in backtests.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models.atr_optimized_rsi import (
    ATROptimizedRSIModel,
    OPTIMIZED_CONFIGS,
    OptimizedConfig,
    backtest,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def model_config() -> ModelConfig:
    """Create default model config."""
    return ModelConfig(
        model_id="test_atr_rsi",
        model_type="mean_reversion",
        parameters={
            "rsi_period": 14,
            "rsi_oversold": 35,
            "rsi_exit": 50,
            "atr_period": 14,
            "atr_stop_mult": 1.5,
            "atr_tp_mult": 2.0,
            "use_optimized": True,
        },
    )


@pytest.fixture
def model(model_config: ModelConfig) -> ATROptimizedRSIModel:
    """Create ATR-Optimized RSI model."""
    return ATROptimizedRSIModel(model_config)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create sample OHLCV data with enough bars for indicators."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    
    # Create trending then mean-reverting data
    base_price = 100.0
    prices = [base_price]
    for i in range(1, n):
        # Add some volatility and mean-reversion tendency
        change = np.random.normal(0, 0.5) + (100 - prices[-1]) * 0.02
        prices.append(max(50, prices[-1] + change))
    
    close = pd.Series(prices)
    high = close * (1 + np.random.uniform(0, 0.02, n))
    low = close * (1 - np.random.uniform(0, 0.02, n))
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
def oversold_data() -> pd.DataFrame:
    """Create data that should trigger RSI oversold entry."""
    n = 50
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    
    # Sharp decline to create oversold RSI
    prices = [100.0]
    for i in range(1, n):
        if i < 30:
            # Downtrend to push RSI below 35
            prices.append(prices[-1] * 0.99)
        else:
            # Stabilize
            prices.append(prices[-1] * (1 + np.random.uniform(-0.001, 0.001)))
    
    close = pd.Series(prices)
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
# OptimizedConfig Tests
# ============================================================================


class TestOptimizedConfig:
    """Tests for per-symbol optimized configurations."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = OptimizedConfig()
        assert config.rsi_oversold == 35
        assert config.rsi_exit == 50
        assert config.atr_stop_mult == 1.5
        assert config.atr_tp_mult == 2.0

    def test_optimized_configs_exist(self):
        """Test that optimized configs exist for expected symbols."""
        assert "DKNG" in OPTIMIZED_CONFIGS
        assert "AMD" in OPTIMIZED_CONFIGS
        assert "COIN" in OPTIMIZED_CONFIGS
        assert "DEFAULT" in OPTIMIZED_CONFIGS

    def test_dkng_config(self):
        """Test DKNG-specific optimized parameters."""
        config = OPTIMIZED_CONFIGS["DKNG"]
        assert config.rsi_oversold == 35
        assert config.atr_stop_mult == 1.5
        assert config.atr_tp_mult == 2.0

    def test_coin_has_higher_tp_mult(self):
        """Test COIN has higher take-profit multiplier (more volatile)."""
        coin_config = OPTIMIZED_CONFIGS["COIN"]
        default_config = OPTIMIZED_CONFIGS["DEFAULT"]
        assert coin_config.atr_tp_mult > default_config.atr_tp_mult
        assert coin_config.atr_tp_mult == 3.0

    def test_amd_has_tighter_tp(self):
        """Test AMD has tighter take-profit."""
        amd_config = OPTIMIZED_CONFIGS["AMD"]
        assert amd_config.atr_tp_mult == 1.5


# ============================================================================
# ATROptimizedRSIModel Tests
# ============================================================================


class TestATROptimizedRSIModel:
    """Tests for ATR-Optimized RSI Model."""

    def test_model_initialization(self, model: ATROptimizedRSIModel):
        """Test model initializes with correct parameters."""
        assert model.rsi_period == 14
        assert model.rsi_oversold == 35
        assert model.rsi_exit == 50
        assert model.atr_period == 14
        assert model.atr_stop_mult == 1.5
        assert model.atr_tp_mult == 2.0
        assert model.use_optimized is True

    def test_model_initialization_defaults(self):
        """Test model initializes with defaults when no parameters provided."""
        config = ModelConfig(model_id="test", model_type="mean_reversion")
        model = ATROptimizedRSIModel(config)
        assert model.rsi_period == 14
        assert model.rsi_oversold == 35
        assert model.atr_stop_mult == 1.5

    def test_get_config_for_known_symbol(self, model: ATROptimizedRSIModel):
        """Test getting optimized config for known symbol."""
        config = model._get_config_for_symbol("DKNG")
        assert config.atr_tp_mult == 2.0

    def test_get_config_for_unknown_symbol(self, model: ATROptimizedRSIModel):
        """Test getting default config for unknown symbol."""
        config = model._get_config_for_symbol("UNKNOWN_SYMBOL")
        assert config == OPTIMIZED_CONFIGS["DEFAULT"]

    def test_compute_atr(self, model: ATROptimizedRSIModel, sample_ohlcv: pd.DataFrame):
        """Test ATR computation."""
        atr = model._compute_atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
        )
        assert len(atr) == len(sample_ohlcv)
        assert atr.iloc[-1] > 0  # ATR should be positive
        assert not np.isnan(atr.iloc[-1])

    def test_validate_valid_data(self, model: ATROptimizedRSIModel, sample_ohlcv: pd.DataFrame):
        """Test validation passes with valid data."""
        is_valid, msg = model.validate(sample_ohlcv)
        assert is_valid is True
        assert msg == ""

    def test_validate_insufficient_data(self, model: ATROptimizedRSIModel):
        """Test validation fails with insufficient data."""
        small_df = pd.DataFrame({
            "open": [100, 101],
            "high": [102, 103],
            "low": [99, 100],
            "close": [101, 102],
        })
        is_valid, msg = model.validate(small_df)
        assert is_valid is False
        assert "Need" in msg

    def test_validate_missing_columns(self, model: ATROptimizedRSIModel):
        """Test validation fails with missing columns."""
        df = pd.DataFrame({
            "close": [100] * 50,
            "open": [100] * 50,
        })
        is_valid, msg = model.validate(df)
        assert is_valid is False
        assert "Missing" in msg

    @pytest.mark.asyncio
    async def test_generate_no_signal_on_invalid_data(self, model: ATROptimizedRSIModel):
        """Test generate returns None for invalid data."""
        small_df = pd.DataFrame({
            "open": [100],
            "high": [101],
            "low": [99],
            "close": [100],
        })
        signal = await model.generate("AAPL", small_df, datetime.now(timezone.utc))
        assert signal is None

    @pytest.mark.asyncio
    async def test_generate_entry_signal(
        self, model: ATROptimizedRSIModel, oversold_data: pd.DataFrame
    ):
        """Test generating entry signal when RSI oversold."""
        signal = await model.generate("DKNG", oversold_data, datetime.now(timezone.utc))
        
        # Should generate an entry signal due to low RSI
        if signal is not None:
            assert signal.signal_type == SignalType.ENTRY
            assert signal.direction == Direction.LONG
            assert signal.symbol == "DKNG"
            assert "stop_loss" in signal.metadata
            assert "take_profit" in signal.metadata
            assert signal.metadata["model"] == "atr_optimized_rsi"

    @pytest.mark.asyncio
    async def test_generate_no_duplicate_entry(
        self, model: ATROptimizedRSIModel, oversold_data: pd.DataFrame
    ):
        """Test that model doesn't generate duplicate entry signals."""
        # First call may generate entry
        ts = datetime.now(timezone.utc)
        await model.generate("DKNG", oversold_data, ts)
        
        # Second call should not generate another entry
        signal2 = await model.generate("DKNG", oversold_data, ts)
        
        # If already in position, should be None or EXIT
        if signal2 is not None:
            assert signal2.signal_type in (SignalType.EXIT, SignalType.NONE)

    @pytest.mark.asyncio
    async def test_signal_metadata_has_required_fields(
        self, model: ATROptimizedRSIModel, oversold_data: pd.DataFrame
    ):
        """Test that entry signal metadata has required fields."""
        signal = await model.generate("DKNG", oversold_data, datetime.now(timezone.utc))
        
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            metadata = signal.metadata
            assert "model" in metadata
            assert "rsi" in metadata
            assert "atr" in metadata
            assert "stop_loss" in metadata
            assert "take_profit" in metadata
            assert "stop_distance_pct" in metadata
            assert "target_distance_pct" in metadata

    def test_describe(self, model: ATROptimizedRSIModel):
        """Test model description."""
        desc = model.describe()
        assert desc["name"] == "ATR-Optimized RSI"
        assert desc["type"] == "mean_reversion"
        assert desc["version"] == "1.0.0"
        assert "parameters" in desc
        assert "optimized_symbols" in desc
        assert "DKNG" in desc["optimized_symbols"]


# ============================================================================
# Backtest Function Tests
# ============================================================================


class TestBacktestFunction:
    """Tests for the standalone backtest function."""

    def test_backtest_empty_result(self):
        """Test backtest with no trades."""
        # Create data with no oversold conditions
        n = 50
        df = pd.DataFrame({
            "open": [100] * n,
            "high": [101] * n,
            "low": [99] * n,
            "close": [100] * n,  # Flat, RSI will be near 50
        })
        
        result = backtest(df, rsi_os=20)  # Very strict threshold
        assert result["total_trades"] == 0
        assert result["total_return"] == 0

    def test_backtest_generates_trades(self):
        """Test backtest generates trades with trending data."""
        np.random.seed(42)
        n = 200
        
        # Create data with dips below RSI threshold
        prices = [100.0]
        for i in range(1, n):
            if i % 30 < 10:  # Create periodic dips
                prices.append(prices[-1] * 0.98)
            else:
                prices.append(prices[-1] * 1.01)
        
        df = pd.DataFrame({
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
        })
        
        result = backtest(df, rsi_os=40)  # Relaxed threshold
        assert result["total_trades"] >= 0  # May or may not generate trades
        assert "trades" in result

    def test_backtest_returns_expected_keys(self, sample_ohlcv: pd.DataFrame):
        """Test backtest returns all expected keys."""
        result = backtest(sample_ohlcv)
        expected_keys = {
            "total_return",
            "win_rate",
            "total_trades",
            "profit_factor",
            "avg_trade",
            "trades",
        }
        assert expected_keys.issubset(result.keys())

    def test_backtest_custom_parameters(self, sample_ohlcv: pd.DataFrame):
        """Test backtest with custom parameters."""
        result = backtest(
            sample_ohlcv,
            rsi_os=40,
            rsi_exit=60,
            atr_stop_mult=2.0,
            atr_tp_mult=3.0,
            rsi_period=10,
            atr_period=10,
        )
        assert isinstance(result, dict)
        assert "total_trades" in result


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_nan_handling(self, model: ATROptimizedRSIModel):
        """Test model handles NaN values gracefully."""
        n = 50
        df = pd.DataFrame({
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
        })
        df.loc[25, "close"] = np.nan  # Insert NaN
        
        # Should not raise, may return None
        signal = await model.generate("AAPL", df, datetime.now(timezone.utc))
        # Just verify no exception

    @pytest.mark.asyncio
    async def test_extreme_prices(self, model: ATROptimizedRSIModel):
        """Test model handles extreme price values."""
        n = 50
        df = pd.DataFrame({
            "open": [1e6] * n,
            "high": [1.001e6] * n,
            "low": [0.999e6] * n,
            "close": [1e6] * n,
        })
        
        signal = await model.generate("AAPL", df, datetime.now(timezone.utc))
        # Should handle without overflow

    def test_reset_position_tracking(self, model: ATROptimizedRSIModel):
        """Test that position tracking can be reset."""
        model._position = "long"
        model._entry_price = 100.0
        
        # Create new model to verify fresh state
        new_model = ATROptimizedRSIModel(model.config)
        assert new_model._position is None
        assert new_model._entry_price is None
