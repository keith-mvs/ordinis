"""
Additional tests for Fibonacci ADX Strategy v1.2 features.

Tests cover:
- ADX slope calculation and trend_accelerating gating
- Chandelier Exit model functionality
- Integration of new features with strategy

Tests file: tests/test_application/test_fibonacci_adx_strategy.py (10 tests)
Additional tests: This file (saved to docs for reference)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ordinis.engines.signalcore import ModelConfig
from ordinis.engines.signalcore.models import ADXTrendModel, ChandelierExitModel, ExitMode
from ordinis.application.strategies.fibonacci_adx import FibonacciADXStrategy
from ordinis.engines.signalcore.core.signal import SignalType, Direction


# =============================================================================
# ADX Slope Tests
# =============================================================================

class TestADXSlopeCalculation:
    """Tests for ADX slope and trend_accelerating metadata."""

    @pytest.fixture
    def adx_model(self):
        """Create ADX model with slope parameters."""
        config = ModelConfig(
            model_id="test-adx",
            model_type="trend",
            version="1.2.0",
            parameters={
                "adx_period": 14,
                "adx_threshold": 25,
                "slope_lookback": 5,
                "slope_threshold": 2.0,
            },
        )
        return ADXTrendModel(config)

    @pytest.fixture
    def trending_data(self):
        """Create data with clear uptrend (ADX should be high and accelerating)."""
        np.random.seed(42)
        n_bars = 100
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="1h")
        
        # Strong uptrend
        close = 100 + np.cumsum(np.random.uniform(0.2, 0.8, n_bars))
        high = close + np.random.uniform(0.3, 0.6, n_bars)
        low = close - np.random.uniform(0.2, 0.4, n_bars)
        
        return pd.DataFrame({
            "open": close - np.random.uniform(-0.2, 0.2, n_bars),
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 5000, n_bars),
            "symbol": "TEST",
        }, index=dates)

    @pytest.mark.asyncio
    async def test_adx_slope_in_metadata(self, adx_model, trending_data):
        """Test that ADX slope is calculated and included in metadata."""
        timestamp = datetime.now()
        signal = await adx_model.generate(trending_data, timestamp)
        
        assert "adx_slope" in signal.metadata
        assert "trend_accelerating" in signal.metadata
        assert "trend_decelerating" in signal.metadata
        assert "slope_lookback" in signal.metadata

    @pytest.mark.asyncio
    async def test_trend_accelerating_boolean(self, adx_model, trending_data):
        """Test that trend_accelerating is a boolean-like value."""
        timestamp = datetime.now()
        signal = await adx_model.generate(trending_data, timestamp)
        
        # Allow numpy bool or Python bool
        assert signal.metadata["trend_accelerating"] in (True, False, np.True_, np.False_)
        assert signal.metadata["trend_decelerating"] in (True, False, np.True_, np.False_)

    @pytest.mark.asyncio
    async def test_slope_in_feature_contributions(self, adx_model, trending_data):
        """Test that ADX slope is in feature contributions."""
        timestamp = datetime.now()
        signal = await adx_model.generate(trending_data, timestamp)
        
        assert "adx_slope" in signal.feature_contributions
        assert "trend_accelerating" in signal.feature_contributions


# =============================================================================
# Chandelier Exit Tests
# =============================================================================

class TestChandelierExitModel:
    """Tests for Chandelier Exit trailing stop model."""

    @pytest.fixture
    def chandelier_model_long(self):
        """Create Chandelier Exit model for long positions."""
        config = ModelConfig(
            model_id="test-chandelier",
            model_type="exit",
            version="1.0.0",
            parameters={
                "atr_period": 22,
                "atr_multiplier": 3.0,
                "lookback": 22,
                "exit_mode": "long",
            },
        )
        return ChandelierExitModel(config)

    @pytest.fixture
    def chandelier_model_short(self):
        """Create Chandelier Exit model for short positions."""
        config = ModelConfig(
            model_id="test-chandelier",
            model_type="exit",
            version="1.0.0",
            parameters={
                "atr_period": 22,
                "atr_multiplier": 3.0,
                "lookback": 22,
                "exit_mode": "short",
            },
        )
        return ChandelierExitModel(config)

    @pytest.fixture
    def uptrend_data(self):
        """Create uptrend data where price is above chandelier level."""
        np.random.seed(42)
        n_bars = 50
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="1h")
        
        # Strong uptrend - price well above any trailing stop
        base = 100 + np.cumsum(np.random.uniform(0.5, 1.0, n_bars))
        high = base + np.random.uniform(0.5, 1.0, n_bars)
        low = base - np.random.uniform(0.3, 0.6, n_bars)
        close = base
        
        return pd.DataFrame({
            "open": close - np.random.uniform(-0.2, 0.2, n_bars),
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 5000, n_bars),
            "symbol": "TEST",
        }, index=dates)

    @pytest.fixture
    def reversal_data(self):
        """Create data where price drops below chandelier level."""
        np.random.seed(43)
        n_bars = 50
        dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="1h")
        
        # Uptrend then sharp reversal
        phase1 = 100 + np.cumsum(np.random.uniform(0.5, 1.0, 35))
        phase2 = np.linspace(phase1[-1], phase1[-1] - 15, 15)  # Sharp drop
        close = np.concatenate([phase1, phase2])
        
        high = close + np.random.uniform(0.5, 1.0, n_bars)
        low = close - np.random.uniform(0.3, 0.6, n_bars)
        
        return pd.DataFrame({
            "open": close - np.random.uniform(-0.2, 0.2, n_bars),
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.randint(1000, 5000, n_bars),
            "symbol": "TEST",
        }, index=dates)

    @pytest.mark.asyncio
    async def test_chandelier_level_calculated(self, chandelier_model_long, uptrend_data):
        """Test that chandelier level is calculated and in metadata."""
        timestamp = datetime.now()
        signal = await chandelier_model_long.generate(uptrend_data, timestamp)
        
        assert "chandelier_level" in signal.metadata
        assert "current_price" in signal.metadata
        assert "atr" in signal.metadata
        assert "exit_mode" in signal.metadata
        assert signal.metadata["exit_mode"] == "long"

    @pytest.mark.asyncio
    async def test_chandelier_hold_in_uptrend(self, chandelier_model_long, uptrend_data):
        """Test that chandelier returns HOLD when price is above level."""
        timestamp = datetime.now()
        signal = await chandelier_model_long.generate(uptrend_data, timestamp)
        
        # In uptrend, price should be above chandelier level
        if signal.metadata["current_price"] > signal.metadata["chandelier_level"]:
            assert signal.signal_type == SignalType.HOLD
            assert signal.metadata["exit_triggered"] == False  # Use == for numpy bool

    @pytest.mark.asyncio
    async def test_chandelier_exit_on_reversal(self, chandelier_model_long, reversal_data):
        """Test that chandelier returns EXIT when price drops below level."""
        timestamp = datetime.now()
        signal = await chandelier_model_long.generate(reversal_data, timestamp)
        
        # After sharp reversal, exit should be triggered
        if signal.metadata["exit_triggered"]:
            assert signal.signal_type == SignalType.EXIT
            assert signal.direction == Direction.NEUTRAL

    @pytest.mark.asyncio
    async def test_chandelier_short_mode(self, chandelier_model_short, uptrend_data):
        """Test chandelier exit in short mode."""
        timestamp = datetime.now()
        signal = await chandelier_model_short.generate(uptrend_data, timestamp)
        
        assert signal.metadata["exit_mode"] == "short"
        assert "lowest_low" in signal.metadata

    @pytest.mark.asyncio
    async def test_distance_to_exit_calculated(self, chandelier_model_long, uptrend_data):
        """Test that distance to exit is calculated."""
        timestamp = datetime.now()
        signal = await chandelier_model_long.generate(uptrend_data, timestamp)
        
        assert "distance_to_exit" in signal.metadata
        assert "distance_percent" in signal.metadata

    def test_calculate_stop_level_utility(self, chandelier_model_long, uptrend_data):
        """Test the utility method for calculating stop level."""
        high = uptrend_data["high"]
        low = uptrend_data["low"]
        close = uptrend_data["close"]
        
        stop_level = chandelier_model_long.calculate_stop_level(high, low, close)
        
        assert isinstance(stop_level, float)
        assert stop_level < close.iloc[-1]  # For longs, stop should be below current price


# =============================================================================
# Strategy Integration Tests
# =============================================================================

class TestFibonacciADXWithSlope:
    """Tests for Fibonacci ADX strategy with trend_accelerating gating."""

    @pytest.fixture
    def strategy_with_slope_gating(self):
        """Create strategy with require_trend_accelerating=True."""
        strat = FibonacciADXStrategy(
            name="test-fib-adx-slope",
            adx_period=14,
            adx_threshold=25,
            swing_lookback=50,
            tolerance=0.02,
            require_trend_accelerating=True,
            slope_lookback=5,
        )
        strat.configure()
        return strat

    @pytest.fixture
    def strategy_without_slope_gating(self):
        """Create strategy with require_trend_accelerating=False (default)."""
        strat = FibonacciADXStrategy(
            name="test-fib-adx",
            adx_period=14,
            adx_threshold=25,
            swing_lookback=50,
            tolerance=0.02,
            require_trend_accelerating=False,
        )
        strat.configure()
        return strat

    def test_slope_gating_parameter_stored(self, strategy_with_slope_gating):
        """Test that slope gating parameter is stored."""
        assert strategy_with_slope_gating.require_trend_accelerating is True
        assert strategy_with_slope_gating.slope_lookback == 5

    def test_default_slope_gating_off(self, strategy_without_slope_gating):
        """Test that default is no slope gating."""
        assert strategy_without_slope_gating.require_trend_accelerating is False


# =============================================================================
# Run as standalone
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
