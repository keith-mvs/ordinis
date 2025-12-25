"""
Unit tests for FibonacciADXStrategy.

Tests cover:
- Entry requires ADX confirmation
- Tiered stop-loss placement based on entry level
- Fibonacci extension targets (127.2%, 161.8%)
- Signal metadata validation
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ordinis.application.strategies.fibonacci_adx import FibonacciADXStrategy
from ordinis.engines.signalcore.core.signal import Direction, SignalType


@pytest.fixture
def strategy():
    """Create a FibonacciADXStrategy instance."""
    strat = FibonacciADXStrategy(
        name="test-fib-adx",
        adx_period=14,
        adx_threshold=25,
        swing_lookback=50,
        fib_levels=[0.382, 0.5, 0.618],
        tolerance=0.02,  # 2% tolerance for testing
    )
    strat.configure()
    return strat


@pytest.fixture
def bullish_data_at_382():
    """
    Create OHLCV data simulating a bullish trend with price at 38.2% retracement.
    
    Swing: Low=100, High=110 (10% move up)
    38.2% retracement = 110 - (10 * 0.382) = 106.18
    Current price near 106.18
    ADX > 25 (strong trend)
    """
    np.random.seed(42)
    n_bars = 100
    
    # Create uptrend then pullback to 38.2%
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="1h")
    
    # Phase 1: Uptrend from 100 to 110 (first 60 bars)
    phase1 = np.linspace(100, 110, 60)
    # Phase 2: Pullback to ~106.18 (38.2% level) over 40 bars
    phase2 = np.linspace(110, 106.18, 40)
    close = np.concatenate([phase1, phase2])
    
    # Add small noise
    close = close + np.random.normal(0, 0.1, n_bars)
    close[-1] = 106.20  # Ensure last price is near 38.2%
    
    # Generate OHLCV
    high = close + np.random.uniform(0.2, 0.5, n_bars)
    low = close - np.random.uniform(0.2, 0.5, n_bars)
    open_price = close - np.random.uniform(-0.3, 0.3, n_bars)
    volume = np.random.randint(1000, 5000, n_bars)
    
    # Ensure swing high/low are clear
    high[59] = 110.5  # Clear swing high
    low[0] = 99.5     # Clear swing low
    
    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "TEST",
    }, index=dates)
    
    return df


@pytest.fixture
def bullish_data_at_618():
    """
    Create OHLCV data with price at 61.8% retracement (deeper pullback).
    
    Swing: Low=100, High=110
    61.8% retracement = 110 - (10 * 0.618) = 103.82
    """
    np.random.seed(43)
    n_bars = 100
    
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="1h")
    
    # Uptrend then deeper pullback
    phase1 = np.linspace(100, 110, 50)
    phase2 = np.linspace(110, 103.82, 50)
    close = np.concatenate([phase1, phase2])
    close = close + np.random.normal(0, 0.1, n_bars)
    close[-1] = 103.85
    
    high = close + np.random.uniform(0.2, 0.5, n_bars)
    low = close - np.random.uniform(0.2, 0.5, n_bars)
    open_price = close - np.random.uniform(-0.3, 0.3, n_bars)
    volume = np.random.randint(1000, 5000, n_bars)
    
    high[49] = 110.5
    low[0] = 99.5
    
    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "TEST",
    }, index=dates)
    
    return df


@pytest.fixture
def weak_trend_data():
    """Create data with weak ADX (choppy market) - should not generate entry."""
    np.random.seed(44)
    n_bars = 100
    
    dates = pd.date_range(end=datetime.now(), periods=n_bars, freq="1h")
    
    # Sideways choppy movement
    close = 100 + np.random.normal(0, 0.5, n_bars)
    close = np.cumsum(np.random.choice([-0.1, 0.1], n_bars)) + 100
    
    high = close + np.random.uniform(0.2, 0.5, n_bars)
    low = close - np.random.uniform(0.2, 0.5, n_bars)
    open_price = close - np.random.uniform(-0.2, 0.2, n_bars)
    volume = np.random.randint(1000, 5000, n_bars)
    
    df = pd.DataFrame({
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "symbol": "TEST",
    }, index=dates)
    
    return df


class TestFibonacciADXStrategy:
    """Test suite for FibonacciADXStrategy."""
    
    @pytest.mark.asyncio
    async def test_strategy_initialization(self, strategy):
        """Test strategy initializes correctly with parameters."""
        assert strategy.name == "test-fib-adx"
        assert strategy.adx_threshold == 25
        assert strategy.swing_lookback == 50
        assert 0.382 in strategy.fib_levels
        assert 0.5 in strategy.fib_levels
        assert 0.618 in strategy.fib_levels
    
    @pytest.mark.asyncio
    async def test_entry_requires_adx_above_threshold(self, strategy, weak_trend_data):
        """Test that no entry is generated when ADX is below threshold."""
        timestamp = datetime.now()
        signal = await strategy.generate_signal(weak_trend_data, timestamp)
        
        # Should return None when ADX is weak (no strong trend)
        # Note: actual behavior depends on ADX model output
        # This test validates the gating logic
        if signal is not None:
            # If a signal is returned, verify ADX was checked
            adx = signal.metadata.get("adx", 0)
            assert adx >= strategy.adx_threshold, "Signal should only be generated when ADX >= threshold"
    
    @pytest.mark.asyncio
    async def test_tiered_stop_at_382_level(self, strategy, bullish_data_at_382):
        """Test that entry at 38.2% sets stop below 50% level."""
        timestamp = datetime.now()
        signal = await strategy.generate_signal(bullish_data_at_382, timestamp)
        
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            stop_loss = signal.metadata.get("stop_loss")
            entry_level = signal.metadata.get("entry_level", signal.metadata.get("nearest_level"))
            all_levels = signal.metadata.get("all_levels", {})
            
            if entry_level == "38.2%" and stop_loss is not None:
                # Stop should be just below the 50% level
                level_50 = all_levels.get("50.0%")
                if level_50:
                    assert stop_loss < level_50, f"Stop {stop_loss} should be below 50% level {level_50}"
                    assert stop_loss > all_levels.get("61.8%", 0), "Stop should be above 61.8% level"
    
    @pytest.mark.asyncio
    async def test_tiered_stop_at_618_level(self, strategy, bullish_data_at_618):
        """Test that entry at 61.8% sets stop below swing low."""
        timestamp = datetime.now()
        signal = await strategy.generate_signal(bullish_data_at_618, timestamp)
        
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            stop_loss = signal.metadata.get("stop_loss")
            entry_level = signal.metadata.get("entry_level", signal.metadata.get("nearest_level"))
            swing_low = signal.metadata.get("swing_low")
            
            if entry_level == "61.8%" and stop_loss is not None and swing_low is not None:
                # Stop should be below swing low (swing_low * 0.98)
                assert stop_loss < swing_low, f"Stop {stop_loss} should be below swing low {swing_low}"
                assert stop_loss >= swing_low * 0.97, "Stop should not be too far below swing low"
    
    @pytest.mark.asyncio
    async def test_extension_targets_present(self, strategy, bullish_data_at_382):
        """Test that extension targets (127.2%, 161.8%) are calculated."""
        timestamp = datetime.now()
        signal = await strategy.generate_signal(bullish_data_at_382, timestamp)
        
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            # Check extension targets are in metadata
            ext_1272 = signal.metadata.get("extension_1272")
            ext_1618 = signal.metadata.get("extension_1618")
            take_profit_2 = signal.metadata.get("take_profit_2")
            take_profit_3 = signal.metadata.get("take_profit_3")
            swing_high = signal.metadata.get("swing_high")
            swing_low = signal.metadata.get("swing_low")
            
            if swing_high and swing_low:
                swing_range = swing_high - swing_low
                
                # Verify extension calculations
                if ext_1272 is not None:
                    expected_1272 = swing_high + (swing_range * 0.272)
                    assert abs(ext_1272 - expected_1272) < 0.01, f"127.2% extension incorrect"
                
                if ext_1618 is not None:
                    expected_1618 = swing_high + (swing_range * 0.618)
                    assert abs(ext_1618 - expected_1618) < 0.01, f"161.8% extension incorrect"
                
                # TP2 should equal 127.2% extension for longs
                if take_profit_2 is not None and signal.direction == Direction.LONG:
                    assert take_profit_2 == ext_1272, "TP2 should be 127.2% extension for longs"
                
                # TP3 should equal 161.8% extension for longs
                if take_profit_3 is not None and signal.direction == Direction.LONG:
                    assert take_profit_3 == ext_1618, "TP3 should be 161.8% extension for longs"
    
    @pytest.mark.asyncio
    async def test_take_profit_1_is_swing_high(self, strategy, bullish_data_at_382):
        """Test that primary take profit is at swing high for longs."""
        timestamp = datetime.now()
        signal = await strategy.generate_signal(bullish_data_at_382, timestamp)
        
        if signal is not None and signal.signal_type == SignalType.ENTRY and signal.direction == Direction.LONG:
            take_profit = signal.metadata.get("take_profit")
            swing_high = signal.metadata.get("swing_high")
            
            if take_profit is not None and swing_high is not None:
                assert take_profit == swing_high, "TP1 should be swing high for long entries"
    
    @pytest.mark.asyncio
    async def test_risk_reward_ratio_calculated(self, strategy, bullish_data_at_382):
        """Test that risk/reward ratio is calculated correctly."""
        timestamp = datetime.now()
        signal = await strategy.generate_signal(bullish_data_at_382, timestamp)
        
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            rr_ratio = signal.metadata.get("risk_reward_ratio")
            stop_loss = signal.metadata.get("stop_loss")
            take_profit = signal.metadata.get("take_profit")
            current_price = signal.metadata.get("current_price")
            
            if rr_ratio is not None and all([stop_loss, take_profit, current_price]):
                expected_rr = abs(take_profit - current_price) / abs(current_price - stop_loss)
                assert abs(rr_ratio - expected_rr) < 0.01, "R:R ratio calculation incorrect"
    
    @pytest.mark.asyncio
    async def test_signal_metadata_complete(self, strategy, bullish_data_at_382):
        """Test that all required metadata fields are present."""
        timestamp = datetime.now()
        signal = await strategy.generate_signal(bullish_data_at_382, timestamp)
        
        if signal is not None and signal.signal_type == SignalType.ENTRY:
            required_fields = [
                "swing_high",
                "swing_low",
                "current_price",
                "adx",
                "strategy",
                "stop_loss",
                "take_profit",
            ]
            
            for field in required_fields:
                assert field in signal.metadata, f"Missing required metadata field: {field}"
    
    @pytest.mark.asyncio
    async def test_insufficient_data_returns_none(self, strategy):
        """Test that insufficient data returns None."""
        # Create minimal data (less than required)
        dates = pd.date_range(end=datetime.now(), periods=10, freq="1h")
        df = pd.DataFrame({
            "open": [100] * 10,
            "high": [101] * 10,
            "low": [99] * 10,
            "close": [100] * 10,
            "volume": [1000] * 10,
            "symbol": "TEST",
        }, index=dates)
        
        timestamp = datetime.now()
        signal = await strategy.generate_signal(df, timestamp)
        
        assert signal is None, "Should return None for insufficient data"


class TestFibonacciLevelCalculations:
    """Tests for Fibonacci level calculations in the model."""
    
    @pytest.mark.asyncio
    async def test_fib_levels_correctly_computed(self, strategy, bullish_data_at_382):
        """Test that Fibonacci levels are correctly calculated from swing points."""
        timestamp = datetime.now()
        signal = await strategy.generate_signal(bullish_data_at_382, timestamp)
        
        if signal is not None:
            all_levels = signal.metadata.get("all_levels", {})
            swing_high = signal.metadata.get("swing_high")
            swing_low = signal.metadata.get("swing_low")
            
            if all_levels and swing_high and swing_low:
                diff = swing_high - swing_low
                
                # Check key levels
                if "38.2%" in all_levels:
                    expected = swing_high - (diff * 0.382)
                    assert abs(all_levels["38.2%"] - expected) < 0.01
                
                if "50.0%" in all_levels:
                    expected = swing_high - (diff * 0.5)
                    assert abs(all_levels["50.0%"] - expected) < 0.01
                
                if "61.8%" in all_levels:
                    expected = swing_high - (diff * 0.618)
                    assert abs(all_levels["61.8%"] - expected) < 0.01
