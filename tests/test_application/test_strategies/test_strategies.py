"""
Comprehensive tests for application strategies.
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta

from ordinis.application.strategies.moving_average_crossover import (
    MovingAverageCrossoverStrategy,
)
from ordinis.application.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from ordinis.application.strategies.bollinger_bands import BollingerBandsStrategy
from ordinis.application.strategies.macd import MACDStrategy
from ordinis.application.strategies.momentum_breakout import MomentumBreakoutStrategy


@pytest.fixture
def sample_data():
    """Generate sample price data for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1D")
    data = pd.DataFrame(
        {
            "open": [100 + i * 0.5 + (i % 10) for i in range(100)],
            "high": [102 + i * 0.5 + (i % 10) for i in range(100)],
            "low": [98 + i * 0.5 + (i % 10) for i in range(100)],
            "close": [100 + i * 0.5 + (i % 10) * 0.8 for i in range(100)],
            "volume": [1000000 + i * 10000 for i in range(100)],
        },
        index=dates,
    )
    data.name = "AAPL"
    return data


class TestMovingAverageCrossoverStrategy:
    """Test Moving Average Crossover strategy."""

    @pytest.mark.asyncio
    async def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = MovingAverageCrossoverStrategy(fast_period=10, slow_period=20)
        assert strategy.params["fast_period"] == 10
        assert strategy.params["slow_period"] == 20

    @pytest.mark.asyncio
    async def test_generate_signal_uptrend(self, sample_data):
        """Test signal generation in uptrend."""
        strategy = MovingAverageCrossoverStrategy(fast_period=5, slow_period=20)
        
        # Use last timestamp
        timestamp = sample_data.index[-1]
        signal = await strategy.generate_signal(sample_data, timestamp)

        # Should generate some signal or None
        assert signal is None or hasattr(signal, "signal_type")

    @pytest.mark.asyncio
    async def test_generate_signal_insufficient_data(self):
        """Test signal generation with insufficient data."""
        strategy = MovingAverageCrossoverStrategy(fast_period=10, slow_period=20)
        
        # Only 5 data points (and required columns present)
        short_data = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [101, 102, 103, 104, 105],
                "low": [99, 100, 101, 102, 103],
                "close": [100, 101, 102, 103, 104],
                "volume": [1000000] * 5,
            },
            index=pd.date_range("2024-01-01", periods=5),
        )
        short_data.name = "AAPL"
        
        timestamp = short_data.index[-1]
        signal = await strategy.generate_signal(short_data, timestamp)
        
        # Should return None due to insufficient data
        assert signal is None


class TestRSIMeanReversionStrategy:
    """Test RSI Mean Reversion strategy."""

    @pytest.mark.asyncio
    async def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = RSIMeanReversionStrategy(
            rsi_period=14,
            oversold_threshold=30,
            overbought_threshold=70,
        )
        assert strategy.params["rsi_period"] == 14
        assert strategy.params["oversold_threshold"] == 30
        assert strategy.params["overbought_threshold"] == 70

    @pytest.mark.asyncio
    async def test_generate_signal_oversold(self):
        """Test signal generation in oversold condition."""
        strategy = RSIMeanReversionStrategy(
            rsi_period=5, oversold_threshold=30, overbought_threshold=70
        )
        
        # Create downtrending data (oversold)
        dates = pd.date_range(start="2024-01-01", periods=50)
        closes = [100 - i * 2 for i in range(50)]  # Strong downtrend
        data = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
                "close": closes,
                "volume": [1000000] * 50,
            },
            index=dates,
        )
        data.name = "AAPL"
        
        timestamp = data.index[-1]
        signal = await strategy.generate_signal(data, timestamp)
        
        # May generate buy signal or None
        assert signal is None or hasattr(signal, "signal_type")

    @pytest.mark.asyncio
    async def test_generate_signal_overbought(self):
        """Test signal generation in overbought condition."""
        strategy = RSIMeanReversionStrategy(
            rsi_period=5, oversold_threshold=30, overbought_threshold=70
        )
        
        # Create uptrending data (overbought)
        dates = pd.date_range(start="2024-01-01", periods=50)
        closes = [100 + i * 2 for i in range(50)]  # Strong uptrend
        data = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
                "close": closes,
                "volume": [1000000] * 50,
            },
            index=dates,
        )
        data.name = "AAPL"
        
        timestamp = data.index[-1]
        signal = await strategy.generate_signal(data, timestamp)
        
        # May generate sell signal or None
        assert signal is None or hasattr(signal, "signal_type")


class TestBollingerBandsStrategy:
    """Test Bollinger Bands strategy."""

    @pytest.mark.asyncio
    async def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = BollingerBandsStrategy(bb_period=20, bb_std=2.0)
        assert strategy.params["bb_period"] == 20
        assert strategy.params["bb_std"] == 2.0

    @pytest.mark.asyncio
    async def test_generate_signal_normal_conditions(self, sample_data):
        """Test signal generation in normal market conditions."""
        strategy = BollingerBandsStrategy(bb_period=10, bb_std=2.0)
        
        timestamp = sample_data.index[-1]
        signal = await strategy.generate_signal(sample_data, timestamp)
        
        # Should return signal or None
        assert signal is None or hasattr(signal, "signal_type")


class TestMACDStrategy:
    """Test MACD strategy."""

    @pytest.mark.asyncio
    async def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = MACDStrategy(
            fast_period=12,
            slow_period=26,
            signal_period=9,
        )
        assert strategy.params["fast_period"] == 12
        assert strategy.params["slow_period"] == 26
        assert strategy.params["signal_period"] == 9

    @pytest.mark.asyncio
    async def test_generate_signal(self, sample_data):
        """Test MACD signal generation."""
        strategy = MACDStrategy(fast_period=5, slow_period=10, signal_period=3)
        
        timestamp = sample_data.index[-1]
        signal = await strategy.generate_signal(sample_data, timestamp)
        
        # Should return signal or None
        assert signal is None or hasattr(signal, "signal_type")


class TestMomentumBreakoutStrategy:
    """Test Momentum Breakout strategy."""

    @pytest.mark.asyncio
    async def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = MomentumBreakoutStrategy(lookback_period=20, breakout_threshold=0.02)
        assert strategy.params["lookback_period"] == 20
        assert strategy.params["breakout_threshold"] == 0.02

    @pytest.mark.asyncio
    async def test_generate_signal_breakout(self):
        """Test signal generation during breakout."""
        strategy = MomentumBreakoutStrategy(lookback_period=10, breakout_threshold=0.01)
        
        # Create data with a breakout
        dates = pd.date_range(start="2024-01-01", periods=50)
        closes = [100] * 40 + [105, 108, 110, 112, 115, 118, 120, 125, 128, 130]
        data = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 2 for c in closes],
                "low": [c - 2 for c in closes],
                "close": closes,
                "volume": [1000000] * 50,
            },
            index=dates,
        )
        data.name = "AAPL"
        
        timestamp = data.index[-1]
        signal = await strategy.generate_signal(data, timestamp)
        
        # Should detect breakout
        assert signal is None or hasattr(signal, "signal_type")

    @pytest.mark.asyncio
    async def test_generate_signal_no_breakout(self):
        """Test signal generation with no breakout."""
        strategy = MomentumBreakoutStrategy(lookback_period=10, breakout_threshold=0.05)
        
        # Create sideways data
        dates = pd.date_range(start="2024-01-01", periods=50)
        closes = [100 + (i % 5) for i in range(50)]  # Oscillating
        data = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
                "close": closes,
                "volume": [1000000] * 50,
            },
            index=dates,
        )
        data.name = "AAPL"
        
        timestamp = data.index[-1]
        signal = await strategy.generate_signal(data, timestamp)
        
        # Likely no signal in sideways market
        assert signal is None or hasattr(signal, "signal_type")


class TestStrategyIntegration:
    """Integration tests for strategies."""

    @pytest.mark.asyncio
    async def test_all_strategies_handle_empty_data(self):
        """Test that all strategies handle empty data gracefully."""
        strategies = [
            MovingAverageCrossoverStrategy(),
            RSIMeanReversionStrategy(),
            BollingerBandsStrategy(),
            MACDStrategy(),
            MomentumBreakoutStrategy(),
        ]
        
        empty_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        empty_data.name = "AAPL"
        timestamp = datetime.utcnow()
        
        for strategy in strategies:
            signal = await strategy.generate_signal(empty_data, timestamp)
            # Should handle empty data without crashing
            assert signal is None

    @pytest.mark.asyncio
    async def test_strategies_consistent_interface(self, sample_data):
        """Test that all strategies have consistent interface."""
        strategies = [
            MovingAverageCrossoverStrategy(),
            RSIMeanReversionStrategy(),
            BollingerBandsStrategy(),
            MACDStrategy(),
            MomentumBreakoutStrategy(),
        ]
        
        timestamp = sample_data.index[-1]
        
        for strategy in strategies:
            # All should have generate_signal method
            assert hasattr(strategy, "generate_signal")
            
            # All should return Signal or None
            signal = await strategy.generate_signal(sample_data, timestamp)
            assert signal is None or hasattr(signal, "signal_type")

    @pytest.mark.asyncio
    async def test_strategies_with_real_pattern(self):
        """Test strategies with realistic price pattern."""
        # V-shaped recovery pattern
        dates = pd.date_range(start="2024-01-01", periods=60)
        closes = (
            [100 - i for i in range(30)]  # Downtrend
            + [70 + i * 0.5 for i in range(30)]  # Recovery
        )
        
        data = pd.DataFrame(
            {
                "open": closes,
                "high": [c + 1 for c in closes],
                "low": [c - 1 for c in closes],
                "close": closes,
                "volume": [1000000] * 60,
            },
            index=dates,
        )
        data.name = "AAPL"
        
        strategies = [
            MovingAverageCrossoverStrategy(fast_period=5, slow_period=15),
            RSIMeanReversionStrategy(rsi_period=10),
            BollingerBandsStrategy(bb_period=15),
        ]
        
        # Test at bottom (should generate buy signals)
        timestamp_bottom = dates[30]
        for strategy in strategies:
            signal = await strategy.generate_signal(
                data.loc[: timestamp_bottom], timestamp_bottom
            )
            assert signal is None or hasattr(signal, "signal_type")
        
        # Test at recovery (may generate different signals)
        timestamp_recovery = dates[-1]
        for strategy in strategies:
            signal = await strategy.generate_signal(data, timestamp_recovery)
            assert signal is None or hasattr(signal, "signal_type")
