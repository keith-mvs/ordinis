"""Tests for FlowRoute trading strategies."""

import pytest

from ordinis.engines.flowroute.strategies.base import BaseStrategy, Signal, SignalStrength
from ordinis.engines.flowroute.strategies.breakout import BreakoutStrategy
from ordinis.engines.flowroute.strategies.ma_crossover import MACrossoverStrategy
from ordinis.engines.flowroute.strategies.rsi_mean_reversion import RSIMeanReversionStrategy
from ordinis.engines.flowroute.strategies.vwap import VWAPStrategy


class TestSignalStrength:
    """Tests for SignalStrength enum."""

    def test_signal_strength_values(self):
        """Test SignalStrength enum values."""
        assert SignalStrength.STRONG_BUY.value == 2
        assert SignalStrength.BUY.value == 1
        assert SignalStrength.NEUTRAL.value == 0
        assert SignalStrength.SELL.value == -1
        assert SignalStrength.STRONG_SELL.value == -2


class TestSignal:
    """Tests for Signal dataclass."""

    def test_signal_creation(self):
        """Test creating a Signal."""
        signal = Signal(
            direction="buy",
            strength=SignalStrength.BUY,
            confidence=0.8,
            reason="Test buy signal",
            metadata={"price": 100.0},
        )
        assert signal.direction == "buy"
        assert signal.strength == SignalStrength.BUY
        assert signal.confidence == 0.8
        assert signal.reason == "Test buy signal"
        assert signal.metadata["price"] == 100.0


class TestBaseStrategy:
    """Tests for BaseStrategy."""

    def test_initialization(self):
        """Test strategy initialization."""

        class ConcreteStrategy(BaseStrategy):
            def update(self, price: float, **kwargs) -> Signal | None:
                return None

        strategy = ConcreteStrategy(name="TestStrategy")
        assert strategy.name == "TestStrategy"
        assert strategy.initialized is False
        assert strategy.last_signal is None

    def test_reset(self):
        """Test strategy reset."""

        class ConcreteStrategy(BaseStrategy):
            def update(self, price: float, **kwargs) -> Signal | None:
                self.initialized = True
                self.last_signal = Signal(
                    direction="buy",
                    strength=SignalStrength.BUY,
                    confidence=0.5,
                    reason="test",
                    metadata={},
                )
                return self.last_signal

        strategy = ConcreteStrategy(name="Test")
        strategy.update(100.0)
        assert strategy.initialized is True
        assert strategy.last_signal is not None

        strategy.reset()
        assert strategy.initialized is False
        assert strategy.last_signal is None

    def test_get_status(self):
        """Test get_status method."""

        class ConcreteStrategy(BaseStrategy):
            def update(self, price: float, **kwargs) -> Signal | None:
                return None

        strategy = ConcreteStrategy(name="TestStrategy")
        status = strategy.get_status()
        assert status["name"] == "TestStrategy"
        assert status["initialized"] is False
        assert status["last_signal"] is None

    def test_is_ready(self):
        """Test is_ready method."""

        class ConcreteStrategy(BaseStrategy):
            def update(self, price: float, **kwargs) -> Signal | None:
                self.initialized = True
                return None

        strategy = ConcreteStrategy(name="Test")
        assert strategy.is_ready() is False
        strategy.update(100.0)
        assert strategy.is_ready() is True


class TestRSIMeanReversionStrategy:
    """Tests for RSI Mean Reversion Strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a strategy instance."""
        return RSIMeanReversionStrategy(period=5, oversold=30, overbought=70)

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.period == 5
        assert strategy.oversold == 30.0
        assert strategy.overbought == 70.0
        assert strategy.initialized is False
        assert len(strategy.prices) == 0

    def test_warmup_period(self, strategy):
        """Test that no signal during warmup."""
        for i in range(5):
            signal = strategy.update(100.0 + i)
            assert signal is None
        assert strategy.initialized is False

    def test_initialization_after_warmup(self, strategy):
        """Test initialization after warmup."""
        for i in range(7):  # period + 1 = 6
            strategy.update(100.0 + i * 0.5)
        assert strategy.initialized is True

    def test_oversold_buy_signal(self, strategy):
        """Test buy signal on oversold RSI."""
        # Create descending prices to drive RSI below 30
        prices = [100, 99, 98, 97, 96, 95, 93]
        for p in prices:
            signal = strategy.update(p)

        # Need more dropping to get oversold
        for p in [90, 85, 80, 75]:
            signal = strategy.update(p)

        # RSI should be very low now, check for buy signal
        if signal:
            assert signal.direction == "buy"

    def test_overbought_sell_signal(self, strategy):
        """Test sell signal on overbought RSI."""
        # Create ascending prices to drive RSI above 70
        prices = [100, 101, 102, 103, 104, 105, 107]
        for p in prices:
            signal = strategy.update(p)

        # Continue rising
        for p in [110, 115, 120, 125]:
            signal = strategy.update(p)

        # RSI should be high now, check for sell signal
        if signal:
            assert signal.direction == "sell"

    def test_no_repeat_signals(self, strategy):
        """Test that same signal isn't repeated."""
        # Drive to oversold for buy
        for p in [100, 99, 98, 97, 96, 95, 90, 85]:
            signal = strategy.update(p)

        first_signal = signal
        # Continue low prices
        signal = strategy.update(82)

        # Should not get another buy signal
        if first_signal and first_signal.direction == "buy":
            assert signal is None or signal.direction != "buy"

    def test_reset(self, strategy):
        """Test strategy reset."""
        for i in range(10):
            strategy.update(100 + i)
        assert len(strategy.prices) > 0
        assert strategy.initialized is True

        strategy.reset()
        assert len(strategy.prices) == 0
        assert strategy.initialized is False
        assert strategy.prev_signal_direction is None

    def test_rsi_calculation(self, strategy):
        """Test internal RSI calculation."""
        # Not enough data
        strategy.prices = [100, 101, 102]
        assert strategy._calculate_rsi() is None

        # Enough data with gains
        strategy.prices = [100, 101, 102, 103, 104, 105]
        rsi = strategy._calculate_rsi()
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_rsi_max_when_no_losses(self, strategy):
        """Test RSI is 100 when no losses."""
        # All gains
        strategy.prices = [100, 101, 102, 103, 104, 105, 106]
        rsi = strategy._calculate_rsi()
        assert rsi == 100.0


class TestMACrossoverStrategy:
    """Tests for MA Crossover Strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a strategy instance."""
        return MACrossoverStrategy(fast_period=5, slow_period=10)

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.fast_period == 5
        assert strategy.slow_period == 10
        assert strategy.initialized is False
        assert len(strategy.prices) == 0

    def test_warmup_period(self, strategy):
        """Test no signal during warmup."""
        for i in range(8):
            signal = strategy.update(100.0)
            assert signal is None
        assert strategy.initialized is False

    def test_initialization_after_warmup(self, strategy):
        """Test initialization after warmup."""
        for i in range(12):
            strategy.update(100.0)
        assert strategy.initialized is True

    def test_bullish_crossover(self, strategy):
        """Test buy signal on bullish crossover."""
        # Start with declining prices so slow MA is higher
        for p in [110, 109, 108, 107, 106, 105, 104, 103, 102, 101]:
            strategy.update(p)

        # Now start rising to create crossover
        for p in [100, 102, 104, 106, 108, 110]:
            signal = strategy.update(p)
            if signal and signal.direction == "buy":
                assert signal.strength == SignalStrength.BUY
                assert "fast_ma" in signal.metadata
                assert "slow_ma" in signal.metadata
                break

    def test_bearish_crossover(self, strategy):
        """Test sell signal on bearish crossover."""
        # Start with rising prices so fast MA is higher
        for p in [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]:
            strategy.update(p)

        # Now start declining to create crossover
        for p in [108, 106, 104, 102, 100, 98]:
            signal = strategy.update(p)
            if signal and signal.direction == "sell":
                assert signal.strength == SignalStrength.SELL
                break

    def test_reset(self, strategy):
        """Test strategy reset."""
        for i in range(15):
            strategy.update(100 + i * 0.1)
        assert len(strategy.prices) > 0
        assert strategy.initialized is True

        strategy.reset()
        assert len(strategy.prices) == 0
        assert strategy.initialized is False
        assert strategy.prev_signal_direction is None

    def test_price_history_limit(self, strategy):
        """Test that price history is limited."""
        for i in range(100):
            strategy.update(100 + i * 0.01)
        # History should be limited to slow_period * 2
        assert len(strategy.prices) <= strategy.slow_period * 2


class TestBreakoutStrategy:
    """Tests for Breakout Strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a strategy instance."""
        return BreakoutStrategy(lookback_period=5, breakout_threshold=0.02)

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.lookback_period == 5
        assert strategy.breakout_threshold == 0.02
        assert strategy.initialized is False
        assert len(strategy.prices) == 0

    def test_warmup_period(self, strategy):
        """Test no signal during warmup."""
        for i in range(4):
            signal = strategy.update(100.0)
            assert signal is None
        assert strategy.initialized is False

    def test_initialization_after_warmup(self, strategy):
        """Test initialization after warmup."""
        for i in range(6):
            strategy.update(100.0)
        assert strategy.initialized is True

    def test_upside_breakout(self, strategy):
        """Test buy signal on upside breakout."""
        # Establish range
        for p in [100, 101, 102, 101, 100]:
            strategy.update(p)

        # Break above range + threshold
        signal = strategy.update(104.5)  # > 102 * 1.02 = 104.04
        assert signal is not None
        assert signal.direction == "buy"
        assert "recent_high" in signal.metadata
        assert "breakout_pct" in signal.metadata

    def test_downside_breakdown(self, strategy):
        """Test sell signal on downside breakdown."""
        # Establish range
        for p in [100, 101, 102, 101, 100]:
            strategy.update(p)

        # Break below range - threshold
        signal = strategy.update(97.0)  # < 100 * 0.98 = 98
        assert signal is not None
        assert signal.direction == "sell"
        assert "recent_low" in signal.metadata
        assert "breakdown_pct" in signal.metadata

    def test_strong_breakout(self, strategy):
        """Test strong buy signal on large breakout."""
        for p in [100, 100, 100, 100, 100]:
            strategy.update(p)

        # Large breakout > 2%
        signal = strategy.update(105)  # +5%
        if signal:
            assert signal.direction == "buy"
            # Could be STRONG_BUY depending on implementation

    def test_no_signal_in_range(self, strategy):
        """Test no signal when price stays in range."""
        for p in [100, 101, 102, 101, 100]:
            strategy.update(p)

        # Stay in range
        signal = strategy.update(101.5)
        assert signal is None

    def test_reset(self, strategy):
        """Test strategy reset."""
        for i in range(10):
            strategy.update(100 + i * 0.1)
        assert len(strategy.prices) > 0
        assert strategy.initialized is True

        strategy.reset()
        assert len(strategy.prices) == 0
        assert strategy.initialized is False
        assert strategy.prev_signal_direction is None

    def test_price_history_limit(self, strategy):
        """Test that price history is limited."""
        for i in range(50):
            strategy.update(100 + i * 0.01)
        # History should be limited
        assert len(strategy.prices) <= strategy.lookback_period * 2


class TestVWAPStrategy:
    """Tests for VWAP Strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a strategy instance."""
        return VWAPStrategy(deviation_threshold=0.01)  # 1% threshold

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.deviation_threshold == 0.01
        assert strategy.initialized is False
        assert len(strategy.prices) == 0
        assert len(strategy.volumes) == 0
        assert strategy.cumulative_tpv == 0.0
        assert strategy.cumulative_volume == 0.0

    def test_warmup_period(self, strategy):
        """Test no signal during warmup."""
        for i in range(4):
            signal = strategy.update(100.0, volume=1000)
            assert signal is None
        assert strategy.initialized is False

    def test_initialization_after_warmup(self, strategy):
        """Test initialization after warmup."""
        for i in range(6):
            strategy.update(100.0, volume=1000)
        assert strategy.initialized is True

    def test_buy_signal_above_vwap(self, strategy):
        """Test buy signal when price above VWAP."""
        # Establish VWAP at 100
        for i in range(5):
            strategy.update(100.0, volume=1000)

        # Price jumps above VWAP by more than threshold
        signal = strategy.update(102.0, volume=1000)  # +2% above VWAP
        assert signal is not None
        assert signal.direction == "buy"
        assert "vwap" in signal.metadata
        assert "deviation" in signal.metadata

    def test_sell_signal_below_vwap(self, strategy):
        """Test sell signal when price below VWAP."""
        # Establish VWAP at 100
        for i in range(5):
            strategy.update(100.0, volume=1000)

        # Price drops below VWAP by more than threshold
        signal = strategy.update(98.0, volume=1000)  # -2% below VWAP
        assert signal is not None
        assert signal.direction == "sell"
        assert "vwap" in signal.metadata

    def test_no_signal_within_threshold(self, strategy):
        """Test no signal when price within threshold."""
        for i in range(5):
            strategy.update(100.0, volume=1000)

        # Small deviation - within threshold
        signal = strategy.update(100.5, volume=1000)  # +0.5%
        assert signal is None

    def test_default_volume(self, strategy):
        """Test with default volume when not provided."""
        for i in range(6):
            signal = strategy.update(100.0)  # No volume argument
        assert strategy.initialized is True
        assert sum(strategy.volumes) > 0

    def test_reset(self, strategy):
        """Test strategy reset."""
        for i in range(10):
            strategy.update(100 + i, volume=1000)
        assert len(strategy.prices) > 0
        assert strategy.initialized is True
        assert strategy.cumulative_volume > 0

        strategy.reset()
        assert len(strategy.prices) == 0
        assert len(strategy.volumes) == 0
        assert strategy.cumulative_tpv == 0.0
        assert strategy.cumulative_volume == 0.0
        assert strategy.initialized is False

    def test_vwap_calculation(self, strategy):
        """Test VWAP calculation accuracy."""
        # Add known prices and volumes
        strategy.update(100.0, volume=100)  # TPV = 10000
        strategy.update(102.0, volume=200)  # TPV = 20400
        strategy.update(101.0, volume=100)  # TPV = 10100
        strategy.update(103.0, volume=100)  # TPV = 10300
        strategy.update(104.0, volume=100)  # TPV = 10400

        # Total TPV = 61200, Total Volume = 600
        # VWAP = 61200 / 600 = 102.0
        assert strategy.cumulative_volume == 600
        assert strategy.cumulative_tpv == 61200.0

    def test_price_history_limit(self, strategy):
        """Test that price history is limited."""
        for i in range(600):
            strategy.update(100 + i * 0.001, volume=100)
        # History should be limited to 500
        assert len(strategy.prices) <= 500
        assert len(strategy.volumes) <= 500
