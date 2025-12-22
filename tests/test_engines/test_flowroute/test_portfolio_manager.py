"""Tests for PortfolioManager.

Tests cover:
- PortfolioSignal dataclass
- PortfolioManager initialization
- Aggregation modes (consensus, majority, weighted, any)
- Signal history management
- Status and reset methods
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ordinis.engines.flowroute.portfolio_manager import PortfolioManager, PortfolioSignal
from ordinis.engines.flowroute.strategies.base import BaseStrategy, Signal, SignalStrength


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""

    def __init__(self, name: str, is_ready_val: bool = True):
        super().__init__(name)
        self._is_ready = is_ready_val
        self._signal: Signal | None = None
        self.initialized = is_ready_val

    def update(self, price: float, **kwargs) -> Signal | None:
        return self._signal

    def set_signal(self, signal: Signal | None):
        self._signal = signal

    def is_ready(self) -> bool:
        return self._is_ready

    def set_ready(self, ready: bool):
        self._is_ready = ready
        self.initialized = ready


class TestPortfolioSignal:
    """Tests for PortfolioSignal dataclass."""

    @pytest.mark.unit
    def test_create_portfolio_signal(self):
        """Test creating a portfolio signal."""
        signal = PortfolioSignal(
            direction="buy",
            strength=0.8,
            confidence=0.9,
            consensus=0.75,
            contributing_strategies=["strategy1", "strategy2"],
            reasons=["reason1", "reason2"],
        )

        assert signal.direction == "buy"
        assert signal.strength == 0.8
        assert signal.confidence == 0.9
        assert signal.consensus == 0.75
        assert len(signal.contributing_strategies) == 2
        assert len(signal.reasons) == 2


class TestPortfolioManagerInit:
    """Tests for PortfolioManager initialization."""

    @pytest.mark.unit
    def test_init_default_mode(self):
        """Test default initialization."""
        strategies = [MockStrategy("s1"), MockStrategy("s2")]
        manager = PortfolioManager(strategies)

        assert manager.mode == "weighted"
        assert manager.min_strategies_ready == 1
        assert len(manager.strategies) == 2
        assert manager.signal_history == []

    @pytest.mark.unit
    def test_init_custom_mode(self):
        """Test initialization with custom mode."""
        strategies = [MockStrategy("s1")]
        manager = PortfolioManager(strategies, mode="consensus", min_strategies_ready=2)

        assert manager.mode == "consensus"
        assert manager.min_strategies_ready == 2


class TestPortfolioManagerUpdate:
    """Tests for PortfolioManager update method."""

    @pytest.mark.unit
    def test_update_no_signals(self):
        """Test update with no signals from strategies."""
        s1 = MockStrategy("s1")
        s1.set_signal(None)
        manager = PortfolioManager([s1])

        result = manager.update(100.0)

        assert result is None

    @pytest.mark.unit
    def test_update_not_enough_ready_strategies(self):
        """Test update when not enough strategies are ready."""
        s1 = MockStrategy("s1", is_ready_val=False)
        manager = PortfolioManager([s1], min_strategies_ready=2)

        result = manager.update(100.0)

        assert result is None

    @pytest.mark.unit
    def test_update_stores_signal_in_history(self):
        """Test that signals are stored in history."""
        s1 = MockStrategy("s1")
        s1.set_signal(Signal(
            direction="buy",
            strength=SignalStrength.BUY,
            confidence=0.8,
            reason="test",
            metadata={},
        ))
        manager = PortfolioManager([s1], mode="any")

        manager.update(100.0)

        assert len(manager.signal_history) == 1

    @pytest.mark.unit
    def test_update_trims_history_at_100(self):
        """Test that signal history is trimmed at 100 entries."""
        s1 = MockStrategy("s1")
        s1.set_signal(Signal(
            direction="buy",
            strength=SignalStrength.BUY,
            confidence=0.8,
            reason="test",
            metadata={},
        ))
        manager = PortfolioManager([s1], mode="any")
        # Pre-fill history
        manager.signal_history = [MagicMock() for _ in range(100)]

        manager.update(100.0)

        assert len(manager.signal_history) == 100


class TestConsensusAggregation:
    """Tests for consensus aggregation mode."""

    @pytest.mark.unit
    def test_consensus_all_buy(self):
        """Test consensus when all strategies agree on buy."""
        s1 = MockStrategy("s1")
        s2 = MockStrategy("s2")
        s1.set_signal(Signal("buy", SignalStrength.BUY, 0.8, "reason1", {}))
        s2.set_signal(Signal("buy", SignalStrength.BUY, 0.9, "reason2", {}))

        manager = PortfolioManager([s1, s2], mode="consensus")
        result = manager.update(100.0)

        assert result is not None
        assert result.direction == "buy"
        assert result.consensus == 1.0

    @pytest.mark.unit
    def test_consensus_all_sell(self):
        """Test consensus when all strategies agree on sell."""
        s1 = MockStrategy("s1")
        s2 = MockStrategy("s2")
        s1.set_signal(Signal("sell", SignalStrength.SELL, 0.8, "reason1", {}))
        s2.set_signal(Signal("sell", SignalStrength.SELL, 0.9, "reason2", {}))

        manager = PortfolioManager([s1, s2], mode="consensus")
        result = manager.update(100.0)

        assert result is not None
        assert result.direction == "sell"
        assert result.strength == -1.0

    @pytest.mark.unit
    def test_consensus_no_agreement(self):
        """Test consensus when strategies disagree."""
        s1 = MockStrategy("s1")
        s2 = MockStrategy("s2")
        s1.set_signal(Signal("buy", SignalStrength.BUY, 0.8, "reason1", {}))
        s2.set_signal(Signal("sell", SignalStrength.SELL, 0.9, "reason2", {}))

        manager = PortfolioManager([s1, s2], mode="consensus")
        result = manager.update(100.0)

        assert result is None


class TestMajorityAggregation:
    """Tests for majority aggregation mode."""

    @pytest.mark.unit
    def test_majority_buy(self):
        """Test majority with buy signals."""
        s1 = MockStrategy("s1")
        s2 = MockStrategy("s2")
        s3 = MockStrategy("s3")
        s1.set_signal(Signal("buy", SignalStrength.BUY, 0.8, "reason1", {}))
        s2.set_signal(Signal("buy", SignalStrength.BUY, 0.9, "reason2", {}))
        s3.set_signal(Signal("sell", SignalStrength.SELL, 0.7, "reason3", {}))

        manager = PortfolioManager([s1, s2, s3], mode="majority")
        result = manager.update(100.0)

        assert result is not None
        assert result.direction == "buy"
        assert result.consensus == 2 / 3

    @pytest.mark.unit
    def test_majority_no_majority(self):
        """Test majority when no clear majority."""
        s1 = MockStrategy("s1")
        s2 = MockStrategy("s2")
        s1.set_signal(Signal("buy", SignalStrength.BUY, 0.8, "reason1", {}))
        s2.set_signal(Signal("sell", SignalStrength.SELL, 0.9, "reason2", {}))

        manager = PortfolioManager([s1, s2], mode="majority")
        result = manager.update(100.0)

        assert result is None


class TestWeightedAggregation:
    """Tests for weighted aggregation mode."""

    @pytest.mark.unit
    def test_weighted_net_buy(self):
        """Test weighted with net buy signal."""
        s1 = MockStrategy("s1")
        s1.set_signal(Signal("buy", SignalStrength.STRONG_BUY, 0.9, "reason1", {}))

        manager = PortfolioManager([s1], mode="weighted")
        result = manager.update(100.0)

        assert result is not None
        assert result.direction == "buy"

    @pytest.mark.unit
    def test_weighted_net_sell(self):
        """Test weighted with net sell signal."""
        s1 = MockStrategy("s1")
        s1.set_signal(Signal("sell", SignalStrength.STRONG_SELL, 0.9, "reason1", {}))

        manager = PortfolioManager([s1], mode="weighted")
        result = manager.update(100.0)

        assert result is not None
        assert result.direction == "sell"

    @pytest.mark.unit
    def test_weighted_below_threshold(self):
        """Test weighted when score is below threshold."""
        s1 = MockStrategy("s1")
        s2 = MockStrategy("s2")
        # Very low confidence signals that should cancel out
        s1.set_signal(Signal("buy", SignalStrength.NEUTRAL, 0.01, "reason1", {}))
        s2.set_signal(Signal("sell", SignalStrength.NEUTRAL, 0.01, "reason2", {}))

        manager = PortfolioManager([s1, s2], mode="weighted")
        result = manager.update(100.0)

        assert result is None


class TestAnyAggregation:
    """Tests for 'any' aggregation mode."""

    @pytest.mark.unit
    def test_any_takes_buy(self):
        """Test any mode takes any buy signal."""
        s1 = MockStrategy("s1")
        s2 = MockStrategy("s2")
        s1.set_signal(Signal("buy", SignalStrength.BUY, 0.9, "reason1", {}))
        s2.set_signal(None)

        manager = PortfolioManager([s1, s2], mode="any")
        result = manager.update(100.0)

        assert result is not None
        assert result.direction == "buy"

    @pytest.mark.unit
    def test_any_takes_sell(self):
        """Test any mode takes sell when no buy."""
        s1 = MockStrategy("s1")
        s1.set_signal(Signal("sell", SignalStrength.SELL, 0.8, "reason1", {}))

        manager = PortfolioManager([s1], mode="any")
        result = manager.update(100.0)

        assert result is not None
        assert result.direction == "sell"

    @pytest.mark.unit
    def test_any_prefers_highest_confidence(self):
        """Test any mode takes highest confidence buy."""
        s1 = MockStrategy("s1")
        s2 = MockStrategy("s2")
        s1.set_signal(Signal("buy", SignalStrength.BUY, 0.5, "low", {}))
        s2.set_signal(Signal("buy", SignalStrength.BUY, 0.9, "high", {}))

        manager = PortfolioManager([s1, s2], mode="any")
        result = manager.update(100.0)

        assert result is not None
        assert result.confidence == 0.9


class TestUnknownMode:
    """Tests for unknown aggregation mode."""

    @pytest.mark.unit
    def test_unknown_mode_returns_none(self):
        """Test that unknown mode returns None."""
        s1 = MockStrategy("s1")
        s1.set_signal(Signal("buy", SignalStrength.BUY, 0.8, "reason", {}))

        manager = PortfolioManager([s1], mode="unknown_mode")
        result = manager.update(100.0)

        assert result is None


class TestPortfolioManagerStatus:
    """Tests for PortfolioManager status method."""

    @pytest.mark.unit
    def test_get_status(self):
        """Test get_status returns correct info."""
        s1 = MockStrategy("s1")
        s2 = MockStrategy("s2", is_ready_val=False)

        manager = PortfolioManager([s1, s2], mode="weighted")
        status = manager.get_status()

        assert status["total_strategies"] == 2
        assert status["ready_strategies"] == 1
        assert status["aggregation_mode"] == "weighted"
        assert len(status["strategy_status"]) == 2
        assert status["signal_count"] == 0


class TestPortfolioManagerReset:
    """Tests for PortfolioManager reset method."""

    @pytest.mark.unit
    def test_reset_clears_history(self):
        """Test reset clears signal history."""
        s1 = MockStrategy("s1")
        manager = PortfolioManager([s1])
        manager.signal_history = [MagicMock(), MagicMock()]

        manager.reset()

        assert manager.signal_history == []

    @pytest.mark.unit
    def test_reset_resets_strategies(self):
        """Test reset calls reset on all strategies."""
        s1 = MockStrategy("s1")
        s1.reset = MagicMock()

        manager = PortfolioManager([s1])
        manager.reset()

        s1.reset.assert_called_once()
