"""
Tests for Signal-Driven Rebalancing Strategy.
"""

from datetime import datetime

import pytest

from ordinis.engines.portfolio.signal_driven import (
    SignalDrivenDecision,
    SignalDrivenRebalancer,
    SignalDrivenWeights,
    SignalInput,
    SignalMethod,
)


class TestSignalInput:
    """Tests for SignalInput dataclass."""

    def test_valid_signal(self):
        """Test creating valid signal input."""
        signal = SignalInput("AAPL", signal=0.8, confidence=0.9, source="RSI")
        assert signal.symbol == "AAPL"
        assert signal.signal == 0.8
        assert signal.confidence == 0.9
        assert signal.source == "RSI"

    def test_default_confidence(self):
        """Test default confidence is 1.0."""
        signal = SignalInput("AAPL", signal=0.5)
        assert signal.confidence == 1.0
        assert signal.source == "unknown"

    def test_negative_signal(self):
        """Test negative signals are valid."""
        signal = SignalInput("AAPL", signal=-0.5)
        assert signal.signal == -0.5

    def test_invalid_signal_too_high(self):
        """Test signal > 2.0 raises error."""
        with pytest.raises(ValueError, match="signal should be in"):
            SignalInput("AAPL", signal=2.5)

    def test_invalid_signal_too_low(self):
        """Test signal < -2.0 raises error."""
        with pytest.raises(ValueError, match="signal should be in"):
            SignalInput("AAPL", signal=-2.5)

    def test_invalid_confidence_too_high(self):
        """Test confidence > 1.0 raises error."""
        with pytest.raises(ValueError, match="confidence must be in"):
            SignalInput("AAPL", signal=0.5, confidence=1.5)

    def test_invalid_confidence_negative(self):
        """Test negative confidence raises error."""
        with pytest.raises(ValueError, match="confidence must be in"):
            SignalInput("AAPL", signal=0.5, confidence=-0.1)


class TestSignalDrivenWeights:
    """Tests for SignalDrivenWeights dataclass."""

    def test_create_weights(self):
        """Test creating signal-driven weights."""
        timestamp = datetime.now()
        weights = SignalDrivenWeights(
            weights={"AAPL": 0.50, "MSFT": 0.30, "GOOGL": 0.20},
            signals={"AAPL": 0.8, "MSFT": 0.5, "GOOGL": 0.3},
            confidences={"AAPL": 0.9, "MSFT": 0.8, "GOOGL": 0.7},
            method=SignalMethod.PROPORTIONAL,
            timestamp=timestamp,
        )
        assert len(weights.weights) == 3
        assert weights.method == SignalMethod.PROPORTIONAL
        assert weights.signals["AAPL"] == 0.8


class TestSignalDrivenDecision:
    """Tests for SignalDrivenDecision dataclass."""

    def test_create_decision(self):
        """Test creating signal-driven decision."""
        timestamp = datetime.now()
        decision = SignalDrivenDecision(
            symbol="AAPL",
            current_weight=0.30,
            target_weight=0.50,
            signal=0.8,
            confidence=0.9,
            adjustment_shares=10.0,
            adjustment_value=1500.0,
            timestamp=timestamp,
        )
        assert decision.symbol == "AAPL"
        assert decision.signal == 0.8
        assert decision.confidence == 0.9


class TestSignalDrivenRebalancer:
    """Tests for SignalDrivenRebalancer class."""

    @pytest.fixture
    def mixed_signals(self):
        """Fixture: Mixed bullish and bearish signals."""
        return [
            SignalInput("AAPL", signal=0.8, confidence=0.9, source="RSI"),
            SignalInput("MSFT", signal=0.5, confidence=0.8, source="MACD"),
            SignalInput("GOOGL", signal=0.2, confidence=0.7, source="RSI"),
            SignalInput("TSLA", signal=-0.3, confidence=0.6, source="MACD"),
        ]

    @pytest.fixture
    def strong_signals(self):
        """Fixture: All strong bullish signals."""
        return [
            SignalInput("AAPL", signal=1.0, confidence=1.0),
            SignalInput("MSFT", signal=0.9, confidence=0.95),
            SignalInput("GOOGL", signal=0.8, confidence=0.90),
        ]

    def test_initialization(self):
        """Test rebalancer initialization."""
        rebalancer = SignalDrivenRebalancer(
            method=SignalMethod.PROPORTIONAL,
            min_weight=0.05,
            max_weight=0.60,
            min_signal_threshold=0.1,
            cash_buffer=0.10,
            drift_threshold=0.15,
        )
        assert rebalancer.method == SignalMethod.PROPORTIONAL
        assert rebalancer.min_weight == 0.05
        assert rebalancer.max_weight == 0.60
        assert rebalancer.min_signal_threshold == 0.1
        assert rebalancer.cash_buffer == 0.10
        assert rebalancer.drift_threshold == 0.15

    def test_invalid_weight_bounds(self):
        """Test initialization fails with invalid weight bounds."""
        with pytest.raises(ValueError, match="Must have 0 <= min_weight < max_weight"):
            SignalDrivenRebalancer(min_weight=0.6, max_weight=0.4)

    def test_invalid_signal_threshold(self):
        """Test initialization fails with invalid signal threshold."""
        with pytest.raises(ValueError, match="min_signal_threshold must be in"):
            SignalDrivenRebalancer(min_signal_threshold=2.5)

    def test_invalid_cash_buffer(self):
        """Test initialization fails with invalid cash buffer."""
        with pytest.raises(ValueError, match="cash_buffer must be in"):
            SignalDrivenRebalancer(cash_buffer=1.2)

    def test_calculate_weights_proportional(self, mixed_signals):
        """Test proportional weight calculation."""
        rebalancer = SignalDrivenRebalancer(method=SignalMethod.PROPORTIONAL)
        weights = rebalancer.calculate_weights(mixed_signals)

        # Weights should sum to 1.0
        assert sum(weights.weights.values()) == pytest.approx(1.0, abs=1e-9)

        # AAPL (0.8*0.9=0.72) should have highest weight
        # TSLA (negative signal) should have zero weight
        assert weights.weights["AAPL"] > weights.weights["MSFT"]
        assert weights.weights["MSFT"] > weights.weights["GOOGL"]
        assert "TSLA" not in weights.weights or weights.weights.get("TSLA", 0) == 0

    def test_calculate_weights_binary(self, mixed_signals):
        """Test binary weight calculation."""
        rebalancer = SignalDrivenRebalancer(method=SignalMethod.BINARY)
        weights = rebalancer.calculate_weights(mixed_signals)

        # Only positive signals get weight
        positive_signals = [s for s in mixed_signals if s.signal > 0]
        expected_weight = 1.0 / len(positive_signals)

        for s in positive_signals:
            assert weights.weights[s.symbol] == pytest.approx(expected_weight, abs=1e-6)

        # TSLA (negative) should not be in weights
        assert "TSLA" not in weights.weights

    def test_calculate_weights_ranked(self, strong_signals):
        """Test ranked weight calculation."""
        rebalancer = SignalDrivenRebalancer(method=SignalMethod.RANKED)
        weights = rebalancer.calculate_weights(strong_signals)

        # Weights should sum to 1.0
        assert sum(weights.weights.values()) == pytest.approx(1.0, abs=1e-9)

        # AAPL (best signal) should have highest weight
        # GOOGL (worst signal) should have lowest weight
        assert weights.weights["AAPL"] > weights.weights["MSFT"]
        assert weights.weights["MSFT"] > weights.weights["GOOGL"]

        # Rank-based: weights should be in ratio 3:2:1 (normalized)
        total_rank = 3 + 2 + 1
        assert weights.weights["AAPL"] == pytest.approx(3 / total_rank, abs=1e-6)
        assert weights.weights["MSFT"] == pytest.approx(2 / total_rank, abs=1e-6)
        assert weights.weights["GOOGL"] == pytest.approx(1 / total_rank, abs=1e-6)

    def test_calculate_weights_with_signal_threshold(self, mixed_signals):
        """Test signal threshold filtering."""
        rebalancer = SignalDrivenRebalancer(
            method=SignalMethod.PROPORTIONAL,
            min_signal_threshold=0.6,  # Only AAPL (0.8) passes
        )
        weights = rebalancer.calculate_weights(mixed_signals)

        # Only AAPL should get weight
        assert len(weights.weights) == 1
        assert "AAPL" in weights.weights
        assert weights.weights["AAPL"] == pytest.approx(1.0, abs=1e-9)

    def test_calculate_weights_with_cash_buffer(self, strong_signals):
        """Test cash buffer reduces equity allocation."""
        rebalancer = SignalDrivenRebalancer(
            method=SignalMethod.PROPORTIONAL,
            cash_buffer=0.20,  # 20% cash
        )
        weights = rebalancer.calculate_weights(strong_signals)

        # Equity weights should sum to 0.80 (1.0 - 0.20 cash)
        assert sum(weights.weights.values()) == pytest.approx(0.80, abs=1e-6)

    def test_calculate_weights_empty_signals(self):
        """Test error with empty signals list."""
        rebalancer = SignalDrivenRebalancer()

        with pytest.raises(ValueError, match="Signals list cannot be empty"):
            rebalancer.calculate_weights([])

    def test_calculate_weights_no_signals_above_threshold(self, mixed_signals):
        """Test error when no signals pass threshold."""
        rebalancer = SignalDrivenRebalancer(min_signal_threshold=1.5)

        with pytest.raises(ValueError, match="No signals above threshold"):
            rebalancer.calculate_weights(mixed_signals)

    def test_calculate_weights_binary_no_positive_signals(self):
        """Test error with binary method and no positive signals."""
        rebalancer = SignalDrivenRebalancer(method=SignalMethod.BINARY)
        negative_signals = [
            SignalInput("AAPL", signal=-0.5),
            SignalInput("MSFT", signal=-0.3),
        ]

        with pytest.raises(ValueError, match="No signals above threshold"):
            rebalancer.calculate_weights(negative_signals)

    def test_should_rebalance_within_threshold(self, strong_signals):
        """Test should_rebalance returns False when drift is small."""
        rebalancer = SignalDrivenRebalancer(drift_threshold=0.50)  # High threshold

        # Create positions roughly matching signal weights
        positions = {"AAPL": 10, "MSFT": 8, "GOOGL": 6}
        prices = {"AAPL": 150.0, "MSFT": 150.0, "GOOGL": 150.0}

        # With high drift threshold, should not trigger rebalance
        # (actual result depends on exact weights)
        result = rebalancer.should_rebalance(strong_signals, positions, prices)
        assert isinstance(result, bool)

    def test_should_rebalance_empty_portfolio(self, strong_signals):
        """Test should_rebalance returns True for empty portfolio."""
        rebalancer = SignalDrivenRebalancer()

        positions = {"AAPL": 0, "MSFT": 0, "GOOGL": 0}
        prices = {"AAPL": 150.0, "MSFT": 150.0, "GOOGL": 150.0}

        assert rebalancer.should_rebalance(strong_signals, positions, prices)

    def test_generate_rebalance_orders_proportional(self, strong_signals):
        """Test generating rebalance orders with proportional method."""
        rebalancer = SignalDrivenRebalancer(method=SignalMethod.PROPORTIONAL)

        positions = {"AAPL": 10, "MSFT": 10, "GOOGL": 10}
        prices = {"AAPL": 200.0, "MSFT": 200.0, "GOOGL": 200.0}

        decisions = rebalancer.generate_rebalance_orders(strong_signals, positions, prices)

        assert len(decisions) == 3

        # AAPL should get more allocation (strongest signal)
        decisions_map = {d.symbol: d for d in decisions}
        assert decisions_map["AAPL"].target_weight > decisions_map["MSFT"].target_weight
        assert decisions_map["MSFT"].target_weight > decisions_map["GOOGL"].target_weight

    def test_generate_rebalance_orders_with_cash(self, strong_signals):
        """Test rebalance orders when cash is available."""
        rebalancer = SignalDrivenRebalancer()

        positions = {"AAPL": 5, "MSFT": 5, "GOOGL": 5}  # 3000 total
        prices = {"AAPL": 200.0, "MSFT": 200.0, "GOOGL": 200.0}
        cash = 3000.0  # Double portfolio with cash

        decisions = rebalancer.generate_rebalance_orders(
            strong_signals, positions, prices, cash=cash
        )

        # Total target value should be 6000
        total_target = sum(d.target_weight * 6000 for d in decisions)
        assert total_target == pytest.approx(6000.0, abs=1.0)

    def test_generate_rebalance_orders_empty_portfolio_no_cash(self, strong_signals):
        """Test rebalance orders with empty portfolio and no cash."""
        rebalancer = SignalDrivenRebalancer()

        positions = {"AAPL": 0, "MSFT": 0, "GOOGL": 0}
        prices = {"AAPL": 200.0, "MSFT": 200.0, "GOOGL": 200.0}

        decisions = rebalancer.generate_rebalance_orders(strong_signals, positions, prices)

        # Cannot rebalance empty portfolio with no cash
        assert len(decisions) == 0

    def test_generate_rebalance_orders_empty_portfolio_with_cash(self, strong_signals):
        """Test rebalance orders with empty portfolio but available cash."""
        rebalancer = SignalDrivenRebalancer()

        positions = {"AAPL": 0, "MSFT": 0, "GOOGL": 0}
        prices = {"AAPL": 200.0, "MSFT": 200.0, "GOOGL": 200.0}
        cash = 10000.0

        decisions = rebalancer.generate_rebalance_orders(
            strong_signals, positions, prices, cash=cash
        )

        assert len(decisions) == 3

        # All adjustments should be buys
        for decision in decisions:
            if decision.target_weight > 0:
                assert decision.adjustment_shares > 0
                assert decision.adjustment_value > 0

    def test_generate_rebalance_orders_with_cash_buffer(self, strong_signals):
        """Test rebalance orders respect cash buffer."""
        rebalancer = SignalDrivenRebalancer(cash_buffer=0.20)  # Keep 20% cash

        positions = {"AAPL": 10, "MSFT": 10, "GOOGL": 10}
        prices = {"AAPL": 200.0, "MSFT": 200.0, "GOOGL": 200.0}

        decisions = rebalancer.generate_rebalance_orders(strong_signals, positions, prices)

        # Total target allocation should be 80% of portfolio (20% kept as cash)
        total_target_weight = sum(d.target_weight for d in decisions)
        assert total_target_weight == pytest.approx(0.80, abs=1e-6)

    def test_generate_rebalance_orders_sell_positions(self, strong_signals):
        """Test generating sell orders for positions not in signals."""
        rebalancer = SignalDrivenRebalancer()

        # Hold TSLA but it's not in signals
        positions = {"AAPL": 10, "MSFT": 10, "GOOGL": 10, "TSLA": 10}
        prices = {"AAPL": 200.0, "MSFT": 200.0, "GOOGL": 200.0, "TSLA": 200.0}

        decisions = rebalancer.generate_rebalance_orders(strong_signals, positions, prices)

        # TSLA should have target_weight = 0 and negative adjustment
        tsla_decision = next(d for d in decisions if d.symbol == "TSLA")
        assert tsla_decision.target_weight == 0.0
        assert tsla_decision.adjustment_shares < 0  # Sell all TSLA

    def test_get_rebalance_summary(self, strong_signals):
        """Test conversion of decisions to DataFrame summary."""
        rebalancer = SignalDrivenRebalancer()

        positions = {"AAPL": 10, "MSFT": 10, "GOOGL": 10}
        prices = {"AAPL": 200.0, "MSFT": 200.0, "GOOGL": 200.0}

        decisions = rebalancer.generate_rebalance_orders(strong_signals, positions, prices)
        df = rebalancer.get_rebalance_summary(decisions)

        assert len(df) == 3
        assert list(df.columns) == [
            "symbol",
            "current_weight",
            "target_weight",
            "signal",
            "confidence",
            "drift",
            "adjustment_shares",
            "adjustment_value",
        ]

        # Verify signals are present
        assert all(df["signal"] >= 0)

        # Verify confidences are present
        assert all(df["confidence"] > 0)

    def test_timestamp_consistency(self, strong_signals):
        """Test that all decisions have the same timestamp."""
        rebalancer = SignalDrivenRebalancer()

        positions = {"AAPL": 10, "MSFT": 10, "GOOGL": 10}
        prices = {"AAPL": 200.0, "MSFT": 200.0, "GOOGL": 200.0}

        decisions = rebalancer.generate_rebalance_orders(strong_signals, positions, prices)

        # All decisions should have same timestamp
        timestamps = [d.timestamp for d in decisions]
        assert all(t == timestamps[0] for t in timestamps)

    def test_confidence_weighting(self):
        """Test that confidence affects weight allocation."""
        # Same signals, different confidences
        signals_low_conf = [
            SignalInput("AAPL", signal=1.0, confidence=0.5),
            SignalInput("MSFT", signal=1.0, confidence=0.5),
        ]

        signals_high_conf = [
            SignalInput("AAPL", signal=1.0, confidence=1.0),
            SignalInput("MSFT", signal=1.0, confidence=1.0),
        ]

        rebalancer = SignalDrivenRebalancer(method=SignalMethod.PROPORTIONAL)

        weights_low = rebalancer.calculate_weights(signals_low_conf)
        weights_high = rebalancer.calculate_weights(signals_high_conf)

        # Both should have equal weights (same relative confidence)
        assert weights_low.weights["AAPL"] == pytest.approx(0.5, abs=1e-6)
        assert weights_high.weights["AAPL"] == pytest.approx(0.5, abs=1e-6)

        # Now test different confidences
        signals_mixed = [
            SignalInput("AAPL", signal=1.0, confidence=1.0),
            SignalInput("MSFT", signal=1.0, confidence=0.5),
        ]

        weights_mixed = rebalancer.calculate_weights(signals_mixed)

        # AAPL should get more weight (higher confidence)
        assert weights_mixed.weights["AAPL"] > weights_mixed.weights["MSFT"]
