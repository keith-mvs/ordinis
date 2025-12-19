"""
Tests for Target Allocation Rebalancing Strategy.
"""

from datetime import datetime

import pytest

from ordinis.engines.portfolio.target_allocation import (
    RebalanceDecision,
    TargetAllocation,
    TargetAllocationRebalancer,
)


class TestTargetAllocation:
    """Tests for TargetAllocation dataclass."""

    def test_valid_allocation(self):
        """Test creating valid target allocation."""
        target = TargetAllocation("AAPL", 0.50)
        assert target.symbol == "AAPL"
        assert target.target_weight == 0.50

    def test_zero_weight(self):
        """Test zero weight is valid."""
        target = TargetAllocation("AAPL", 0.0)
        assert target.target_weight == 0.0

    def test_full_weight(self):
        """Test 1.0 weight is valid."""
        target = TargetAllocation("AAPL", 1.0)
        assert target.target_weight == 1.0

    def test_invalid_negative_weight(self):
        """Test negative weight raises error."""
        with pytest.raises(ValueError, match="target_weight must be in"):
            TargetAllocation("AAPL", -0.1)

    def test_invalid_weight_above_one(self):
        """Test weight > 1.0 raises error."""
        with pytest.raises(ValueError, match="target_weight must be in"):
            TargetAllocation("AAPL", 1.5)


class TestRebalanceDecision:
    """Tests for RebalanceDecision dataclass."""

    def test_create_decision(self):
        """Test creating rebalance decision."""
        timestamp = datetime.now()
        decision = RebalanceDecision(
            symbol="AAPL",
            current_weight=0.40,
            target_weight=0.50,
            adjustment_shares=10.0,
            adjustment_value=1500.0,
            timestamp=timestamp,
        )
        assert decision.symbol == "AAPL"
        assert decision.current_weight == 0.40
        assert decision.target_weight == 0.50
        assert decision.adjustment_shares == 10.0
        assert decision.adjustment_value == 1500.0
        assert decision.timestamp == timestamp


class TestTargetAllocationRebalancer:
    """Tests for TargetAllocationRebalancer class."""

    @pytest.fixture
    def three_stock_targets(self):
        """Fixture: 40/30/30 allocation across three stocks."""
        return [
            TargetAllocation("AAPL", 0.40),
            TargetAllocation("MSFT", 0.30),
            TargetAllocation("GOOGL", 0.30),
        ]

    @pytest.fixture
    def equal_weight_targets(self):
        """Fixture: Equal weight 25% allocation across four stocks."""
        return [
            TargetAllocation("AAPL", 0.25),
            TargetAllocation("MSFT", 0.25),
            TargetAllocation("GOOGL", 0.25),
            TargetAllocation("TSLA", 0.25),
        ]

    def test_initialization(self, three_stock_targets):
        """Test rebalancer initialization with valid targets."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets, drift_threshold=0.05)
        assert len(rebalancer.targets) == 3
        assert rebalancer.targets["AAPL"] == 0.40
        assert rebalancer.targets["MSFT"] == 0.30
        assert rebalancer.targets["GOOGL"] == 0.30
        assert rebalancer.drift_threshold == 0.05

    def test_invalid_weights_sum(self):
        """Test initialization fails when weights don't sum to 1.0."""
        invalid_targets = [
            TargetAllocation("AAPL", 0.40),
            TargetAllocation("MSFT", 0.40),  # Sum = 0.80, not 1.0
        ]
        with pytest.raises(ValueError, match="Target weights must sum to 1.0"):
            TargetAllocationRebalancer(invalid_targets)

    def test_calculate_drift_balanced_portfolio(self, three_stock_targets):
        """Test drift calculation when portfolio is perfectly balanced."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets)

        # Portfolio perfectly matches targets: 40/30/30
        positions = {
            "AAPL": 20,
            "MSFT": 10,
            "GOOGL": 15,
        }  # 20*200=4000, 10*300=3000, 15*200=3000 -> total=10000
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 200.0}

        drift = rebalancer.calculate_drift(positions, prices)

        assert drift["AAPL"] == pytest.approx(0.0, abs=1e-9)
        assert drift["MSFT"] == pytest.approx(0.0, abs=1e-9)
        assert drift["GOOGL"] == pytest.approx(0.0, abs=1e-9)

    def test_calculate_drift_imbalanced_portfolio(self, three_stock_targets):
        """Test drift calculation when portfolio is out of balance."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets)

        # AAPL overweight (50% instead of 40%), MSFT/GOOGL underweight
        positions = {
            "AAPL": 25,
            "MSFT": 8,
            "GOOGL": 12,
        }  # 25*200=5000, 8*300=2400, 12*200=2400 -> total=9800
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 200.0}

        drift = rebalancer.calculate_drift(positions, prices)

        # AAPL: 5000/9800 = 0.5102 vs target 0.40 -> drift +0.1102
        # MSFT: 2400/9800 = 0.2449 vs target 0.30 -> drift -0.0551
        # GOOGL: 2400/9800 = 0.2449 vs target 0.30 -> drift -0.0551
        assert drift["AAPL"] == pytest.approx(0.1102, abs=1e-3)
        assert drift["MSFT"] == pytest.approx(-0.0551, abs=1e-3)
        assert drift["GOOGL"] == pytest.approx(-0.0551, abs=1e-3)

    def test_calculate_drift_empty_portfolio(self, three_stock_targets):
        """Test drift calculation with empty portfolio."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets)

        positions = {"AAPL": 0, "MSFT": 0, "GOOGL": 0}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 200.0}

        drift = rebalancer.calculate_drift(positions, prices)

        # Empty portfolio means all symbols need full allocation (negative drift = need to buy)
        assert drift["AAPL"] == -0.40
        assert drift["MSFT"] == -0.30
        assert drift["GOOGL"] == -0.30

    def test_calculate_drift_missing_positions(self, three_stock_targets):
        """Test drift calculation when some positions are missing."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets)

        # Only hold AAPL, missing MSFT and GOOGL
        positions = {"AAPL": 50}  # 50*200=10000
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 200.0}

        drift = rebalancer.calculate_drift(positions, prices)

        # AAPL: 10000/10000 = 1.0 vs target 0.40 -> drift +0.60
        # MSFT: 0/10000 = 0.0 vs target 0.30 -> drift -0.30
        # GOOGL: 0/10000 = 0.0 vs target 0.30 -> drift -0.30
        assert drift["AAPL"] == pytest.approx(0.60, abs=1e-9)
        assert drift["MSFT"] == pytest.approx(-0.30, abs=1e-9)
        assert drift["GOOGL"] == pytest.approx(-0.30, abs=1e-9)

    def test_should_rebalance_within_threshold(self, three_stock_targets):
        """Test should_rebalance returns False when drift is within threshold."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets, drift_threshold=0.05)

        # Small drift: AAPL 41% instead of 40% (drift = 0.01 < 0.05 threshold)
        positions = {
            "AAPL": 20.5,
            "MSFT": 10,
            "GOOGL": 14.75,
        }  # 20.5*200=4100, 10*300=3000, 14.75*200=2950 -> total=10050
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 200.0}

        assert not rebalancer.should_rebalance(positions, prices)

    def test_should_rebalance_exceeds_threshold(self, three_stock_targets):
        """Test should_rebalance returns True when drift exceeds threshold."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets, drift_threshold=0.05)

        # Large drift: AAPL 50% instead of 40% (drift = 0.10 > 0.05 threshold)
        positions = {"AAPL": 25, "MSFT": 8, "GOOGL": 12}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 200.0}

        assert rebalancer.should_rebalance(positions, prices)

    def test_generate_rebalance_orders_balanced(self, three_stock_targets):
        """Test rebalance orders when portfolio is already balanced."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets)

        positions = {"AAPL": 20, "MSFT": 10, "GOOGL": 15}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 200.0}

        decisions = rebalancer.generate_rebalance_orders(positions, prices)

        assert len(decisions) == 3
        for decision in decisions:
            assert decision.adjustment_shares == pytest.approx(0.0, abs=1e-6)
            assert decision.adjustment_value == pytest.approx(0.0, abs=1e-6)

    def test_generate_rebalance_orders_imbalanced(self, three_stock_targets):
        """Test rebalance orders when portfolio needs rebalancing."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets)

        # AAPL overweight, MSFT/GOOGL underweight
        positions = {"AAPL": 25, "MSFT": 8, "GOOGL": 12}  # Total = 9800
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 200.0}

        decisions = rebalancer.generate_rebalance_orders(positions, prices)

        assert len(decisions) == 3

        # Find decisions by symbol
        decisions_map = {d.symbol: d for d in decisions}

        # AAPL: current 5000/9800=51.02%, target 40% -> need to SELL
        # Target value: 9800 * 0.40 = 3920, current = 5000, adjustment = -1080
        aapl = decisions_map["AAPL"]
        assert aapl.current_weight == pytest.approx(0.5102, abs=1e-3)
        assert aapl.target_weight == 0.40
        assert aapl.adjustment_value == pytest.approx(-1080.0, abs=1.0)
        assert aapl.adjustment_shares == pytest.approx(-5.4, abs=0.01)  # -1080 / 200

        # MSFT: current 2400/9800=24.49%, target 30% -> need to BUY
        # Target value: 9800 * 0.30 = 2940, current = 2400, adjustment = +540
        msft = decisions_map["MSFT"]
        assert msft.current_weight == pytest.approx(0.2449, abs=1e-3)
        assert msft.target_weight == 0.30
        assert msft.adjustment_value == pytest.approx(540.0, abs=1.0)
        assert msft.adjustment_shares == pytest.approx(1.8, abs=0.01)  # 540 / 300

        # GOOGL: current 2400/9800=24.49%, target 30% -> need to BUY
        googl = decisions_map["GOOGL"]
        assert googl.current_weight == pytest.approx(0.2449, abs=1e-3)
        assert googl.target_weight == 0.30
        assert googl.adjustment_value == pytest.approx(540.0, abs=1.0)
        assert googl.adjustment_shares == pytest.approx(2.7, abs=0.01)  # 540 / 200

    def test_generate_rebalance_orders_with_cash(self, three_stock_targets):
        """Test rebalance orders when cash is available."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets)

        # Portfolio worth 6000, adding 4000 cash -> total 10000
        positions = {
            "AAPL": 10,
            "MSFT": 5,
            "GOOGL": 5,
        }  # 10*200=2000, 5*300=1500, 5*500=2500 -> 6000
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}
        cash = 4000.0

        decisions = rebalancer.generate_rebalance_orders(positions, prices, cash=cash)

        decisions_map = {d.symbol: d for d in decisions}

        # Total portfolio = 10000 (6000 equity + 4000 cash)
        # AAPL: target 40% = 4000, current 2000 -> buy 2000 worth = 10 shares
        aapl = decisions_map["AAPL"]
        assert aapl.adjustment_value == pytest.approx(2000.0, abs=1.0)
        assert aapl.adjustment_shares == pytest.approx(10.0, abs=0.01)

        # MSFT: target 30% = 3000, current 1500 -> buy 1500 worth = 5 shares
        msft = decisions_map["MSFT"]
        assert msft.adjustment_value == pytest.approx(1500.0, abs=1.0)
        assert msft.adjustment_shares == pytest.approx(5.0, abs=0.01)

        # GOOGL: target 30% = 3000, current 2500 -> buy 500 worth = 1 share
        googl = decisions_map["GOOGL"]
        assert googl.adjustment_value == pytest.approx(500.0, abs=1.0)
        assert googl.adjustment_shares == pytest.approx(1.0, abs=0.01)

    def test_generate_rebalance_orders_empty_portfolio_no_cash(self, three_stock_targets):
        """Test rebalance orders with empty portfolio and no cash."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets)

        positions = {"AAPL": 0, "MSFT": 0, "GOOGL": 0}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 200.0}

        decisions = rebalancer.generate_rebalance_orders(positions, prices, cash=0.0)

        # Cannot rebalance empty portfolio with no cash
        assert len(decisions) == 0

    def test_generate_rebalance_orders_empty_portfolio_with_cash(self, three_stock_targets):
        """Test rebalance orders with empty portfolio but available cash."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets)

        positions = {"AAPL": 0, "MSFT": 0, "GOOGL": 0}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 200.0}
        cash = 10000.0

        decisions = rebalancer.generate_rebalance_orders(positions, prices, cash=cash)

        decisions_map = {d.symbol: d for d in decisions}

        # AAPL: 40% of 10000 = 4000 -> 20 shares at $200
        aapl = decisions_map["AAPL"]
        assert aapl.adjustment_value == pytest.approx(4000.0, abs=1.0)
        assert aapl.adjustment_shares == pytest.approx(20.0, abs=0.01)

        # MSFT: 30% of 10000 = 3000 -> 10 shares at $300
        msft = decisions_map["MSFT"]
        assert msft.adjustment_value == pytest.approx(3000.0, abs=1.0)
        assert msft.adjustment_shares == pytest.approx(10.0, abs=0.01)

        # GOOGL: 30% of 10000 = 3000 -> 15 shares at $200
        googl = decisions_map["GOOGL"]
        assert googl.adjustment_value == pytest.approx(3000.0, abs=1.0)
        assert googl.adjustment_shares == pytest.approx(15.0, abs=0.01)

    def test_get_rebalance_summary(self, three_stock_targets):
        """Test conversion of decisions to DataFrame summary."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets)

        positions = {"AAPL": 25, "MSFT": 8, "GOOGL": 12}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 200.0}

        decisions = rebalancer.generate_rebalance_orders(positions, prices)
        df = rebalancer.get_rebalance_summary(decisions)

        assert len(df) == 3
        assert list(df.columns) == [
            "symbol",
            "current_weight",
            "target_weight",
            "drift",
            "adjustment_shares",
            "adjustment_value",
        ]

        # Check AAPL row
        aapl_row = df[df["symbol"] == "AAPL"].iloc[0]
        assert aapl_row["current_weight"] == pytest.approx(0.5102, abs=1e-3)
        assert aapl_row["target_weight"] == 0.40
        assert aapl_row["drift"] == pytest.approx(0.1102, abs=1e-3)
        assert aapl_row["adjustment_shares"] < 0  # Should sell

    def test_equal_weight_rebalancing(self, equal_weight_targets):
        """Test rebalancing with equal weight targets."""
        rebalancer = TargetAllocationRebalancer(equal_weight_targets)

        # One stock dominates, others underweight
        positions = {
            "AAPL": 40,
            "MSFT": 5,
            "GOOGL": 5,
            "TSLA": 5,
        }  # AAPL=8000, others=1500 each -> total=12500
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0, "TSLA": 300.0}

        decisions = rebalancer.generate_rebalance_orders(positions, prices)
        decisions_map = {d.symbol: d for d in decisions}

        # Each should be 25% of 12500 = 3125
        # AAPL: 8000 -> 3125 (sell 4875 = 24.375 shares)
        assert decisions_map["AAPL"].adjustment_value == pytest.approx(-4875.0, abs=1.0)

        # Others: 1500 -> 3125 (buy 1625 each)
        for sym in ["MSFT", "GOOGL", "TSLA"]:
            assert decisions_map[sym].adjustment_value == pytest.approx(1625.0, abs=1.0)

    def test_timestamp_consistency(self, three_stock_targets):
        """Test that all decisions have the same timestamp."""
        rebalancer = TargetAllocationRebalancer(three_stock_targets)

        positions = {"AAPL": 25, "MSFT": 8, "GOOGL": 12}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 200.0}

        decisions = rebalancer.generate_rebalance_orders(positions, prices)

        # All decisions should have same timestamp (generated in same call)
        timestamps = [d.timestamp for d in decisions]
        assert all(t == timestamps[0] for t in timestamps)
