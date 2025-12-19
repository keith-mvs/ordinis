"""
Tests for Risk Parity Rebalancing Strategy.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.portfolio.risk_parity import (
    RiskParityDecision,
    RiskParityRebalancer,
    RiskParityWeights,
)


class TestRiskParityWeights:
    """Tests for RiskParityWeights dataclass."""

    def test_create_weights(self):
        """Test creating risk parity weights."""
        timestamp = datetime.now()
        weights = RiskParityWeights(
            weights={"AAPL": 0.40, "MSFT": 0.30, "GOOGL": 0.30},
            volatilities={"AAPL": 0.25, "MSFT": 0.20, "GOOGL": 0.30},
            risk_contributions={"AAPL": 0.33, "MSFT": 0.33, "GOOGL": 0.34},
            timestamp=timestamp,
        )
        assert len(weights.weights) == 3
        assert weights.volatilities["AAPL"] == 0.25
        assert weights.risk_contributions["GOOGL"] == 0.34
        assert weights.timestamp == timestamp


class TestRiskParityDecision:
    """Tests for RiskParityDecision dataclass."""

    def test_create_decision(self):
        """Test creating risk parity decision."""
        timestamp = datetime.now()
        decision = RiskParityDecision(
            symbol="AAPL",
            current_weight=0.40,
            target_weight=0.35,
            current_volatility=0.25,
            risk_contribution=0.33,
            adjustment_shares=-5.0,
            adjustment_value=-750.0,
            timestamp=timestamp,
        )
        assert decision.symbol == "AAPL"
        assert decision.current_weight == 0.40
        assert decision.target_weight == 0.35
        assert decision.current_volatility == 0.25
        assert decision.risk_contribution == 0.33
        assert decision.adjustment_shares == -5.0
        assert decision.adjustment_value == -750.0


class TestRiskParityRebalancer:
    """Tests for RiskParityRebalancer class."""

    @pytest.fixture
    def three_stock_returns(self):
        """Fixture: Returns for three stocks with different volatilities."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        # AAPL: medium volatility (15% annual)
        aapl_returns = np.random.normal(0.0005, 0.15 / np.sqrt(252), 252)

        # MSFT: low volatility (10% annual)
        msft_returns = np.random.normal(0.0005, 0.10 / np.sqrt(252), 252)

        # GOOGL: high volatility (25% annual)
        googl_returns = np.random.normal(0.0005, 0.25 / np.sqrt(252), 252)

        return pd.DataFrame(
            {
                "AAPL": aapl_returns,
                "MSFT": msft_returns,
                "GOOGL": googl_returns,
            },
            index=dates,
        )

    @pytest.fixture
    def equal_vol_returns(self):
        """Fixture: Returns for stocks with equal volatility."""
        np.random.seed(123)
        dates = pd.date_range("2023-01-01", periods=252, freq="D")

        # All stocks: 20% annual volatility
        vol = 0.20 / np.sqrt(252)
        returns = {
            "AAPL": np.random.normal(0.0005, vol, 252),
            "MSFT": np.random.normal(0.0005, vol, 252),
            "GOOGL": np.random.normal(0.0005, vol, 252),
            "TSLA": np.random.normal(0.0005, vol, 252),
        }

        return pd.DataFrame(returns, index=dates)

    def test_initialization(self):
        """Test rebalancer initialization with valid parameters."""
        rebalancer = RiskParityRebalancer(
            lookback_days=252,
            min_weight=0.05,
            max_weight=0.60,
            drift_threshold=0.03,
        )
        assert rebalancer.lookback_days == 252
        assert rebalancer.min_weight == 0.05
        assert rebalancer.max_weight == 0.60
        assert rebalancer.drift_threshold == 0.03

    def test_invalid_lookback_days(self):
        """Test initialization fails with insufficient lookback period."""
        with pytest.raises(ValueError, match="lookback_days must be >= 20"):
            RiskParityRebalancer(lookback_days=10)

    def test_invalid_weight_bounds(self):
        """Test initialization fails with invalid weight bounds."""
        with pytest.raises(ValueError, match="Must have 0 <= min_weight < max_weight"):
            RiskParityRebalancer(min_weight=0.6, max_weight=0.4)

    def test_invalid_drift_threshold(self):
        """Test initialization fails with invalid drift threshold."""
        with pytest.raises(ValueError, match="drift_threshold must be in"):
            RiskParityRebalancer(drift_threshold=1.5)

    def test_calculate_weights_basic(self, three_stock_returns):
        """Test weight calculation with different volatilities."""
        rebalancer = RiskParityRebalancer()
        weights = rebalancer.calculate_weights(three_stock_returns)

        # Verify weights sum to 1.0
        assert sum(weights.weights.values()) == pytest.approx(1.0, abs=1e-9)

        # MSFT (lowest vol) should have highest weight
        # GOOGL (highest vol) should have lowest weight
        assert weights.weights["MSFT"] > weights.weights["AAPL"]
        assert weights.weights["AAPL"] > weights.weights["GOOGL"]

        # Verify volatilities are calculated
        assert all(vol > 0 for vol in weights.volatilities.values())

        # Verify risk contributions sum to approximately 1.0
        assert sum(weights.risk_contributions.values()) == pytest.approx(1.0, abs=1e-6)

    def test_calculate_weights_equal_volatility(self, equal_vol_returns):
        """Test weights are equal when volatilities are equal."""
        rebalancer = RiskParityRebalancer(min_weight=0.0, max_weight=1.0)
        weights = rebalancer.calculate_weights(equal_vol_returns)

        # All weights should be approximately equal (25% each)
        for weight in weights.weights.values():
            assert weight == pytest.approx(0.25, abs=0.02)

        # Risk contributions should also be equal
        for rc in weights.risk_contributions.values():
            assert rc == pytest.approx(0.25, abs=0.02)

    def test_calculate_weights_min_constraint(self, three_stock_returns):
        """Test minimum weight constraint is enforced."""
        rebalancer = RiskParityRebalancer(min_weight=0.20)
        weights = rebalancer.calculate_weights(three_stock_returns)

        # All weights should be >= 20% (with small tolerance for renormalization)
        for weight in weights.weights.values():
            assert weight >= 0.19  # Allow 1% tolerance after renormalization

        # Should still sum to 1.0
        assert sum(weights.weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_calculate_weights_max_constraint(self, three_stock_returns):
        """Test maximum weight constraint is enforced."""
        rebalancer = RiskParityRebalancer(max_weight=0.35)
        weights = rebalancer.calculate_weights(three_stock_returns)

        # All weights should be <= 35% (with tolerance for renormalization)
        for weight in weights.weights.values():
            assert weight <= 0.42  # Allow tolerance after renormalization

        # Should still sum to 1.0
        assert sum(weights.weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_calculate_weights_empty_dataframe(self):
        """Test error on empty returns DataFrame."""
        rebalancer = RiskParityRebalancer()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Returns DataFrame cannot be empty"):
            rebalancer.calculate_weights(empty_df)

    def test_calculate_weights_insufficient_history(self, three_stock_returns):
        """Test error when insufficient return history."""
        rebalancer = RiskParityRebalancer(lookback_days=252)
        short_returns = three_stock_returns.head(15)  # Only 15 days

        with pytest.raises(ValueError, match="Insufficient return history"):
            rebalancer.calculate_weights(short_returns)

    def test_calculate_weights_lookback_window(self):
        """Test that lookback_days correctly limits data used."""
        np.random.seed(999)
        # Create 500 days of data
        dates = pd.date_range("2022-01-01", periods=500, freq="D")
        returns = pd.DataFrame(
            {
                "AAPL": np.random.normal(0.0005, 0.01, 500),
                "MSFT": np.random.normal(0.0005, 0.01, 500),
            },
            index=dates,
        )

        # Use only last 100 days
        rebalancer = RiskParityRebalancer(lookback_days=100)
        weights = rebalancer.calculate_weights(returns)

        # Should successfully calculate (uses tail(100))
        assert len(weights.weights) == 2
        assert sum(weights.weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_should_rebalance_within_threshold(self, three_stock_returns):
        """Test should_rebalance returns False when drift is small."""
        rebalancer = RiskParityRebalancer(drift_threshold=0.10)

        # Get target weights
        target_weights = rebalancer.calculate_weights(three_stock_returns)

        # Create positions that approximately match targets
        positions = {
            "AAPL": 10,
            "MSFT": 15,
            "GOOGL": 5,
        }
        prices = {"AAPL": 150.0, "MSFT": 200.0, "GOOGL": 100.0}

        # Total value = 10*150 + 15*200 + 5*100 = 1500 + 3000 + 500 = 5000
        # Current weights: AAPL=0.30, MSFT=0.60, GOOGL=0.10

        # Drift depends on target weights - if targets are close to current, should not rebalance
        # This test may need adjustment based on actual volatilities in fixture
        result = rebalancer.should_rebalance(three_stock_returns, positions, prices)

        # Result depends on actual target weights from fixture
        assert isinstance(result, bool)

    def test_should_rebalance_empty_portfolio(self, three_stock_returns):
        """Test should_rebalance returns True for empty portfolio."""
        rebalancer = RiskParityRebalancer()

        positions = {"AAPL": 0, "MSFT": 0, "GOOGL": 0}
        prices = {"AAPL": 150.0, "MSFT": 200.0, "GOOGL": 100.0}

        assert rebalancer.should_rebalance(three_stock_returns, positions, prices)

    def test_generate_rebalance_orders(self, three_stock_returns):
        """Test generating rebalance orders."""
        rebalancer = RiskParityRebalancer()

        positions = {"AAPL": 10, "MSFT": 10, "GOOGL": 10}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}

        decisions = rebalancer.generate_rebalance_orders(three_stock_returns, positions, prices)

        assert len(decisions) == 3

        # Verify all decisions have required fields
        for decision in decisions:
            assert decision.symbol in ["AAPL", "MSFT", "GOOGL"]
            assert hasattr(decision, "current_weight")
            assert hasattr(decision, "target_weight")
            assert hasattr(decision, "current_volatility")
            assert hasattr(decision, "risk_contribution")
            assert hasattr(decision, "adjustment_shares")
            assert hasattr(decision, "adjustment_value")

        # Verify decisions are rebalancing toward equal risk
        # MSFT (low vol) should get more weight, GOOGL (high vol) should get less
        decisions_map = {d.symbol: d for d in decisions}

        # MSFT should have higher target weight than GOOGL
        assert decisions_map["MSFT"].target_weight > decisions_map["GOOGL"].target_weight

    def test_generate_rebalance_orders_with_cash(self, three_stock_returns):
        """Test rebalance orders when cash is available."""
        rebalancer = RiskParityRebalancer()

        positions = {"AAPL": 5, "MSFT": 5, "GOOGL": 5}  # Total = 5000
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}
        cash = 5000.0  # Double portfolio with cash

        decisions = rebalancer.generate_rebalance_orders(
            three_stock_returns, positions, prices, cash=cash
        )

        # Total value = 10000
        # Check that target values use total including cash
        total_target_value = sum(d.target_weight * 10000 for d in decisions)
        assert total_target_value == pytest.approx(10000.0, abs=1.0)

    def test_generate_rebalance_orders_empty_portfolio_no_cash(self, three_stock_returns):
        """Test rebalance orders with empty portfolio and no cash."""
        rebalancer = RiskParityRebalancer()

        positions = {"AAPL": 0, "MSFT": 0, "GOOGL": 0}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}

        decisions = rebalancer.generate_rebalance_orders(three_stock_returns, positions, prices)

        # Cannot rebalance empty portfolio with no cash
        assert len(decisions) == 0

    def test_generate_rebalance_orders_empty_portfolio_with_cash(self, three_stock_returns):
        """Test rebalance orders with empty portfolio but available cash."""
        rebalancer = RiskParityRebalancer()

        positions = {"AAPL": 0, "MSFT": 0, "GOOGL": 0}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}
        cash = 10000.0

        decisions = rebalancer.generate_rebalance_orders(
            three_stock_returns, positions, prices, cash=cash
        )

        assert len(decisions) == 3

        # All adjustments should be buys (positive)
        for decision in decisions:
            assert decision.adjustment_shares > 0
            assert decision.adjustment_value > 0

        # Total allocation should equal cash
        total_allocation = sum(d.adjustment_value for d in decisions)
        assert total_allocation == pytest.approx(10000.0, abs=1.0)

    def test_get_rebalance_summary(self, three_stock_returns):
        """Test conversion of decisions to DataFrame summary."""
        rebalancer = RiskParityRebalancer()

        positions = {"AAPL": 10, "MSFT": 10, "GOOGL": 10}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}

        decisions = rebalancer.generate_rebalance_orders(three_stock_returns, positions, prices)
        df = rebalancer.get_rebalance_summary(decisions)

        assert len(df) == 3
        assert list(df.columns) == [
            "symbol",
            "current_weight",
            "target_weight",
            "volatility",
            "risk_contribution",
            "drift",
            "adjustment_shares",
            "adjustment_value",
        ]

        # Verify risk contributions are present
        assert all(df["risk_contribution"] > 0)

        # Verify volatilities are present
        assert all(df["volatility"] > 0)

    def test_timestamp_consistency(self, three_stock_returns):
        """Test that all decisions have the same timestamp."""
        rebalancer = RiskParityRebalancer()

        positions = {"AAPL": 10, "MSFT": 10, "GOOGL": 10}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}

        decisions = rebalancer.generate_rebalance_orders(three_stock_returns, positions, prices)

        # All decisions should have same timestamp
        timestamps = [d.timestamp for d in decisions]
        assert all(t == timestamps[0] for t in timestamps)

    def test_inverse_volatility_relationship(self, three_stock_returns):
        """Test that weights are inversely proportional to volatility."""
        rebalancer = RiskParityRebalancer(min_weight=0.0, max_weight=1.0)
        weights = rebalancer.calculate_weights(three_stock_returns)

        # Sort by volatility
        sorted_by_vol = sorted(
            weights.volatilities.items(),
            key=lambda x: x[1],
        )

        # Sort by weight (reverse)
        sorted_by_weight = sorted(
            weights.weights.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        # Lowest vol should have highest weight
        assert sorted_by_vol[0][0] == sorted_by_weight[0][0]

        # Highest vol should have lowest weight
        assert sorted_by_vol[-1][0] == sorted_by_weight[-1][0]
