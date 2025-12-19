"""
Tests for Threshold-Based Rebalancing Strategy.
"""

from datetime import UTC, datetime, timedelta

import pytest

from ordinis.engines.portfolio.threshold_based import (
    ThresholdBasedRebalancer,
    ThresholdConfig,
    ThresholdDecision,
    ThresholdStatus,
)


class TestThresholdConfig:
    """Tests for ThresholdConfig dataclass."""

    def test_create_config(self):
        """Test creating threshold config with valid parameters."""
        config = ThresholdConfig(
            symbol="AAPL",
            target_weight=0.40,
            lower_band=-0.05,
            upper_band=0.05,
        )
        assert config.symbol == "AAPL"
        assert config.target_weight == 0.40
        assert config.lower_band == -0.05
        assert config.upper_band == 0.05

    def test_invalid_target_weight(self):
        """Test config fails with invalid target weight."""
        with pytest.raises(ValueError, match="target_weight must be in"):
            ThresholdConfig("AAPL", target_weight=1.5, lower_band=-0.05, upper_band=0.05)

    def test_invalid_lower_band_positive(self):
        """Test config fails with positive lower band."""
        with pytest.raises(ValueError, match="lower_band must be negative"):
            ThresholdConfig("AAPL", target_weight=0.40, lower_band=0.05, upper_band=0.10)

    def test_invalid_upper_band_negative(self):
        """Test config fails with negative upper band."""
        with pytest.raises(ValueError, match="upper_band must be positive"):
            ThresholdConfig("AAPL", target_weight=0.40, lower_band=-0.05, upper_band=-0.10)

    def test_invalid_band_exceeds_limit(self):
        """Test config fails when bands exceed ±100%."""
        with pytest.raises(ValueError, match="Bands cannot exceed"):
            ThresholdConfig("AAPL", target_weight=0.40, lower_band=-1.5, upper_band=0.05)


class TestThresholdStatus:
    """Tests for ThresholdStatus dataclass."""

    def test_create_status(self):
        """Test creating threshold status."""
        status = ThresholdStatus(
            symbol="AAPL",
            current_weight=0.45,
            target_weight=0.40,
            drift=0.05,
            breaches_lower=False,
            breaches_upper=False,
            days_since_rebalance=15,
            exceeds_time_threshold=False,
            should_rebalance=False,
        )
        assert status.symbol == "AAPL"
        assert status.drift == 0.05
        assert not status.should_rebalance


class TestThresholdDecision:
    """Tests for ThresholdDecision dataclass."""

    def test_create_decision(self):
        """Test creating threshold decision."""
        timestamp = datetime.now(tz=UTC)
        decision = ThresholdDecision(
            symbol="AAPL",
            current_weight=0.45,
            target_weight=0.40,
            drift=0.05,
            adjustment_shares=-5.0,
            adjustment_value=-750.0,
            trigger_reason="Above upper band (+5.0%)",
            timestamp=timestamp,
        )
        assert decision.symbol == "AAPL"
        assert decision.trigger_reason == "Above upper band (+5.0%)"
        assert decision.timestamp == timestamp


class TestThresholdBasedRebalancer:
    """Tests for ThresholdBasedRebalancer class."""

    @pytest.fixture
    def three_stock_configs(self):
        """Fixture: Three-stock portfolio with symmetric bands."""
        return [
            ThresholdConfig("AAPL", 0.40, lower_band=-0.05, upper_band=0.05),
            ThresholdConfig("MSFT", 0.30, lower_band=-0.05, upper_band=0.05),
            ThresholdConfig("GOOGL", 0.30, lower_band=-0.05, upper_band=0.05),
        ]

    @pytest.fixture
    def asymmetric_configs(self):
        """Fixture: Configs with asymmetric bands."""
        return [
            ThresholdConfig("AAPL", 0.50, lower_band=-0.10, upper_band=0.05),
            ThresholdConfig("MSFT", 0.30, lower_band=-0.03, upper_band=0.08),
            ThresholdConfig("GOOGL", 0.20, lower_band=-0.05, upper_band=0.10),
        ]

    def test_initialization(self, three_stock_configs):
        """Test rebalancer initialization with valid configs."""
        rebalancer = ThresholdBasedRebalancer(
            threshold_configs=three_stock_configs,
            min_days_between_rebalance=30,
            min_trade_value=100.0,
        )
        assert rebalancer.min_days_between_rebalance == 30
        assert rebalancer.min_trade_value == 100.0
        assert len(rebalancer.configs) == 3

    def test_invalid_min_days(self, three_stock_configs):
        """Test initialization fails with negative min_days."""
        with pytest.raises(ValueError, match="min_days_between_rebalance must be >= 0"):
            ThresholdBasedRebalancer(three_stock_configs, min_days_between_rebalance=-1)

    def test_invalid_min_trade_value(self, three_stock_configs):
        """Test initialization fails with negative min_trade_value."""
        with pytest.raises(ValueError, match="min_trade_value must be >= 0"):
            ThresholdBasedRebalancer(three_stock_configs, min_trade_value=-10.0)

    def test_invalid_weights_sum(self):
        """Test initialization fails when weights don't sum to 1.0."""
        configs = [
            ThresholdConfig("AAPL", 0.40, -0.05, 0.05),
            ThresholdConfig("MSFT", 0.40, -0.05, 0.05),
        ]
        with pytest.raises(ValueError, match="Target weights must sum to 1.0"):
            ThresholdBasedRebalancer(configs)

    def test_check_thresholds_within_bands(self, three_stock_configs):
        """Test threshold check when all symbols within bands."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        # Portfolio close to targets
        positions = {"AAPL": 10, "MSFT": 8, "GOOGL": 8}
        prices = {"AAPL": 200.0, "MSFT": 187.5, "GOOGL": 187.5}

        # Total = 10*200 + 8*187.5 + 8*187.5 = 2000 + 1500 + 1500 = 5000
        # Weights: AAPL=0.40, MSFT=0.30, GOOGL=0.30 (exact targets)

        statuses = rebalancer.check_thresholds(positions, prices)

        assert len(statuses) == 3
        for status in statuses:
            assert not status.breaches_lower
            assert not status.breaches_upper
            assert status.should_rebalance is False

    def test_check_thresholds_breaches_upper_band(self, three_stock_configs):
        """Test threshold check when symbol breaches upper band."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        # AAPL overweight
        positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}

        # Total = 30*200 + 5*300 + 5*300 = 6000 + 1500 + 1500 = 9000
        # Weights: AAPL=0.6667, MSFT=0.1667, GOOGL=0.1667
        # Drift: AAPL=+0.2667 (exceeds +0.05), MSFT=-0.1333, GOOGL=-0.1333

        statuses = rebalancer.check_thresholds(positions, prices)
        aapl_status = next(s for s in statuses if s.symbol == "AAPL")

        assert aapl_status.breaches_upper is True
        assert aapl_status.should_rebalance is True

    def test_check_thresholds_breaches_lower_band(self, three_stock_configs):
        """Test threshold check when symbol breaches lower band."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        # AAPL underweight
        positions = {"AAPL": 5, "MSFT": 15, "GOOGL": 15}
        prices = {"AAPL": 200.0, "MSFT": 200.0, "GOOGL": 200.0}

        # Total = 5*200 + 15*200 + 15*200 = 1000 + 3000 + 3000 = 7000
        # Weights: AAPL=0.1429, MSFT=0.4286, GOOGL=0.4286
        # Drift: AAPL=-0.2571 (below -0.05), MSFT=+0.1286, GOOGL=+0.1286

        statuses = rebalancer.check_thresholds(positions, prices)
        aapl_status = next(s for s in statuses if s.symbol == "AAPL")

        assert aapl_status.breaches_lower is True
        assert aapl_status.should_rebalance is True

    def test_check_thresholds_time_constraint(self, three_stock_configs):
        """Test time constraint prevents rebalancing."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs, min_days_between_rebalance=30)

        # AAPL overweight but recently rebalanced
        positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}

        # Last rebalance was 15 days ago
        last_rebalance = datetime.now(tz=UTC) - timedelta(days=15)

        statuses = rebalancer.check_thresholds(positions, prices, last_rebalance)
        aapl_status = next(s for s in statuses if s.symbol == "AAPL")

        assert aapl_status.breaches_upper is True
        assert aapl_status.exceeds_time_threshold is False
        assert aapl_status.should_rebalance is False  # Time constraint not met

    def test_check_thresholds_time_constraint_met(self, three_stock_configs):
        """Test rebalancing proceeds when time constraint is met."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs, min_days_between_rebalance=30)

        positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}

        # Last rebalance was 45 days ago
        last_rebalance = datetime.now(tz=UTC) - timedelta(days=45)

        statuses = rebalancer.check_thresholds(positions, prices, last_rebalance)
        aapl_status = next(s for s in statuses if s.symbol == "AAPL")

        assert aapl_status.breaches_upper is True
        assert aapl_status.exceeds_time_threshold is True
        assert aapl_status.should_rebalance is True

    def test_check_thresholds_empty_portfolio(self, three_stock_configs):
        """Test threshold check with empty portfolio."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        positions = {"AAPL": 0, "MSFT": 0, "GOOGL": 0}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}

        statuses = rebalancer.check_thresholds(positions, prices)

        assert len(statuses) == 3
        for status in statuses:
            assert status.should_rebalance is True
            assert status.breaches_lower is True

    def test_should_rebalance_within_bands(self, three_stock_configs):
        """Test should_rebalance returns False when within bands."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        positions = {"AAPL": 10, "MSFT": 8, "GOOGL": 8}
        prices = {"AAPL": 200.0, "MSFT": 187.5, "GOOGL": 187.5}

        assert rebalancer.should_rebalance(positions, prices) is False

    def test_should_rebalance_breaches_band(self, three_stock_configs):
        """Test should_rebalance returns True when band breached."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}

        assert rebalancer.should_rebalance(positions, prices) is True

    def test_should_rebalance_time_constraint(self, three_stock_configs):
        """Test should_rebalance respects time constraint."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs, min_days_between_rebalance=30)

        positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}

        last_rebalance = datetime.now(tz=UTC) - timedelta(days=15)

        assert rebalancer.should_rebalance(positions, prices, last_rebalance) is False

    def test_generate_rebalance_orders_no_breach(self, three_stock_configs):
        """Test no orders generated when within bands."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        positions = {"AAPL": 10, "MSFT": 8, "GOOGL": 8}
        prices = {"AAPL": 200.0, "MSFT": 187.5, "GOOGL": 187.5}

        decisions = rebalancer.generate_rebalance_orders(positions, prices)

        assert len(decisions) == 0

    def test_generate_rebalance_orders_with_breach(self, three_stock_configs):
        """Test orders generated when band breached."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}

        decisions = rebalancer.generate_rebalance_orders(positions, prices)

        assert len(decisions) == 3  # All symbols rebalanced for portfolio alignment

        decisions_map = {d.symbol: d for d in decisions}

        # AAPL should be sold (overweight)
        assert decisions_map["AAPL"].adjustment_shares < 0

        # MSFT and GOOGL should be bought (underweight)
        assert decisions_map["MSFT"].adjustment_shares > 0
        assert decisions_map["GOOGL"].adjustment_shares > 0

    def test_generate_rebalance_orders_with_cash(self, three_stock_configs):
        """Test rebalance orders when cash is available."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        positions = {"AAPL": 15, "MSFT": 5, "GOOGL": 5}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}
        cash = 3000.0

        # Total = 15*200 + 5*300 + 5*300 + 3000 = 3000 + 1500 + 1500 + 3000 = 9000

        decisions = rebalancer.generate_rebalance_orders(positions, prices, cash=cash)

        # Should generate decisions using total value including cash
        assert len(decisions) > 0

        total_adjustment = sum(d.adjustment_value for d in decisions)
        # Total adjustments should approximately equal cash (will be allocated)
        assert abs(total_adjustment) <= cash + 100  # Small tolerance

    def test_generate_rebalance_orders_min_trade_value(self, three_stock_configs):
        """Test orders filtered by minimum trade value."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs, min_trade_value=1000.0)

        # Small drift that generates small trades
        positions = {"AAPL": 10, "MSFT": 7, "GOOGL": 9}
        prices = {"AAPL": 201.0, "MSFT": 215.0, "GOOGL": 167.0}

        # Total = 10*201 + 7*215 + 9*167 = 2010 + 1505 + 1503 = 5018
        # Weights: AAPL=0.4005, MSFT=0.2999, GOOGL=0.2996
        # Close to targets but may generate small adjustments

        # Force a breach by using extreme positions
        positions = {"AAPL": 25, "MSFT": 5, "GOOGL": 5}
        prices = {"AAPL": 100.0, "MSFT": 150.0, "GOOGL": 150.0}

        decisions = rebalancer.generate_rebalance_orders(positions, prices)

        # Should only include decisions where abs(adjustment_value) >= 1000.0
        for decision in decisions:
            assert abs(decision.adjustment_value) >= 1000.0

    def test_generate_rebalance_orders_empty_portfolio_no_cash(self, three_stock_configs):
        """Test no orders for empty portfolio with no cash."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        positions = {"AAPL": 0, "MSFT": 0, "GOOGL": 0}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}

        decisions = rebalancer.generate_rebalance_orders(positions, prices)

        assert len(decisions) == 0

    def test_generate_rebalance_orders_empty_portfolio_with_cash(self, three_stock_configs):
        """Test orders generated for empty portfolio with cash."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        positions = {"AAPL": 0, "MSFT": 0, "GOOGL": 0}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}
        cash = 10000.0

        decisions = rebalancer.generate_rebalance_orders(positions, prices, cash=cash)

        assert len(decisions) == 3

        # All should be buys
        for decision in decisions:
            assert decision.adjustment_shares > 0

        # Total allocation should equal cash
        total_allocation = sum(d.adjustment_value for d in decisions)
        assert total_allocation == pytest.approx(10000.0, abs=1.0)

    def test_asymmetric_bands(self, asymmetric_configs):
        """Test rebalancer with asymmetric bands."""
        rebalancer = ThresholdBasedRebalancer(asymmetric_configs)

        # AAPL: target=0.50, lower=-0.10, upper=+0.05
        # Set AAPL to 0.58 (drift=+0.08, exceeds +0.05 upper band)
        positions = {"AAPL": 29, "MSFT": 10, "GOOGL": 5}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 200.0}

        # Total = 29*200 + 10*300 + 5*200 = 5800 + 3000 + 1000 = 9800
        # Weights: AAPL=0.5918, MSFT=0.3061, GOOGL=0.1020
        # Drift: AAPL=+0.0918 (exceeds +0.05)

        statuses = rebalancer.check_thresholds(positions, prices)
        aapl_status = next(s for s in statuses if s.symbol == "AAPL")

        assert aapl_status.breaches_upper is True
        assert aapl_status.should_rebalance is True

    def test_get_threshold_status_summary(self, three_stock_configs):
        """Test conversion of statuses to DataFrame."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}

        statuses = rebalancer.check_thresholds(positions, prices)
        df = rebalancer.get_threshold_status_summary(statuses)

        assert len(df) == 3
        assert list(df.columns) == [
            "symbol",
            "current_weight",
            "target_weight",
            "drift",
            "breaches_lower",
            "breaches_upper",
            "days_since_rebalance",
            "exceeds_time_threshold",
            "should_rebalance",
        ]

    def test_get_rebalance_summary(self, three_stock_configs):
        """Test conversion of decisions to DataFrame summary."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}

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
            "trigger_reason",
        ]

        # Verify trigger reasons are present
        assert all(df["trigger_reason"].notna())

    def test_timestamp_consistency(self, three_stock_configs):
        """Test all decisions have the same timestamp."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}

        decisions = rebalancer.generate_rebalance_orders(positions, prices)

        timestamps = [d.timestamp for d in decisions]
        assert all(t == timestamps[0] for t in timestamps)

    def test_trigger_reason_upper_band(self, three_stock_configs):
        """Test trigger reason for upper band breach."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
        prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 300.0}

        decisions = rebalancer.generate_rebalance_orders(positions, prices)
        aapl_decision = next(d for d in decisions if d.symbol == "AAPL")

        assert "Above upper band" in aapl_decision.trigger_reason

    def test_trigger_reason_lower_band(self, three_stock_configs):
        """Test trigger reason for lower band breach."""
        rebalancer = ThresholdBasedRebalancer(three_stock_configs)

        positions = {"AAPL": 5, "MSFT": 15, "GOOGL": 15}
        prices = {"AAPL": 200.0, "MSFT": 200.0, "GOOGL": 200.0}

        decisions = rebalancer.generate_rebalance_orders(positions, prices)
        aapl_decision = next(d for d in decisions if d.symbol == "AAPL")

        assert "Below lower band" in aapl_decision.trigger_reason
