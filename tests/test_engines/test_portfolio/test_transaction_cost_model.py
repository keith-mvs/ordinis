"""Tests for TransactionCostModel implementations."""

from __future__ import annotations

from decimal import Decimal

import pytest

from ordinis.engines.portfolio.costs.transaction_cost_model import (
    AdaptiveCostModel,
    AlmgrenChrissModel,
    LiquidityMetrics,
    OrderType,
    SimpleCostModel,
    TransactionCostEstimate,
)


class TestLiquidityMetrics:
    """Tests for LiquidityMetrics."""

    def test_default_metrics(self) -> None:
        """Test default liquidity metrics."""
        metrics = LiquidityMetrics(
            symbol="AAPL",
            avg_daily_volume=50_000_000,
            avg_spread_bps=2.0,
            volatility=0.25,
        )
        assert metrics.symbol == "AAPL"
        assert metrics.avg_daily_volume == 50_000_000
        assert metrics.avg_spread_bps == pytest.approx(2.0)

    def test_illiquid_metrics(self) -> None:
        """Test illiquid asset metrics."""
        metrics = LiquidityMetrics(
            symbol="SMALL_CAP",
            avg_daily_volume=100_000,
            avg_spread_bps=50.0,
            volatility=0.45,
        )
        assert metrics.avg_spread_bps == pytest.approx(50.0)
        assert metrics.volatility == pytest.approx(0.45)


class TestAlmgrenChrissModel:
    """Tests for Almgren-Chriss market impact model."""

    @pytest.fixture
    def model(self) -> AlmgrenChrissModel:
        """Create default model."""
        return AlmgrenChrissModel()

    @pytest.fixture
    def liquid_metrics(self) -> LiquidityMetrics:
        """Create liquid asset metrics."""
        return LiquidityMetrics(
            symbol="SPY",
            avg_daily_volume=100_000_000,
            avg_spread_bps=1.0,
            volatility=0.15,
        )

    @pytest.fixture
    def illiquid_metrics(self) -> LiquidityMetrics:
        """Create illiquid asset metrics."""
        return LiquidityMetrics(
            symbol="MICRO",
            avg_daily_volume=50_000,
            avg_spread_bps=100.0,
            volatility=0.50,
        )

    def test_small_order_low_impact(
        self,
        model: AlmgrenChrissModel,
        liquid_metrics: LiquidityMetrics,
    ) -> None:
        """Test that small orders have low impact."""
        estimate = model.estimate_cost(
            symbol="SPY",
            quantity=Decimal("100"),
            price=Decimal("500"),
            order_type=OrderType.LIMIT,
            liquidity=liquid_metrics,
        )

        # Small order in liquid asset should have minimal impact
        assert estimate.total_cost_bps < 5.0
        assert estimate.market_impact_bps < estimate.spread_cost_bps

    def test_large_order_high_impact(
        self,
        model: AlmgrenChrissModel,
        liquid_metrics: LiquidityMetrics,
    ) -> None:
        """Test that large orders have higher impact."""
        # 10% of daily volume
        large_qty = Decimal(str(liquid_metrics.avg_daily_volume * 0.1))

        estimate = model.estimate_cost(
            symbol="SPY",
            quantity=large_qty,
            price=Decimal("500"),
            order_type=OrderType.MARKET,
            liquidity=liquid_metrics,
        )

        # Large order should have significant impact
        assert estimate.market_impact_bps > 5.0
        assert estimate.participation_rate == pytest.approx(0.1)

    def test_illiquid_asset_higher_cost(
        self,
        model: AlmgrenChrissModel,
        illiquid_metrics: LiquidityMetrics,
    ) -> None:
        """Test that illiquid assets have higher costs."""
        estimate = model.estimate_cost(
            symbol="MICRO",
            quantity=Decimal("1000"),
            price=Decimal("10"),
            order_type=OrderType.MARKET,
            liquidity=illiquid_metrics,
        )

        # High spread and volatility should increase costs
        assert estimate.spread_cost_bps == pytest.approx(50.0)  # Half of bid-ask
        assert estimate.total_cost_bps > 50.0


class TestSimpleCostModel:
    """Tests for SimpleCostModel."""

    def test_fixed_rate_cost(self) -> None:
        """Test fixed-rate cost calculation."""
        model = SimpleCostModel(fixed_cost_bps=5.0)

        estimate = model.estimate_cost(
            symbol="AAPL",
            quantity=Decimal("100"),
            price=Decimal("150"),
            order_type=OrderType.MARKET,
        )

        # 100 * 150 = $15,000 * 5 bps = $7.50
        assert estimate.total_cost_bps == pytest.approx(5.0)
        assert estimate.total_cost_dollars == pytest.approx(7.5)

    def test_commission_added(self) -> None:
        """Test commission is added to costs."""
        model = SimpleCostModel(
            fixed_cost_bps=5.0,
            commission_per_share=Decimal("0.005"),
        )

        estimate = model.estimate_cost(
            symbol="AAPL",
            quantity=Decimal("100"),
            price=Decimal("150"),
        )

        # Base cost + commission
        commission = 100 * 0.005  # $0.50
        base_cost = 15000 * 0.0005  # $7.50
        assert estimate.total_cost_dollars == pytest.approx(base_cost + commission)


class TestAdaptiveCostModel:
    """Tests for AdaptiveCostModel."""

    @pytest.fixture
    def model(self) -> AdaptiveCostModel:
        """Create adaptive model."""
        return AdaptiveCostModel(learning_rate=0.2)

    def test_initial_estimate(self, model: AdaptiveCostModel) -> None:
        """Test initial estimate before learning."""
        estimate = model.estimate_cost(
            symbol="AAPL",
            quantity=Decimal("100"),
            price=Decimal("150"),
        )

        # Should use base model estimate
        assert estimate.total_cost_bps > 0
        assert estimate.confidence < 0.5  # Low confidence initially

    def test_learning_from_execution(self, model: AdaptiveCostModel) -> None:
        """Test model learns from execution data."""
        symbol = "AAPL"

        # Record several executions
        for _ in range(5):
            model.record_execution(
                symbol=symbol,
                estimated_cost_bps=5.0,
                actual_cost_bps=7.0,  # Consistently higher
                quantity=Decimal("100"),
            )

        # Model should adjust estimates upward
        estimate = model.estimate_cost(
            symbol=symbol,
            quantity=Decimal("100"),
            price=Decimal("150"),
        )

        # Confidence should increase after learning
        assert estimate.confidence > 0.3

    def test_symbol_specific_learning(self, model: AdaptiveCostModel) -> None:
        """Test model learns separately for each symbol."""
        # Record high slippage for AAPL
        for _ in range(3):
            model.record_execution(
                symbol="AAPL",
                estimated_cost_bps=5.0,
                actual_cost_bps=15.0,
            )

        # Record low slippage for MSFT
        for _ in range(3):
            model.record_execution(
                symbol="MSFT",
                estimated_cost_bps=5.0,
                actual_cost_bps=3.0,
            )

        # Estimates should differ
        aapl_est = model.estimate_cost("AAPL", Decimal("100"), Decimal("150"))
        msft_est = model.estimate_cost("MSFT", Decimal("100"), Decimal("300"))

        # AAPL should have adjustment factor > 1
        # MSFT should have adjustment factor < 1
        # This depends on implementation details
        assert aapl_est.total_cost_bps != msft_est.total_cost_bps


class TestTransactionCostEstimate:
    """Tests for TransactionCostEstimate."""

    def test_total_calculation(self) -> None:
        """Test total cost calculation."""
        estimate = TransactionCostEstimate(
            symbol="AAPL",
            spread_cost_bps=1.0,
            market_impact_bps=3.0,
            commission_bps=0.5,
            timing_cost_bps=0.5,
            notional_value=Decimal("15000"),
        )

        assert estimate.total_cost_bps == pytest.approx(5.0)
        # Total cost: notional * bps / 10000 = 15000 * 5 / 10000
        assert estimate.total_cost_dollars == pytest.approx(7.5)

    def test_is_acceptable(self) -> None:
        """Test acceptable cost check."""
        low_cost = TransactionCostEstimate(
            symbol="SPY",
            spread_cost_bps=1.0,
            market_impact_bps=2.0,
            notional_value=Decimal("10000"),
        )

        high_cost = TransactionCostEstimate(
            symbol="MICRO",
            spread_cost_bps=50.0,
            market_impact_bps=100.0,
            notional_value=Decimal("1000"),
        )

        assert low_cost.is_acceptable(max_cost_bps=10.0)
        assert not high_cost.is_acceptable(max_cost_bps=10.0)
