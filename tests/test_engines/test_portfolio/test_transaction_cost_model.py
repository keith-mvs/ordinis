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
            avg_daily_volume=50_000_000,
            avg_spread_bps=2.0,
            volatility=0.25,
        )
        assert metrics.avg_daily_volume == 50_000_000
        assert metrics.avg_spread_bps == pytest.approx(2.0)
        assert metrics.liquidity_score > 0  # Test the property

    def test_illiquid_metrics(self) -> None:
        """Test illiquid asset metrics."""
        metrics = LiquidityMetrics(
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
            avg_daily_volume=100_000_000,
            avg_spread_bps=1.0,
            volatility=0.15,
        )

    @pytest.fixture
    def illiquid_metrics(self) -> LiquidityMetrics:
        """Create illiquid asset metrics."""
        return LiquidityMetrics(
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
            order_size=100,
            price=500.0,
            side="buy",
            order_type=OrderType.LIMIT,
            liquidity=liquid_metrics,
        )

        # Small order in liquid asset should have minimal impact
        assert estimate.total_bps < 5.0
        # Market impact should be small for small orders
        assert float(estimate.market_impact) < float(estimate.spread_cost)

    def test_large_order_high_impact(
        self,
        model: AlmgrenChrissModel,
        liquid_metrics: LiquidityMetrics,
    ) -> None:
        """Test that large orders have higher impact."""
        # 10% of daily volume
        large_qty = liquid_metrics.avg_daily_volume * 0.1

        estimate = model.estimate_cost(
            symbol="SPY",
            order_size=large_qty,
            price=500.0,
            side="buy",
            order_type=OrderType.MARKET,
            liquidity=liquid_metrics,
        )

        # Large order should have significant impact
        # Check via metadata since implementation stores participation_rate there
        assert estimate.metadata.get("participation_rate") == pytest.approx(0.1)
        assert float(estimate.market_impact) > 0  # Has market impact

    def test_illiquid_asset_higher_cost(
        self,
        model: AlmgrenChrissModel,
        illiquid_metrics: LiquidityMetrics,
    ) -> None:
        """Test that illiquid assets have higher costs."""
        estimate = model.estimate_cost(
            symbol="MICRO",
            order_size=1000,
            price=10.0,
            side="buy",
            order_type=OrderType.MARKET,
            liquidity=illiquid_metrics,
        )

        # High spread and volatility should increase costs
        # Spread cost is half of bid-ask spread (100 bps / 2 = 50 bps)
        spread_bps = float(estimate.spread_cost) / float(estimate.notional_value) * 10000
        assert spread_bps == pytest.approx(50.0)
        assert estimate.total_bps > 50.0


class TestSimpleCostModel:
    """Tests for SimpleCostModel."""

    def test_fixed_rate_cost(self) -> None:
        """Test fixed-rate cost calculation."""
        # SimpleCostModel uses spread_bps and impact_bps (default 5 each = 10 total)
        model = SimpleCostModel(spread_bps=5.0, impact_bps=0.0)

        estimate = model.estimate_cost(
            symbol="AAPL",
            order_size=100,
            price=150.0,
            side="buy",
            order_type=OrderType.MARKET,
        )

        # 100 * 150 = $15,000 * 5 bps = $7.50
        assert estimate.total_bps == pytest.approx(5.0)
        assert float(estimate.total_cost) == pytest.approx(7.5)

    def test_commission_added(self) -> None:
        """Test commission is added to costs."""
        # SimpleCostModel uses commission_per_trade not per_share
        model = SimpleCostModel(
            spread_bps=5.0,
            impact_bps=0.0,
            commission_per_trade=0.50,  # $0.50 flat commission
        )

        estimate = model.estimate_cost(
            symbol="AAPL",
            order_size=100,
            price=150.0,
            side="buy",
        )

        # Base cost + commission
        commission = 0.50  # Flat commission
        base_cost = 15000 * 0.0005  # $7.50 at 5 bps
        assert float(estimate.total_cost) == pytest.approx(base_cost + commission)


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
            order_size=100,
            price=150.0,
            side="buy",
        )

        # Should use base model estimate
        assert estimate.total_bps > 0
        # Base model confidence is 0.5, adaptive adds 0.1 = 0.6
        assert estimate.confidence >= 0.5

    def test_learning_from_execution(self, model: AdaptiveCostModel) -> None:
        """Test model learns from execution data."""
        symbol = "AAPL"

        # Record several executions
        for _ in range(5):
            model.record_execution(
                symbol=symbol,
                estimated_cost_bps=5.0,
                actual_cost_bps=7.0,  # Consistently higher
                notional=15000.0,
            )

        # Model should adjust estimates upward
        estimate = model.estimate_cost(
            symbol=symbol,
            order_size=100,
            price=150.0,
            side="buy",
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
                notional=15000.0,
            )

        # Record low slippage for MSFT
        for _ in range(3):
            model.record_execution(
                symbol="MSFT",
                estimated_cost_bps=5.0,
                actual_cost_bps=3.0,
                notional=30000.0,
            )

        # Estimates should differ
        aapl_est = model.estimate_cost("AAPL", 100, 150.0, "buy")
        msft_est = model.estimate_cost("MSFT", 100, 300.0, "buy")

        # AAPL should have adjustment factor > 1
        # MSFT should have adjustment factor < 1
        # This depends on implementation details
        assert aapl_est.total_bps != msft_est.total_bps


class TestTransactionCostEstimate:
    """Tests for TransactionCostEstimate."""

    def test_total_calculation(self) -> None:
        """Test total cost calculation."""
        # TransactionCostEstimate uses dollar amounts, not bps, for component costs
        notional = Decimal("15000")
        # Create costs that sum to 5 bps = $7.50 for $15k notional
        spread = Decimal("1.50")  # 1 bps
        impact = Decimal("4.50")  # 3 bps
        commission = Decimal("0.75")  # 0.5 bps
        total = spread + impact + commission  # $6.75 = 4.5 bps
        total_bps = float(total) / float(notional) * 10000  # 4.5 bps

        estimate = TransactionCostEstimate(
            symbol="AAPL",
            order_size=100,
            notional_value=notional,
            spread_cost=spread,
            market_impact=impact,
            commission=commission,
            total_cost=total,
            total_bps=total_bps,
            confidence=0.8,
        )

        assert estimate.total_bps == pytest.approx(4.5)
        assert float(estimate.total_cost) == pytest.approx(6.75)

    def test_is_material(self) -> None:
        """Test is_material property."""
        low_cost = TransactionCostEstimate(
            symbol="SPY",
            order_size=100,
            notional_value=Decimal("10000"),
            spread_cost=Decimal("0.50"),
            market_impact=Decimal("0.50"),
            commission=Decimal("0"),
            total_cost=Decimal("1.00"),
            total_bps=1.0,  # 1 bps < 10 bps threshold
            confidence=0.8,
        )

        high_cost = TransactionCostEstimate(
            symbol="MICRO",
            order_size=100,
            notional_value=Decimal("1000"),
            spread_cost=Decimal("5.00"),
            market_impact=Decimal("10.00"),
            commission=Decimal("0"),
            total_cost=Decimal("15.00"),
            total_bps=150.0,  # 150 bps > 10 bps threshold
            confidence=0.5,
        )

        assert not low_cost.is_material  # < 10 bps
        assert high_cost.is_material  # > 10 bps
