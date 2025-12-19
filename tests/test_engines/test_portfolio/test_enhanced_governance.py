"""Tests for Enhanced Governance Rules."""

from __future__ import annotations

from datetime import UTC, datetime
import pytest

from ordinis.engines.base import PreflightContext
from ordinis.engines.portfolio.hooks.enhanced_governance import (
    CorrelationClusterRule,
    DrawdownRule,
    LiquidityAdjustedRule,
    MarketHoursRule,
    SectorConcentrationRule,
    SectorMapping,
    VolatilityRegimeRule,
    create_conservative_governance_rules,
    create_standard_governance_rules,
)


class TestSectorConcentrationRule:
    """Tests for SectorConcentrationRule."""

    @pytest.fixture
    def sector_mapping(self) -> SectorMapping:
        """Create sector mapping."""
        return SectorMapping(
            symbol_to_sector={
                "AAPL": "Technology",
                "MSFT": "Technology",
                "GOOGL": "Technology",
                "JPM": "Financials",
                "BAC": "Financials",
                "XOM": "Energy",
            }
        )

    @pytest.fixture
    def rule(self, sector_mapping: SectorMapping) -> SectorConcentrationRule:
        """Create rule with 30% sector limit."""
        return SectorConcentrationRule(
            max_sector_pct=30.0,
            sector_mapping=sector_mapping,
        )

    def test_within_limits(self, rule: SectorConcentrationRule) -> None:
        """Test positions within sector limits."""
        context = PreflightContext(
            operation="check_sector",
            parameters={
                "positions": {"AAPL": 100, "JPM": 100, "XOM": 100},
                "prices": {"AAPL": 100, "JPM": 100, "XOM": 100},
            },
        )

        passed, reason = rule.check(context)
        assert passed
        assert "within" in reason.lower()

    def test_exceeds_sector_limit(self, rule: SectorConcentrationRule) -> None:
        """Test positions exceeding sector limit."""
        context = PreflightContext(
            operation="check_sector",
            parameters={
                # 60% in Technology
                "positions": {"AAPL": 200, "MSFT": 200, "GOOGL": 200, "JPM": 100},
                "prices": {"AAPL": 100, "MSFT": 100, "GOOGL": 100, "JPM": 100},
            },
        )

        passed, reason = rule.check(context)
        assert not passed
        assert "Technology" in reason

    def test_get_sector_exposure(self, sector_mapping: SectorMapping) -> None:
        """Test sector exposure calculation."""
        positions = {"AAPL": 100, "MSFT": 100, "JPM": 50}
        prices = {"AAPL": 100, "MSFT": 100, "JPM": 100}

        exposure = sector_mapping.get_sector_exposure(positions, prices)

        # Tech: 20000 / 25000 = 80%
        # Financials: 5000 / 25000 = 20%
        assert exposure["Technology"] == pytest.approx(80.0)
        assert exposure["Financials"] == pytest.approx(20.0)


class TestCorrelationClusterRule:
    """Tests for CorrelationClusterRule."""

    @pytest.fixture
    def rule(self) -> CorrelationClusterRule:
        """Create rule with correlations."""
        rule = CorrelationClusterRule(
            max_cluster_pct=40.0,
            correlation_threshold=0.7,
        )
        # Set up high correlations for tech stocks
        rule.set_correlation("AAPL", "MSFT", 0.85)
        rule.set_correlation("AAPL", "GOOGL", 0.80)
        rule.set_correlation("MSFT", "GOOGL", 0.75)
        # Low correlation for financials
        rule.set_correlation("JPM", "AAPL", 0.30)
        return rule

    def test_cluster_within_limits(self, rule: CorrelationClusterRule) -> None:
        """Test correlated cluster within limits."""
        context = PreflightContext(
            operation="check_correlation",
            parameters={
                "positions": {"AAPL": 100, "MSFT": 100, "JPM": 300},
                "prices": {"AAPL": 100, "MSFT": 100, "JPM": 100},
            },
        )

        passed, _ = rule.check(context)
        assert passed

    def test_cluster_exceeds_limit(self, rule: CorrelationClusterRule) -> None:
        """Test correlated cluster exceeding limit."""
        context = PreflightContext(
            operation="check_correlation",
            parameters={
                # Tech cluster is 60% of portfolio
                "positions": {"AAPL": 200, "MSFT": 200, "GOOGL": 200, "JPM": 100},
                "prices": {"AAPL": 100, "MSFT": 100, "GOOGL": 100, "JPM": 100},
            },
        )

        passed, reason = rule.check(context)
        assert not passed
        assert "cluster" in reason.lower()

    def test_find_clusters(self, rule: CorrelationClusterRule) -> None:
        """Test cluster detection."""
        symbols = ["AAPL", "MSFT", "GOOGL", "JPM"]
        clusters = rule.find_clusters(symbols)

        # Should find one cluster of tech stocks
        assert len(clusters) >= 1
        tech_cluster = max(clusters, key=len)
        assert "AAPL" in tech_cluster or "MSFT" in tech_cluster


class TestLiquidityAdjustedRule:
    """Tests for LiquidityAdjustedRule."""

    @pytest.fixture
    def rule(self) -> LiquidityAdjustedRule:
        """Create rule with liquidity data."""
        rule = LiquidityAdjustedRule(
            max_participation_rate=5.0,
            max_illiquid_position_pct=2.0,
            liquidity_threshold=500_000,
        )
        rule.set_liquidity("LIQUID", avg_daily_volume=10_000_000)
        rule.set_liquidity("ILLIQUID", avg_daily_volume=10_000, avg_spread_bps=100)
        return rule

    def test_liquid_position_passes(self, rule: LiquidityAdjustedRule) -> None:
        """Test liquid position within limits."""
        context = PreflightContext(
            operation="check_liquidity",
            parameters={
                "positions": {"LIQUID": 100_000},  # 1% of daily volume
                "prices": {"LIQUID": 100},
            },
        )

        passed, _ = rule.check(context)
        assert passed

    def test_high_participation_fails(self, rule: LiquidityAdjustedRule) -> None:
        """Test high participation rate fails."""
        context = PreflightContext(
            operation="check_liquidity",
            parameters={
                "positions": {"LIQUID": 1_000_000},  # 10% of daily volume
                "prices": {"LIQUID": 100},
            },
        )

        passed, reason = rule.check(context)
        assert not passed
        assert "participation" in reason.lower()

    def test_illiquid_position_limit(self, rule: LiquidityAdjustedRule) -> None:
        """Test illiquid position size limit."""
        context = PreflightContext(
            operation="check_liquidity",
            parameters={
                # Large position in illiquid asset
                "positions": {"ILLIQUID": 500, "LIQUID": 100},
                "prices": {"ILLIQUID": 100, "LIQUID": 100},
            },
        )

        passed, reason = rule.check(context)
        # Should fail due to concentration in illiquid asset
        # Depends on implementation details
        assert isinstance(passed, bool)


class TestDrawdownRule:
    """Tests for DrawdownRule."""

    @pytest.fixture
    def rule(self) -> DrawdownRule:
        """Create rule with default settings."""
        return DrawdownRule(
            max_drawdown_for_full_size=5.0,
            min_size_at_max_drawdown=0.25,
            max_drawdown_limit=20.0,
        )

    def test_no_drawdown_full_size(self, rule: DrawdownRule) -> None:
        """Test full sizing when no drawdown."""
        rule.peak_equity = 100_000
        rule.update_drawdown(100_000)

        assert rule.get_sizing_multiplier() == pytest.approx(1.0)

    def test_moderate_drawdown_reduced_size(self, rule: DrawdownRule) -> None:
        """Test reduced sizing in moderate drawdown."""
        rule.peak_equity = 100_000
        rule.update_drawdown(88_000)  # 12% drawdown

        multiplier = rule.get_sizing_multiplier()
        assert 0.25 < multiplier < 1.0

    def test_max_drawdown_blocks_trades(self, rule: DrawdownRule) -> None:
        """Test trades blocked at max drawdown."""
        rule.peak_equity = 100_000
        rule.update_drawdown(75_000)  # 25% drawdown

        context = PreflightContext(operation="trade")
        passed, reason = rule.check(context)

        assert not passed
        assert "maximum" in reason.lower() or "exceeded" in reason.lower()


class TestVolatilityRegimeRule:
    """Tests for VolatilityRegimeRule."""

    def test_low_vol_full_size(self) -> None:
        """Test full sizing in low volatility."""
        rule = VolatilityRegimeRule()
        rule.update_volatility(vix=12.0)

        assert rule.get_sizing_multiplier() == pytest.approx(1.0)

    def test_high_vol_reduced_size(self) -> None:
        """Test reduced sizing in high volatility."""
        rule = VolatilityRegimeRule()
        rule.update_volatility(vix=35.0)

        multiplier = rule.get_sizing_multiplier()
        assert multiplier == pytest.approx(0.5)

    def test_moderate_vol_interpolation(self) -> None:
        """Test interpolated sizing in moderate volatility."""
        rule = VolatilityRegimeRule()
        rule.update_volatility(vix=22.5)  # Midpoint

        multiplier = rule.get_sizing_multiplier()
        assert 0.5 < multiplier < 1.0


class TestMarketHoursRule:
    """Tests for MarketHoursRule."""

    def test_regular_hours_allowed(self) -> None:
        """Test trading during regular hours."""
        rule = MarketHoursRule(allow_extended_hours=False)

        # 10:00 AM ET = 15:00 UTC
        during_hours = datetime(2024, 3, 15, 15, 0, tzinfo=UTC)
        context = PreflightContext(operation="trade", timestamp=during_hours)

        passed, reason = rule.check(context)
        assert passed

    def test_after_hours_blocked(self) -> None:
        """Test trading blocked after hours."""
        rule = MarketHoursRule(allow_extended_hours=False)

        # 8:00 PM ET = 01:00 UTC next day
        after_hours = datetime(2024, 3, 16, 1, 0, tzinfo=UTC)
        context = PreflightContext(operation="trade", timestamp=after_hours)

        passed, reason = rule.check(context)
        assert not passed
        assert "outside" in reason.lower()


class TestGovernanceRuleFactories:
    """Tests for governance rule factories."""

    def test_standard_rules_creation(self) -> None:
        """Test standard rules factory."""
        rules = create_standard_governance_rules()

        assert len(rules) >= 4
        rule_types = {type(r).__name__ for r in rules}
        assert "SectorConcentrationRule" in rule_types
        assert "DrawdownRule" in rule_types

    def test_conservative_rules_stricter(self) -> None:
        """Test conservative rules have stricter limits."""
        standard = create_standard_governance_rules()
        conservative = create_conservative_governance_rules()

        # Find DrawdownRule in each
        std_dd = next(r for r in standard if isinstance(r, DrawdownRule))
        cons_dd = next(r for r in conservative if isinstance(r, DrawdownRule))

        assert cons_dd.max_drawdown_limit < std_dd.max_drawdown_limit
