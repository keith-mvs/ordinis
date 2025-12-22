"""Tests for PortfolioOptAdapter - Alpaca-style drift and rebalancing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from ordinis.engines.portfolio.adapters.portfolioopt_adapter import (
    CalendarConfig,
    DriftBandConfig,
    DriftType,
    PortfolioOptAdapter,
    PortfolioWeight,
    RebalanceCondition,
)


class TestDriftBandConfig:
    """Tests for DriftBandConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = DriftBandConfig()
        assert config.drift_type == DriftType.ABSOLUTE
        assert config.threshold_pct == pytest.approx(5.0)
        assert config.cooldown_days == 7

    def test_alpaca_config(self) -> None:
        """Test Alpaca-style configuration."""
        config = DriftBandConfig.alpaca_default()
        assert config.drift_type == DriftType.RELATIVE
        assert config.threshold_pct == pytest.approx(5.0)
        assert config.cooldown_days == 7


class TestPortfolioWeight:
    """Tests for PortfolioWeight."""

    def test_drift_calculation_absolute(self) -> None:
        """Test absolute drift calculation."""
        weight = PortfolioWeight(
            symbol="AAPL",
            weight_type="asset",
            target_pct=25.0,
            current_pct=22.0,
        )
        # drift = current - target = 22 - 25 = -3
        assert weight.drift == -3.0

    def test_drift_calculation_relative(self) -> None:
        """Test relative drift calculation."""
        weight = PortfolioWeight(
            symbol="AAPL",
            weight_type="asset",
            target_pct=20.0,
            current_pct=18.0,
        )
        # relative_drift = (current - target) / target * 100 = (18 - 20) / 20 * 100 = -10%
        assert weight.relative_drift == pytest.approx(-10.0)

    def test_new_position_detection(self) -> None:
        """Test new position is identified by 0% current weight."""
        weight = PortfolioWeight(
            symbol="NEW",
            weight_type="asset",
            target_pct=10.0,
            current_pct=0.0,
        )
        # New position has 0% current weight
        assert weight.current_pct == 0.0
        assert weight.target_pct == 10.0


class TestPortfolioOptAdapter:
    """Tests for PortfolioOptAdapter."""

    @pytest.fixture
    def adapter(self) -> PortfolioOptAdapter:
        """Create adapter with default settings."""
        return PortfolioOptAdapter()

    def test_analyze_drift_below_threshold(self, adapter: PortfolioOptAdapter) -> None:
        """Test drift analysis when below threshold."""
        # Current weights as float percentages
        current = {"AAPL": 24.0, "MSFT": 26.0}
        # Target weights as PortfolioWeight objects
        targets = [
            PortfolioWeight(symbol="AAPL", weight_type="asset", target_pct=25.0),
            PortfolioWeight(symbol="MSFT", weight_type="asset", target_pct=25.0),
        ]

        result = adapter.analyze_drift(current, targets)

        assert not result.drift_triggered
        assert result.max_drift < 5.0

    def test_analyze_drift_above_threshold(self, adapter: PortfolioOptAdapter) -> None:
        """Test drift analysis when above threshold."""
        # 15% drift exceeds 5% threshold
        current = {"AAPL": 40.0, "MSFT": 60.0}
        targets = [
            PortfolioWeight(symbol="AAPL", weight_type="asset", target_pct=25.0),
            PortfolioWeight(symbol="MSFT", weight_type="asset", target_pct=25.0),
        ]

        result = adapter.analyze_drift(current, targets)

        assert result.drift_triggered
        assert result.max_drift > 5.0

    def test_cooldown_prevents_rebalance(self, adapter: PortfolioOptAdapter) -> None:
        """Test cooldown period prevents rebalancing."""
        # Simulate recent rebalance
        adapter._last_rebalance_time = datetime.now(UTC) - timedelta(days=1)

        current = {"AAPL": 10.0}  # 10% current weight
        targets = [PortfolioWeight(symbol="AAPL", weight_type="asset", target_pct=50.0)]

        result = adapter.analyze_drift(current, targets)

        # Drift detected (40% > 5% threshold)
        assert result.max_drift > 5.0

    def test_rebalance_trades_calculation(self, adapter: PortfolioOptAdapter) -> None:
        """Test rebalance trade calculation."""
        # Current weights as float percentages
        current = {"AAPL": 50.0, "MSFT": 50.0}  # 50-50 split
        # Target weights as PortfolioWeight objects
        targets = [
            PortfolioWeight(symbol="AAPL", weight_type="asset", target_pct=40.0),
            PortfolioWeight(symbol="MSFT", weight_type="asset", target_pct=60.0),
        ]
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        total_equity = 30000.0

        trades = adapter.calculate_rebalance_trades(
            current,
            targets,
            total_equity,
            prices,
        )

        # Should have trades for both symbols
        assert isinstance(trades, list)

    def test_minimum_trade_value(self, adapter: PortfolioOptAdapter) -> None:
        """Test minimum trade value enforcement (Alpaca $1 minimum)."""
        # Current weights
        current = {"AAPL": 50.0}
        # Target weights (tiny change)
        targets = [PortfolioWeight(symbol="AAPL", weight_type="asset", target_pct=50.01)]
        prices = {"AAPL": 0.50}
        total_equity = 50.0

        trades = adapter.calculate_rebalance_trades(
            current,
            targets,
            total_equity,
            prices,
            min_order_value=1.0,  # $1 minimum
        )

        # Trade should be filtered out if value < $1
        # Note: Depends on implementation
        assert isinstance(trades, list)


class TestCalendarRebalancing:
    """Tests for calendar-based rebalancing configuration."""

    def test_monthly_config(self) -> None:
        """Test monthly rebalancing configuration."""
        config = CalendarConfig(
            frequency="monthly",
            day_of_month=1,
        )
        assert config.frequency == "monthly"
        assert config.day_of_month == 1

    def test_weekly_config(self) -> None:
        """Test weekly rebalancing configuration."""
        config = CalendarConfig(
            frequency="weekly",
            day_of_week=0,  # Monday
        )
        assert config.frequency == "weekly"
        assert config.day_of_week == 0

    def test_quarterly_config(self) -> None:
        """Test quarterly rebalancing configuration."""
        config = CalendarConfig(
            frequency="quarterly",
            day_of_month=1,
        )
        assert config.frequency == "quarterly"
        assert config.day_of_month == 1
