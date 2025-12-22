"""Tests for Portfolio Sizing module.

Tests cover:
- SizingMethod enum
- RebalanceTrigger enum
- PositionSizeResult dataclass
- PortfolioAllocation dataclass
- RebalanceAction dataclass
- RebalancePlan dataclass
- SizingConfig dataclass
- RebalanceConfig dataclass
- VolatilityCalculator
- PositionSizer
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.portfolio.sizing import (
    SizingMethod,
    RebalanceTrigger,
    PositionSizeResult,
    PortfolioAllocation,
    RebalanceAction,
    RebalancePlan,
    SizingConfig,
    RebalanceConfig,
    VolatilityCalculator,
    PositionSizer,
)


class TestSizingMethod:
    """Tests for SizingMethod enum."""

    @pytest.mark.unit
    def test_all_methods_defined(self):
        """Test all sizing methods are defined."""
        assert SizingMethod.FIXED_DOLLAR is not None
        assert SizingMethod.FIXED_PERCENT is not None
        assert SizingMethod.VOLATILITY_ADJUSTED is not None
        assert SizingMethod.KELLY is not None
        assert SizingMethod.HALF_KELLY is not None
        assert SizingMethod.RISK_PARITY is not None
        assert SizingMethod.MEAN_VARIANCE is not None
        assert SizingMethod.EQUAL_WEIGHT is not None

    @pytest.mark.unit
    def test_method_count(self):
        """Test correct number of methods."""
        assert len(SizingMethod) == 8


class TestRebalanceTrigger:
    """Tests for RebalanceTrigger enum."""

    @pytest.mark.unit
    def test_all_triggers_defined(self):
        """Test all triggers are defined."""
        assert RebalanceTrigger.SCHEDULED is not None
        assert RebalanceTrigger.THRESHOLD is not None
        assert RebalanceTrigger.SIGNAL is not None
        assert RebalanceTrigger.VOLATILITY is not None
        assert RebalanceTrigger.DRAWDOWN is not None

    @pytest.mark.unit
    def test_trigger_count(self):
        """Test correct number of triggers."""
        assert len(RebalanceTrigger) == 5


class TestPositionSizeResult:
    """Tests for PositionSizeResult dataclass."""

    @pytest.mark.unit
    def test_create_result(self):
        """Test creating a position size result."""
        result = PositionSizeResult(
            symbol="AAPL",
            shares=100,
            notional_value=Decimal("15000.00"),
            weight=0.15,
            risk_contribution=0.12,
            sizing_method=SizingMethod.VOLATILITY_ADJUSTED,
            confidence=0.85,
        )

        assert result.symbol == "AAPL"
        assert result.shares == 100
        assert result.notional_value == Decimal("15000.00")
        assert result.weight == 0.15
        assert result.sizing_method == SizingMethod.VOLATILITY_ADJUSTED

    @pytest.mark.unit
    def test_is_valid_true(self):
        """Test is_valid returns True for valid result."""
        result = PositionSizeResult(
            symbol="MSFT",
            shares=50,
            notional_value=Decimal("10000.00"),
            weight=0.10,
            risk_contribution=0.08,
            sizing_method=SizingMethod.FIXED_PERCENT,
            confidence=0.7,
        )

        assert result.is_valid is True

    @pytest.mark.unit
    def test_is_valid_false_zero_shares(self):
        """Test is_valid returns False for zero shares."""
        result = PositionSizeResult(
            symbol="GOOGL",
            shares=0,
            notional_value=Decimal("0"),
            weight=0.0,
            risk_contribution=0.0,
            sizing_method=SizingMethod.KELLY,
            confidence=0.5,
        )

        assert result.is_valid is False

    @pytest.mark.unit
    def test_is_valid_false_zero_confidence(self):
        """Test is_valid returns False for zero confidence."""
        result = PositionSizeResult(
            symbol="AMZN",
            shares=10,
            notional_value=Decimal("1000.00"),
            weight=0.01,
            risk_contribution=0.01,
            sizing_method=SizingMethod.HALF_KELLY,
            confidence=0.0,
        )

        assert result.is_valid is False


class TestPortfolioAllocation:
    """Tests for PortfolioAllocation dataclass."""

    @pytest.fixture
    def sample_positions(self):
        """Create sample positions."""
        return {
            "AAPL": PositionSizeResult(
                symbol="AAPL",
                shares=100,
                notional_value=Decimal("15000.00"),
                weight=0.15,
                risk_contribution=0.12,
                sizing_method=SizingMethod.VOLATILITY_ADJUSTED,
                confidence=0.8,
            ),
            "MSFT": PositionSizeResult(
                symbol="MSFT",
                shares=50,
                notional_value=Decimal("10000.00"),
                weight=0.10,
                risk_contribution=0.08,
                sizing_method=SizingMethod.VOLATILITY_ADJUSTED,
                confidence=0.75,
            ),
        }

    @pytest.mark.unit
    def test_create_allocation(self, sample_positions):
        """Test creating a portfolio allocation."""
        allocation = PortfolioAllocation(
            positions=sample_positions,
            total_notional=Decimal("25000.00"),
            expected_return=0.12,
            expected_risk=0.15,
            sharpe_ratio=0.8,
            max_position_weight=0.15,
            diversification_ratio=1.2,
        )

        assert allocation.total_notional == Decimal("25000.00")
        assert allocation.expected_return == 0.12
        assert allocation.sharpe_ratio == 0.8

    @pytest.mark.unit
    def test_position_count(self, sample_positions):
        """Test position_count property."""
        allocation = PortfolioAllocation(
            positions=sample_positions,
            total_notional=Decimal("25000.00"),
            expected_return=0.12,
            expected_risk=0.15,
            sharpe_ratio=0.8,
            max_position_weight=0.15,
            diversification_ratio=1.2,
        )

        assert allocation.position_count == 2

    @pytest.mark.unit
    def test_get_weights(self, sample_positions):
        """Test get_weights method."""
        allocation = PortfolioAllocation(
            positions=sample_positions,
            total_notional=Decimal("25000.00"),
            expected_return=0.12,
            expected_risk=0.15,
            sharpe_ratio=0.8,
            max_position_weight=0.15,
            diversification_ratio=1.2,
        )

        weights = allocation.get_weights()
        assert weights["AAPL"] == 0.15
        assert weights["MSFT"] == 0.10


class TestRebalanceAction:
    """Tests for RebalanceAction dataclass."""

    @pytest.mark.unit
    def test_create_buy_action(self):
        """Test creating a buy action."""
        action = RebalanceAction(
            symbol="AAPL",
            current_shares=100,
            target_shares=150,
            shares_delta=50,
            action="buy",
            priority=1,
            reason="Underweight",
        )

        assert action.symbol == "AAPL"
        assert action.shares_delta == 50
        assert action.action == "buy"

    @pytest.mark.unit
    def test_requires_trade_true(self):
        """Test requires_trade returns True when delta non-zero."""
        action = RebalanceAction(
            symbol="MSFT",
            current_shares=50,
            target_shares=30,
            shares_delta=-20,
            action="sell",
            priority=2,
            reason="Overweight",
        )

        assert action.requires_trade is True

    @pytest.mark.unit
    def test_requires_trade_false(self):
        """Test requires_trade returns False when delta is zero."""
        action = RebalanceAction(
            symbol="GOOGL",
            current_shares=75,
            target_shares=75,
            shares_delta=0,
            action="hold",
            priority=3,
            reason="On target",
        )

        assert action.requires_trade is False


class TestSizingConfig:
    """Tests for SizingConfig dataclass."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration values."""
        config = SizingConfig()

        assert config.method == SizingMethod.VOLATILITY_ADJUSTED
        assert config.max_position_pct == 0.10
        assert config.min_position_pct == 0.01
        assert config.max_portfolio_risk == 0.15
        assert config.risk_free_rate == 0.05
        assert config.kelly_fraction == 0.5
        assert config.vol_lookback_days == 20
        assert config.vol_target == 0.15

    @pytest.mark.unit
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SizingConfig(
            method=SizingMethod.KELLY,
            max_position_pct=0.20,
            kelly_fraction=0.25,
        )

        assert config.method == SizingMethod.KELLY
        assert config.max_position_pct == 0.20
        assert config.kelly_fraction == 0.25


class TestRebalanceConfig:
    """Tests for RebalanceConfig dataclass."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default rebalance configuration."""
        config = RebalanceConfig()

        assert config.trigger == RebalanceTrigger.THRESHOLD
        assert config.threshold_pct == 0.05
        assert config.min_interval_hours == 4
        assert config.max_turnover_pct == 0.20
        assert config.priority_by_drift is True


class TestVolatilityCalculator:
    """Tests for VolatilityCalculator."""

    @pytest.fixture
    def sample_returns(self):
        """Create sample return series."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.randn(100) * 0.02, index=dates)
        return returns

    @pytest.fixture
    def sample_ohlc(self):
        """Create sample OHLC data."""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high = close + np.abs(np.random.randn(100)) * 0.5
        low = close - np.abs(np.random.randn(100)) * 0.5
        open_ = close - np.random.randn(100) * 0.2

        return pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }, index=dates)

    @pytest.mark.unit
    def test_realized_volatility(self, sample_returns):
        """Test realized volatility calculation."""
        vol = VolatilityCalculator.realized_volatility(sample_returns, window=20)

        assert len(vol) == len(sample_returns)
        # First window-1 values should be NaN
        assert pd.isna(vol.iloc[:19]).all()
        # After warmup, values should be positive
        assert (vol.dropna() >= 0).all()

    @pytest.mark.unit
    def test_realized_volatility_annualized(self, sample_returns):
        """Test annualized volatility is larger than daily."""
        daily_vol = VolatilityCalculator.realized_volatility(
            sample_returns, window=20, annualize=False
        )
        annual_vol = VolatilityCalculator.realized_volatility(
            sample_returns, window=20, annualize=True
        )

        # Annualized should be larger (multiplied by sqrt(252))
        ratio = (annual_vol / daily_vol).dropna()
        assert (ratio > 1).all()
        # Should be close to sqrt(252) ~ 15.87
        assert np.allclose(ratio.mean(), np.sqrt(252), rtol=0.1)

    @pytest.mark.unit
    def test_parkinson_volatility(self, sample_ohlc):
        """Test Parkinson volatility calculation."""
        vol = VolatilityCalculator.parkinson_volatility(
            sample_ohlc["high"], sample_ohlc["low"], window=20
        )

        assert len(vol) == len(sample_ohlc)
        # After warmup, values should be positive
        assert (vol.dropna() >= 0).all()

    @pytest.mark.unit
    def test_garman_klass_volatility(self, sample_ohlc):
        """Test Garman-Klass volatility calculation."""
        vol = VolatilityCalculator.garman_klass_volatility(
            sample_ohlc["open"],
            sample_ohlc["high"],
            sample_ohlc["low"],
            sample_ohlc["close"],
            window=20,
        )

        assert len(vol) == len(sample_ohlc)
        # Values should be non-negative (NaN for warmup)
        valid_values = vol.dropna()
        assert (valid_values >= 0).all() or len(valid_values) == 0

    @pytest.mark.unit
    def test_ewma_volatility(self, sample_returns):
        """Test EWMA volatility calculation."""
        vol = VolatilityCalculator.ewma_volatility(sample_returns, span=20)

        assert len(vol) == len(sample_returns)
        # EWMA starts from first observation
        assert (vol.dropna() >= 0).all()


class TestPositionSizer:
    """Tests for PositionSizer."""

    @pytest.fixture
    def sizer(self):
        """Create default position sizer."""
        return PositionSizer()

    @pytest.fixture
    def sizer_with_data(self):
        """Create position sizer with data."""
        sizer = PositionSizer()
        sizer.set_portfolio_value(100000)

        # Add returns data
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        returns = pd.Series(np.random.randn(100) * 0.02, index=dates)
        sizer.set_returns("AAPL", returns)

        return sizer

    @pytest.mark.unit
    def test_init_default_config(self, sizer):
        """Test initialization with default config."""
        assert sizer.config is not None
        assert sizer.config.method == SizingMethod.VOLATILITY_ADJUSTED

    @pytest.mark.unit
    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = SizingConfig(method=SizingMethod.KELLY)
        sizer = PositionSizer(config)

        assert sizer.config.method == SizingMethod.KELLY

    @pytest.mark.unit
    def test_set_portfolio_value(self, sizer):
        """Test setting portfolio value."""
        sizer.set_portfolio_value(100000)
        assert sizer._portfolio_value == Decimal("100000")

    @pytest.mark.unit
    def test_set_portfolio_value_float(self, sizer):
        """Test setting portfolio value with float."""
        sizer.set_portfolio_value(150000.50)
        assert sizer._portfolio_value == Decimal("150000.50")

    @pytest.mark.unit
    def test_set_returns(self, sizer):
        """Test setting returns data."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(50) * 0.02)
        sizer.set_returns("AAPL", returns)

        assert "AAPL" in sizer._returns
        assert len(sizer._returns["AAPL"]) == 50

    @pytest.mark.unit
    def test_calculate_size_fixed_dollar(self, sizer_with_data):
        """Test calculating fixed dollar size."""
        result = sizer_with_data.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            signal_strength=1.0,
            method=SizingMethod.FIXED_DOLLAR,
        )

        assert result.symbol == "AAPL"
        assert result.shares > 0
        assert result.sizing_method == SizingMethod.FIXED_DOLLAR

    @pytest.mark.unit
    def test_calculate_size_fixed_percent(self, sizer_with_data):
        """Test calculating fixed percent size."""
        result = sizer_with_data.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            signal_strength=0.8,
            method=SizingMethod.FIXED_PERCENT,
        )

        assert result.symbol == "AAPL"
        assert result.shares > 0
        assert result.sizing_method == SizingMethod.FIXED_PERCENT
        assert result.weight <= sizer_with_data.config.max_position_pct

    @pytest.mark.unit
    def test_calculate_size_respects_signal_strength(self, sizer_with_data):
        """Test that signal strength affects position size."""
        result_high = sizer_with_data.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            signal_strength=1.0,
            method=SizingMethod.FIXED_DOLLAR,
        )

        result_low = sizer_with_data.calculate_size(
            symbol="AAPL",
            current_price=150.0,
            signal_strength=0.5,
            method=SizingMethod.FIXED_DOLLAR,
        )

        # Higher signal strength should result in more shares
        assert result_high.shares >= result_low.shares


class TestRebalancePlan:
    """Tests for RebalancePlan dataclass."""

    @pytest.fixture
    def sample_allocation(self):
        """Create sample portfolio allocation."""
        positions = {
            "AAPL": PositionSizeResult(
                symbol="AAPL",
                shares=100,
                notional_value=Decimal("15000.00"),
                weight=0.15,
                risk_contribution=0.12,
                sizing_method=SizingMethod.VOLATILITY_ADJUSTED,
                confidence=0.8,
            ),
        }
        return PortfolioAllocation(
            positions=positions,
            total_notional=Decimal("100000.00"),
            expected_return=0.12,
            expected_risk=0.15,
            sharpe_ratio=0.8,
            max_position_weight=0.15,
            diversification_ratio=1.2,
        )

    @pytest.fixture
    def sample_actions(self):
        """Create sample rebalance actions."""
        return [
            RebalanceAction(
                symbol="AAPL",
                current_shares=100,
                target_shares=120,
                shares_delta=20,
                action="buy",
                priority=1,
                reason="Underweight",
            ),
        ]

    @pytest.mark.unit
    def test_create_plan(self, sample_allocation, sample_actions):
        """Test creating a rebalance plan."""
        plan = RebalancePlan(
            trigger=RebalanceTrigger.THRESHOLD,
            actions=sample_actions,
            current_allocation=sample_allocation,
            target_allocation=sample_allocation,
            estimated_turnover=0.05,
            estimated_cost=Decimal("10.00"),
        )

        assert plan.trigger == RebalanceTrigger.THRESHOLD
        assert len(plan.actions) == 1
        assert plan.estimated_turnover == 0.05

    @pytest.mark.unit
    def test_requires_action_true(self, sample_allocation, sample_actions):
        """Test requires_action returns True when actions have trades."""
        plan = RebalancePlan(
            trigger=RebalanceTrigger.SIGNAL,
            actions=sample_actions,
            current_allocation=sample_allocation,
            target_allocation=sample_allocation,
            estimated_turnover=0.05,
            estimated_cost=Decimal("10.00"),
        )

        assert plan.requires_action is True

    @pytest.mark.unit
    def test_requires_action_false(self, sample_allocation):
        """Test requires_action returns False when no trades needed."""
        no_trade_actions = [
            RebalanceAction(
                symbol="AAPL",
                current_shares=100,
                target_shares=100,
                shares_delta=0,
                action="hold",
                priority=1,
                reason="On target",
            ),
        ]

        plan = RebalancePlan(
            trigger=RebalanceTrigger.SCHEDULED,
            actions=no_trade_actions,
            current_allocation=sample_allocation,
            target_allocation=sample_allocation,
            estimated_turnover=0.0,
            estimated_cost=Decimal("0.00"),
        )

        assert plan.requires_action is False
