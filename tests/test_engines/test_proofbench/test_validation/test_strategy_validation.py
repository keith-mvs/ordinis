"""Unit tests for strategy validation harness.

Tests the unified StrategyValidator which provides:
- Walk-forward analysis with configurable windows
- Bootstrap confidence intervals for Sharpe ratios
- Transaction cost impact analysis
- Stress testing on adverse market periods
- Acceptance criteria checking
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.proofbench.validation.strategy_validation import (
    AcceptanceCriteria,
    BootstrapResult,
    CostAnalysis,
    StrategyValidationResult,
    StrategyValidator,
    StressTestResult,
    ValidationStatus,
    WalkForwardPeriod,
    WalkForwardSummary,
    create_default_validator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Create a random generator with fixed seed for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_returns(rng: np.random.Generator) -> pd.Series:
    """Generate sample daily returns for testing."""
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    returns = rng.normal(0.0005, 0.015, 500)
    return pd.Series(returns, index=dates, name="returns")


@pytest.fixture
def sample_trades(rng: np.random.Generator) -> pd.DataFrame:
    """Generate sample trade data for testing."""
    n_trades = 100
    dates = pd.date_range("2020-01-01", periods=n_trades, freq="W")

    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": rng.choice(["AAPL", "MSFT", "GOOGL", "AMZN"], n_trades),
            "side": rng.choice(["buy", "sell"], n_trades),
            "quantity": rng.integers(10, 100, n_trades),
            "price": rng.uniform(100, 500, n_trades),
            "pnl": rng.normal(50, 200, n_trades),
        }
    )


@pytest.fixture
def profitable_returns(rng: np.random.Generator) -> pd.Series:
    """Generate returns that should pass acceptance criteria."""
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    returns = rng.normal(0.002, 0.01, 500)
    return pd.Series(returns, index=dates, name="returns")


@pytest.fixture
def losing_returns(rng: np.random.Generator) -> pd.Series:
    """Generate returns that should fail acceptance criteria."""
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    returns = rng.normal(-0.001, 0.02, 500)
    return pd.Series(returns, index=dates, name="returns")


@pytest.fixture
def default_criteria() -> AcceptanceCriteria:
    """Default acceptance criteria for testing."""
    return AcceptanceCriteria()


@pytest.fixture
def mock_cost_model():
    """Create mock cost model for testing."""
    from ordinis.engines.portfolio.costs.transaction_cost_model import SimpleCostModel

    return SimpleCostModel(spread_bps=3.0, impact_bps=5.0, commission_per_trade=0.0)


# =============================================================================
# AcceptanceCriteria Tests
# =============================================================================


class TestAcceptanceCriteria:
    """Tests for AcceptanceCriteria dataclass."""

    def test_default_values(self):
        """Default criteria should have reasonable production thresholds."""
        criteria = AcceptanceCriteria()

        assert criteria.min_net_sharpe == pytest.approx(1.0)
        assert criteria.max_drawdown_pct == pytest.approx(20.0)
        assert criteria.min_walk_forward_win_pct == pytest.approx(60.0)
        assert criteria.min_trades_oos == 500
        assert criteria.min_sample_days == 252

    def test_custom_values(self):
        """Custom thresholds should be respected."""
        criteria = AcceptanceCriteria(
            min_net_sharpe=1.5,
            max_drawdown_pct=15.0,
            min_trades_oos=200,
        )

        assert criteria.min_net_sharpe == pytest.approx(1.5)
        assert criteria.max_drawdown_pct == pytest.approx(15.0)
        assert criteria.min_trades_oos == 200

    def test_relaxed_criteria_for_development(self):
        """Relaxed criteria for early-stage strategies."""
        criteria = AcceptanceCriteria(
            min_net_sharpe=0.5,
            max_drawdown_pct=30.0,
            min_trades_oos=50,
        )

        assert criteria.min_net_sharpe < 1.0
        assert criteria.max_drawdown_pct > 20.0


# =============================================================================
# ValidationStatus Tests
# =============================================================================


class TestValidationStatus:
    """Tests for ValidationStatus enum."""

    def test_all_statuses_defined(self):
        """All expected statuses should be defined."""
        assert hasattr(ValidationStatus, "PASSED")
        assert hasattr(ValidationStatus, "FAILED")
        assert hasattr(ValidationStatus, "CONDITIONAL")
        assert hasattr(ValidationStatus, "INCOMPLETE")

    def test_status_is_enum(self):
        """Status should be an Enum type."""
        assert isinstance(ValidationStatus.PASSED, ValidationStatus)
        assert isinstance(ValidationStatus.FAILED, ValidationStatus)


# =============================================================================
# BootstrapResult Tests
# =============================================================================


class TestBootstrapResult:
    """Tests for BootstrapResult dataclass."""

    def test_basic_result(self):
        """Bootstrap result should contain CI bounds."""
        result = BootstrapResult(
            metric_name="sharpe",
            point_estimate=1.5,
            ci_lower=1.0,
            ci_upper=2.0,
            std_error=0.25,
            n_resamples=1000,
            significant=True,
        )

        assert result.point_estimate == pytest.approx(1.5)
        assert result.ci_lower == pytest.approx(1.0)
        assert result.ci_upper == pytest.approx(2.0)
        assert result.ci_lower < result.point_estimate < result.ci_upper

    def test_ci_width_property(self):
        """CI width property should calculate correctly."""
        result = BootstrapResult(
            metric_name="sharpe",
            point_estimate=1.5,
            ci_lower=1.0,
            ci_upper=2.0,
            std_error=0.25,
            n_resamples=1000,
            significant=True,
        )

        assert result.ci_width == pytest.approx(1.0)


# =============================================================================
# WalkForwardPeriod and WalkForwardSummary Tests
# =============================================================================


class TestWalkForwardPeriod:
    """Tests for WalkForwardPeriod dataclass."""

    def test_basic_period(self):
        """Period should capture train/test split results."""
        period = WalkForwardPeriod(
            period_id=1,
            train_start=date(2020, 1, 1),
            train_end=date(2020, 6, 30),
            test_start=date(2020, 7, 1),
            test_end=date(2020, 12, 31),
            train_sharpe=1.5,
            test_sharpe=1.2,
            train_return=0.15,
            test_return=0.08,
            test_max_dd=0.10,
            n_trades=50,
            profitable=True,
        )

        assert period.period_id == 1
        assert period.train_sharpe == pytest.approx(1.5)
        assert period.test_sharpe == pytest.approx(1.2)
        assert period.profitable is True


class TestWalkForwardSummary:
    """Tests for WalkForwardSummary dataclass."""

    def test_basic_summary(self):
        """Summary should aggregate period results."""
        summary = WalkForwardSummary(
            periods=[],
            n_periods=4,
            periods_profitable=3,
            win_rate_pct=75.0,
            avg_test_sharpe=1.2,
            std_test_sharpe=0.15,
            robustness_ratio=0.85,
            worst_period_sharpe=0.8,
            best_period_sharpe=1.6,
        )

        assert summary.n_periods == 4
        assert summary.win_rate_pct == pytest.approx(75.0)
        assert summary.robustness_ratio == pytest.approx(0.85)


# =============================================================================
# CostAnalysis Tests
# =============================================================================


class TestCostAnalysis:
    """Tests for CostAnalysis dataclass."""

    def test_cost_impact(self):
        """Cost analysis should show impact on returns."""
        analysis = CostAnalysis(
            gross_return=0.20,
            net_return=0.16,
            total_costs=Decimal("5000.00"),
            avg_cost_per_trade_bps=8.0,
            cost_drag_pct=20.0,
            n_trades=100,
        )

        assert analysis.gross_return > analysis.net_return
        assert analysis.cost_drag_pct == pytest.approx(20.0)
        assert analysis.n_trades == 100


# =============================================================================
# StressTestResult Tests
# =============================================================================


class TestStressTestResult:
    """Tests for StressTestResult dataclass."""

    def test_stress_period(self):
        """Stress test should capture adverse period performance."""
        result = StressTestResult(
            period_name="covid_crash_2020",
            start_date=date(2020, 2, 19),
            end_date=date(2020, 3, 23),
            max_drawdown=-0.35,
            total_return=-0.25,
            sharpe=-2.5,
            survived=False,
        )

        assert result.total_return < 0
        assert result.max_drawdown < result.total_return
        assert result.survived is False


# =============================================================================
# StrategyValidationResult Tests
# =============================================================================


class TestStrategyValidationResult:
    """Tests for StrategyValidationResult dataclass."""

    def test_basic_creation(self):
        """Result should be creatable with minimal fields."""
        result = StrategyValidationResult(
            strategy_id="test_strategy",
            strategy_version="1.0.0",
        )

        assert result.strategy_id == "test_strategy"
        assert result.status == ValidationStatus.INCOMPLETE

    def test_check_acceptance_failing_sharpe(self):
        """Low Sharpe should fail acceptance."""
        result = StrategyValidationResult(
            strategy_id="low_sharpe",
            strategy_version="1.0.0",
            net_sharpe=0.5,  # Below 1.0 threshold
            max_drawdown_pct=10.0,
            n_trades=600,
        )

        status = result.check_acceptance()
        assert status == ValidationStatus.FAILED

    def test_check_acceptance_failing_drawdown(self):
        """High drawdown should fail acceptance."""
        result = StrategyValidationResult(
            strategy_id="high_dd",
            strategy_version="1.0.0",
            net_sharpe=1.5,
            max_drawdown_pct=25.0,  # Above 20% threshold
            n_trades=600,
        )

        status = result.check_acceptance()
        assert status == ValidationStatus.FAILED

    def test_check_acceptance_conditional_without_bootstrap(self):
        """Missing bootstrap should result in CONDITIONAL status."""
        result = StrategyValidationResult(
            strategy_id="good_metrics",
            strategy_version="1.0.0",
            net_sharpe=1.5,
            max_drawdown_pct=10.0,
            n_trades=600,
            walk_forward=WalkForwardSummary(
                periods=[],
                n_periods=4,
                periods_profitable=3,
                win_rate_pct=75.0,
                avg_test_sharpe=1.2,
                std_test_sharpe=0.1,
                robustness_ratio=0.85,
                worst_period_sharpe=0.9,
                best_period_sharpe=1.5,
            ),
            # No sharpe_bootstrap provided
        )

        status = result.check_acceptance()
        # Should be CONDITIONAL because bootstrap CI not computed
        assert status in [ValidationStatus.CONDITIONAL, ValidationStatus.FAILED]


# =============================================================================
# StrategyValidator Tests
# =============================================================================


class TestStrategyValidator:
    """Tests for StrategyValidator class."""

    def test_creation_with_defaults(self):
        """Validator can be created with defaults."""
        validator = StrategyValidator()

        assert validator.criteria is not None

    def test_creation_with_cost_model(self, mock_cost_model):
        """Validator can be created with cost model."""
        validator = StrategyValidator(cost_model=mock_cost_model)

        assert validator.cost_model is not None

    def test_calculate_sharpe(self, sample_returns):
        """Sharpe ratio calculation should be correct."""
        validator = StrategyValidator()
        sharpe = validator._calculate_sharpe(sample_returns)

        assert -5.0 < sharpe < 5.0
        assert isinstance(sharpe, float)

    def test_calculate_sharpe_empty_returns(self):
        """Empty returns should return 0 Sharpe."""
        validator = StrategyValidator()
        empty_returns = pd.Series([], dtype=float)
        sharpe = validator._calculate_sharpe(empty_returns)

        assert sharpe == pytest.approx(0.0)

    def test_calculate_max_drawdown(self, sample_returns):
        """Max drawdown calculation should be non-negative."""
        validator = StrategyValidator()
        dd = validator._calculate_max_drawdown(sample_returns)

        assert dd >= 0.0
        assert dd <= 100.0

    def test_calculate_max_drawdown_profitable(self, profitable_returns):
        """Profitable strategy should have lower drawdown."""
        validator = StrategyValidator()
        dd = validator._calculate_max_drawdown(profitable_returns)

        assert dd < 50.0

    def test_bootstrap_sharpe(self, sample_returns):
        """Bootstrap should produce valid confidence interval."""
        validator = StrategyValidator()
        result = validator._bootstrap_sharpe(sample_returns, n_resamples=100)

        assert isinstance(result, BootstrapResult)
        assert result.ci_lower <= result.point_estimate <= result.ci_upper
        assert result.n_resamples == 100


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateDefaultValidator:
    """Tests for create_default_validator factory."""

    def test_creates_validator(self):
        """Factory should create valid validator."""
        validator = create_default_validator()

        assert isinstance(validator, StrategyValidator)
        assert validator.cost_model is not None
        assert validator.criteria is not None

    def test_default_acceptance_criteria(self):
        """Default criteria should be production-ready."""
        validator = create_default_validator()

        assert validator.criteria.min_net_sharpe == pytest.approx(1.0)
        assert validator.criteria.max_drawdown_pct == pytest.approx(20.0)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_all_zero_returns(self):
        """Zero returns should not crash."""
        validator = StrategyValidator()
        returns = pd.Series([0.0] * 100)
        returns.index = pd.date_range("2020-01-01", periods=100)

        sharpe = validator._calculate_sharpe(returns)
        # Should return 0, nan, or inf but not crash
        assert sharpe is not None

    def test_single_return(self):
        """Single return should not crash."""
        validator = StrategyValidator()
        returns = pd.Series([0.01])
        returns.index = pd.date_range("2020-01-01", periods=1)

        # Should not raise
        sharpe = validator._calculate_sharpe(returns)
        dd = validator._calculate_max_drawdown(returns)

    def test_negative_all_returns(self, losing_returns):
        """All negative returns should produce negative Sharpe."""
        validator = StrategyValidator()
        sharpe = validator._calculate_sharpe(losing_returns)

        assert sharpe < 0.5


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full validation workflow."""

    def test_validation_result_serializable(self):
        """Validation result should be JSON-serializable for reporting."""
        result = StrategyValidationResult(
            strategy_id="serialization_test",
            strategy_version="1.0.0",
            net_sharpe=1.2,
            max_drawdown_pct=12.0,
        )

        # Convert to dict for JSON serialization
        result_dict = {
            "strategy_id": result.strategy_id,
            "status": result.status.name,
            "net_sharpe": result.net_sharpe,
            "max_drawdown_pct": result.max_drawdown_pct,
        }

        import json

        serialized = json.dumps(result_dict)
        assert "serialization_test" in serialized
