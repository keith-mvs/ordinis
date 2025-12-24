"""
Strategy Validation Harness.

Unified validation framework for GTM strategies providing:
- Walk-forward analysis with configurable windows
- Transaction cost-adjusted metrics
- Bootstrap confidence intervals
- Acceptance criteria checking
- Lookahead bias auditing

This module addresses gaps identified in the GTM Strategy Test Depth Assessment:
- P0: Unified ValidationResult harness
- P0: Transaction cost modeling integration
- P1: Bootstrap confidence intervals
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ordinis.engines.proofbench.analytics.performance import PerformanceMetrics
    from ordinis.engines.portfolio.costs.transaction_cost_model import (
        TransactionCostEstimate,
        TransactionCostModel,
    )


class ValidationStatus(Enum):
    """Strategy validation status."""

    PASSED = auto()
    FAILED = auto()
    CONDITIONAL = auto()  # Passed with warnings
    INCOMPLETE = auto()  # Missing required tests


@dataclass
class AcceptanceCriteria:
    """Defines thresholds for strategy acceptance.

    Based on GTM Strategy Test Depth Assessment recommendations:
    - Net Sharpe ≥ 1.0 (after 8 bps costs)
    - Max drawdown ≤ 20%
    - Walk-forward periods profitable ≥ 60%
    - Bootstrap 95% CI for Sharpe excludes zero
    """

    min_net_sharpe: float = 1.0
    max_drawdown_pct: float = 20.0
    min_walk_forward_win_pct: float = 60.0
    min_trades_oos: int = 500
    min_bootstrap_sharpe_lower: float = 0.0  # 95% CI lower bound
    max_stress_drawdown_pct: float = 30.0  # During stress periods
    min_sample_days: int = 252


@dataclass
class BootstrapResult:
    """Results from bootstrap analysis.

    Provides confidence intervals for key metrics to assess
    statistical significance of backtest results.
    """

    metric_name: str
    point_estimate: float
    ci_lower: float  # 2.5th percentile
    ci_upper: float  # 97.5th percentile
    std_error: float
    n_resamples: int
    significant: bool  # True if CI excludes zero (or other null)

    @property
    def ci_width(self) -> float:
        """Width of confidence interval."""
        return self.ci_upper - self.ci_lower


@dataclass
class WalkForwardPeriod:
    """Results for a single walk-forward period."""

    period_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_sharpe: float
    test_sharpe: float
    train_return: float
    test_return: float
    test_max_dd: float
    n_trades: int
    profitable: bool


@dataclass
class WalkForwardSummary:
    """Aggregated walk-forward results."""

    periods: list[WalkForwardPeriod]
    n_periods: int
    periods_profitable: int
    win_rate_pct: float
    avg_test_sharpe: float
    std_test_sharpe: float
    robustness_ratio: float  # avg(test_sharpe) / avg(train_sharpe)
    worst_period_sharpe: float
    best_period_sharpe: float


@dataclass
class CostAnalysis:
    """Transaction cost impact analysis."""

    gross_return: float
    net_return: float
    total_costs: Decimal
    avg_cost_per_trade_bps: float
    cost_drag_pct: float  # % of gross return consumed by costs
    n_trades: int


@dataclass
class StressTestResult:
    """Results from stress testing."""

    period_name: str
    start_date: date
    end_date: date
    max_drawdown: float
    total_return: float
    sharpe: float
    survived: bool  # Met stress criteria


@dataclass
class StrategyValidationResult:
    """Comprehensive strategy validation result.

    This is the unified output from the validation harness,
    containing all metrics needed to assess production readiness.
    """

    # Identification
    strategy_id: str
    strategy_version: str
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Backtest scope
    backtest_start: date | None = None
    backtest_end: date | None = None
    symbols_tested: list[str] = field(default_factory=list)

    # Core metrics (gross)
    gross_sharpe: float = 0.0
    gross_return_pct: float = 0.0
    gross_cagr_pct: float = 0.0

    # Cost-adjusted metrics (net)
    net_sharpe: float = 0.0
    net_return_pct: float = 0.0
    net_cagr_pct: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    volatility_pct: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    n_trades: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return_pct: float = 0.0

    # Walk-forward results
    walk_forward: WalkForwardSummary | None = None

    # Bootstrap confidence intervals
    sharpe_bootstrap: BootstrapResult | None = None
    return_bootstrap: BootstrapResult | None = None

    # Cost analysis
    cost_analysis: CostAnalysis | None = None

    # Stress test results
    stress_tests: list[StressTestResult] = field(default_factory=list)

    # Validation status
    status: ValidationStatus = ValidationStatus.INCOMPLETE
    acceptance_criteria: AcceptanceCriteria = field(default_factory=AcceptanceCriteria)
    criteria_results: dict[str, tuple[bool, str]] = field(default_factory=dict)

    # Audit
    lookahead_audit_passed: bool | None = None
    data_provenance: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def check_acceptance(self) -> ValidationStatus:
        """Evaluate all acceptance criteria and return status."""
        criteria = self.acceptance_criteria
        results = {}
        all_passed = True
        has_warnings = False

        # Net Sharpe check
        sharpe_pass = self.net_sharpe >= criteria.min_net_sharpe
        results["net_sharpe"] = (
            sharpe_pass,
            f"Net Sharpe {self.net_sharpe:.2f} {'≥' if sharpe_pass else '<'} {criteria.min_net_sharpe}",
        )
        if not sharpe_pass:
            all_passed = False

        # Max drawdown check
        dd_pass = self.max_drawdown_pct <= criteria.max_drawdown_pct
        results["max_drawdown"] = (
            dd_pass,
            f"Max DD {self.max_drawdown_pct:.1f}% {'≤' if dd_pass else '>'} {criteria.max_drawdown_pct}%",
        )
        if not dd_pass:
            all_passed = False

        # Walk-forward check
        if self.walk_forward:
            wf_pass = self.walk_forward.win_rate_pct >= criteria.min_walk_forward_win_pct
            results["walk_forward_win_rate"] = (
                wf_pass,
                f"WF win rate {self.walk_forward.win_rate_pct:.1f}% {'≥' if wf_pass else '<'} {criteria.min_walk_forward_win_pct}%",
            )
            if not wf_pass:
                all_passed = False
        else:
            results["walk_forward_win_rate"] = (False, "Walk-forward not performed")
            has_warnings = True

        # Sample size check
        trades_pass = self.n_trades >= criteria.min_trades_oos
        results["sample_size"] = (
            trades_pass,
            f"OOS trades {self.n_trades} {'≥' if trades_pass else '<'} {criteria.min_trades_oos}",
        )
        if not trades_pass:
            has_warnings = True  # Warning, not failure

        # Bootstrap CI check
        if self.sharpe_bootstrap:
            ci_pass = self.sharpe_bootstrap.ci_lower >= criteria.min_bootstrap_sharpe_lower
            results["bootstrap_ci"] = (
                ci_pass,
                f"Sharpe 95% CI lower {self.sharpe_bootstrap.ci_lower:.2f} {'≥' if ci_pass else '<'} {criteria.min_bootstrap_sharpe_lower}",
            )
            if not ci_pass:
                all_passed = False
        else:
            results["bootstrap_ci"] = (False, "Bootstrap CI not computed")
            has_warnings = True

        # Stress test check
        if self.stress_tests:
            stress_pass = all(st.survived for st in self.stress_tests)
            failed_stress = [st.period_name for st in self.stress_tests if not st.survived]
            results["stress_tests"] = (
                stress_pass,
                f"Stress tests {'all passed' if stress_pass else f'failed: {failed_stress}'}",
            )
            if not stress_pass:
                all_passed = False
        else:
            results["stress_tests"] = (False, "No stress tests performed")
            has_warnings = True

        # Lookahead audit
        if self.lookahead_audit_passed is not None:
            results["lookahead_audit"] = (
                self.lookahead_audit_passed,
                f"Lookahead audit {'passed' if self.lookahead_audit_passed else 'FAILED'}",
            )
            if not self.lookahead_audit_passed:
                all_passed = False
        else:
            results["lookahead_audit"] = (False, "Lookahead audit not performed")
            has_warnings = True

        self.criteria_results = results

        if all_passed and not has_warnings:
            self.status = ValidationStatus.PASSED
        elif all_passed:
            self.status = ValidationStatus.CONDITIONAL
        else:
            self.status = ValidationStatus.FAILED

        return self.status

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "validated_at": self.validated_at.isoformat(),
            "status": self.status.name,
            "backtest_period": {
                "start": str(self.backtest_start) if self.backtest_start else None,
                "end": str(self.backtest_end) if self.backtest_end else None,
            },
            "symbols": self.symbols_tested,
            "metrics": {
                "gross_sharpe": self.gross_sharpe,
                "net_sharpe": self.net_sharpe,
                "gross_return_pct": self.gross_return_pct,
                "net_return_pct": self.net_return_pct,
                "max_drawdown_pct": self.max_drawdown_pct,
                "n_trades": self.n_trades,
                "win_rate_pct": self.win_rate_pct,
                "profit_factor": self.profit_factor,
            },
            "walk_forward": {
                "n_periods": self.walk_forward.n_periods if self.walk_forward else 0,
                "win_rate_pct": self.walk_forward.win_rate_pct if self.walk_forward else 0,
                "robustness_ratio": self.walk_forward.robustness_ratio if self.walk_forward else 0,
            },
            "bootstrap": {
                "sharpe_ci_lower": self.sharpe_bootstrap.ci_lower if self.sharpe_bootstrap else None,
                "sharpe_ci_upper": self.sharpe_bootstrap.ci_upper if self.sharpe_bootstrap else None,
            },
            "cost_analysis": {
                "cost_drag_pct": self.cost_analysis.cost_drag_pct if self.cost_analysis else None,
            },
            "criteria_results": {k: v[1] for k, v in self.criteria_results.items()},
            "warnings": self.warnings,
            "errors": self.errors,
        }


class StrategyValidator:
    """
    Unified strategy validation harness.

    Performs comprehensive validation including:
    - Walk-forward analysis
    - Transaction cost adjustment
    - Bootstrap confidence intervals
    - Stress testing
    - Acceptance criteria checking
    """

    # Standard stress test periods
    STRESS_PERIODS = {
        "covid_crash_2020": (date(2020, 2, 19), date(2020, 3, 23)),
        "fed_hike_2022": (date(2022, 1, 3), date(2022, 6, 16)),
        "svb_crisis_2023": (date(2023, 3, 8), date(2023, 3, 15)),
    }

    def __init__(
        self,
        cost_model: TransactionCostModel | None = None,
        criteria: AcceptanceCriteria | None = None,
        risk_free_rate: float = 0.02,
    ):
        """Initialize validator.

        Args:
            cost_model: Transaction cost model for net returns
            criteria: Acceptance criteria thresholds
            risk_free_rate: Annual risk-free rate for Sharpe
        """
        self.cost_model = cost_model
        self.criteria = criteria or AcceptanceCriteria()
        self.risk_free_rate = risk_free_rate

    def validate(
        self,
        strategy_id: str,
        strategy_version: str,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        symbols: list[str],
        train_window_days: int = 252,
        test_window_days: int = 63,
        n_bootstrap: int = 1000,
        run_stress_tests: bool = True,
    ) -> StrategyValidationResult:
        """Run comprehensive validation.

        Args:
            strategy_id: Strategy identifier
            strategy_version: Strategy version
            equity_curve: Series of equity values indexed by date
            trades: DataFrame with columns: date, symbol, side, quantity, price, pnl
            symbols: List of symbols traded
            train_window_days: Walk-forward training window
            test_window_days: Walk-forward test window
            n_bootstrap: Number of bootstrap resamples
            run_stress_tests: Whether to run stress tests

        Returns:
            StrategyValidationResult with all metrics
        """
        result = StrategyValidationResult(
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            symbols_tested=symbols,
            acceptance_criteria=self.criteria,
        )

        if equity_curve.empty or trades.empty:
            result.errors.append("Empty equity curve or trades")
            result.status = ValidationStatus.INCOMPLETE
            return result

        # Set backtest period
        result.backtest_start = equity_curve.index.min().date()
        result.backtest_end = equity_curve.index.max().date()
        result.n_trades = len(trades)

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Gross metrics
        result.gross_sharpe = self._calculate_sharpe(returns)
        result.gross_return_pct = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        result.max_drawdown_pct = self._calculate_max_drawdown(equity_curve) * 100
        result.volatility_pct = returns.std() * np.sqrt(252) * 100

        # Trade statistics
        if "pnl" in trades.columns:
            winning = trades[trades["pnl"] > 0]
            result.win_rate_pct = len(winning) / len(trades) * 100 if len(trades) > 0 else 0
            gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
            gross_loss = abs(trades[trades["pnl"] < 0]["pnl"].sum())
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Cost analysis
        if self.cost_model and "price" in trades.columns and "quantity" in trades.columns:
            result.cost_analysis = self._analyze_costs(trades, result.gross_return_pct)
            result.net_return_pct = result.cost_analysis.net_return
            # Adjust Sharpe for costs
            cost_drag_daily = result.cost_analysis.cost_drag_pct / 252
            net_returns = returns - cost_drag_daily / 100
            result.net_sharpe = self._calculate_sharpe(net_returns)
        else:
            # No cost model - use gross as net with warning
            result.net_sharpe = result.gross_sharpe
            result.net_return_pct = result.gross_return_pct
            result.warnings.append("No transaction cost model applied")

        # Walk-forward analysis
        result.walk_forward = self._run_walk_forward(
            returns, train_window_days, test_window_days
        )

        # Bootstrap confidence intervals
        result.sharpe_bootstrap = self._bootstrap_sharpe(returns, n_bootstrap)
        result.return_bootstrap = self._bootstrap_returns(returns, n_bootstrap)

        # Stress tests
        if run_stress_tests:
            result.stress_tests = self._run_stress_tests(equity_curve)

        # Check acceptance criteria
        result.check_acceptance()

        return result

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if returns.empty or returns.std() == 0:
            return 0.0
        excess_returns = returns - self.risk_free_rate / 252
        return float(excess_returns.mean() / returns.std() * np.sqrt(252))

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        return float(abs(drawdown.min()))

    def _analyze_costs(self, trades: pd.DataFrame, gross_return: float) -> CostAnalysis:
        """Analyze transaction costs."""
        total_costs = Decimal("0")
        cost_estimates = []

        for _, trade in trades.iterrows():
            estimate = self.cost_model.estimate_cost(
                symbol=trade.get("symbol", "UNKNOWN"),
                order_size=abs(trade["quantity"]),
                price=trade["price"],
                side=trade.get("side", "buy"),
            )
            cost_estimates.append(estimate)
            total_costs += estimate.total_cost

        n_trades = len(trades)
        avg_bps = np.mean([e.total_bps for e in cost_estimates]) if cost_estimates else 0

        # Calculate cost drag
        total_notional = (trades["price"] * trades["quantity"].abs()).sum()
        cost_drag = float(total_costs) / total_notional * 100 if total_notional > 0 else 0

        net_return = gross_return - cost_drag

        return CostAnalysis(
            gross_return=gross_return,
            net_return=net_return,
            total_costs=total_costs,
            avg_cost_per_trade_bps=avg_bps,
            cost_drag_pct=cost_drag,
            n_trades=n_trades,
        )

    def _run_walk_forward(
        self,
        returns: pd.Series,
        train_days: int,
        test_days: int,
    ) -> WalkForwardSummary:
        """Run walk-forward analysis."""
        periods = []
        idx = 0
        period_id = 0

        while idx + train_days + test_days <= len(returns):
            train_slice = returns.iloc[idx : idx + train_days]
            test_slice = returns.iloc[idx + train_days : idx + train_days + test_days]

            train_sharpe = self._calculate_sharpe(train_slice)
            test_sharpe = self._calculate_sharpe(test_slice)
            train_ret = float((1 + train_slice).prod() - 1) * 100
            test_ret = float((1 + test_slice).prod() - 1) * 100

            # Calculate test drawdown
            test_equity = (1 + test_slice).cumprod()
            test_dd = self._calculate_max_drawdown(test_equity) * 100

            periods.append(
                WalkForwardPeriod(
                    period_id=period_id,
                    train_start=train_slice.index[0].date(),
                    train_end=train_slice.index[-1].date(),
                    test_start=test_slice.index[0].date(),
                    test_end=test_slice.index[-1].date(),
                    train_sharpe=train_sharpe,
                    test_sharpe=test_sharpe,
                    train_return=train_ret,
                    test_return=test_ret,
                    test_max_dd=test_dd,
                    n_trades=0,  # Would need trade data per period
                    profitable=test_ret > 0,
                )
            )

            idx += test_days
            period_id += 1

        if not periods:
            return WalkForwardSummary(
                periods=[],
                n_periods=0,
                periods_profitable=0,
                win_rate_pct=0,
                avg_test_sharpe=0,
                std_test_sharpe=0,
                robustness_ratio=0,
                worst_period_sharpe=0,
                best_period_sharpe=0,
            )

        profitable_periods = [p for p in periods if p.profitable]
        test_sharpes = [p.test_sharpe for p in periods]
        train_sharpes = [p.train_sharpe for p in periods]

        avg_train = np.mean(train_sharpes)
        avg_test = np.mean(test_sharpes)
        robustness = avg_test / avg_train if avg_train != 0 else 0

        return WalkForwardSummary(
            periods=periods,
            n_periods=len(periods),
            periods_profitable=len(profitable_periods),
            win_rate_pct=len(profitable_periods) / len(periods) * 100,
            avg_test_sharpe=float(avg_test),
            std_test_sharpe=float(np.std(test_sharpes)),
            robustness_ratio=float(robustness),
            worst_period_sharpe=float(min(test_sharpes)),
            best_period_sharpe=float(max(test_sharpes)),
        )

    def _bootstrap_sharpe(self, returns: pd.Series, n_resamples: int) -> BootstrapResult:
        """Bootstrap confidence intervals for Sharpe ratio."""
        if len(returns) < 20:
            return BootstrapResult(
                metric_name="sharpe_ratio",
                point_estimate=self._calculate_sharpe(returns),
                ci_lower=0,
                ci_upper=0,
                std_error=0,
                n_resamples=0,
                significant=False,
            )

        rng = np.random.default_rng(42)
        boot_sharpes = []

        for _ in range(n_resamples):
            sample = returns.sample(n=len(returns), replace=True, random_state=rng)
            boot_sharpes.append(self._calculate_sharpe(sample))

        boot_sharpes = np.array(boot_sharpes)
        point_est = self._calculate_sharpe(returns)

        return BootstrapResult(
            metric_name="sharpe_ratio",
            point_estimate=point_est,
            ci_lower=float(np.percentile(boot_sharpes, 2.5)),
            ci_upper=float(np.percentile(boot_sharpes, 97.5)),
            std_error=float(np.std(boot_sharpes)),
            n_resamples=n_resamples,
            significant=float(np.percentile(boot_sharpes, 2.5)) > 0,
        )

    def _bootstrap_returns(self, returns: pd.Series, n_resamples: int) -> BootstrapResult:
        """Bootstrap confidence intervals for total return."""
        if len(returns) < 20:
            total_ret = float((1 + returns).prod() - 1) * 100
            return BootstrapResult(
                metric_name="total_return",
                point_estimate=total_ret,
                ci_lower=0,
                ci_upper=0,
                std_error=0,
                n_resamples=0,
                significant=False,
            )

        rng = np.random.default_rng(42)
        boot_returns = []

        for _ in range(n_resamples):
            sample = returns.sample(n=len(returns), replace=True, random_state=rng)
            boot_returns.append(float((1 + sample).prod() - 1) * 100)

        boot_returns = np.array(boot_returns)
        point_est = float((1 + returns).prod() - 1) * 100

        return BootstrapResult(
            metric_name="total_return",
            point_estimate=point_est,
            ci_lower=float(np.percentile(boot_returns, 2.5)),
            ci_upper=float(np.percentile(boot_returns, 97.5)),
            std_error=float(np.std(boot_returns)),
            n_resamples=n_resamples,
            significant=float(np.percentile(boot_returns, 2.5)) > 0,
        )

    def _run_stress_tests(self, equity_curve: pd.Series) -> list[StressTestResult]:
        """Run stress tests on standard adverse periods."""
        results = []

        for name, (start, end) in self.STRESS_PERIODS.items():
            # Filter equity to stress period
            mask = (equity_curve.index.date >= start) & (equity_curve.index.date <= end)
            stress_equity = equity_curve[mask]

            if len(stress_equity) < 5:
                continue  # Skip if not enough data in period

            stress_returns = stress_equity.pct_change().dropna()
            max_dd = self._calculate_max_drawdown(stress_equity) * 100
            total_ret = (stress_equity.iloc[-1] / stress_equity.iloc[0] - 1) * 100
            sharpe = self._calculate_sharpe(stress_returns)

            results.append(
                StressTestResult(
                    period_name=name,
                    start_date=start,
                    end_date=end,
                    max_drawdown=max_dd,
                    total_return=total_ret,
                    sharpe=sharpe,
                    survived=max_dd <= self.criteria.max_stress_drawdown_pct,
                )
            )

        return results


def create_default_validator() -> StrategyValidator:
    """Create validator with default settings.

    Uses SimpleCostModel with 8 bps round-trip (3 bps spread + 5 bps impact).
    """
    from ordinis.engines.portfolio.costs.transaction_cost_model import SimpleCostModel

    cost_model = SimpleCostModel(
        spread_bps=3.0,  # Half-spread
        impact_bps=5.0,  # Market impact
        commission_per_trade=0.0,  # Zero commission (Alpaca)
    )

    return StrategyValidator(
        cost_model=cost_model,
        criteria=AcceptanceCriteria(),
    )
