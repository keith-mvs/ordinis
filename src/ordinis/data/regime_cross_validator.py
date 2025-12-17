"""
Regime-Aware Cross-Validation for Strategy Testing.

Implements walk-forward and regime-stratified cross-validation to ensure
strategies are tested across diverse market conditions.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .training_data_generator import DataChunk, MarketRegime, TrainingDataGenerator


@dataclass
class ValidationResult:
    """Results from a single validation fold."""

    fold_id: int
    regime: MarketRegime
    chunk: DataChunk
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    benchmark_return: float
    alpha: float


@dataclass
class CrossValidationReport:
    """Comprehensive cross-validation report."""

    strategy_name: str
    results: list[ValidationResult]
    regime_performance: dict = field(default_factory=dict)
    overall_metrics: dict = field(default_factory=dict)

    def __post_init__(self):
        self._compute_regime_performance()
        self._compute_overall_metrics()

    def _compute_regime_performance(self):
        """Compute performance breakdown by regime."""
        regime_results = {}
        for regime in MarketRegime:
            regime_folds = [r for r in self.results if r.regime == regime]
            if regime_folds:
                regime_results[regime] = {
                    "count": len(regime_folds),
                    "avg_return": np.mean([r.total_return for r in regime_folds]),
                    "avg_sharpe": np.mean([r.sharpe_ratio for r in regime_folds]),
                    "avg_alpha": np.mean([r.alpha for r in regime_folds]),
                    "win_rate_vs_benchmark": sum(1 for r in regime_folds if r.alpha > 0)
                    / len(regime_folds)
                    * 100,
                    "avg_max_drawdown": np.mean([r.max_drawdown for r in regime_folds]),
                }
        self.regime_performance = regime_results

    def _compute_overall_metrics(self):
        """Compute overall performance metrics."""
        if not self.results:
            return

        returns = [r.total_return for r in self.results]
        alphas = [r.alpha for r in self.results]
        sharpes = [r.sharpe_ratio for r in self.results]
        drawdowns = [r.max_drawdown for r in self.results]

        self.overall_metrics = {
            "total_folds": len(self.results),
            "avg_return": np.mean(returns),
            "std_return": np.std(returns),
            "avg_alpha": np.mean(alphas),
            "positive_alpha_rate": sum(1 for a in alphas if a > 0) / len(alphas) * 100,
            "avg_sharpe": np.mean(sharpes),
            "avg_max_drawdown": np.mean(drawdowns),
            "worst_drawdown": min(drawdowns),
            "consistency_score": self._compute_consistency_score(),
        }

    def _compute_consistency_score(self) -> float:
        """
        Compute consistency score across regimes.
        Higher score = more consistent performance across different conditions.
        """
        if not self.regime_performance:
            return 0.0

        alphas = [v["avg_alpha"] for v in self.regime_performance.values()]
        if len(alphas) < 2:
            return 0.0

        # Consistency = 1 / (1 + coefficient of variation)
        mean_alpha = np.mean(alphas)
        std_alpha = np.std(alphas)

        if mean_alpha <= 0:
            return 0.0

        cv = std_alpha / abs(mean_alpha) if mean_alpha != 0 else float("inf")
        return 1 / (1 + cv) * 100

    def print_report(self):
        """Print formatted cross-validation report."""
        print("=" * 80)
        print(f"CROSS-VALIDATION REPORT: {self.strategy_name}")
        print("=" * 80)

        # Overall metrics
        print("\n[OVERALL PERFORMANCE]")
        print("-" * 40)
        m = self.overall_metrics
        print(f"  Total Folds:          {m.get('total_folds', 0)}")
        print(f"  Avg Return:           {m.get('avg_return', 0):+.2f}%")
        print(f"  Return Std Dev:       {m.get('std_return', 0):.2f}%")
        print(f"  Avg Alpha vs B&H:     {m.get('avg_alpha', 0):+.2f}%")
        print(f"  % Folds w/ +Alpha:    {m.get('positive_alpha_rate', 0):.1f}%")
        print(f"  Avg Sharpe:           {m.get('avg_sharpe', 0):.2f}")
        print(f"  Avg Max Drawdown:     {m.get('avg_max_drawdown', 0):.2f}%")
        print(f"  Worst Drawdown:       {m.get('worst_drawdown', 0):.2f}%")
        print(f"  Consistency Score:    {m.get('consistency_score', 0):.1f}/100")

        # Regime breakdown
        print("\n[PERFORMANCE BY REGIME]")
        print("-" * 80)
        print(
            f"{'Regime':<12} {'Folds':>6} {'Avg Ret':>10} {'Avg Alpha':>10} {'Win% vs B&H':>12} {'Avg DD':>10}"
        )
        print("-" * 80)

        for regime in MarketRegime:
            if regime in self.regime_performance:
                p = self.regime_performance[regime]
                print(
                    f"{regime.value:<12} {p['count']:>6} {p['avg_return']:>+9.2f}% "
                    f"{p['avg_alpha']:>+9.2f}% {p['win_rate_vs_benchmark']:>11.1f}% "
                    f"{p['avg_max_drawdown']:>9.2f}%"
                )

        # Verdict
        print("\n[VERDICT]")
        print("-" * 40)
        alpha_rate = m.get("positive_alpha_rate", 0)
        consistency = m.get("consistency_score", 0)

        if alpha_rate >= 60 and consistency >= 50:
            verdict = "PROMISING - Strategy shows potential across conditions"
        elif alpha_rate >= 50:
            verdict = "MARGINAL - Strategy needs regime-specific tuning"
        elif alpha_rate >= 40:
            verdict = "WEAK - Strategy underperforms in most conditions"
        else:
            verdict = "POOR - Strategy fails to add value consistently"

        print(f"  {verdict}")
        print()


class RegimeCrossValidator:
    """
    Cross-validator that ensures strategies are tested across all market regimes.
    """

    def __init__(
        self, strategy_callback: Callable, strategy_name: str, initial_capital: float = 100000.0
    ):
        self.strategy_callback = strategy_callback
        self.strategy_name = strategy_name
        self.initial_capital = initial_capital

    def validate(
        self, chunks: list[DataChunk], min_folds_per_regime: int = 5
    ) -> CrossValidationReport:
        """
        Run cross-validation across data chunks.

        Args:
            chunks: List of DataChunk objects to test on
            min_folds_per_regime: Minimum number of folds per regime type

        Returns:
            CrossValidationReport with results
        """
        # Import here to avoid circular dependency
        from ordinis.engines.proofbench.core.simulator import SimulationConfig, SimulationEngine

        results = []

        for i, chunk in enumerate(chunks):
            try:
                # Setup simulation
                config = SimulationConfig(
                    initial_capital=self.initial_capital,
                    bar_frequency="1d",
                    enable_logging=False,
                )

                engine = SimulationEngine(config)
                engine.load_data(chunk.symbol, chunk.data)

                # Reset strategy state if needed
                fresh_strategy = self._create_fresh_strategy()
                engine.set_strategy(fresh_strategy)

                # Run simulation
                sim_results = engine.run()
                metrics = sim_results.metrics

                # Calculate benchmark
                benchmark_return = chunk.metrics["total_return"] * 100
                alpha = metrics.total_return - benchmark_return

                result = ValidationResult(
                    fold_id=i,
                    regime=chunk.regime,
                    chunk=chunk,
                    total_return=metrics.total_return,
                    sharpe_ratio=metrics.sharpe_ratio,
                    max_drawdown=metrics.max_drawdown,
                    win_rate=metrics.win_rate,
                    num_trades=metrics.num_trades,
                    benchmark_return=benchmark_return,
                    alpha=alpha,
                )
                results.append(result)

            except Exception as e:
                print(f"[WARN] Fold {i} failed: {e}")
                continue

        return CrossValidationReport(strategy_name=self.strategy_name, results=results)

    def _create_fresh_strategy(self) -> Callable:
        """Create a fresh instance of the strategy callback."""
        # This is a placeholder - actual implementation depends on strategy structure
        return self.strategy_callback


class WalkForwardValidator:
    """
    Walk-forward validation with expanding or rolling windows.

    Tests strategy on future data it hasn't seen during training.
    """

    def __init__(
        self,
        train_months: int = 12,
        test_months: int = 3,
        step_months: int = 3,
        expanding: bool = False,
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.expanding = expanding

    def generate_folds(
        self, data: pd.DataFrame, min_train_bars: int = 252
    ) -> list[tuple[pd.DataFrame, pd.DataFrame, dict]]:
        """
        Generate train/test folds for walk-forward validation.

        Returns:
            List of (train_data, test_data, fold_info) tuples
        """
        folds = []

        train_bars = self.train_months * 21
        test_bars = self.test_months * 21
        step_bars = self.step_months * 21

        start_idx = 0 if self.expanding else 0

        while True:
            if self.expanding:
                train_start = 0
            else:
                train_start = start_idx

            train_end = train_start + train_bars + (start_idx if self.expanding else 0)
            test_start = train_end
            test_end = test_start + test_bars

            if test_end > len(data):
                break

            train_data = data.iloc[train_start:train_end].copy()
            test_data = data.iloc[test_start:test_end].copy()

            if len(train_data) >= min_train_bars:
                fold_info = {
                    "train_start": data.index[train_start],
                    "train_end": data.index[train_end - 1],
                    "test_start": data.index[test_start],
                    "test_end": data.index[test_end - 1],
                    "train_bars": len(train_data),
                    "test_bars": len(test_data),
                }
                folds.append((train_data, test_data, fold_info))

            start_idx += step_bars

        return folds


def run_comprehensive_validation(
    strategy_factory: Callable,
    strategy_name: str,
    symbols: list[str] | None = None,
    chunks_per_symbol: int = 50,
    random_seed: int = 42,
) -> CrossValidationReport:
    """
    Run comprehensive regime-stratified cross-validation.

    Args:
        strategy_factory: Function that returns a fresh strategy callback
        strategy_name: Name of the strategy
        symbols: List of symbols to test on
        chunks_per_symbol: Number of chunks per symbol
        random_seed: Random seed for reproducibility

    Returns:
        CrossValidationReport with comprehensive results
    """
    from .training_data_generator import TrainingConfig

    # Generate training data
    config = TrainingConfig(symbols=symbols or ["SPY"], random_seed=random_seed)

    generator = TrainingDataGenerator(config)
    chunks = generator.generate_multi_symbol_dataset(
        chunks_per_symbol=chunks_per_symbol, balance_regimes=True
    )

    # Run validation
    validator = RegimeCrossValidator(
        strategy_callback=strategy_factory(), strategy_name=strategy_name
    )

    report = validator.validate(chunks)
    report.print_report()

    return report


if __name__ == "__main__":
    print("Regime Cross-Validator module loaded.")
    print("Use run_comprehensive_validation() to test strategies.")
