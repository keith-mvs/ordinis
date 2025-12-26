#!/usr/bin/env python3
"""
Optimization Controller for Network Parity Strategy

Manages the iterative optimization loop with convergence detection
and history tracking.

Author: Ordinis Quantitative Research
Version: 1.0.0
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from config import (
    OptimizationConfig,
    NetworkParityParams,
    OUTPUT_DIR,
)
from data_pipeline import DataPipeline, DataPipelineResult
from equity_universe import EquityUniverse, create_default_universe
from backtesting import BacktestEngine, BacktestResult, run_single_symbol_backtest
from nemo_integration import (
    NemoOptimizer,
    NemoResponse,
    NemoSuggestion,
    generate_random_perturbation,
)
from reporting import ReportGenerator

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class IterationResult:
    """Result of a single optimization iteration."""

    iteration: int
    params: dict[str, Any]
    backtest_result: BacktestResult
    score: float
    nemo_suggestions: list[NemoSuggestion] = field(default_factory=list)
    improvement: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "iteration": self.iteration,
            "params": self.params,
            "score": self.score,
            "improvement": self.improvement,
            "timestamp": self.timestamp,
            "metrics": {
                "total_return": self.backtest_result.total_return,
                "sharpe_ratio": self.backtest_result.sharpe_ratio,
                "sortino_ratio": self.backtest_result.sortino_ratio,
                "max_drawdown": self.backtest_result.max_drawdown,
                "win_rate": self.backtest_result.win_rate,
                "profit_factor": self.backtest_result.profit_factor,
                "num_trades": self.backtest_result.num_trades,
            },
            "nemo_suggestions": [s.to_dict() for s in self.nemo_suggestions],
        }


@dataclass
class OptimizationState:
    """State of the optimization process."""

    iteration: int = 0
    current_params: dict[str, Any] = field(default_factory=dict)
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float = -np.inf
    best_result: BacktestResult | None = None
    history: list[IterationResult] = field(default_factory=list)
    converged: bool = False
    convergence_reason: str = ""
    patience_counter: int = 0

    def update(
        self,
        result: IterationResult,
        patience_limit: int,
        convergence_threshold: float,
    ) -> None:
        """
        Update state with new iteration result.

        Args:
            result: Iteration result
            patience_limit: Early stopping patience
            convergence_threshold: Minimum improvement threshold
        """
        self.history.append(result)

        if result.score > self.best_score:
            improvement = result.score - self.best_score
            result.improvement = improvement

            if improvement >= convergence_threshold:
                self.best_score = result.score
                self.best_params = result.params.copy()
                self.best_result = result.backtest_result
                self.patience_counter = 0
                logger.info(
                    f"Iteration {result.iteration}: New best score {result.score:.4f} "
                    f"(improvement: {improvement:.4f})"
                )
            else:
                self.patience_counter += 1
                logger.info(
                    f"Iteration {result.iteration}: Score {result.score:.4f} "
                    f"(marginal improvement: {improvement:.6f})"
                )
        else:
            self.patience_counter += 1
            logger.info(
                f"Iteration {result.iteration}: Score {result.score:.4f} "
                f"(no improvement, patience: {self.patience_counter}/{patience_limit})"
            )

        # Check convergence
        if self.patience_counter >= patience_limit:
            self.converged = True
            self.convergence_reason = f"Early stopping after {patience_limit} iterations without improvement"

    def get_history_summary(self) -> list[dict[str, Any]]:
        """Get summary of optimization history for Nemo context."""
        return [
            {
                "iteration": r.iteration,
                "total_return": r.backtest_result.total_return,
                "sharpe_ratio": r.backtest_result.sharpe_ratio,
                "sortino_ratio": r.backtest_result.sortino_ratio,
                "score": r.score,
            }
            for r in self.history[-10:]  # Last 10 iterations
        ]


@dataclass
class OptimizationResult:
    """Final result of optimization process."""

    config: OptimizationConfig
    state: OptimizationState
    baseline_result: BacktestResult | None = None
    final_result: BacktestResult | None = None
    total_time_seconds: float = 0.0

    @property
    def improvement_vs_baseline(self) -> float:
        """Calculate improvement over baseline."""
        if self.baseline_result is None or self.final_result is None:
            return 0.0
        baseline_score = self.baseline_result.compute_score()
        final_score = self.final_result.compute_score()
        return final_score - baseline_score

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "OPTIMIZATION SUMMARY",
            "=" * 60,
            f"Configuration ID: {self.config.config_id}",
            f"Total Iterations: {self.state.iteration}",
            f"Converged: {self.state.converged}",
            f"Reason: {self.state.convergence_reason}",
            f"Time: {self.total_time_seconds:.1f}s",
            "",
            "BASELINE PERFORMANCE:",
        ]

        if self.baseline_result:
            lines.extend([
                f"  Return: {self.baseline_result.total_return:.2%}",
                f"  Sharpe: {self.baseline_result.sharpe_ratio:.2f}",
                f"  Sortino: {self.baseline_result.sortino_ratio:.2f}",
                f"  MaxDD: {self.baseline_result.max_drawdown:.2%}",
                f"  Score: {self.baseline_result.compute_score():.4f}",
            ])

        lines.append("")
        lines.append("OPTIMIZED PERFORMANCE:")

        if self.final_result:
            lines.extend([
                f"  Return: {self.final_result.total_return:.2%}",
                f"  Sharpe: {self.final_result.sharpe_ratio:.2f}",
                f"  Sortino: {self.final_result.sortino_ratio:.2f}",
                f"  MaxDD: {self.final_result.max_drawdown:.2%}",
                f"  Score: {self.final_result.compute_score():.4f}",
            ])

        lines.extend([
            "",
            f"IMPROVEMENT: {self.improvement_vs_baseline:.4f}",
            "=" * 60,
        ])

        return "\n".join(lines)


# =============================================================================
# OPTIMIZATION CONTROLLER
# =============================================================================

class OptimizationController:
    """
    Controls the iterative optimization process.

    Orchestrates data loading, backtesting, Nemo integration,
    and result reporting.
    """

    def __init__(
        self,
        config: OptimizationConfig | None = None,
    ):
        """
        Initialize optimization controller.

        Args:
            config: Optimization configuration
        """
        self.config = config or OptimizationConfig()
        self.state = OptimizationState()

        # Initialize components
        self.pipeline = DataPipeline(config=self.config.backtesting)
        self.universe = create_default_universe()
        self.nemo = NemoOptimizer(config=self.config.nemo)
        self.reporter = ReportGenerator(self.config)

        # Data cache
        self._data: DataPipelineResult | None = None

    def _load_data(self) -> DataPipelineResult:
        """Load market data for backtesting."""
        if self._data is not None:
            return self._data

        logger.info("Loading market data...")

        # Get available symbols from data
        available = self.pipeline.get_available_symbols()
        self.universe.update_availability(available)

        # Get symbols that have data
        symbols = self.universe.get_available_symbols()
        if not symbols:
            raise ValueError("No symbols available in data")

        logger.info(f"Available symbols: {len(symbols)}")

        # Load data
        self._data = self.pipeline.load_universe(
            symbols,
            aggregate_mins=5,
        )

        return self._data

    def _run_backtest(
        self,
        params: NetworkParityParams,
        data: DataPipelineResult,
    ) -> BacktestResult:
        """Run backtest with given parameters."""
        engine = BacktestEngine(params, self.config.backtesting)
        return engine.run(data, data.symbols_loaded)

    def _run_per_symbol_backtests(
        self,
        params: NetworkParityParams,
        data: DataPipelineResult,
    ) -> dict[str, BacktestResult]:
        """Run backtest for each symbol individually."""
        results = {}
        for symbol in data.symbols_loaded:
            result = run_single_symbol_backtest(
                data,
                symbol,
                params,
                self.config.backtesting,
            )
            results[symbol] = result
        return results

    async def _get_nemo_suggestions(
        self,
        params: dict[str, Any],
        result: BacktestResult,
    ) -> list[NemoSuggestion]:
        """Get parameter suggestions from Nemo."""
        if not self.nemo.is_available:
            logger.warning("Nemo not available, using random perturbation")
            return []

        response = await self.nemo.get_suggestions_async(
            params,
            result,
            self.state.get_history_summary(),
        )

        if response.success:
            return response.suggestions
        else:
            logger.warning(f"Nemo error: {response.error}")
            return []

    def _apply_suggestions(
        self,
        params: dict[str, Any],
        suggestions: list[NemoSuggestion],
    ) -> dict[str, Any]:
        """Apply Nemo suggestions or random perturbation."""
        if suggestions:
            return self.nemo.apply_suggestions(params, suggestions)
        else:
            # Fallback to random perturbation
            return generate_random_perturbation(
                params,
                scale=0.1,
                seed=self.config.random_seed + self.state.iteration,
            )

    async def run_async(self) -> OptimizationResult:
        """
        Run the optimization loop asynchronously.

        Returns:
            OptimizationResult with final state
        """
        import time
        start_time = time.time()

        logger.info("Starting Network Parity optimization...")
        logger.info(f"Config ID: {self.config.config_id}")

        # Save configuration
        config_path = self.config.save()
        logger.info(f"Configuration saved: {config_path}")

        # Load data
        data = self._load_data()
        if data.n_symbols < 2:
            raise ValueError(f"Need at least 2 symbols, got {data.n_symbols}")

        logger.info(f"Loaded {data.n_symbols} symbols, {data.total_bars:,} total bars")

        # Initialize state with baseline parameters
        self.state.current_params = self.config.baseline_params.to_dict()

        # Run baseline
        logger.info("Running baseline backtest...")
        baseline_params = NetworkParityParams.from_dict(self.state.current_params)
        baseline_result = self._run_backtest(baseline_params, data)
        baseline_score = baseline_result.compute_score()

        logger.info(f"Baseline: Return={baseline_result.total_return:.2%}, "
                   f"Sharpe={baseline_result.sharpe_ratio:.2f}, "
                   f"Score={baseline_score:.4f}")

        # Save baseline result
        self.reporter.save_iteration(
            iteration=0,
            params=self.state.current_params,
            result=baseline_result,
            suggestions=[],
            is_baseline=True,
        )

        # Run per-symbol backtests for baseline
        per_symbol_results = self._run_per_symbol_backtests(baseline_params, data)
        self.reporter.save_per_symbol_results(0, per_symbol_results)

        # Initialize best
        self.state.best_params = self.state.current_params.copy()
        self.state.best_score = baseline_score
        self.state.best_result = baseline_result

        baseline_for_result = baseline_result

        # Optimization loop
        for iteration in range(1, self.config.hyperparams.max_iterations + 1):
            self.state.iteration = iteration

            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration}")
            logger.info(f"{'='*60}")

            # Get Nemo suggestions
            suggestions = await self._get_nemo_suggestions(
                self.state.current_params,
                self.state.best_result or baseline_result,
            )

            # Apply suggestions
            self.state.current_params = self._apply_suggestions(
                self.state.current_params,
                suggestions,
            )

            # Run backtest
            current_params = NetworkParityParams.from_dict(self.state.current_params)
            result = self._run_backtest(current_params, data)
            score = result.compute_score()

            # Record iteration
            iter_result = IterationResult(
                iteration=iteration,
                params=self.state.current_params.copy(),
                backtest_result=result,
                score=score,
                nemo_suggestions=suggestions,
            )

            # Update state
            self.state.update(
                iter_result,
                self.config.hyperparams.early_stopping_patience,
                self.config.hyperparams.convergence_threshold,
            )

            # Save iteration report
            self.reporter.save_iteration(
                iteration=iteration,
                params=self.state.current_params,
                result=result,
                suggestions=suggestions,
            )

            # Save per-symbol results
            per_symbol_results = self._run_per_symbol_backtests(current_params, data)
            self.reporter.save_per_symbol_results(iteration, per_symbol_results)

            # Check convergence
            if self.state.converged:
                logger.info(f"Converged: {self.state.convergence_reason}")
                break

            # If score improved, use these params as base for next iteration
            if score > self.state.best_score:
                pass  # Already updated in state.update()

        # Final result
        total_time = time.time() - start_time

        opt_result = OptimizationResult(
            config=self.config,
            state=self.state,
            baseline_result=baseline_for_result,
            final_result=self.state.best_result,
            total_time_seconds=total_time,
        )

        # Save summary
        self.reporter.save_summary(opt_result)

        logger.info("\n" + opt_result.summary())

        return opt_result

    def run(self) -> OptimizationResult:
        """
        Run the optimization loop (sync wrapper).

        Returns:
            OptimizationResult with final state
        """
        return asyncio.run(self.run_async())


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_DIR / "optimization.log"),
        ],
    )

    # Run optimization
    config = OptimizationConfig()
    controller = OptimizationController(config)

    try:
        result = controller.run()
        print("\nOptimization complete!")
        print(result.summary())
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        sys.exit(1)
