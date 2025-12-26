#!/usr/bin/env python3
"""
JSON Reporting and Traceability for Network Parity Optimization

Generates comprehensive JSON reports for audit trails and reproducibility.

Author: Ordinis Quantitative Research
Version: 1.0.0
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from config import OptimizationConfig, OUTPUT_DIR

if TYPE_CHECKING:
    from backtesting import BacktestResult
    from nemo_integration import NemoSuggestion
    from optimization import OptimizationResult

logger = logging.getLogger(__name__)


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """
    Generates JSON reports with full traceability.

    Creates:
    - Configuration JSON (full config snapshot)
    - Iteration sequence JSONs (per-iteration results)
    - Per-symbol result JSONs (individual symbol backtests)
    - Summary JSON (final optimization results)
    """

    SCHEMA_VERSION = "1.0"

    def __init__(self, config: OptimizationConfig):
        """
        Initialize report generator.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.output_dir = OUTPUT_DIR

        # Create directory structure
        self.configs_dir = self.output_dir / "configs"
        self.iterations_dir = self.output_dir / "iterations"
        self.baseline_dir = self.output_dir / "baseline"
        self.summary_dir = self.output_dir / "summary"

        for d in [self.configs_dir, self.iterations_dir, self.baseline_dir, self.summary_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _generate_sequence_id(self, iteration: int) -> str:
        """Generate unique sequence ID for an iteration."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        hash_input = f"{self.config.config_id}_{iteration}_{timestamp}"
        hash_suffix = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        return f"seq_{timestamp}_{hash_suffix}"

    def save_config(self) -> Path:
        """
        Save configuration JSON.

        Returns:
            Path to saved config file
        """
        path = self.configs_dir / f"{self.config.config_id}.json"
        config_dict = self.config.to_dict()

        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"Saved configuration: {path}")
        return path

    def save_iteration(
        self,
        iteration: int,
        params: dict[str, Any],
        result: "BacktestResult",
        suggestions: list["NemoSuggestion"],
        is_baseline: bool = False,
    ) -> Path:
        """
        Save iteration sequence JSON.

        Args:
            iteration: Iteration number
            params: Parameters used in this iteration
            result: Backtest result
            suggestions: Nemo suggestions received
            is_baseline: Whether this is the baseline iteration

        Returns:
            Path to saved sequence file
        """
        sequence_id = self._generate_sequence_id(iteration)

        # Determine previous score for improvement calculation
        improvement = 0.0
        if iteration > 0:
            # Load previous iteration to calculate improvement
            prev_path = self.iterations_dir / f"Iteration_{iteration-1}" / "sequence.json"
            if prev_path.exists():
                with open(prev_path) as f:
                    prev_data = json.load(f)
                    prev_score = prev_data.get("score", 0)
                    current_score = result.compute_score()
                    improvement = current_score - prev_score

        sequence_data = {
            "schema_version": self.SCHEMA_VERSION,
            "sequence_id": sequence_id,
            "configuration_id": self.config.config_id,
            "iteration_number": iteration,
            "timestamp": datetime.now().isoformat() + "Z",
            "is_baseline": is_baseline,

            "tested_parameters": params,

            "performance_metrics": {
                "total_return": result.total_return,
                "annualized_return": result.annualized_return,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "max_drawdown": result.max_drawdown,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "num_trades": result.num_trades,
                "avg_trade_pnl": result.avg_trade_pnl,
                "avg_trade_pnl_pct": result.avg_trade_pnl_pct,
            },

            "network_metrics": {
                "avg_centrality": result.avg_centrality,
                "network_density": result.network_density,
                "rebalance_count": result.rebalance_count,
            },

            "backtest_results": {
                "period": f"{result.params.get('start_date', 'N/A')} to {result.params.get('end_date', 'N/A')}",
                "symbols_traded": len(result.trades),
                "equity_curve_sample": result.equity_curve[:100] if result.equity_curve else [],
                "trades_sample": [t.to_dict() for t in result.trades[:20]],
            },

            "nemo_suggestions": [s.to_dict() for s in suggestions],

            "score": result.compute_score(),
            "convergence_status": {
                "improvement_vs_previous": improvement,
            },
        }

        # Determine output path
        if is_baseline:
            iter_dir = self.baseline_dir
            filename = f"network_parity_baseline_{self.config.timestamp}.json"
        else:
            iter_dir = self.iterations_dir / f"Iteration_{iteration}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            filename = "sequence.json"

        path = iter_dir / filename

        with open(path, "w") as f:
            json.dump(sequence_data, f, indent=2, default=str)

        logger.info(f"Saved iteration {iteration}: {path}")
        return path

    def save_per_symbol_results(
        self,
        iteration: int,
        results: dict[str, "BacktestResult"],
    ) -> list[Path]:
        """
        Save per-symbol backtest results.

        Args:
            iteration: Iteration number
            results: Symbol -> BacktestResult mapping

        Returns:
            List of paths to saved files
        """
        if iteration == 0:
            iter_dir = self.baseline_dir / "per_symbol"
        else:
            iter_dir = self.iterations_dir / f"Iteration_{iteration}" / "per_symbol"

        iter_dir.mkdir(parents=True, exist_ok=True)
        paths = []

        for symbol, result in results.items():
            symbol_data = {
                "schema_version": self.SCHEMA_VERSION,
                "configuration_id": self.config.config_id,
                "iteration_number": iteration,
                "symbol": symbol,
                "timestamp": datetime.now().isoformat() + "Z",

                "params": result.params,

                "metrics": {
                    "total_return": result.total_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "sortino_ratio": result.sortino_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "num_trades": result.num_trades,
                    "avg_trade_pnl": result.avg_trade_pnl,
                },

                "trades": [t.to_dict() for t in result.trades],
                "equity_curve": result.equity_curve[:200] if result.equity_curve else [],

                "score": result.compute_score(),
            }

            path = iter_dir / f"{symbol}.json"
            with open(path, "w") as f:
                json.dump(symbol_data, f, indent=2, default=str)

            paths.append(path)

        logger.info(f"Saved {len(paths)} per-symbol results for iteration {iteration}")
        return paths

    def save_summary(
        self,
        opt_result: "OptimizationResult",
    ) -> Path:
        """
        Save optimization summary JSON.

        Args:
            opt_result: Final optimization result

        Returns:
            Path to saved summary file
        """
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Prepare per-symbol summary
        symbol_summary = {}
        if opt_result.state.best_result:
            # Would need to access per-symbol results from files
            # For now, leave empty or load from saved files
            pass

        summary_data = {
            "schema_version": self.SCHEMA_VERSION,
            "summary_id": f"sum_{timestamp}",
            "configuration_id": self.config.config_id,
            "timestamp": datetime.now().isoformat() + "Z",

            "optimization_summary": {
                "total_iterations": opt_result.state.iteration,
                "converged": opt_result.state.converged,
                "convergence_reason": opt_result.state.convergence_reason,
                "total_time_seconds": opt_result.total_time_seconds,
            },

            "baseline_performance": {
                "total_return": opt_result.baseline_result.total_return if opt_result.baseline_result else 0,
                "sharpe_ratio": opt_result.baseline_result.sharpe_ratio if opt_result.baseline_result else 0,
                "sortino_ratio": opt_result.baseline_result.sortino_ratio if opt_result.baseline_result else 0,
                "max_drawdown": opt_result.baseline_result.max_drawdown if opt_result.baseline_result else 0,
                "score": opt_result.baseline_result.compute_score() if opt_result.baseline_result else 0,
            },

            "optimized_performance": {
                "total_return": opt_result.final_result.total_return if opt_result.final_result else 0,
                "sharpe_ratio": opt_result.final_result.sharpe_ratio if opt_result.final_result else 0,
                "sortino_ratio": opt_result.final_result.sortino_ratio if opt_result.final_result else 0,
                "max_drawdown": opt_result.final_result.max_drawdown if opt_result.final_result else 0,
                "score": opt_result.final_result.compute_score() if opt_result.final_result else 0,
            },

            "improvement": {
                "score_delta": opt_result.improvement_vs_baseline,
                "return_delta": (
                    (opt_result.final_result.total_return if opt_result.final_result else 0)
                    - (opt_result.baseline_result.total_return if opt_result.baseline_result else 0)
                ),
            },

            "best_parameters": opt_result.state.best_params,

            "iteration_history": [
                {
                    "iteration": r.iteration,
                    "score": r.score,
                    "return": r.backtest_result.total_return,
                    "sharpe": r.backtest_result.sharpe_ratio,
                    "sortino": r.backtest_result.sortino_ratio,
                }
                for r in opt_result.state.history
            ],

            "configuration_reference": {
                "config_file": f"configs/{self.config.config_id}.json",
                "iterations_dir": "iterations/",
                "baseline_dir": "baseline/",
            },

            "reproducibility": {
                "random_seed": self.config.random_seed,
                "gpu_available": True,  # Would detect at runtime
                "python_version": "3.11+",
                "schema_version": self.SCHEMA_VERSION,
            },
        }

        path = self.summary_dir / f"optimization_summary_{timestamp}.json"

        with open(path, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)

        logger.info(f"Saved optimization summary: {path}")
        return path

    def generate_report_index(self) -> Path:
        """
        Generate an index file listing all reports.

        Returns:
            Path to index file
        """
        index = {
            "schema_version": self.SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat() + "Z",
            "configuration_id": self.config.config_id,
            "output_directory": str(self.output_dir),

            "files": {
                "configs": sorted([f.name for f in self.configs_dir.glob("*.json")]),
                "iterations": sorted([d.name for d in self.iterations_dir.iterdir() if d.is_dir()]),
                "baseline": sorted([f.name for f in self.baseline_dir.glob("*.json")]),
                "summary": sorted([f.name for f in self.summary_dir.glob("*.json")]),
            },

            "traceability": {
                "config_to_iterations": f"configs/{self.config.config_id}.json -> iterations/Iteration_*/sequence.json",
                "iterations_to_symbols": "iterations/Iteration_*/sequence.json -> iterations/Iteration_*/per_symbol/*.json",
                "summary_references": "summary/optimization_summary_*.json contains configuration_reference",
            },
        }

        path = self.output_dir / "report_index.json"

        with open(path, "w") as f:
            json.dump(index, f, indent=2)

        logger.info(f"Generated report index: {path}")
        return path


def load_iteration_report(path: Path) -> dict[str, Any]:
    """
    Load an iteration report from file.

    Args:
        path: Path to sequence.json file

    Returns:
        Report data as dictionary
    """
    with open(path) as f:
        return json.load(f)


def load_summary_report(path: Path) -> dict[str, Any]:
    """
    Load optimization summary from file.

    Args:
        path: Path to summary JSON file

    Returns:
        Summary data as dictionary
    """
    with open(path) as f:
        return json.load(f)


def verify_traceability(output_dir: Path) -> tuple[bool, list[str]]:
    """
    Verify traceability links in reports.

    Args:
        output_dir: Output directory to check

    Returns:
        (is_valid, list_of_issues)
    """
    issues = []

    # Check for config file
    configs = list((output_dir / "configs").glob("*.json"))
    if not configs:
        issues.append("No configuration files found")
        return False, issues

    config_id = configs[0].stem

    # Check iterations reference config
    for iter_dir in (output_dir / "iterations").iterdir():
        if iter_dir.is_dir():
            seq_path = iter_dir / "sequence.json"
            if seq_path.exists():
                with open(seq_path) as f:
                    data = json.load(f)
                    if data.get("configuration_id") != config_id:
                        issues.append(f"Config ID mismatch in {seq_path}")

    # Check summary references
    for summary in (output_dir / "summary").glob("*.json"):
        with open(summary) as f:
            data = json.load(f)
            if data.get("configuration_id") != config_id:
                issues.append(f"Config ID mismatch in {summary}")

    return len(issues) == 0, issues


if __name__ == "__main__":
    from config import OptimizationConfig
    from backtesting import BacktestResult

    # Test report generation
    config = OptimizationConfig()
    reporter = ReportGenerator(config)

    # Save config
    config_path = reporter.save_config()
    print(f"Config saved: {config_path}")

    # Create mock result
    result = BacktestResult(
        total_return=0.15,
        sharpe_ratio=1.5,
        sortino_ratio=2.2,
        max_drawdown=0.08,
        win_rate=0.58,
        profit_factor=1.8,
        num_trades=42,
    )

    # Save baseline
    baseline_path = reporter.save_iteration(
        iteration=0,
        params=config.baseline_params.to_dict(),
        result=result,
        suggestions=[],
        is_baseline=True,
    )
    print(f"Baseline saved: {baseline_path}")

    # Generate index
    index_path = reporter.generate_report_index()
    print(f"Index saved: {index_path}")

    # Verify traceability
    is_valid, issues = verify_traceability(OUTPUT_DIR)
    print(f"\nTraceability valid: {is_valid}")
    if issues:
        print(f"Issues: {issues}")
