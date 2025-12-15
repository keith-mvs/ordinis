"""
Walk-Forward Validation Framework - Robust Out-of-Sample Testing
This script implements walk-forward analysis for backtesting studies.

Usage:
    python walkforward_validation.py --training-period 3 --test-period 6 --data price_data.csv
"""

from dataclasses import dataclass
from datetime import datetime
import json
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window"""

    window_id: int
    training_start: datetime
    training_end: datetime
    test_start: datetime
    test_end: datetime

    def __str__(self):
        return (
            f"Window {self.window_id}: "
            f"Train {self.training_start.date()}-{self.training_end.date()}, "
            f"Test {self.test_start.date()}-{self.test_end.date()}"
        )


@dataclass
class WindowResults:
    """Results from a single walk-forward window"""

    window_id: int
    training_sharpe: float
    training_return: float
    training_volatility: float
    test_sharpe: float
    test_return: float
    test_volatility: float
    test_max_dd: float
    test_win_rate: float
    optimal_parameters: dict
    consistency_ratio: float  # test_sharpe / training_sharpe

    def to_dict(self):
        return {
            "window_id": self.window_id,
            "training_sharpe": self.training_sharpe,
            "training_return": self.training_return,
            "training_volatility": self.training_volatility,
            "test_sharpe": self.test_sharpe,
            "test_return": self.test_return,
            "test_volatility": self.test_volatility,
            "test_max_dd": self.test_max_dd,
            "test_win_rate": self.test_win_rate,
            "consistency_ratio": self.consistency_ratio,
            "optimal_parameters": self.optimal_parameters,
        }


class WalkForwardAnalysis:
    """Implements walk-forward validation methodology"""

    def __init__(
        self,
        data: pd.DataFrame,  # Time series data (index: dates)
        training_period_years: int = 3,
        test_period_months: int = 6,
        rollforward_months: int = 6,
    ):
        """
        Initialize walk-forward analysis

        Args:
            data: Time series data with dates as index
            training_period_years: Years of data for training window
            test_period_months: Months of data for testing window
            rollforward_months: Months to roll forward between windows
        """
        self.data = data
        self.training_days = training_period_years * 252
        self.test_days = test_period_months * 21
        self.rollforward_days = rollforward_months * 21
        self.windows: list[WalkForwardWindow] = []
        self.results: list[WindowResults] = []

        self._create_windows()

    def _create_windows(self):
        """Create walk-forward windows"""
        dates = self.data.index
        total_observations = len(dates)

        window_id = 1
        position = 0

        while position + self.training_days + self.test_days <= total_observations:
            training_start_idx = position
            training_end_idx = position + self.training_days
            test_start_idx = training_end_idx
            test_end_idx = test_start_idx + self.test_days

            window = WalkForwardWindow(
                window_id=window_id,
                training_start=dates[training_start_idx],
                training_end=dates[training_end_idx - 1],
                test_start=dates[test_start_idx],
                test_end=dates[test_end_idx - 1],
            )

            self.windows.append(window)
            logger.info(f"Created {window}")

            position += self.rollforward_days
            window_id += 1

    def get_training_data(self, window: WalkForwardWindow) -> pd.DataFrame:
        """Extract training data for window"""
        return self.data[window.training_start : window.training_end]

    def get_test_data(self, window: WalkForwardWindow) -> pd.DataFrame:
        """Extract test data for window"""
        return self.data[window.test_start : window.test_end]

    def validate_window_integrity(self, window: WalkForwardWindow) -> bool:
        """Verify no data leakage between train/test"""
        training_data = self.get_training_data(window)
        test_data = self.get_test_data(window)

        # Check for overlaps
        train_dates = set(training_data.index)
        test_dates = set(test_data.index)

        if train_dates & test_dates:
            logger.warning(f"Window {window.window_id} has overlapping dates!")
            return False

        # Check chronological order
        if training_data.index[-1] >= test_data.index[0]:
            logger.warning(f"Window {window.window_id} not properly ordered!")
            return False

        return True

    def calculate_aggregate_metrics(self) -> dict:
        """Calculate summary statistics across all windows"""

        if not self.results:
            logger.warning("No results to aggregate - run analysis first")
            return {}

        test_sharpes = [r.test_sharpe for r in self.results]
        test_returns = [r.test_return for r in self.results]
        test_vols = [r.test_volatility for r in self.results]
        test_dds = [r.test_max_dd for r in self.results]
        consistency_ratios = [r.consistency_ratio for r in self.results]

        return {
            "num_windows": len(self.results),
            "avg_test_sharpe": np.mean(test_sharpes),
            "std_test_sharpe": np.std(test_sharpes),
            "min_test_sharpe": np.min(test_sharpes),
            "max_test_sharpe": np.max(test_sharpes),
            "avg_test_return": np.mean(test_returns),
            "avg_test_volatility": np.mean(test_vols),
            "avg_test_max_dd": np.mean(test_dds),
            "avg_consistency": np.mean(consistency_ratios),
            "sharpe_cv": np.std(test_sharpes) / np.mean(test_sharpes)
            if np.mean(test_sharpes) > 0
            else 0,
        }

    def generate_report(self) -> str:
        """Generate text report of walk-forward analysis"""

        aggregate = self.calculate_aggregate_metrics()

        report = []
        report.append("=" * 80)
        report.append("WALK-FORWARD ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        report.append("WINDOW CONFIGURATION:")
        report.append(f"  Training period: {self.training_days // 252} years")
        report.append(f"  Test period: {self.test_days // 21} months")
        report.append(f"  Roll-forward: {self.rollforward_days // 21} months")
        report.append(f"  Number of windows: {aggregate['num_windows']}")
        report.append("")

        report.append("OUT-OF-SAMPLE PERFORMANCE (Most Important):")
        report.append(f"  Average Sharpe Ratio: {aggregate['avg_test_sharpe']:.2f}")
        report.append(f"  Sharpe Std Dev: {aggregate['std_test_sharpe']:.2f}")
        report.append(
            f"  Sharpe Range: [{aggregate['min_test_sharpe']:.2f}, {aggregate['max_test_sharpe']:.2f}]"
        )
        report.append(f"  Coefficient of Variation: {aggregate['sharpe_cv']:.1%}")
        report.append("")

        report.append("RETURN METRICS (Out-of-Sample):")
        report.append(f"  Average Annual Return: {aggregate['avg_test_return']:.2f}%")
        report.append(f"  Average Volatility: {aggregate['avg_test_volatility']:.2f}%")
        report.append(f"  Average Max Drawdown: {aggregate['avg_test_max_dd']:.2f}%")
        report.append("")

        report.append("OVERFITTING ASSESSMENT:")
        report.append(f"  Average Consistency Ratio: {aggregate['avg_consistency']:.1%}")
        if aggregate["avg_consistency"] > 0.8:
            report.append("  ✓ Good generalization (ratio > 80%)")
        elif aggregate["avg_consistency"] > 0.6:
            report.append("  ⚠ Moderate overfitting (ratio 60-80%)")
        else:
            report.append("  ✗ Potential overfitting (ratio < 60%)")
        report.append("")

        report.append("WINDOW-BY-WINDOW RESULTS:")
        report.append("-" * 80)
        report.append(
            f"{'Win':>4} {'Train SR':>10} {'Test SR':>10} {'Consist':>10} {'Ret %':>10} {'DD %':>10}"
        )
        report.append("-" * 80)

        for r in self.results:
            report.append(
                f"{r.window_id:>4} {r.training_sharpe:>10.2f} {r.test_sharpe:>10.2f} "
                f"{r.consistency_ratio:>10.1%} {r.test_return:>10.2f} {r.test_max_dd:>10.2f}"
            )

        report.append("-" * 80)
        report.append("")

        report.append("VALIDATION CHECKLIST:")
        report.append(
            f"  [ ] Out-of-sample Sharpe > 1.0: "
            f"{'✓' if aggregate['avg_test_sharpe'] > 1.0 else '✗'}"
        )
        report.append(
            f"  [ ] Consistency ratio > 80%: "
            f"{'✓' if aggregate['avg_consistency'] > 0.8 else '✗'}"
        )
        report.append(f"  [ ] Sharpe CV < 20%: " f"{'✓' if aggregate['sharpe_cv'] < 0.2 else '✗'}")
        report.append(
            f"  [ ] Returns positive: " f"{'✓' if aggregate['avg_test_return'] > 0 else '✗'}"
        )
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def export_results(self, filepath: str):
        """Export results to JSON"""

        data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "num_windows": len(self.results),
                "training_period_years": self.training_days // 252,
                "test_period_months": self.test_days // 21,
            },
            "aggregate_metrics": self.calculate_aggregate_metrics(),
            "window_results": [r.to_dict() for r in self.results],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results exported to {filepath}")


class ParameterOptimizer:
    """Optimizes parameters on training data (in-sample)"""

    @staticmethod
    def optimize_weights(
        returns_data: pd.DataFrame, optimization_criterion: str = "sharpe"
    ) -> dict[str, float]:
        """
        Optimize signal weights on training data

        Args:
            returns_data: DataFrame with returns for each signal
            optimization_criterion: 'sharpe', 'sortino', 'cagr'

        Returns:
            Optimal weights dictionary
        """

        # Simplified: equal weighting as baseline
        # In production: use scipy.optimize.minimize for complex optimization

        signals = returns_data.columns
        num_signals = len(signals)
        weights = {signal: 1.0 / num_signals for signal in signals}

        logger.info(f"Optimized weights (using {optimization_criterion}): {weights}")

        return weights

    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.03) -> float:
        """Calculate Sharpe ratio"""
        annual_return = np.mean(returns) * 252
        annual_vol = np.std(returns) * np.sqrt(252)

        if annual_vol == 0:
            return 0

        return (annual_return - risk_free_rate) / annual_vol


def main():
    """Example usage"""
    logger.info("Walk-Forward Validation Framework loaded")
    logger.info("Use this module to validate backtesting strategies")

    # Example: Create synthetic data
    dates = pd.date_range("2014-01-01", "2024-12-31", freq="B")  # Business days
    data = pd.DataFrame(np.random.randn(len(dates), 1), index=dates, columns=["returns"])

    # Initialize walk-forward
    wfa = WalkForwardAnalysis(
        data=data, training_period_years=3, test_period_months=6, rollforward_months=6
    )

    logger.info(f"Created {len(wfa.windows)} walk-forward windows")
    logger.info(f"First window: {wfa.windows[0]}")
    logger.info(f"Last window: {wfa.windows[-1]}")


if __name__ == "__main__":
    main()
