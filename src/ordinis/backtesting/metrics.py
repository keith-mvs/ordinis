"""Extended metrics for backtesting including IC, hit rate, decay."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ordinis.engines.proofbench.analytics.performance import PerformanceMetrics


@dataclass
class BacktestMetrics(PerformanceMetrics):
    """Extended metrics for backtesting with per-model analytics.

    Adds Information Coefficient, hit rate, decay curves, and model comparisons.
    """

    # Per-model metrics
    model_metrics: dict[str, dict] = field(default_factory=dict)

    # IC and hit rate
    ic_mean: float = 0.0  # Information Coefficient (mean)
    ic_std: float = 0.0  # IC standard deviation
    ic_pvalue: float = 1.0  # IC statistical significance
    hit_rate: float = 0.0  # % of signals that were profitable
    false_positive_rate: float = 0.0  # % of signals that lost money

    # Turnover and transaction costs
    avg_turnover: float = 0.0  # Average portfolio turnover per period
    total_transaction_costs: float = 0.0  # Total fees/slippage paid

    # Decay metrics
    signal_decay_halflife: float = 0.0  # Days until signal value decays 50%
    avg_signal_age: float = 0.0  # Average age of active signals


@dataclass
class ModelPerformance:
    """Per-model performance tracking."""

    model_id: str
    signals_generated: int = 0
    profitable_signals: int = 0
    avg_return: float = 0.0
    ic_score: float = 0.0
    consistency: float = 0.0  # Fraction of trades that were positive


class MetricsEngine:
    """Computes extended metrics for backtesting."""

    @staticmethod
    def compute_ic(
        predictions: pd.Series,
        returns: pd.Series,
    ) -> tuple[float, float, float]:
        """Compute Information Coefficient (Spearman correlation).

        Args:
            predictions: Model predictions (scores)
            returns: Realized returns

        Returns:
            (ic_mean, ic_std, p_value)
        """
        from scipy.stats import spearmanr

        valid_mask = ~(predictions.isna() | returns.isna())
        pred_clean = predictions[valid_mask]
        ret_clean = returns[valid_mask]

        if len(pred_clean) < 2:
            return 0.0, 0.0, 1.0

        ic, pvalue = spearmanr(pred_clean, ret_clean)
        return float(ic), 0.0, float(pvalue)

    @staticmethod
    def compute_hit_rate(signals: list, outcomes: list) -> tuple[float, float]:
        """Compute hit rate and false positive rate.

        Args:
            signals: List of signals (boolean or direction)
            outcomes: List of outcomes (True=profitable)

        Returns:
            (hit_rate, false_positive_rate)
        """
        if not signals:
            return 0.0, 0.0

        signal_array = np.array([1 if s else -1 for s in signals])
        outcome_array = np.array([1 if o else -1 for o in outcomes])

        # Hit rate = signals that predicted correctly
        correct = (signal_array * outcome_array) > 0
        hit_rate = float(correct.mean()) if len(correct) > 0 else 0.0

        # False positive = signals that were wrong
        fp_rate = 1.0 - hit_rate

        return hit_rate, fp_rate

    @staticmethod
    def compute_turnover(
        positions_history: list[dict],
        prices: pd.Series,
    ) -> float:
        """Compute average portfolio turnover.

        Args:
            positions_history: List of position snapshots over time
            prices: Current prices

        Returns:
            Average turnover as fraction of portfolio
        """
        if len(positions_history) < 2:
            return 0.0

        turnovers = []

        for i in range(1, len(positions_history)):
            prev = positions_history[i - 1]
            curr = positions_history[i]

            # Sum absolute changes in each position
            symbols = set(prev.keys()) | set(curr.keys())
            total_change = 0.0
            total_value = 0.0

            for symbol in symbols:
                prev_qty = prev.get(symbol, 0.0)
                curr_qty = curr.get(symbol, 0.0)
                price = prices.get(symbol, 1.0)

                change = abs(curr_qty - prev_qty) * price
                total_change += change

                if symbol in curr:
                    total_value += curr_qty * price

            if total_value > 0:
                turnovers.append(total_change / total_value)

        return float(np.mean(turnovers)) if turnovers else 0.0

    @staticmethod
    def compute_signal_decay(
        signal_ages: list[float],
    ) -> tuple[float, float]:
        """Compute signal decay halflife.

        Args:
            signal_ages: Ages of signals in days

        Returns:
            (halflife_days, avg_age_days)
        """
        if not signal_ages:
            return 0.0, 0.0

        # Assume exponential decay; fit to find halflife
        # For now, just return percentiles
        signal_ages_arr = np.array(signal_ages)

        # Halflife is where 50% of signal value remains
        # Approximate as the age where 50% of signals are younger
        halflife = float(np.percentile(signal_ages_arr, 50))
        avg_age = float(np.mean(signal_ages_arr))

        return halflife, avg_age

    @staticmethod
    def aggregate_model_metrics(
        model_signals: dict[str, list],
        trades_by_model: dict[str, list],
    ) -> dict[str, dict]:
        """Aggregate performance by model.

        Args:
            model_signals: Dict[model_id] -> signals
            trades_by_model: Dict[model_id] -> trades

        Returns:
            Dict[model_id] -> metrics
        """
        metrics = {}

        for model_id, signals in model_signals.items():
            trades = trades_by_model.get(model_id, [])

            # Count profitable trades
            winners = sum(1 for t in trades if t.get("pnl", 0) > 0)
            hit_rate = winners / len(trades) if trades else 0.0

            # Average return per signal
            avg_return = sum(t.get("pnl_pct", 0) for t in trades) / len(trades) if trades else 0.0

            metrics[model_id] = {
                "signals_generated": len(signals),
                "profitable_trades": winners,
                "hit_rate": hit_rate,
                "avg_return": avg_return,
                "total_return": sum(t.get("pnl", 0) for t in trades),
            }

        return metrics
