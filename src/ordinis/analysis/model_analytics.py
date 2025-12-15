"""Model performance analytics for continuous improvement."""

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


@dataclass
class ModelPerformanceRecord:
    """Performance record for a single signal."""

    model_id: str
    signal_timestamp: datetime
    signal_score: float
    signal_direction: str  # 'long', 'short', 'neutral'
    entry_price: float
    exit_price: float
    exit_timestamp: datetime
    pnl: float
    pnl_pct: float
    holding_days: int
    information_coefficient: float = 0.0  # Calculated later


@dataclass
class ModelMetrics:
    """Aggregated metrics for a model.

    Attributes:
        model_id: Model identifier
        period_start: Start of measurement period
        period_end: End of measurement period

        # Returns
        total_return: Total return of trades
        avg_return: Average return per trade
        sharpe_ratio: Risk-adjusted return (Sharpe)

        # Accuracy
        hit_rate: % of profitable trades
        win_count: Number of winning trades
        loss_count: Number of losing trades

        # IC and decay
        ic_score: Information Coefficient
        ic_decay_halflife: Days until IC decays 50%

        # Consistency
        win_streak: Current win streak
        consistency_score: How consistent is the model

        # Volume and frequency
        signals_generated: Number of signals
        avg_signal_age: Average age of signals before they decay
    """

    model_id: str
    period_start: datetime
    period_end: datetime

    # Returns
    total_return: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0

    # Accuracy
    hit_rate: float = 0.0
    win_count: int = 0
    loss_count: int = 0

    # IC and decay
    ic_score: float = 0.0
    ic_decay_halflife: float = 0.0

    # Consistency
    win_streak: int = 0
    consistency_score: float = 0.0

    # Volume
    signals_generated: int = 0
    avg_signal_age: float = 0.0


class ModelPerformanceAnalyzer:
    """Analyzes model performance for IC, hit rate, and decay."""

    def __init__(self):
        """Initialize analyzer."""
        self.records: list[ModelPerformanceRecord] = []

    def add_record(self, record: ModelPerformanceRecord):
        """Add a performance record.

        Args:
            record: Performance record for a signal
        """
        self.records.append(record)

    def compute_hit_rate(self, model_id: str | None = None) -> tuple[float, int, int]:
        """Compute hit rate for a model.

        Args:
            model_id: Filter to specific model (None = all)

        Returns:
            (hit_rate, wins, losses)
        """
        records = self._filter_records(model_id)

        if not records:
            return 0.0, 0, 0

        winners = sum(1 for r in records if r.pnl > 0)
        losers = sum(1 for r in records if r.pnl < 0)

        hit_rate = winners / len(records) if records else 0.0

        return hit_rate, winners, losers

    def compute_ic(
        self,
        model_id: str | None = None,
        lookback_days: int = 30,
    ) -> tuple[float, float, float]:
        """Compute Information Coefficient (Spearman correlation).

        IC measures correlation between signal strength and actual returns.

        Args:
            model_id: Filter to specific model
            lookback_days: Lookback period

        Returns:
            (ic_mean, ic_std, decay_halflife_days)
        """
        records = self._filter_records(model_id)

        if not records:
            return 0.0, 0.0, 0.0

        # Filter by lookback period
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent = [r for r in records if r.signal_timestamp >= cutoff]

        if len(recent) < 2:
            return 0.0, 0.0, 0.0

        # IC = Spearman correlation between signal and returns
        signals = np.array([r.signal_score for r in recent])
        returns = np.array([r.pnl_pct for r in recent])

        # Simple Pearson correlation (would use Spearman in production)
        if len(signals) > 1 and signals.std() > 0 and returns.std() > 0:
            ic = np.corrcoef(signals, returns)[0, 1]
            ic = np.nan_to_num(ic, nan=0.0)
        else:
            ic = 0.0

        # IC decay: how long does signal remain predictive?
        # Group by signal age and compute IC for each cohort
        ages_ic = {}
        for record in recent:
            age_days = (record.exit_timestamp - record.signal_timestamp).days
            if age_days not in ages_ic:
                ages_ic[age_days] = []
            ages_ic[age_days].append(record.pnl_pct)

        # Estimate halflife (simplified: age at which median IC is 50% of initial)
        initial_ic = ic
        halflife = 0.0

        if initial_ic != 0 and len(ages_ic) > 1:
            for age in sorted(ages_ic.keys()):
                age_returns = ages_ic[age]
                age_ic = np.mean(age_returns) / initial_ic if initial_ic != 0 else 0
                if abs(age_ic) < 0.5 * abs(initial_ic):
                    halflife = age
                    break

        return float(ic), 0.0, float(halflife)  # std=0 for now

    def compute_sharpe_ratio(
        self,
        model_id: str | None = None,
        risk_free_rate: float = 0.02,
    ) -> float:
        """Compute Sharpe ratio.

        Args:
            model_id: Filter to specific model
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        records = self._filter_records(model_id)

        if not records:
            return 0.0

        returns = np.array([r.pnl_pct for r in records])

        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        # Assume daily returns, annualize
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)

        return float(sharpe)

    def compute_consistency(self, model_id: str | None = None) -> float:
        """Compute consistency score (fraction of positive return periods).

        Args:
            model_id: Filter to specific model

        Returns:
            Consistency score 0-1
        """
        records = self._filter_records(model_id)

        if not records:
            return 0.0

        # Group records by day and check if positive
        daily_returns: dict[str, float] = {}

        for record in records:
            day = record.exit_timestamp.date()
            if day not in daily_returns:
                daily_returns[day] = 0.0
            daily_returns[day] += record.pnl

        positive_days = sum(1 for ret in daily_returns.values() if ret > 0)

        consistency = positive_days / len(daily_returns) if daily_returns else 0.0

        return float(consistency)

    def generate_model_report(
        self,
        model_id: str,
        lookback_days: int = 30,
    ) -> ModelMetrics:
        """Generate comprehensive report for a model.

        Args:
            model_id: Model identifier
            lookback_days: Lookback period

        Returns:
            ModelMetrics
        """
        records = self._filter_records(model_id, lookback_days)

        if not records:
            return ModelMetrics(
                model_id=model_id,
                period_start=datetime.now(),
                period_end=datetime.now(),
            )

        # Compute metrics
        hit_rate, wins, losses = self.compute_hit_rate(model_id)
        total_return = sum(r.pnl for r in records)
        avg_return = total_return / len(records) if records else 0.0
        sharpe = self.compute_sharpe_ratio(model_id)
        ic, ic_std, ic_decay = self.compute_ic(model_id, lookback_days)
        consistency = self.compute_consistency(model_id)

        # Win streak
        win_streak = 0
        for r in reversed(records):
            if r.pnl > 0:
                win_streak += 1
            else:
                break

        # Signal age
        ages = [(r.exit_timestamp - r.signal_timestamp).days for r in records]
        avg_age = np.mean(ages) if ages else 0.0

        return ModelMetrics(
            model_id=model_id,
            period_start=min(r.signal_timestamp for r in records),
            period_end=max(r.exit_timestamp for r in records),
            total_return=float(total_return),
            avg_return=float(avg_return),
            sharpe_ratio=float(sharpe),
            hit_rate=float(hit_rate),
            win_count=wins,
            loss_count=losses,
            ic_score=float(ic),
            ic_decay_halflife=float(ic_decay),
            consistency_score=float(consistency),
            win_streak=win_streak,
            signals_generated=len(records),
            avg_signal_age=float(avg_age),
        )

    def get_all_model_rankings(
        self,
        metric: str = "ic_score",
        lookback_days: int = 30,
    ) -> list[tuple[str, float]]:
        """Rank models by specific metric.

        Args:
            metric: Metric to rank by (ic_score, hit_rate, sharpe_ratio, etc.)
            lookback_days: Lookback period

        Returns:
            List of (model_id, metric_value) sorted desc
        """
        models = set(r.model_id for r in self.records)

        rankings = []

        for model_id in models:
            report = self.generate_model_report(model_id, lookback_days)
            value = getattr(report, metric, 0.0)
            rankings.append((model_id, value))

        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings

    def _filter_records(
        self,
        model_id: str | None = None,
        lookback_days: int = 30,
    ) -> list[ModelPerformanceRecord]:
        """Filter records by model and lookback."""
        records = self.records

        if model_id:
            records = [r for r in records if r.model_id == model_id]

        if lookback_days:
            cutoff = datetime.now() - timedelta(days=lookback_days)
            records = [r for r in records if r.signal_timestamp >= cutoff]

        return records

    def to_dataframe(self, model_id: str | None = None) -> pd.DataFrame:
        """Export records to DataFrame.

        Args:
            model_id: Filter to specific model

        Returns:
            DataFrame with records
        """
        records = self._filter_records(model_id, lookback_days=None)

        data = []

        for r in records:
            data.append(
                {
                    "model_id": r.model_id,
                    "signal_timestamp": r.signal_timestamp,
                    "signal_score": r.signal_score,
                    "signal_direction": r.signal_direction,
                    "entry_price": r.entry_price,
                    "exit_price": r.exit_price,
                    "pnl": r.pnl,
                    "pnl_pct": r.pnl_pct,
                    "holding_days": r.holding_days,
                    "ic": r.information_coefficient,
                }
            )

        return pd.DataFrame(data)
