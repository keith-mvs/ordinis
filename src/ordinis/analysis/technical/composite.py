"""
Composite indicator aggregation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CompositeResult:
    """Result of a composite aggregation."""

    value: float | str
    method: str


class CompositeIndicator:
    """Combine multiple indicator outputs into a single composite score/signal."""

    @staticmethod
    def weighted_sum(
        values: dict[str, float], weights: dict[str, float] | None = None, normalize: bool = True
    ) -> CompositeResult:
        """
        Weighted sum aggregation for numeric indicators.

        Args:
            values: Mapping of indicator name -> numeric value.
            weights: Optional mapping of indicator name -> weight (defaults to 1.0).
            normalize: If True, weights are normalized to sum to 1.
        """
        if not values:
            return CompositeResult(0.0, "weighted_sum")
        weights = weights or {}
        total_weight = 0.0
        weighted = 0.0
        for name, val in values.items():
            w = weights.get(name, 1.0)
            total_weight += w
            weighted += val * w
        if normalize and total_weight > 0:
            weighted /= total_weight
        return CompositeResult(weighted, "weighted_sum")

    @staticmethod
    def vote(signals: list[str], neutral: str = "neutral") -> CompositeResult:
        """
        Majority vote aggregation for discrete signals.

        Args:
            signals: List of signal strings.
            neutral: Returned if tie or empty.
        """
        if not signals:
            return CompositeResult(neutral, "vote")
        counts: dict[str, int] = {}
        for sig in signals:
            counts[sig] = counts.get(sig, 0) + 1
        sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        top, top_count = sorted_counts[0]
        if len(sorted_counts) > 1 and sorted_counts[1][1] == top_count:
            return CompositeResult(neutral, "vote")
        return CompositeResult(top, "vote")

    @staticmethod
    def min_value(values: list[float]) -> CompositeResult:
        """Return minimum value composite."""
        return CompositeResult(min(values) if values else 0.0, "min")

    @staticmethod
    def max_value(values: list[float]) -> CompositeResult:
        """Return maximum value composite."""
        return CompositeResult(max(values) if values else 0.0, "max")
