"""
Support and resistance level detection.

Finds recent pivot-based levels and summarizes their strength.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SupportResistanceLevels:
    """Support and resistance summary."""

    support: float | None
    resistance: float | None
    support_touches: int
    resistance_touches: int


class SupportResistanceLocator:
    """Locate support and resistance using pivot highs/lows."""

    @staticmethod
    def find_levels(
        high: pd.Series,
        low: pd.Series,
        window: int = 3,
        tolerance: float = 0.003,
    ) -> SupportResistanceLevels:
        """
        Identify recent support/resistance from pivot highs/lows.

        Args:
            high: High price series.
            low: Low price series.
            window: Bars on each side for pivot detection.
            tolerance: Merge levels within this relative distance.
        """
        if len(high) < window * 2 + 1:
            return SupportResistanceLevels(None, None, 0, 0)

        pivots_high: list[float] = []
        pivots_low: list[float] = []

        for i in range(window, len(high) - window):
            segment_high = high.iloc[i - window : i + window + 1]
            segment_low = low.iloc[i - window : i + window + 1]
            if high.iloc[i] == segment_high.max():
                pivots_high.append(float(high.iloc[i]))
            if low.iloc[i] == segment_low.min():
                pivots_low.append(float(low.iloc[i]))

        def _merge_levels(levels: list[float]) -> tuple[float | None, int]:
            if not levels:
                return None, 0
            merged: list[float] = []
            touches = 0
            for level in sorted(levels):
                if not merged or abs(level - merged[-1]) / merged[-1] > tolerance:
                    merged.append(level)
                else:
                    merged[-1] = (merged[-1] + level) / 2
                touches += 1
            return merged[-1], touches

        resistance, res_touches = _merge_levels(pivots_high)
        support, sup_touches = _merge_levels(pivots_low)

        return SupportResistanceLevels(
            support=support,
            resistance=resistance,
            support_touches=sup_touches,
            resistance_touches=res_touches,
        )
