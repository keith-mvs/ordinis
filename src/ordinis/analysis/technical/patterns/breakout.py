"""
Breakout detection utilities built on support/resistance levels.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BreakoutSignal:
    """Breakout summary."""

    direction: str | None  # "bullish", "bearish", or None
    level: float | None
    confirmed: bool


class BreakoutDetector:
    """Detect simple support/resistance breakouts."""

    @staticmethod
    def detect(
        close: pd.Series,
        support: float | None,
        resistance: float | None,
        tolerance: float = 0.002,
    ) -> BreakoutSignal:
        """
        Detect breakout on the latest bar.

        Args:
            close: Close price series.
            support: Support level to watch.
            resistance: Resistance level to watch.
            tolerance: Relative buffer to reduce false positives.
        """
        if close.empty:
            return BreakoutSignal(None, None, False)
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2]) if len(close) > 1 else last_close

        if resistance is not None and last_close > resistance * (1 + tolerance):
            confirmed = prev_close <= resistance * (1 + tolerance / 2)
            return BreakoutSignal("bullish", resistance, confirmed)

        if support is not None and last_close < support * (1 - tolerance):
            confirmed = prev_close >= support * (1 - tolerance / 2)
            return BreakoutSignal("bearish", support, confirmed)

        return BreakoutSignal(None, None, False)
