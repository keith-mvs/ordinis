"""
Multi-timeframe analysis utilities.

Provides alignment scoring across multiple timeframes to gauge
trend consensus (e.g., 1H / 4H / 1D).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators.moving_averages import MovingAverages


@dataclass
class TimeframeSignal:
    """Trend snapshot for a single timeframe."""

    timeframe: str
    trend_direction: str  # "bullish", "bearish", "neutral"
    trend_strength: float  # 0-100
    ma_alignment: str  # "bullish", "bearish", "mixed"


@dataclass
class MultiTimeframeResult:
    """Aggregated multi-timeframe assessment."""

    signals: list[TimeframeSignal]
    majority_trend: str  # "bullish", "bearish", "neutral"
    agreement_score: float  # 0-1 alignment across timeframes
    bias: str  # "strong_bullish", "bullish", "neutral", "bearish", "strong_bearish"


class MultiTimeframeAnalyzer:
    """Evaluate trend agreement across multiple timeframes."""

    def __init__(self, periods: list[int] | None = None):
        self.ma = MovingAverages()
        self.periods = periods or [10, 20, 50]

    def analyze(self, data_by_timeframe: dict[str, pd.DataFrame]) -> MultiTimeframeResult:
        """
        Analyze multiple timeframes for trend alignment.

        Args:
            data_by_timeframe: Mapping of timeframe label -> OHLCV DataFrame.

        Returns:
            MultiTimeframeResult summarizing alignment and bias.
        """
        signals: list[TimeframeSignal] = []
        for tf, df in data_by_timeframe.items():
            if df.empty:
                continue
            ma_analysis = self.ma.multi_ma_analysis(df, periods=self.periods)
            strength = ma_analysis["trend_strength"]
            alignment = ma_analysis["alignment"]

            if strength > 70:
                direction = "bullish"
            elif strength < 30:
                direction = "bearish"
            else:
                direction = "neutral"

            signals.append(
                TimeframeSignal(
                    timeframe=tf,
                    trend_direction=direction,
                    trend_strength=strength,
                    ma_alignment=alignment,
                )
            )

        if not signals:
            return MultiTimeframeResult([], "neutral", 0.0, "neutral")

        # Majority vote on trend direction
        direction_counts: dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}
        for sig in signals:
            direction_counts[sig.trend_direction] += 1
        majority_trend = max(direction_counts, key=lambda k: direction_counts[k])
        agreement_score = direction_counts[majority_trend] / len(signals)

        bias = self._calculate_bias(majority_trend, agreement_score)

        return MultiTimeframeResult(
            signals=signals,
            majority_trend=majority_trend,
            agreement_score=agreement_score,
            bias=bias,
        )

    @staticmethod
    def _calculate_bias(majority_trend: str, agreement_score: float) -> str:
        """Map majority trend and agreement strength to a bias label."""
        if majority_trend == "neutral":
            return "neutral"
        if agreement_score >= 0.75:
            return f"strong_{majority_trend}"
        if agreement_score >= 0.5:
            return majority_trend
        return "neutral"
