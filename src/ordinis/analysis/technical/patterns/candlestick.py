"""
Candlestick pattern detection.

Provides lightweight heuristics for common single- and multi-candle
patterns. The detectors focus on the most recent bars to surface
actionable signals.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PatternMatch:
    """Result of a pattern check."""

    name: str
    matched: bool
    confidence: float = 1.0


class CandlestickPatterns:
    """Detect common candlestick patterns on OHLC data."""

    @staticmethod
    def detect(data: pd.DataFrame) -> list[str]:
        """
        Detect patterns on the latest bar(s).

        Args:
            data: DataFrame with columns [open, high, low, close]

        Returns:
            List of pattern names matched on the latest bar.
        """
        matches: list[str] = []
        if len(data) < 2:
            return matches

        o = data["open"]
        h = data["high"]
        low_vals = data["low"]
        c = data["close"]

        checks = [
            CandlestickPatterns.doji(o, h, low_vals, c),
            CandlestickPatterns.hammer(o, h, low_vals, c),
            CandlestickPatterns.inverted_hammer(o, h, low_vals, c),
            CandlestickPatterns.hanging_man(o, h, low_vals, c),
            CandlestickPatterns.shooting_star(o, h, low_vals, c),
            CandlestickPatterns.bullish_engulfing(o, h, low_vals, c),
            CandlestickPatterns.bearish_engulfing(o, h, low_vals, c),
            CandlestickPatterns.bullish_harami(o, h, low_vals, c),
            CandlestickPatterns.bearish_harami(o, h, low_vals, c),
            CandlestickPatterns.piercing_line(o, h, low_vals, c),
            CandlestickPatterns.dark_cloud_cover(o, h, low_vals, c),
            CandlestickPatterns.morning_star(o, h, low_vals, c),
            CandlestickPatterns.evening_star(o, h, low_vals, c),
            CandlestickPatterns.tweezer_top(o, h, low_vals, c),
            CandlestickPatterns.tweezer_bottom(o, h, low_vals, c),
        ]
        for result in checks:
            if result.matched:
                matches.append(result.name)
        return matches

    @staticmethod
    def _last(
        o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series
    ) -> tuple[float, float, float, float]:
        return float(o.iloc[-1]), float(h.iloc[-1]), float(low_vals.iloc[-1]), float(c.iloc[-1])

    @staticmethod
    def _prev(
        o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series
    ) -> tuple[float, float, float, float]:
        return float(o.iloc[-2]), float(h.iloc[-2]), float(low_vals.iloc[-2]), float(c.iloc[-2])

    @staticmethod
    def _body(open_: float, close: float) -> float:
        return abs(close - open_)

    @staticmethod
    def _range(high: float, low: float) -> float:
        return high - low

    @staticmethod
    def doji(o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series) -> PatternMatch:
        open_, high, low, close = CandlestickPatterns._last(o, h, low_vals, c)
        body = CandlestickPatterns._body(open_, close)
        total = CandlestickPatterns._range(high, low)
        small_body = body <= 0.1 * total if total > 0 else False
        return PatternMatch("doji", small_body)

    @staticmethod
    def hammer(o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series) -> PatternMatch:
        open_, high, low, close = CandlestickPatterns._last(o, h, low_vals, c)
        body = CandlestickPatterns._body(open_, close)
        total = CandlestickPatterns._range(high, low)
        lower_shadow = open_ - low if open_ > close else close - low
        upper_shadow = high - open_ if open_ > close else high - close
        cond = total > 0 and lower_shadow >= 2 * body and upper_shadow <= body
        return PatternMatch("hammer", cond)

    @staticmethod
    def inverted_hammer(
        o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series
    ) -> PatternMatch:
        open_, high, low, close = CandlestickPatterns._last(o, h, low_vals, c)
        body = CandlestickPatterns._body(open_, close)
        upper_shadow = high - max(open_, close)
        lower_shadow = min(open_, close) - low
        cond = upper_shadow >= 2 * body and lower_shadow <= body
        return PatternMatch("inverted_hammer", cond)

    @staticmethod
    def hanging_man(o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series) -> PatternMatch:
        # Same geometry as hammer; in practice trend context differentiates.
        result = CandlestickPatterns.hammer(o, h, low_vals, c)
        return PatternMatch("hanging_man", result.matched, result.confidence)

    @staticmethod
    def shooting_star(
        o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series
    ) -> PatternMatch:
        result = CandlestickPatterns.inverted_hammer(o, h, low_vals, c)
        return PatternMatch("shooting_star", result.matched, result.confidence)

    @staticmethod
    def bullish_engulfing(
        o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series
    ) -> PatternMatch:
        if len(o) < 2:
            return PatternMatch("bullish_engulfing", False)
        po, _ph, _pl, pc = CandlestickPatterns._prev(o, h, low_vals, c)
        co, _ch, _cl, cc = CandlestickPatterns._last(o, h, low_vals, c)
        cond = pc > po and cc > co and cc > pc and co < po and cc > pc and co < pc
        cond = cond or (pc < po and cc > co and cc >= po and co <= pc)
        return PatternMatch("bullish_engulfing", cond)

    @staticmethod
    def bearish_engulfing(
        o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series
    ) -> PatternMatch:
        if len(o) < 2:
            return PatternMatch("bearish_engulfing", False)
        po, _ph, _pl, pc = CandlestickPatterns._prev(o, h, low_vals, c)
        co, _ch, _cl, cc = CandlestickPatterns._last(o, h, low_vals, c)
        cond = pc < po and cc < co and cc < pc and co > po and cc < pc and co > pc
        cond = cond or (pc > po and cc < co and co >= pc and cc <= po)
        return PatternMatch("bearish_engulfing", cond)

    @staticmethod
    def bullish_harami(
        o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series
    ) -> PatternMatch:
        if len(o) < 2:
            return PatternMatch("bullish_harami", False)
        po, _ph, _pl, pc = CandlestickPatterns._prev(o, h, low_vals, c)
        co, _ch, _cl, cc = CandlestickPatterns._last(o, h, low_vals, c)
        prev_bearish = pc < po
        small_body_inside = min(co, cc) >= min(po, pc) and max(co, cc) <= max(po, pc)
        cond = prev_bearish and cc > co and small_body_inside
        return PatternMatch("bullish_harami", cond)

    @staticmethod
    def bearish_harami(
        o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series
    ) -> PatternMatch:
        if len(o) < 2:
            return PatternMatch("bearish_harami", False)
        po, _ph, _pl, pc = CandlestickPatterns._prev(o, h, low_vals, c)
        co, _ch, _cl, cc = CandlestickPatterns._last(o, h, low_vals, c)
        prev_bullish = pc > po
        small_body_inside = min(co, cc) >= min(po, pc) and max(co, cc) <= max(po, pc)
        cond = prev_bullish and cc < co and small_body_inside
        return PatternMatch("bearish_harami", cond)

    @staticmethod
    def piercing_line(
        o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series
    ) -> PatternMatch:
        if len(o) < 2:
            return PatternMatch("piercing_line", False)
        po, _ph, _pl, pc = CandlestickPatterns._prev(o, h, low_vals, c)
        co, _ch, _cl, cc = CandlestickPatterns._last(o, h, low_vals, c)
        midpoint = (po + pc) / 2
        cond = pc < po and co < pc and cc > midpoint and cc < po
        return PatternMatch("piercing_line", cond)

    @staticmethod
    def dark_cloud_cover(
        o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series
    ) -> PatternMatch:
        if len(o) < 2:
            return PatternMatch("dark_cloud_cover", False)
        po, _ph, _pl, pc = CandlestickPatterns._prev(o, h, low_vals, c)
        co, _ch, _cl, cc = CandlestickPatterns._last(o, h, low_vals, c)
        midpoint = (po + pc) / 2
        cond = pc > po and co > pc and cc < midpoint and cc > po
        return PatternMatch("dark_cloud_cover", cond)

    @staticmethod
    def morning_star(o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series) -> PatternMatch:
        if len(o) < 3:
            return PatternMatch("morning_star", False)
        # Use last three bars
        o1, h1, l1, c1 = (
            float(o.iloc[-3]),
            float(h.iloc[-3]),
            float(low_vals.iloc[-3]),
            float(c.iloc[-3]),
        )
        o2, h2, l2, c2 = (
            float(o.iloc[-2]),
            float(h.iloc[-2]),
            float(low_vals.iloc[-2]),
            float(c.iloc[-2]),
        )
        o3, h3, l3, c3 = CandlestickPatterns._last(o, h, low_vals, c)
        long_bearish = c1 < o1 and CandlestickPatterns._body(o1, c1) > 0.6 * (h1 - l1)
        gap_down = max(o2, c2) < c1
        long_bullish = c3 > o3 and CandlestickPatterns._body(o3, c3) > 0.6 * (h3 - l3)
        closes_into_body = c3 >= (o1 + c1) / 2
        cond = long_bearish and gap_down and long_bullish and closes_into_body
        return PatternMatch("morning_star", cond)

    @staticmethod
    def evening_star(o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series) -> PatternMatch:
        if len(o) < 3:
            return PatternMatch("evening_star", False)
        o1, h1, l1, c1 = (
            float(o.iloc[-3]),
            float(h.iloc[-3]),
            float(low_vals.iloc[-3]),
            float(c.iloc[-3]),
        )
        o2, h2, l2, c2 = (
            float(o.iloc[-2]),
            float(h.iloc[-2]),
            float(low_vals.iloc[-2]),
            float(c.iloc[-2]),
        )
        o3, h3, l3, c3 = CandlestickPatterns._last(o, h, low_vals, c)
        long_bullish = c1 > o1 and CandlestickPatterns._body(o1, c1) > 0.6 * (h1 - l1)
        gap_up = min(o2, c2) > c1
        long_bearish = c3 < o3 and CandlestickPatterns._body(o3, c3) > 0.6 * (h3 - l3)
        closes_into_body = c3 <= (o1 + c1) / 2
        cond = long_bullish and gap_up and long_bearish and closes_into_body
        return PatternMatch("evening_star", cond)

    @staticmethod
    def tweezer_top(
        o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series, tolerance: float = 0.001
    ) -> PatternMatch:
        """Detect tweezer top pattern (low_vals unused, only checks highs)."""
        if len(o) < 2:
            return PatternMatch("tweezer_top", False)
        h1 = float(h.iloc[-2])
        h2 = float(h.iloc[-1])
        cond = abs(h1 - h2) <= max(h1, h2) * tolerance
        cond = (
            cond and float(c.iloc[-2]) > float(o.iloc[-2]) and float(c.iloc[-1]) < float(o.iloc[-1])
        )
        return PatternMatch("tweezer_top", cond)

    @staticmethod
    def tweezer_bottom(
        o: pd.Series, h: pd.Series, low_vals: pd.Series, c: pd.Series, tolerance: float = 0.001
    ) -> PatternMatch:
        """Detect tweezer bottom pattern (h unused, only checks lows)."""
        if len(o) < 2:
            return PatternMatch("tweezer_bottom", False)
        l1 = float(low_vals.iloc[-2])
        l2 = float(low_vals.iloc[-1])
        cond = abs(l1 - l2) <= max(l1, l2) * tolerance
        cond = (
            cond and float(c.iloc[-2]) < float(o.iloc[-2]) and float(c.iloc[-1]) > float(o.iloc[-1])
        )
        return PatternMatch("tweezer_bottom", cond)
