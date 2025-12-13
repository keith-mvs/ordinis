"""Tests for candlestick pattern detection."""

import pandas as pd

from ordinis.analysis.technical.patterns.candlestick import CandlestickPatterns


def _df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"])


def test_detects_doji_and_hammer():
    data = _df(
        [
            (10.0, 10.2, 9.9, 10.05),  # doji-ish
            (10.5, 10.6, 9.0, 10.4),  # hammer geometry
        ]
    )
    patterns = CandlestickPatterns.detect(data)
    assert "doji" in patterns
    assert "hammer" in patterns or "hanging_man" in patterns


def test_bullish_and_bearish_engulfing():
    data = _df(
        [
            (10.0, 10.2, 9.8, 9.7),  # bearish candle
            (9.6, 10.5, 9.5, 10.4),  # bullish engulfing
            (10.0, 10.6, 9.9, 10.4),  # small bullish candle
            (10.5, 10.6, 9.4, 9.5),  # bearish engulfing
        ]
    )
    patterns_first = CandlestickPatterns.detect(data.iloc[:2])
    assert "bullish_engulfing" in patterns_first

    patterns_second = CandlestickPatterns.detect(data.iloc[2:])
    assert "bearish_engulfing" in patterns_second


def test_morning_and_evening_star():
    morning = _df(
        [
            (12.0, 12.1, 11.5, 11.6),  # long red
            (11.3, 11.5, 11.1, 11.2),  # gap down small
            (11.1, 12.2, 11.0, 12.0),  # strong green closing into prior body
        ]
    )
    evening = _df(
        [
            (11.0, 12.2, 10.9, 12.0),  # long green
            (12.1, 12.3, 12.0, 12.05),  # gap up small
            (12.0, 12.1, 11.0, 11.1),  # long red closing into prior body
        ]
    )
    morning_patterns = CandlestickPatterns.detect(morning)
    evening_patterns = CandlestickPatterns.detect(evening)
    assert "morning_star" in morning_patterns
    assert "evening_star" in evening_patterns
