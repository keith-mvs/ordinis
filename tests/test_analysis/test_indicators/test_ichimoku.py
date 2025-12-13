"""Tests for Ichimoku Cloud calculations and signals."""

import pandas as pd
import pytest

from ordinis.analysis.technical.indicators.trend import TrendIndicators


def _build_series(direction: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Create simple monotonic OHLC series for testing."""
    base = pd.Series(range(1, 81), dtype=float)
    close = base if direction == "up" else base[::-1].reset_index(drop=True)
    high = close + 1
    low = close - 1
    return high, low, close


def test_ichimoku_values_match_manual_calculation():
    """Ichimoku line outputs should align with manual rolling calculations."""
    high, low, close = _build_series("up")
    values, _ = TrendIndicators.ichimoku(high, low, close)

    conversion_line = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    base_line = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    span_a = ((conversion_line + base_line) / 2).shift(26)
    span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
    chikou = close.shift(-26).dropna().iloc[-1]

    assert values.tenkan_sen == pytest.approx(conversion_line.iloc[-1])
    assert values.kijun_sen == pytest.approx(base_line.iloc[-1])
    assert values.senkou_span_a == pytest.approx(span_a.iloc[-1])
    assert values.senkou_span_b == pytest.approx(span_b.iloc[-1])
    assert values.chikou_span == pytest.approx(chikou)


def test_ichimoku_signal_bullish_trend():
    """Uptrend places price above a bullish cloud."""
    high, low, close = _build_series("up")
    _, signal = TrendIndicators.ichimoku(high, low, close)

    assert signal.trend == "bullish"
    assert signal.position == "above_cloud"
    assert signal.cloud_bias == "bullish"
    assert signal.lagging_confirmation is True
    assert signal.baseline_cross in (None, "bullish")


def test_ichimoku_signal_bearish_trend():
    """Downtrend places price below a bearish cloud."""
    high, low, close = _build_series("down")
    _, signal = TrendIndicators.ichimoku(high, low, close)

    assert signal.trend == "bearish"
    assert signal.position == "below_cloud"
    assert signal.cloud_bias == "bearish"
    assert signal.lagging_confirmation is True
