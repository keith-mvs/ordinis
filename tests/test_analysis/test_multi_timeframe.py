"""Tests for multi-timeframe analyzer."""

import pandas as pd

from ordinis.analysis.technical.multi_timeframe import MultiTimeframeAnalyzer


def _trend_df(direction: str) -> pd.DataFrame:
    """Create a simple OHLCV DataFrame trending up or down."""
    base = pd.Series(range(1, 60), dtype=float)
    close = base if direction == "up" else base.iloc[::-1].reset_index(drop=True)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000,
        }
    )


def test_multi_timeframe_all_bullish():
    analyzer = MultiTimeframeAnalyzer()
    data = {"1h": _trend_df("up"), "4h": _trend_df("up"), "1d": _trend_df("up")}

    result = analyzer.analyze(data)

    assert result.majority_trend == "bullish"
    assert result.bias == "strong_bullish"
    assert result.agreement_score == 1.0
    assert all(sig.trend_direction == "bullish" for sig in result.signals)


def test_multi_timeframe_mixed_bias_bearish():
    analyzer = MultiTimeframeAnalyzer()
    data = {"1h": _trend_df("down"), "4h": _trend_df("down"), "1d": _trend_df("up")}

    result = analyzer.analyze(data)

    assert result.majority_trend == "bearish"
    assert result.bias in {"bearish", "strong_bearish"}
    assert result.agreement_score >= 2 / 3
