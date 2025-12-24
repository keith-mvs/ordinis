"""Unit tests for the SignalCore regime detector.

Targets `ordinis.engines.signalcore.regime_detector` (not the application strategy
regime detector).

We cover:
- metric validation + compute_metrics happy path
- classification branches (quiet choppy / trending / volatile trending / mean-reverting / choppy)
- recommendations
- analyze_multiple exception handling
- regime_filter decision rules
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore.regime_detector import (
    MarketRegime,
    RegimeAnalysis,
    RegimeDetector,
    RegimeMetrics,
    regime_filter,
)


def _make_ohlcv(n: int = 80, seed: int = 7, drift: float = 0.0008, vol: float = 0.01) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=vol, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))

    # Keep bars realistic
    high = close * (1.0 + rng.uniform(0.0005, 0.01, size=n))
    low = close * (1.0 - rng.uniform(0.0005, 0.01, size=n))
    open_ = (high + low) / 2.0
    volu = rng.integers(1_000, 10_000, size=n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volu},
        index=pd.date_range("2024-01-01", periods=n, freq="5min"),
    )


def _base_metrics(**overrides) -> RegimeMetrics:
    base = dict(
        symbol="X",
        timeframe="5min",
        period_return=5.0,
        direction_change_rate=0.40,
        big_move_frequency=0.12,
        autocorrelation=0.02,
        avg_range_pct=0.50,
        volatility=1.0,
        bounce_after_drop=0.02,
        reversal_after_rally=0.02,
        adx=25.0,
        plus_di=30.0,
        minus_di=10.0,
        atr_pct=0.5,
    )
    base.update(overrides)
    return RegimeMetrics(**base)


class TestSignalcoreRegimeDetectorCompute:
    def test_compute_metrics_validates_minimum_bars(self):
        det = RegimeDetector()
        df = _make_ohlcv(n=40)
        with pytest.raises(ValueError, match="Need at least 50 bars"):
            det.compute_metrics(df)

    def test_compute_metrics_validates_required_columns(self):
        det = RegimeDetector()
        df = _make_ohlcv(n=80).drop(columns=["open"])
        with pytest.raises(ValueError, match="Missing required columns"):
            det.compute_metrics(df)

    def test_compute_metrics_happy_path_includes_adx_and_atr(self):
        det = RegimeDetector()
        df = _make_ohlcv(n=80)
        m = det.compute_metrics(df, symbol="ABC", timeframe="5min")
        assert m.symbol == "ABC"
        assert m.timeframe == "5min"
        assert isinstance(m.adx, float)
        assert isinstance(m.atr_pct, float)
        # ADX/ATR can be 0 with pathological data, but should usually be finite
        assert np.isfinite(m.adx)
        assert np.isfinite(m.atr_pct)


class TestSignalcoreRegimeDetectorClassification:
    def test_classify_quiet_choppy_with_adx(self):
        det = RegimeDetector()
        metrics = _base_metrics(adx=10.0, direction_change_rate=0.60, atr_pct=0.2, big_move_frequency=0.05)
        regime, conf = det.classify_regime(metrics)
        assert regime == MarketRegime.QUIET_CHOPPY
        assert conf >= 0.7

    def test_classify_volatile_trending_with_adx(self):
        det = RegimeDetector()
        metrics = _base_metrics(adx=35.0, atr_pct=1.2, direction_change_rate=0.30)
        regime, conf = det.classify_regime(metrics)
        assert regime == MarketRegime.VOLATILE_TRENDING
        assert 0.6 <= conf <= 0.9

    def test_classify_trending_with_adx(self):
        det = RegimeDetector()
        metrics = _base_metrics(adx=30.0, atr_pct=0.4, direction_change_rate=0.35)
        regime, conf = det.classify_regime(metrics)
        assert regime == MarketRegime.TRENDING
        assert conf >= 0.6

    def test_classify_mean_reverting_with_adx(self):
        det = RegimeDetector()
        metrics = _base_metrics(adx=12.0, autocorrelation=-0.08, bounce_after_drop=0.06, atr_pct=0.6)
        regime, conf = det.classify_regime(metrics)
        assert regime == MarketRegime.MEAN_REVERTING
        assert conf >= 0.5

    def test_classify_choppy_with_adx(self):
        det = RegimeDetector()
        # Avoid QUIET_CHOPPY fallback: ensure signals are not "weak".
        metrics = _base_metrics(
            adx=15.0,
            direction_change_rate=0.60,
            atr_pct=0.6,
            big_move_frequency=0.12,
            bounce_after_drop=0.10,
            reversal_after_rally=0.10,
        )
        regime, conf = det.classify_regime(metrics)
        assert regime == MarketRegime.CHOPPY
        assert conf >= 0.6


class TestSignalcoreRegimeDetectorRecommendations:
    @pytest.mark.parametrize(
        "regime,expected_trade",
        [
            (MarketRegime.QUIET_CHOPPY, "AVOID"),
            (MarketRegime.CHOPPY, "CAUTION"),
            (MarketRegime.TRENDING, "TRADE"),
            (MarketRegime.VOLATILE_TRENDING, "TRADE"),
            (MarketRegime.MEAN_REVERTING, "TRADE"),
        ],
    )
    def test_get_recommendations_trade_flag(self, regime: MarketRegime, expected_trade: str):
        det = RegimeDetector()
        metrics = _base_metrics()
        rec, avoid, trade, reasoning = det.get_recommendations(regime, metrics)
        assert isinstance(rec, list)
        assert isinstance(avoid, list)
        assert trade == expected_trade
        assert isinstance(reasoning, str) and reasoning


class TestSignalcoreRegimeDetectorAnalyzeMultiple:
    def test_analyze_multiple_skips_bad_symbol(self, capsys: pytest.CaptureFixture[str]):
        det = RegimeDetector()
        good = _make_ohlcv(n=80)
        bad = _make_ohlcv(n=40)  # too short -> ValueError

        results = det.analyze_multiple({"GOOD": good, "BAD": bad})
        assert "GOOD" in results
        assert "BAD" not in results

        out = capsys.readouterr().out
        assert "Could not analyze BAD" in out


class TestRegimeFilter:
    def test_regime_filter_rsi_skips_trending(self, monkeypatch: pytest.MonkeyPatch):
        def fake_analyze(_df: pd.DataFrame, symbol: str = "UNKNOWN", timeframe: str = "5min") -> RegimeAnalysis:
            m = _base_metrics(symbol=symbol)
            return RegimeAnalysis(
                metrics=m,
                regime=MarketRegime.TRENDING,
                confidence=0.8,
                recommended_strategies=[],
                avoid_strategies=[],
                trade_recommendation="TRADE",
                reasoning="x",
            )

        monkeypatch.setattr(RegimeDetector, "analyze", fake_analyze)

        ok, reason = regime_filter(_make_ohlcv(n=80), strategy_type="rsi", symbol="AAA")
        assert ok is False
        assert "TRENDING" in reason

    def test_regime_filter_momentum_checks_big_move_frequency(self, monkeypatch: pytest.MonkeyPatch):
        def fake_analyze(_df: pd.DataFrame, symbol: str = "UNKNOWN", timeframe: str = "5min") -> RegimeAnalysis:
            m = _base_metrics(symbol=symbol, big_move_frequency=0.05, adx=25.0)
            return RegimeAnalysis(
                metrics=m,
                regime=MarketRegime.CHOPPY,
                confidence=0.7,
                recommended_strategies=[],
                avoid_strategies=[],
                trade_recommendation="CAUTION",
                reasoning="x",
            )

        monkeypatch.setattr(RegimeDetector, "analyze", fake_analyze)

        ok, reason = regime_filter(_make_ohlcv(n=80), strategy_type="momentum", symbol="BBB")
        assert ok is False
        assert "low big-move frequency" in reason

    def test_regime_filter_on_analyze_error_defaults_to_trade(self, monkeypatch: pytest.MonkeyPatch):
        def boom(
            _self: RegimeDetector,
            _df: pd.DataFrame,
            symbol: str = "UNKNOWN",
            timeframe: str = "5min",
        ) -> RegimeAnalysis:
            raise RuntimeError("nope")

        monkeypatch.setattr(RegimeDetector, "analyze", boom)

        ok, reason = regime_filter(_make_ohlcv(n=80), strategy_type="trend", symbol="CCC")
        assert ok is True
        assert "Could not analyze regime" in reason
