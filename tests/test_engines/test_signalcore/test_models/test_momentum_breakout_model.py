"""Tests for MomentumBreakoutModel.

We patch SMA to avoid dependency/NaN behavior from adapter wrappers and to
make volume/trend gating deterministic.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.features.technical import TechnicalIndicators
from ordinis.engines.signalcore.models.momentum_breakout import MomentumBreakoutModel


def _df_from_series(close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    return pd.DataFrame(
        {
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _patch_sma(monkeypatch: pytest.MonkeyPatch) -> None:
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window, min_periods=window).mean()

    monkeypatch.setattr(TechnicalIndicators, "sma", staticmethod(sma))


@pytest.mark.asyncio
class TestMomentumBreakoutModel:
    def _model(self, **params) -> MomentumBreakoutModel:
        cfg = ModelConfig(
            model_id="mom_breakout_test",
            model_type="technical",
            version="1.0.0",
            parameters={
                "breakout_period": 5,
                "volume_period": 5,
                "volume_mult": 1.2,
                "trend_filter_period": 3,
                "require_trend": True,
                "enable_shorts": True,
                "enable_longs": True,
                **params,
            },
            min_data_points=20,
        )
        return MomentumBreakoutModel(cfg)

    async def test_long_entry_then_exit_on_trend_reversal(self, monkeypatch: pytest.MonkeyPatch):
        _patch_sma(monkeypatch)
        model = self._model()

        # Uptrend base
        close = np.linspace(100, 110, 30)
        high = close + 0.5
        low = close - 0.5
        volume = np.full(30, 100.0)

        # Trigger breakout on last bar
        high[-1] = high[-6:-1].max() + 5.0
        close[-1] = high[-1]  # strong close
        volume[-1] = 1000.0

        data = _df_from_series(close, high, low, volume)
        sig1 = await model.generate("AAPL", data, datetime(2024, 3, 1))

        assert sig1.signal_type == SignalType.ENTRY
        assert sig1.direction == Direction.LONG
        assert model._in_long is True

        # Now force trend reversal: price falls below SMA(3)
        close2 = close.copy()
        high2 = high.copy()
        low2 = low.copy()
        volume2 = volume.copy()
        close2[-1] = close2[-2] - 20.0
        high2[-1] = close2[-1] + 0.5
        low2[-1] = close2[-1] - 0.5
        volume2[-1] = 100.0  # avoid new breakout

        data2 = _df_from_series(close2, high2, low2, volume2)
        sig2 = await model.generate("AAPL", data2, datetime(2024, 3, 2))

        assert sig2.signal_type == SignalType.EXIT
        assert sig2.direction == Direction.NEUTRAL
        assert model._in_long is False

    async def test_short_entry_then_exit_on_trend_reversal(self, monkeypatch: pytest.MonkeyPatch):
        _patch_sma(monkeypatch)
        model = self._model()

        close = np.linspace(110, 100, 30)
        high = close + 0.5
        low = close - 0.5
        volume = np.full(30, 100.0)

        # Trigger breakdown on last bar
        low[-1] = low[-6:-1].min() - 5.0
        close[-1] = low[-1]
        volume[-1] = 1000.0

        data = _df_from_series(close, high, low, volume)
        sig1 = await model.generate("AAPL", data, datetime(2024, 3, 1))

        assert sig1.signal_type == SignalType.ENTRY
        assert sig1.direction == Direction.SHORT
        assert sig1.score < 0
        assert model._in_short is True

        # Reversal: price rises above SMA(3)
        close2 = close.copy()
        high2 = high.copy()
        low2 = low.copy()
        volume2 = volume.copy()
        close2[-1] = close2[-2] + 20.0
        high2[-1] = close2[-1] + 0.5
        low2[-1] = close2[-1] - 0.5
        volume2[-1] = 100.0

        data2 = _df_from_series(close2, high2, low2, volume2)
        sig2 = await model.generate("AAPL", data2, datetime(2024, 3, 2))

        assert sig2.signal_type == SignalType.EXIT
        assert sig2.direction == Direction.NEUTRAL
        assert model._in_short is False

    async def test_require_trend_false_allows_entry_against_sma(self, monkeypatch: pytest.MonkeyPatch):
        _patch_sma(monkeypatch)
        model = self._model(require_trend=False, trend_filter_period=3)

        close = np.linspace(110, 100, 30)  # bearish trend (below SMA)
        high = close + 0.5
        low = close - 0.5
        volume = np.full(30, 100.0)

        # Breakout high anyway
        high[-1] = high[-6:-1].max() + 5.0
        close[-1] = high[-1]
        volume[-1] = 1000.0

        data = _df_from_series(close, high, low, volume)
        sig = await model.generate("AAPL", data, datetime(2024, 3, 1))
        assert sig.signal_type == SignalType.ENTRY
        assert sig.direction == Direction.LONG

    async def test_validation_missing_columns_raises(self, monkeypatch: pytest.MonkeyPatch):
        _patch_sma(monkeypatch)
        model = self._model()
        data = pd.DataFrame({"close": [1.0] * 200})
        with pytest.raises(ValueError, match="Missing columns"):
            await model.generate("AAPL", data, datetime(2024, 3, 1))

    async def test_validation_insufficient_data_raises(self, monkeypatch: pytest.MonkeyPatch):
        _patch_sma(monkeypatch)
        model = self._model()
        close = np.array([100.0] * 10)
        data = _df_from_series(close, close + 0.5, close - 0.5, np.array([100.0] * 10))
        with pytest.raises(ValueError, match="Insufficient data"):
            await model.generate("AAPL", data, datetime(2024, 3, 1))
