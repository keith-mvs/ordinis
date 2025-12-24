"""Tests for ATRBreakoutModel.

We patch indicator calculations to make breakouts deterministic.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.features.technical import TechnicalIndicators
from ordinis.engines.signalcore.models.atr_breakout import ATRBreakoutModel


def _ohlcv_df(close: list[float], *, symbol: str = "AAPL") -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(close), freq="D")
    close_s = pd.Series(close, index=idx)
    return pd.DataFrame(
        {
            "open": close_s.values,
            "high": (close_s + 0.5).values,
            "low": (close_s - 0.5).values,
            "close": close_s.values,
            "volume": np.full(len(close_s), 1_000_000, dtype=float),
            "symbol": [symbol] * len(close_s),
        },
        index=idx,
    )


@pytest.mark.asyncio
class TestATRBreakoutModel:
    def _model(self) -> ATRBreakoutModel:
        cfg = ModelConfig(
            model_id="atr_breakout_test",
            model_type="technical",
            version="1.0.0",
            parameters={"ema_period": 3, "atr_period": 3, "multiplier": 2.0},
            min_data_points=1,
        )
        return ATRBreakoutModel(cfg)

    async def test_crossed_up_emits_entry_long(self, monkeypatch: pytest.MonkeyPatch):
        model = self._model()

        # upper = 102, lower = 98
        def fake_atr(high, low, close, window=14):
            return pd.Series(np.ones(len(close)), index=close.index)

        def fake_ema(close, span=20):
            return pd.Series(np.full(len(close), 100.0), index=close.index)

        monkeypatch.setattr(TechnicalIndicators, "atr", staticmethod(fake_atr))
        monkeypatch.setattr(TechnicalIndicators, "ema", staticmethod(fake_ema))

        # prev 101.5 <= 102, current 103 > 102 -> crossed_up
        data = _ohlcv_df([100.0] * 25 + [101.5, 103.0])
        sig = await model.generate(data, datetime(2024, 3, 1))

        assert sig is not None
        assert sig.signal_type == SignalType.ENTRY
        assert sig.direction == Direction.LONG
        assert sig.score > 0.5
        assert sig.probability >= 0.65
        assert sig.metadata["indicator"] == "ATR_BREAKOUT"

    async def test_crossed_down_emits_entry_short(self, monkeypatch: pytest.MonkeyPatch):
        model = self._model()

        def fake_atr(high, low, close, window=14):
            return pd.Series(np.ones(len(close)), index=close.index)

        def fake_ema(close, span=20):
            return pd.Series(np.full(len(close), 100.0), index=close.index)

        monkeypatch.setattr(TechnicalIndicators, "atr", staticmethod(fake_atr))
        monkeypatch.setattr(TechnicalIndicators, "ema", staticmethod(fake_ema))

        # prev 98.5 >= 98, current 97 < 98 -> crossed_down
        data = _ohlcv_df([100.0] * 25 + [98.5, 97.0])
        sig = await model.generate(data, datetime(2024, 3, 1))

        assert sig is not None
        assert sig.signal_type == SignalType.ENTRY
        assert sig.direction == Direction.SHORT
        assert sig.score > 0.5
        assert sig.probability >= 0.65

    async def test_breakout_up_continuation_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        model = self._model()

        def fake_atr(high, low, close, window=14):
            return pd.Series(np.ones(len(close)), index=close.index)

        def fake_ema(close, span=20):
            return pd.Series(np.full(len(close), 100.0), index=close.index)

        monkeypatch.setattr(TechnicalIndicators, "atr", staticmethod(fake_atr))
        monkeypatch.setattr(TechnicalIndicators, "ema", staticmethod(fake_ema))

        # already above upper on previous bar and still above -> breakout_up but not crossed_up
        data = _ohlcv_df([100.0] * 25 + [103.0, 103.5])
        sig = await model.generate(data, datetime(2024, 3, 1))
        assert sig is None

    async def test_invalid_data_missing_columns_raises(self):
        model = self._model()
        data = pd.DataFrame({"close": [1.0] * 50})
        with pytest.raises(ValueError, match="Missing columns"):
            await model.generate(data, datetime(2024, 3, 1))

    async def test_invalid_data_nulls_raises(self):
        model = self._model()
        data = _ohlcv_df([100.0] * 30)
        data.loc[data.index[-1], "close"] = np.nan
        with pytest.raises(ValueError, match="null"):
            await model.generate(data, datetime(2024, 3, 1))
