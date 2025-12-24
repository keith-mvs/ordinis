"""Tests for GARCHVolatilityModel fallback path.

We force ARCH_AVAILABLE=False to avoid requiring the optional arch dependency.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import ordinis.engines.signalcore.models.forecasting.volatility_model as volmod
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models.forecasting.volatility_model import (
    EGARCHVolatilityModel,
    GARCHVolatilityModel,
    TGARCHVolatilityModel,
)


def _make_ohlcv(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.01, n)
    close = 100.0 * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.0005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
    volume = rng.integers(100_000, 200_000, n)

    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.mark.asyncio
class TestVolatilityModels:
    async def test_garch_generate_uses_fallback_without_arch(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(volmod, "ARCH_AVAILABLE", False)
        monkeypatch.setattr(volmod, "arch_model", None)

        data = _make_ohlcv()
        model = GARCHVolatilityModel(horizon=3, high_vol_threshold=0.20)

        sig = await model.generate("SPY", data, datetime(2024, 6, 1))

        assert sig.symbol == "SPY"
        assert sig.signal_type in {SignalType.ENTRY, SignalType.HOLD}
        assert sig.direction in {Direction.LONG, Direction.NEUTRAL}
        assert sig.metadata["volatility_model"] == "rolling_std_fallback"
        assert "forecast_volatility" in sig.metadata

        v = model.estimate_volatility(data)
        assert isinstance(v, float)
        assert v >= 0.0

        desc = model.describe()
        assert desc["arch_available"] is False

    async def test_egarch_and_tgarch_instantiate(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(volmod, "ARCH_AVAILABLE", False)
        monkeypatch.setattr(volmod, "arch_model", None)

        data = _make_ohlcv()
        ts = datetime(2024, 6, 1)

        eg = EGARCHVolatilityModel(horizon=2)
        tg = TGARCHVolatilityModel(horizon=2)

        sig1 = await eg.generate("SPY", data, ts)
        sig2 = await tg.generate("SPY", data, ts)

        assert sig1.metadata["volatility_model"] == "rolling_std_fallback"
        assert sig2.metadata["volatility_model"] == "rolling_std_fallback"
