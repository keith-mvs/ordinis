"""Tests for OptionsSignalModel."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models.options_signal import OptionsSignalModel


def _make_ohlcv(n: int = 120, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Low-drift random walk around 100
    returns = rng.normal(0.0, 0.002, n)
    close = 100.0 * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.0005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n)))
    volume = rng.integers(1_000_000, 2_000_000, n)

    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.mark.asyncio
class TestOptionsSignalModel:
    async def test_generate_iron_condor_in_neutral_high_iv_mode(self, monkeypatch: pytest.MonkeyPatch):
        data = _make_ohlcv(140)
        cfg = ModelConfig(
            model_id="options_signal_test",
            model_type="options_signal",
            parameters={
                # Make 'high IV' gating trivially pass.
                "min_iv_rank": 0,
                "min_iv_percentile": 0,
                "enable_iron_condor": True,
                "enable_covered_call": False,
                "enable_csp": False,
                # Force neutral regime selection.
                "bullish_rsi_threshold": 100,
                "bearish_rsi_threshold": 0,
                "neutral_adx_threshold": 1_000,
                # Reduce required data.
                "vol_lookback": 20,
                "rsi_period": 14,
                "adx_period": 14,
                "atr_period": 14,
            },
            min_data_points=60,
        )
        model = OptionsSignalModel(cfg)

        # Ensure ADX stays low so the model selects the neutral path.
        monkeypatch.setattr(model, "_compute_adx", lambda _h, _l, _c: pd.Series(0.0, index=data.index))

        sig = await model.generate("SPY", data, datetime(2024, 6, 1))

        assert sig is not None
        assert sig.signal_type == SignalType.ENTRY
        assert sig.direction == Direction.NEUTRAL
        assert sig.metadata["model"] == "options_signal"
        assert sig.metadata["strategy_type"] in {"iron_condor", "covered_call", "cash_secured_put"}

    async def test_generate_returns_none_when_strategy_disabled(self):
        data = _make_ohlcv(140)
        cfg = ModelConfig(
            model_id="options_signal_test2",
            model_type="options_signal",
            parameters={
                "min_iv_rank": 0,
                "min_iv_percentile": 0,
                "enable_iron_condor": False,
                "enable_covered_call": False,
                "enable_csp": False,
                "enable_spreads": False,
            },
            min_data_points=60,
        )
        model = OptionsSignalModel(cfg)

        sig = await model.generate("SPY", data, datetime(2024, 6, 1))
        assert sig is None

    async def test_validation_missing_columns_raises(self):
        cfg = ModelConfig(model_id="options_signal_bad", model_type="options_signal", parameters={})
        model = OptionsSignalModel(cfg)
        data = pd.DataFrame({"close": [1.0] * 300})
        with pytest.raises(ValueError, match="Missing columns"):
            await model.generate("SPY", data, datetime(2024, 6, 1))
