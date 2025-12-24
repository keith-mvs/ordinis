"""Tests for OUPairsModel.

We monkeypatch cointegration and OU parameter estimation to make signals deterministic
and avoid optional dependency drift.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import ordinis.engines.signalcore.models.ou_pairs as ou_pairs
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models.ou_pairs import OUPairsModel, OUParams


@pytest.mark.asyncio
class TestOUPairsModel:
    def _model(self, **params) -> OUPairsModel:
        cfg = ModelConfig(
            model_id="ou_pairs_test",
            model_type="pairs",
            parameters={
                "coint_lookback": 30,
                "coint_pvalue": 0.05,
                "hedge_lookback": 30,
                "ou_lookback": 30,
                "min_halflife": 2,
                "max_halflife": 60,
                "entry_z": 2.0,
                "exit_z": 0.5,
                "stop_z": 4.0,
                **params,
            },
            min_data_points=10,
        )
        return OUPairsModel(cfg)

    async def test_generate_pair_signal_long_spread(self, monkeypatch: pytest.MonkeyPatch):
        # Force a valid pair regardless of optional statsmodels.
        monkeypatch.setattr(ou_pairs, "test_cointegration", lambda _a, _b: (0.01, 1.0))
        monkeypatch.setattr(
            ou_pairs,
            "estimate_ou_params",
            lambda _spread: OUParams(theta=0.5, mu=0.0, sigma=1.0, halflife=5.0, r_squared=0.8),
        )

        model = self._model()

        idx = pd.date_range("2024-01-01", periods=80, freq="D")
        prices_a = pd.Series(np.full(80, 100.0), index=idx)
        prices_b = pd.Series(np.full(80, 100.0), index=idx)
        prices_b.iloc[-1] = 112.0  # make spread very negative at the end

        sig = await model.generate_pair_signal("AAA", "BBB", prices_a, prices_b, datetime(2024, 4, 1))
        assert sig is not None
        assert sig.signal_type == SignalType.ENTRY
        assert sig.direction == Direction.LONG
        assert sig.symbol == "AAA/BBB"
        assert sig.metadata["spread_direction"] == 1
        assert sig.metadata["strategy"] == "ou_pairs"

    async def test_generate_pair_signal_hold_when_not_extreme(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ou_pairs, "test_cointegration", lambda _a, _b: (0.01, 1.0))
        monkeypatch.setattr(
            ou_pairs,
            "estimate_ou_params",
            lambda _spread: OUParams(theta=0.5, mu=0.0, sigma=1.0, halflife=5.0, r_squared=0.8),
        )

        model = self._model(entry_z=10.0)  # make entry hard

        idx = pd.date_range("2024-01-01", periods=80, freq="D")
        prices_a = pd.Series(np.full(80, 100.0), index=idx)
        prices_b = pd.Series(np.full(80, 100.0), index=idx)

        sig = await model.generate_pair_signal("AAA", "BBB", prices_a, prices_b, datetime(2024, 4, 1))
        assert sig is not None
        assert sig.signal_type == SignalType.HOLD
        assert sig.direction == Direction.NEUTRAL
        assert sig.metadata["is_valid_pair"] is True

    async def test_generate_pair_signal_returns_none_for_insufficient_data(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(ou_pairs, "test_cointegration", lambda _a, _b: (0.01, 1.0))
        monkeypatch.setattr(
            ou_pairs,
            "estimate_ou_params",
            lambda _spread: OUParams(theta=0.5, mu=0.0, sigma=1.0, halflife=5.0, r_squared=0.8),
        )

        model = self._model(coint_lookback=1000)

        idx = pd.date_range("2024-01-01", periods=80, freq="D")
        prices_a = pd.Series(np.full(80, 100.0), index=idx)
        prices_b = pd.Series(np.full(80, 100.0), index=idx)

        sig = await model.generate_pair_signal("AAA", "BBB", prices_a, prices_b, datetime(2024, 4, 1))
        assert sig is None
