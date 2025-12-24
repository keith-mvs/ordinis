"""Tests for NetworkRiskParityModel and CorrelationNetwork."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models.network_parity import (
    CorrelationNetwork,
    NetworkRiskParityModel,
    analyze_correlation_network,
)

class TestNetworkRiskParityModel:
    def test_correlation_network_weights_favor_peripheral_assets(self):
        # Create returns where A and B are highly correlated, C is mostly independent.
        rng = np.random.default_rng(42)
        n = 120
        base = rng.normal(0, 0.01, n)
        a = base + rng.normal(0, 0.002, n)
        b = base + rng.normal(0, 0.002, n)
        c = rng.normal(0, 0.01, n)

        returns = pd.DataFrame({"A": a, "B": b, "C": c})

        network = CorrelationNetwork(threshold=0.3, centrality_method="eigenvector").fit(returns)
        weights = network.get_weights(decay=0.5, min_weight=0.02, max_weight=0.30)
        centrality = network.get_centrality()
        stats = network.get_network_stats()

        assert stats.n_nodes == 3
        assert stats.n_edges >= 1
        assert set(weights) == {"A", "B", "C"}
        assert abs(sum(weights.values()) - 1.0) < 1e-9

        # C should be more peripheral than at least one of (A,B)
        assert centrality["C"] <= max(centrality["A"], centrality["B"])
        assert weights["C"] >= min(weights["A"], weights["B"])

    @pytest.mark.asyncio
    async def test_model_generate_portfolio_weights_builds_signals(self):
        rng = np.random.default_rng(1)
        n = 50
        # Returns: make A positive momentum, B negative momentum, C flat.
        a = np.concatenate([rng.normal(0, 0.01, n - 3), np.array([0.02, 0.02, 0.02])])
        b = np.concatenate([rng.normal(0, 0.01, n - 3), np.array([-0.02, -0.02, -0.02])])
        c = rng.normal(0, 0.002, n)
        returns = pd.DataFrame({"A": a, "B": b, "C": c})

        cfg = ModelConfig(
            model_id="network_parity_test",
            model_type="portfolio",
            parameters={
                "corr_lookback": 40,
                "corr_threshold": 0.2,
                "recalc_frequency": 1,
                "momentum_lookback": 3,
                "min_weight": 0.02,
                "max_weight": 0.30,
            },
            min_data_points=10,
        )
        model = NetworkRiskParityModel(cfg)

        ts = datetime(2024, 3, 1)
        signals = await model.generate_portfolio_weights(returns, ts)

        assert set(signals) == {"A", "B", "C"}
        assert all(sig.timestamp == ts for sig in signals.values())
        assert all(sig.symbol in {"A", "B", "C"} for sig in signals.values())
        assert all(sig.signal_type in {SignalType.ENTRY, SignalType.EXIT, SignalType.HOLD} for sig in signals.values())
        assert all(sig.direction in {Direction.LONG, Direction.NEUTRAL} for sig in signals.values())

        # Ensure we at least get one ENTRY given our strong positive momentum tail.
        assert any(sig.signal_type == SignalType.ENTRY for sig in signals.values())

    @pytest.mark.asyncio
    async def test_model_generate_requires_weights_first(self):
        cfg = ModelConfig(model_id="network_parity_test2", model_type="portfolio", parameters={})
        model = NetworkRiskParityModel(cfg)

        # No weights computed yet -> None
        df = pd.DataFrame({"close": np.linspace(100, 110, 50)})
        sig = await model.generate("A", df, datetime(2024, 3, 1))
        assert sig is None

    def test_analyze_correlation_network_helper(self):
        rng = np.random.default_rng(7)
        returns = pd.DataFrame(rng.normal(0, 0.01, size=(80, 4)), columns=list("WXYZ"))
        result = analyze_correlation_network(returns, threshold=0.2)

        assert "stats" in result
        assert "weights" in result
        assert "centrality" in result
        assert "correlation_matrix" in result
        assert set(result["weights"].keys()) == set(returns.columns)
