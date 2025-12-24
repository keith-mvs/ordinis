"""Unit tests for Sprint runner internals.

These tests focus on deterministic, offline components:
- Metrics calculation
- Walk-forward split logic
- A small synthetic end-to-end call for one strategy backtest (GARCH)

We avoid network (yfinance) and LLM calls.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.sprint.core.runner import AcceleratedSprintRunner, SprintConfig, StrategyResult


def _make_price_df(n: int = 320, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Ensure strictly positive prices for log() usage in some strategies.
    log_returns = rng.normal(loc=0.0003, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))

    high = close * (1.0 + rng.uniform(0.0005, 0.01, size=n))
    low = close * (1.0 - rng.uniform(0.0005, 0.01, size=n))
    return pd.DataFrame(
        {
            "Close": close,
            "High": high,
            "Low": low,
        },
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )


class TestStrategyResult:
    def test_to_dict_contains_expected_fields(self):
        r = StrategyResult(
            name="X",
            params={"a": 1},
            total_return=0.1,
            annualized_return=0.05,
            sharpe_ratio=1.2,
        )
        d = r.to_dict()
        assert d["name"] == "X"
        assert d["params"] == {"a": 1}
        assert "total_return" in d
        assert "sharpe_ratio" in d
        # equity_curve intentionally omitted from to_dict
        assert "equity_curve" not in d


class TestSprintRunnerMetrics:
    @pytest.fixture
    def runner(self) -> AcceleratedSprintRunner:
        cfg = SprintConfig(use_gpu=False, use_ai=False, generate_visualizations=False)
        return AcceleratedSprintRunner(cfg)

    def test_calculate_metrics_short_equity_curve_returns_defaults(self, runner: AcceleratedSprintRunner):
        metrics = runner._calculate_metrics(np.array([100_000.0]))
        assert metrics["total_return"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["profit_factor"] == 1.0
        assert metrics["total_trades"] == 0

    def test_calculate_metrics_constant_equity_has_zero_risk_metrics(self, runner: AcceleratedSprintRunner):
        equity = np.full(252, 100_000.0)
        metrics = runner._calculate_metrics(equity)
        assert metrics["total_return"] == 0.0
        assert metrics["max_drawdown"] == 0.0
        assert metrics["sharpe_ratio"] == 0.0
        assert 0.0 <= metrics["return_stability"] <= 1.0

    def test_calculate_metrics_includes_trade_stats_when_trades_present(self, runner: AcceleratedSprintRunner):
        equity = np.linspace(100_000.0, 110_000.0, 100)
        trades = [
            {"pnl": 100.0},
            {"pnl": 50.0},
            {"pnl": -25.0},
        ]
        metrics = runner._calculate_metrics(equity, trades)
        assert metrics["total_trades"] == 3
        assert 0.0 <= metrics["win_rate"] <= 1.0
        assert metrics["profit_factor"] >= 1.0
        assert isinstance(metrics["expectancy"], float)


class TestSprintRunnerWalkForward:
    def test_walk_forward_disabled_returns_none_tuple(self):
        cfg = SprintConfig(use_gpu=False, use_ai=False, walk_forward=False)
        runner = AcceleratedSprintRunner(cfg)
        train, test, overfit = runner._run_walk_forward("x", lambda _d, _p: {"sharpe_ratio": 1.0}, {})
        assert train is None and test is None and overfit is None

    def test_walk_forward_overfit_ratio_branches(self):
        cfg = SprintConfig(use_gpu=False, use_ai=False, walk_forward=True, train_ratio=0.7)
        runner = AcceleratedSprintRunner(cfg)
        runner.price_data = {"A": _make_price_df(220)}

        calls = {"n": 0}

        def backtest_func(_data: pd.DataFrame, _params: dict) -> dict:
            calls["n"] += 1
            # First call = train, second call = test
            return {"sharpe_ratio": 1.0 if calls["n"] == 1 else 0.0}

        train, test, overfit = runner._run_walk_forward("x", backtest_func, {})
        assert train == pytest.approx(1.0)
        assert test == pytest.approx(0.0)
        assert overfit == float("inf")


class TestSprintRunnerGarchSmoke:
    def test_backtest_garch_runs_on_synthetic_data(self):
        # Keep config deterministic and offline.
        cfg = SprintConfig(use_gpu=False, use_ai=False, generate_visualizations=False, walk_forward=True)
        runner = AcceleratedSprintRunner(cfg)
        runner.price_data = {"TEST": _make_price_df(320)}

        result = runner.backtest_garch(
            params={
                "garch_lookback": 60,
                "breakout_threshold": 1.25,
                "atr_stop_mult": 2.0,
                "atr_tp_mult": 3.0,
            }
        )

        assert isinstance(result, StrategyResult)
        assert result.name == "GARCH Breakout"
        assert result.equity_curve is not None
        assert len(result.equity_curve) == len(runner.price_data["TEST"])
        assert isinstance(result.sharpe_ratio, float)
        # Walk-forward fields should be populated when walk_forward=True and data is present
        assert result.train_sharpe is not None
        assert result.test_sharpe is not None
        assert result.overfit_ratio is not None
