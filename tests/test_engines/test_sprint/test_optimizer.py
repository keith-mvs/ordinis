from __future__ import annotations

import asyncio

import numpy as np
import pytest

from ordinis.engines.sprint.core.optimizer import AIStrategyOptimizer, AIOptimizerConfig, StrategyProfile


@pytest.mark.unit
def test_build_optimization_prompt_includes_history() -> None:
    profile = StrategyProfile(
        name="TestStrategy",
        description="Desc",
        param_definitions={
            "lookback": {"min": 5, "max": 100, "default": 20, "description": "lb"},
            "threshold": {"min": 0.1, "max": 2.0, "default": 1.0, "description": "th"},
        },
        objective="sharpe_ratio",
        constraints={"max_drawdown": 0.2},
    )

    opt = AIStrategyOptimizer(AIOptimizerConfig(max_iterations=1))
    history = [
        {"params": {"lookback": 10, "threshold": 1.2}, "sharpe_ratio": 1.0},
        {"params": {"lookback": 50, "threshold": 0.5}, "sharpe_ratio": -0.2},
        {"params": {"lookback": 20, "threshold": 1.0}, "sharpe_ratio": 0.3},
    ]

    prompt = opt._build_optimization_prompt(profile, history, n_suggestions=3)

    assert "Previous Results" in prompt
    assert "BEST" in prompt
    assert "WORST" in prompt
    assert "Return ONLY a JSON array" in prompt


@pytest.mark.unit
def test_parse_suggestions_sanitizes_and_clamps() -> None:
    profile = StrategyProfile(
        name="TestStrategy",
        description="Desc",
        param_definitions={
            "lookback": {"min": 5, "max": 100, "default": 20},
            "threshold": {"min": 0.1, "max": 2.0, "default": 1.0},
        },
    )

    opt = AIStrategyOptimizer(AIOptimizerConfig())

    content = """
    ```json
    [
      {"lookback": "200", "threshold": 3.5,},
      {"lookback": 10, "threshold": 0.05}
    ]
    ```
    """

    parsed = opt._parse_suggestions(content, profile)

    assert len(parsed) == 2
    # clamped
    assert parsed[0]["lookback"] == 100
    assert parsed[0]["threshold"] == 2.0
    assert parsed[1]["lookback"] == 10
    assert parsed[1]["threshold"] == 0.1


@pytest.mark.unit
def test_heuristic_suggestions_repeatable_with_seed() -> None:
    profile = StrategyProfile(
        name="Test",
        description="",
        param_definitions={
            "x": {"min": 0.0, "max": 1.0, "default": 0.5},
            "y": {"min": 0.0, "max": 10.0, "default": 5.0},
        },
        objective="sharpe",
    )
    opt = AIStrategyOptimizer(AIOptimizerConfig())

    np.random.seed(123)
    s1 = opt._heuristic_suggestions(profile, history=[], n_suggestions=3)
    np.random.seed(123)
    s2 = opt._heuristic_suggestions(profile, history=[], n_suggestions=3)

    assert s1 == s2


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_optimization_early_stops(monkeypatch) -> None:
    profile = StrategyProfile(
        name="Test",
        description="",
        param_definitions={
            "x": {"min": 0.0, "max": 1.0, "default": 0.5},
        },
        objective="sharpe_ratio",
        constraints={"max_drawdown": 0.25, "min_win_rate": 45},
    )

    cfg = AIOptimizerConfig(max_iterations=10, min_sharpe=0.5, max_drawdown=0.25, min_win_rate=0.45)
    opt = AIStrategyOptimizer(cfg)

    # Force heuristic path.
    opt._model_available = False

    # Deterministic suggestions.
    async def _suggest(profile, history, n_suggestions=5):
        return [{"x": 0.9}]

    monkeypatch.setattr(opt, "suggest_parameters", _suggest)

    def backtest_fn(params):
        # Always returns a passing result for early stopping.
        return {"sharpe_ratio": 0.8, "max_drawdown": 0.1, "win_rate": 60.0}

    result = await opt.run_optimization(
        profile=profile,
        backtest_fn=backtest_fn,
        initial_params={"x": 0.1},
        max_iterations=10,
    )

    assert result["best_params"] == {"x": 0.9}
    assert result["best_result"]["sharpe_ratio"] == 0.8
    assert result["iterations"] <= 10
