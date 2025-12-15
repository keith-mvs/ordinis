"""
Tests for QPO adapter wrappers.

These tests avoid importing the real QPO dependency stack by stubbing the
import helper.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ordinis.quant import QPOEnvironmentError, QPOPortfolioOptimizer, QPOScenarioGenerator
import ordinis.quant.qpo_adapter as qa


def test_missing_qpo_path_raises():
    """Scenario generation should fail clearly when the blueprint path is absent."""
    missing = Path("/tmp/not-present-qpo")
    gen = QPOScenarioGenerator(qpo_src=missing)

    with pytest.raises(QPOEnvironmentError):
        gen.generate(pd.DataFrame(), [], n_paths=1)


def test_optimizer_executes_with_stubbed_import(monkeypatch):
    """Optimizer wrapper should call into stubbed QPO modules and return weights/metrics."""

    class StubParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class StubPortfolio:
        def __init__(self):
            self.weights = {"A": 0.6, "B": 0.4}
            self.expected_return = 0.012
            self.portfolio_cvar = 0.08

    class StubCVaR:
        def __init__(self, returns_dict, params, api_settings=None):
            self.returns_dict = returns_dict
            self.params = params
            self.api_settings = api_settings

        def solve_optimization_problem(self, solver_settings=None, print_results=False):
            return SimpleNamespace(objective=-0.5), StubPortfolio()

    def fake_import(module_name: str, qpo_src: Path):
        if module_name == "cvar_optimizer":
            return SimpleNamespace(CVaR=StubCVaR)
        if module_name == "cvar_parameters":
            return SimpleNamespace(CvarParameters=StubParams)
        raise AssertionError(f"Unexpected module import: {module_name}")

    monkeypatch.setattr(qa, "_import_qpo_module", fake_import)

    optimizer = QPOPortfolioOptimizer(qpo_src=Path("/tmp/qpo-stub"))
    df = pd.DataFrame({"A": [0.01, 0.02], "B": [-0.01, 0.03]})

    result = optimizer.optimize_from_returns(df, target_return=0.001, execute=True)

    assert result["weights"] == {"A": 0.6, "B": 0.4}
    assert result["metrics"]["expected_return"] == pytest.approx(0.012)
    assert result["metrics"]["cvar"] == pytest.approx(0.08)
