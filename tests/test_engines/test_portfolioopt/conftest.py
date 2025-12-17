"""Shared fixtures for PortfolioOpt engine tests.

This module provides mock implementations and fixtures for testing
the PortfolioOpt engine with mocked NVIDIA QPO dependencies.
"""

from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from ordinis.engines.portfolioopt import (
    OptimizationResult,
    PortfolioOptEngine,
    PortfolioOptEngineConfig,
    PortfolioOptGovernanceHook,
)


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Provide sample returns data for testing.

    Returns:
        DataFrame with 30 periods and 5 assets.
    """
    np.random.seed(42)
    n_periods = 30
    n_assets = 5

    returns = np.random.randn(n_periods, n_assets) * 0.02 + 0.001
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

    return pd.DataFrame(returns, columns=assets)


@pytest.fixture
def sample_weights() -> dict[str, float]:
    """Provide sample portfolio weights.

    Returns:
        Dictionary of asset weights summing to 1.0.
    """
    return {
        "AAPL": 0.25,
        "MSFT": 0.20,
        "GOOGL": 0.20,
        "AMZN": 0.20,
        "TSLA": 0.15,
    }


@pytest.fixture
def sample_optimization_result(sample_weights: dict[str, float]) -> OptimizationResult:
    """Provide sample optimization result.

    Args:
        sample_weights: Sample portfolio weights fixture.

    Returns:
        OptimizationResult instance.
    """
    return OptimizationResult(
        weights=sample_weights,
        expected_return=0.0015,
        cvar=0.025,
        objective=-0.0235,
        solver_api="cvxpy",
        optimization_time=0.5,
        constraints_satisfied=True,
        warnings=[],
    )


@pytest.fixture
def mock_qpo_optimizer() -> MagicMock:
    """Provide mock QPO optimizer.

    Returns:
        MagicMock for QPOPortfolioOptimizer.
    """
    optimizer = MagicMock()

    # Mock optimize_from_returns to return valid result
    def mock_optimize(
        returns: pd.DataFrame,
        target_return: float | None = None,
        max_weight: float | None = None,
        risk_aversion: float | None = None,
        api: str = "cvxpy",
        solver_settings: dict[str, Any] | None = None,
        execute: bool = True,
    ) -> dict[str, Any]:
        # Return empty dict for validation call (execute=False)
        if (
            not execute
            or len(returns) == 0
            or (len(returns.columns) == 1 and "test" in returns.columns)
        ):
            return {}

        # Create weights that respect max_weight constraint
        n_assets = len(returns.columns)
        max_w = max_weight or 0.20
        weights = {col: max_w for col in returns.columns}

        # Normalize to sum to 1
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}

        return {
            "weights": weights,
            "metrics": {
                "expected_return": target_return or 0.001,
                "cvar": 0.025,
                "objective": -0.02,
            },
        }

    optimizer.optimize_from_returns = Mock(side_effect=mock_optimize)
    return optimizer


@pytest.fixture
def mock_qpo_scenario_gen() -> MagicMock:
    """Provide mock QPO scenario generator.

    Returns:
        MagicMock for QPOScenarioGenerator.
    """
    gen = MagicMock()

    def mock_generate(
        fitting_data: pd.DataFrame,
        generation_dates: list[pd.Timestamp],
        n_paths: int = 1000,
        method: str = "log_gbm",
    ) -> np.ndarray:
        n_dates = len(generation_dates)
        n_assets = len(fitting_data.columns)

        # Generate random paths
        np.random.seed(42)
        return np.random.randn(n_paths, n_dates, n_assets)

    gen.generate = Mock(side_effect=mock_generate)
    return gen


@pytest.fixture
def portfolioopt_config() -> PortfolioOptEngineConfig:
    """Provide default PortfolioOpt configuration.

    Returns:
        PortfolioOptEngineConfig instance with default settings.
    """
    return PortfolioOptEngineConfig(
        engine_id="test-portfolioopt",
        engine_name="TestPortfolioOptEngine",
        default_api="cvxpy",
        target_return=0.001,
        max_weight=0.20,
        risk_aversion=0.5,
    )


@pytest.fixture
def portfolioopt_config_strict() -> PortfolioOptEngineConfig:
    """Provide strict PortfolioOpt configuration.

    Returns:
        PortfolioOptEngineConfig with strict constraints.
    """
    return PortfolioOptEngineConfig(
        engine_id="test-portfolioopt-strict",
        default_api="cvxpy",
        target_return=0.0005,
        max_weight=0.15,
        max_concentration=0.20,
        min_diversification=5,
        max_cvar=0.05,
        require_preflight=True,
    )


@pytest.fixture
def portfolioopt_config_no_governance() -> PortfolioOptEngineConfig:
    """Provide PortfolioOpt configuration with governance disabled.

    Returns:
        PortfolioOptEngineConfig with governance disabled.
    """
    return PortfolioOptEngineConfig(
        engine_id="test-portfolioopt-nogov",
        enable_governance=False,
        require_preflight=False,
    )


@pytest.fixture
def governance_hook() -> PortfolioOptGovernanceHook:
    """Provide PortfolioOpt governance hook.

    Returns:
        PortfolioOptGovernanceHook instance.
    """
    hook = PortfolioOptGovernanceHook.__new__(PortfolioOptGovernanceHook)
    hook.engine_name = "TestPortfolioOpt"
    hook._policy_version = "1.0.0"
    from ordinis.engines.portfolioopt.hooks.governance import (
        DataQualityRule,
        RiskLimitRule,
        SolverValidationRule,
    )

    hook.risk_rule = RiskLimitRule()
    hook.data_rule = DataQualityRule()
    hook.solver_rule = SolverValidationRule()
    hook._audit_log = []
    return hook


@pytest.fixture
def governance_hook_strict() -> PortfolioOptGovernanceHook:
    """Provide strict governance hook.

    Returns:
        PortfolioOptGovernanceHook with strict rules.
    """
    from ordinis.engines.portfolioopt.hooks.governance import (
        DataQualityRule,
        RiskLimitRule,
        SolverValidationRule,
    )

    hook = PortfolioOptGovernanceHook.__new__(PortfolioOptGovernanceHook)
    hook.engine_name = "TestPortfolioOpt"
    hook._policy_version = "1.0.0"
    hook.risk_rule = RiskLimitRule(
        max_target_return=0.01,
        max_weight_per_asset=0.15,
        min_assets=5,
    )
    hook.data_rule = DataQualityRule(
        min_periods=30,
        max_periods=5000,
    )
    hook.solver_rule = SolverValidationRule()
    hook._audit_log = []
    return hook


@pytest.fixture
def portfolioopt_engine(
    portfolioopt_config: PortfolioOptEngineConfig,
    monkeypatch: pytest.MonkeyPatch,
    mock_qpo_optimizer: MagicMock,
    mock_qpo_scenario_gen: MagicMock,
) -> PortfolioOptEngine:
    """Provide PortfolioOpt engine with mocked QPO.

    Args:
        portfolioopt_config: Config fixture.
        monkeypatch: Pytest monkeypatch fixture.
        mock_qpo_optimizer: Mock optimizer fixture.
        mock_qpo_scenario_gen: Mock scenario gen fixture.

    Returns:
        PortfolioOptEngine instance with mocked dependencies.
    """
    # Mock the QPO classes in the engine module where they are used
    monkeypatch.setattr(
        "ordinis.engines.portfolioopt.core.engine.QPOPortfolioOptimizer",
        lambda qpo_src=None: mock_qpo_optimizer,
    )
    monkeypatch.setattr(
        "ordinis.engines.portfolioopt.core.engine.QPOScenarioGenerator",
        lambda qpo_src=None: mock_qpo_scenario_gen,
    )

    engine = PortfolioOptEngine(portfolioopt_config)

    return engine


@pytest.fixture
def portfolioopt_engine_with_governance(
    portfolioopt_config: PortfolioOptEngineConfig,
    governance_hook: PortfolioOptGovernanceHook,
    monkeypatch: pytest.MonkeyPatch,
    mock_qpo_optimizer: MagicMock,
    mock_qpo_scenario_gen: MagicMock,
) -> PortfolioOptEngine:
    """Provide PortfolioOpt engine with governance hook.

    Args:
        portfolioopt_config: Config fixture.
        governance_hook: Governance hook fixture.
        monkeypatch: Pytest monkeypatch fixture.
        mock_qpo_optimizer: Mock optimizer fixture.
        mock_qpo_scenario_gen: Mock scenario gen fixture.

    Returns:
        PortfolioOptEngine with governance enabled.
    """
    # Mock the QPO classes in the engine module where they are used
    monkeypatch.setattr(
        "ordinis.engines.portfolioopt.core.engine.QPOPortfolioOptimizer",
        lambda qpo_src=None: mock_qpo_optimizer,
    )
    monkeypatch.setattr(
        "ordinis.engines.portfolioopt.core.engine.QPOScenarioGenerator",
        lambda qpo_src=None: mock_qpo_scenario_gen,
    )

    engine = PortfolioOptEngine(portfolioopt_config, governance_hook)

    return engine


@pytest.fixture
async def initialized_engine(portfolioopt_engine: PortfolioOptEngine) -> PortfolioOptEngine:
    """Provide initialized PortfolioOpt engine.

    Args:
        portfolioopt_engine: Engine fixture.

    Yields:
        Initialized PortfolioOptEngine instance.
    """
    await portfolioopt_engine.initialize()
    yield portfolioopt_engine

    # Cleanup
    if portfolioopt_engine.is_running:
        await portfolioopt_engine.shutdown()


@pytest.fixture
def sample_generation_dates() -> list[pd.Timestamp]:
    """Provide sample generation dates for scenario testing.

    Returns:
        List of 10 daily timestamps.
    """
    return pd.date_range("2024-01-01", periods=10, freq="D").tolist()
