"""Tests for PortfolioOptEngine.

This module tests the core engine functionality including:
- Engine lifecycle (initialization, shutdown, health checks)
- Portfolio optimization
- Scenario generation
- Governance integration
- Error handling
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from ordinis.engines.base import EngineState, HealthLevel
from ordinis.engines.portfolioopt import (
    OptimizationResult,
    PortfolioOptEngine,
    PortfolioOptEngineConfig,
    ScenarioResult,
)
from ordinis.engines.portfolioopt.hooks.governance import PortfolioOptGovernanceHook
from ordinis.quant import QPOEnvironmentError


class TestEngineLifecycle:
    """Test engine lifecycle state transitions."""

    @pytest.mark.asyncio
    async def test_initial_state(self, portfolioopt_config: PortfolioOptEngineConfig) -> None:
        """Test engine starts in correct initial state."""
        engine = PortfolioOptEngine(portfolioopt_config)

        assert engine.config == portfolioopt_config
        assert engine._optimizer is None
        assert engine._scenario_gen is None
        assert engine._qpo_available is False
        assert len(engine._optimization_history) == 0

    @pytest.mark.asyncio
    async def test_initialize_success(self, portfolioopt_engine: PortfolioOptEngine) -> None:
        """Test successful engine initialization."""
        await portfolioopt_engine.initialize()

        assert portfolioopt_engine.state == EngineState.READY
        assert portfolioopt_engine.is_running
        assert portfolioopt_engine._qpo_available is True
        assert portfolioopt_engine._optimizer is not None
        assert portfolioopt_engine._scenario_gen is not None

        await portfolioopt_engine.shutdown()

    @pytest.mark.asyncio
    async def test_initialize_qpo_unavailable(
        self, portfolioopt_config: PortfolioOptEngineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test initialization when QPO is unavailable."""
        from ordinis.quant import QPOPortfolioOptimizer

        def mock_init(self, qpo_src=None):
            raise QPOEnvironmentError("QPO not available")

        monkeypatch.setattr(QPOPortfolioOptimizer, "__init__", mock_init)

        engine = PortfolioOptEngine(portfolioopt_config)
        await engine.initialize()

        assert engine.state == EngineState.READY
        assert engine._qpo_available is False

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown_success(self, initialized_engine: PortfolioOptEngine) -> None:
        """Test successful engine shutdown."""
        await initialized_engine.shutdown()

        assert initialized_engine.state == EngineState.STOPPED
        assert not initialized_engine.is_running
        assert initialized_engine._optimizer is None
        assert initialized_engine._scenario_gen is None

    @pytest.mark.asyncio
    async def test_is_available(self, initialized_engine: PortfolioOptEngine) -> None:
        """Test is_available method."""
        assert initialized_engine.is_available()

        await initialized_engine.shutdown()
        assert not initialized_engine.is_available()


class TestEngineHealthChecks:
    """Test engine health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_qpo_unavailable(
        self, portfolioopt_config: PortfolioOptEngineConfig
    ) -> None:
        """Test health check when QPO unavailable."""
        engine = PortfolioOptEngine(portfolioopt_config)
        engine._qpo_available = False
        await engine.initialize()

        status = await engine.health_check()

        assert status.level == HealthLevel.DEGRADED
        assert "QPO environment not available" in status.message
        assert status.details["qpo_available"] is False

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_success(self, initialized_engine: PortfolioOptEngine) -> None:
        """Test successful health check."""
        status = await initialized_engine.health_check()

        assert status.level == HealthLevel.HEALTHY
        assert "operational" in status.message.lower()
        assert status.details["qpo_available"] is True
        assert "default_api" in status.details
        assert status.details["optimizations_run"] == 0

    @pytest.mark.asyncio
    async def test_health_check_tracks_operations(
        self, initialized_engine: PortfolioOptEngine, sample_returns: pd.DataFrame
    ) -> None:
        """Test health check tracks number of optimizations."""
        # Run an optimization
        await initialized_engine.optimize(sample_returns)

        status = await initialized_engine.health_check()

        assert status.details["optimizations_run"] == 1


class TestOptimization:
    """Test portfolio optimization functionality."""

    @pytest.mark.asyncio
    async def test_optimize_basic(
        self, initialized_engine: PortfolioOptEngine, sample_returns: pd.DataFrame
    ) -> None:
        """Test basic optimization."""
        result = await initialized_engine.optimize(sample_returns)

        assert isinstance(result, OptimizationResult)
        assert len(result.weights) == len(sample_returns.columns)
        assert result.solver_api == "cvxpy"
        assert result.optimization_time >= 0
        assert isinstance(result.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_optimize_with_custom_params(
        self, initialized_engine: PortfolioOptEngine, sample_returns: pd.DataFrame
    ) -> None:
        """Test optimization with custom parameters."""
        result = await initialized_engine.optimize(
            sample_returns,
            target_return=0.002,
            max_weight=0.15,
            risk_aversion=1.0,
            api="cvxpy",
        )

        assert result.solver_api == "cvxpy"
        assert result.expected_return is not None

    @pytest.mark.asyncio
    async def test_optimize_not_available(
        self, portfolioopt_config: PortfolioOptEngineConfig, sample_returns: pd.DataFrame
    ) -> None:
        """Test optimization fails when engine not available."""
        engine = PortfolioOptEngine(portfolioopt_config)

        with pytest.raises(RuntimeError, match="not available"):
            await engine.optimize(sample_returns)

    @pytest.mark.asyncio
    async def test_optimize_validates_concentration(
        self, initialized_engine: PortfolioOptEngine, sample_returns: pd.DataFrame
    ) -> None:
        """Test optimization validates concentration constraint."""

        # Mock optimizer to return concentrated portfolio
        def mock_optimize_concentrated(**kwargs):
            return {
                "weights": {"AAPL": 0.80, "MSFT": 0.20},  # 80% concentration
                "metrics": {
                    "expected_return": 0.001,
                    "cvar": 0.025,
                    "objective": -0.024,
                },
            }

        initialized_engine._optimizer.optimize_from_returns = Mock(
            side_effect=mock_optimize_concentrated
        )

        result = await initialized_engine.optimize(sample_returns)

        assert result.constraints_satisfied is False
        assert any("Concentration limit exceeded" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_optimize_validates_diversification(
        self, initialized_engine: PortfolioOptEngine, sample_returns: pd.DataFrame
    ) -> None:
        """Test optimization validates diversification constraint."""

        # Mock optimizer to return under-diversified portfolio
        def mock_optimize_concentrated(**kwargs):
            return {
                "weights": {"AAPL": 0.50, "MSFT": 0.50},  # Only 2 assets
                "metrics": {
                    "expected_return": 0.001,
                    "cvar": 0.025,
                    "objective": -0.024,
                },
            }

        initialized_engine._optimizer.optimize_from_returns = Mock(
            side_effect=mock_optimize_concentrated
        )

        result = await initialized_engine.optimize(sample_returns)

        assert result.constraints_satisfied is False
        assert any("Diversification below minimum" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_optimize_validates_cvar(
        self, initialized_engine: PortfolioOptEngine, sample_returns: pd.DataFrame
    ) -> None:
        """Test optimization validates CVaR constraint."""

        # Mock optimizer to return high CVaR
        def mock_optimize_high_cvar(**kwargs):
            return {
                "weights": {"AAPL": 0.20, "MSFT": 0.20, "GOOGL": 0.20, "AMZN": 0.20, "TSLA": 0.20},
                "metrics": {
                    "expected_return": 0.001,
                    "cvar": 0.15,  # Exceeds default max_cvar of 0.10
                    "objective": -0.149,
                },
            }

        initialized_engine._optimizer.optimize_from_returns = Mock(
            side_effect=mock_optimize_high_cvar
        )

        result = await initialized_engine.optimize(sample_returns)

        assert result.constraints_satisfied is False
        assert any("CVaR exceeds limit" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_optimize_adds_to_history(
        self, initialized_engine: PortfolioOptEngine, sample_returns: pd.DataFrame
    ) -> None:
        """Test optimization adds result to history."""
        assert len(initialized_engine.get_optimization_history()) == 0

        await initialized_engine.optimize(sample_returns)

        history = initialized_engine.get_optimization_history()
        assert len(history) == 1
        assert isinstance(history[0], OptimizationResult)

    @pytest.mark.asyncio
    async def test_get_last_optimization(
        self, initialized_engine: PortfolioOptEngine, sample_returns: pd.DataFrame
    ) -> None:
        """Test get_last_optimization returns most recent result."""
        assert initialized_engine.get_last_optimization() is None

        result1 = await initialized_engine.optimize(sample_returns, target_return=0.001)
        assert initialized_engine.get_last_optimization() == result1

        result2 = await initialized_engine.optimize(sample_returns, target_return=0.002)
        assert initialized_engine.get_last_optimization() == result2

    @pytest.mark.asyncio
    async def test_optimize_with_solver_settings(
        self, initialized_engine: PortfolioOptEngine, sample_returns: pd.DataFrame
    ) -> None:
        """Test optimization with custom solver settings."""
        solver_settings = {"verbose": True, "max_iters": 1000}

        result = await initialized_engine.optimize(sample_returns, solver_settings=solver_settings)

        assert isinstance(result, OptimizationResult)

        # Verify solver_settings was passed to optimizer
        call_args = initialized_engine._optimizer.optimize_from_returns.call_args
        assert call_args[1]["solver_settings"] == solver_settings


class TestScenarioGeneration:
    """Test scenario generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_scenarios_basic(
        self,
        initialized_engine: PortfolioOptEngine,
        sample_returns: pd.DataFrame,
        sample_generation_dates: list[pd.Timestamp],
    ) -> None:
        """Test basic scenario generation."""
        result = await initialized_engine.generate_scenarios(
            sample_returns, sample_generation_dates
        )

        assert isinstance(result, ScenarioResult)
        assert result.n_paths == initialized_engine.config.n_paths
        assert result.n_assets == len(sample_returns.columns)
        assert result.method == initialized_engine.config.simulation_method
        assert result.generation_time >= 0

    @pytest.mark.asyncio
    async def test_generate_scenarios_custom_params(
        self,
        initialized_engine: PortfolioOptEngine,
        sample_returns: pd.DataFrame,
        sample_generation_dates: list[pd.Timestamp],
    ) -> None:
        """Test scenario generation with custom parameters."""
        result = await initialized_engine.generate_scenarios(
            sample_returns, sample_generation_dates, n_paths=5000, method="geometric_brownian"
        )

        assert result.n_paths == 5000
        assert result.method == "geometric_brownian"

    @pytest.mark.asyncio
    async def test_generate_scenarios_not_available(
        self,
        portfolioopt_config: PortfolioOptEngineConfig,
        sample_returns: pd.DataFrame,
        sample_generation_dates: list[pd.Timestamp],
    ) -> None:
        """Test scenario generation fails when engine not available."""
        engine = PortfolioOptEngine(portfolioopt_config)

        with pytest.raises(RuntimeError, match="not available"):
            await engine.generate_scenarios(sample_returns, sample_generation_dates)

    @pytest.mark.asyncio
    async def test_generate_scenarios_validates_paths(
        self,
        initialized_engine: PortfolioOptEngine,
        sample_returns: pd.DataFrame,
        sample_generation_dates: list[pd.Timestamp],
    ) -> None:
        """Test scenario generation validates path dimensions."""
        result = await initialized_engine.generate_scenarios(
            sample_returns, sample_generation_dates, n_paths=2000
        )

        paths = result.simulated_paths
        assert paths.shape[0] == 2000  # n_paths
        assert paths.shape[1] == len(sample_generation_dates)  # n_dates
        assert paths.shape[2] == len(sample_returns.columns)  # n_assets


class TestGovernanceIntegration:
    """Test governance hook integration."""

    @pytest.mark.asyncio
    async def test_optimize_with_preflight(
        self,
        portfolioopt_engine_with_governance: PortfolioOptEngine,
        sample_returns: pd.DataFrame,
    ) -> None:
        """Test optimization runs preflight check."""
        await portfolioopt_engine_with_governance.initialize()

        result = await portfolioopt_engine_with_governance.optimize(sample_returns)

        assert isinstance(result, OptimizationResult)

        await portfolioopt_engine_with_governance.shutdown()

    @pytest.mark.asyncio
    async def test_optimize_preflight_denied(
        self,
        portfolioopt_config: PortfolioOptEngineConfig,
        governance_hook_strict: "PortfolioOptGovernanceHook",
        sample_returns: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
        mock_qpo_optimizer: MagicMock,
        mock_qpo_scenario_gen: MagicMock,
    ) -> None:
        """Test optimization blocked by preflight."""
        # Mock the QPO classes in the engine module where they are used
        monkeypatch.setattr(
            "ordinis.engines.portfolioopt.core.engine.QPOPortfolioOptimizer",
            lambda qpo_src=None: mock_qpo_optimizer,
        )
        monkeypatch.setattr(
            "ordinis.engines.portfolioopt.core.engine.QPOScenarioGenerator",
            lambda qpo_src=None: mock_qpo_scenario_gen,
        )

        # Create engine with governance
        config = PortfolioOptEngineConfig(
            require_preflight=True,
            target_return=0.10,  # Exceeds strict limit
        )
        engine = PortfolioOptEngine(config, governance_hook_strict)
        engine._optimizer = mock_qpo_optimizer
        engine._scenario_gen = mock_qpo_scenario_gen
        engine._qpo_available = True

        await engine.initialize()

        with pytest.raises(RuntimeError, match="blocked by governance"):
            await engine.optimize(sample_returns, target_return=0.10)

        await engine.shutdown()

    @pytest.mark.asyncio
    async def test_optimize_creates_audit_record(
        self,
        portfolioopt_engine_with_governance: PortfolioOptEngine,
        governance_hook: "PortfolioOptGovernanceHook",
        sample_returns: pd.DataFrame,
    ) -> None:
        """Test optimization creates audit record."""
        await portfolioopt_engine_with_governance.initialize()

        initial_count = len(governance_hook.get_audit_log())

        await portfolioopt_engine_with_governance.optimize(sample_returns)

        audit_log = governance_hook.get_audit_log()
        assert len(audit_log) > initial_count

        last_record = audit_log[-1]
        assert last_record.action == "optimize"
        assert last_record.engine == portfolioopt_engine_with_governance.name

        await portfolioopt_engine_with_governance.shutdown()

    @pytest.mark.asyncio
    async def test_optimize_without_governance(
        self,
        portfolioopt_config_no_governance: PortfolioOptEngineConfig,
        sample_returns: pd.DataFrame,
        monkeypatch: pytest.MonkeyPatch,
        mock_qpo_optimizer: MagicMock,
        mock_qpo_scenario_gen: MagicMock,
    ) -> None:
        """Test optimization without governance hook."""
        # Mock the QPO classes in the engine module where they are used
        monkeypatch.setattr(
            "ordinis.engines.portfolioopt.core.engine.QPOPortfolioOptimizer",
            lambda qpo_src=None: mock_qpo_optimizer,
        )
        monkeypatch.setattr(
            "ordinis.engines.portfolioopt.core.engine.QPOScenarioGenerator",
            lambda qpo_src=None: mock_qpo_scenario_gen,
        )

        engine = PortfolioOptEngine(portfolioopt_config_no_governance)
        engine._optimizer = mock_qpo_optimizer
        engine._scenario_gen = mock_qpo_scenario_gen
        engine._qpo_available = True

        await engine.initialize()
        engine._qpo_available = True

        result = await engine.optimize(sample_returns)

        assert isinstance(result, OptimizationResult)

        await engine.shutdown()


class TestOperationTracking:
    """Test operation tracking and metrics."""

    @pytest.mark.asyncio
    async def test_optimize_tracks_operation(
        self, initialized_engine: PortfolioOptEngine, sample_returns: pd.DataFrame
    ) -> None:
        """Test optimization is tracked as an operation."""
        initial_metrics = initialized_engine.get_metrics()
        initial_count = initial_metrics.requests_total

        await initialized_engine.optimize(sample_returns)

        metrics = initialized_engine.get_metrics()
        assert metrics.requests_total == initial_count + 1
        assert metrics.requests_failed == 0

    @pytest.mark.asyncio
    async def test_scenario_generation_tracks_operation(
        self,
        initialized_engine: PortfolioOptEngine,
        sample_returns: pd.DataFrame,
        sample_generation_dates: list[pd.Timestamp],
    ) -> None:
        """Test scenario generation is tracked as an operation."""
        initial_metrics = initialized_engine.get_metrics()
        initial_count = initial_metrics.requests_total

        await initialized_engine.generate_scenarios(sample_returns, sample_generation_dates)

        metrics = initialized_engine.get_metrics()
        assert metrics.requests_total == initial_count + 1
        assert metrics.requests_failed == 0


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_optimize_optimizer_not_initialized(
        self, portfolioopt_config: PortfolioOptEngineConfig, sample_returns: pd.DataFrame
    ) -> None:
        """Test optimization fails gracefully when optimizer not initialized."""
        engine = PortfolioOptEngine(portfolioopt_config)
        engine._qpo_available = True  # Fake availability
        engine._state = EngineState.READY

        with pytest.raises(RuntimeError, match="Optimizer not initialized"):
            await engine.optimize(sample_returns)

    @pytest.mark.asyncio
    async def test_scenario_gen_not_initialized(
        self,
        portfolioopt_config: PortfolioOptEngineConfig,
        sample_returns: pd.DataFrame,
        sample_generation_dates: list[pd.Timestamp],
    ) -> None:
        """Test scenario generation fails gracefully when generator not initialized."""
        engine = PortfolioOptEngine(portfolioopt_config)
        engine._qpo_available = True  # Fake availability
        engine._state = EngineState.READY

        with pytest.raises(RuntimeError, match="Scenario generator not initialized"):
            await engine.generate_scenarios(sample_returns, sample_generation_dates)

    @pytest.mark.asyncio
    async def test_optimize_with_empty_returns(
        self, initialized_engine: PortfolioOptEngine
    ) -> None:
        """Test optimization with empty returns DataFrame."""
        empty_returns = pd.DataFrame()

        # Should not raise, optimizer will handle
        result = await initialized_engine.optimize(empty_returns)
        assert isinstance(result, OptimizationResult)


class TestEngineProperties:
    """Test engine property accessors."""

    def test_config_property(self, portfolioopt_engine: PortfolioOptEngine) -> None:
        """Test config property accessor."""
        assert portfolioopt_engine.config.engine_id == "test-portfolioopt"

    def test_get_optimization_history_empty(self, portfolioopt_engine: PortfolioOptEngine) -> None:
        """Test get_optimization_history when empty."""
        history = portfolioopt_engine.get_optimization_history()

        assert isinstance(history, list)
        assert len(history) == 0

    def test_get_optimization_history_returns_copy(
        self, portfolioopt_engine: PortfolioOptEngine
    ) -> None:
        """Test get_optimization_history returns a copy."""
        history1 = portfolioopt_engine.get_optimization_history()
        history2 = portfolioopt_engine.get_optimization_history()

        assert history1 is not history2
