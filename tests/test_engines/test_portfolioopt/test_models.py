"""Tests for PortfolioOpt data models.

This module tests OptimizationResult and ScenarioResult dataclasses.
"""

from datetime import UTC, datetime

import numpy as np

from ordinis.engines.portfolioopt import OptimizationResult, ScenarioResult


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_create_result_basic(self, sample_weights: dict[str, float]) -> None:
        """Test creating basic optimization result."""
        result = OptimizationResult(
            weights=sample_weights,
            expected_return=0.001,
            cvar=0.025,
            objective=-0.024,
            solver_api="cvxpy",
            optimization_time=0.5,
        )

        assert result.weights == sample_weights
        assert result.expected_return == 0.001
        assert result.cvar == 0.025
        assert result.objective == -0.024
        assert result.solver_api == "cvxpy"
        assert result.optimization_time == 0.5
        assert result.constraints_satisfied is True
        assert result.warnings == []

    def test_create_result_with_warnings(self, sample_weights: dict[str, float]) -> None:
        """Test creating result with constraint warnings."""
        warnings = ["CVaR exceeds limit", "Concentration too high"]

        result = OptimizationResult(
            weights=sample_weights,
            expected_return=0.001,
            cvar=0.15,
            objective=-0.149,
            solver_api="cvxpy",
            optimization_time=0.5,
            constraints_satisfied=False,
            warnings=warnings,
        )

        assert result.constraints_satisfied is False
        assert len(result.warnings) == 2
        assert "CVaR exceeds limit" in result.warnings

    def test_result_timestamp_auto_generated(self, sample_weights: dict[str, float]) -> None:
        """Test timestamp is auto-generated."""
        before = datetime.now(UTC)

        result = OptimizationResult(
            weights=sample_weights,
            expected_return=0.001,
            cvar=0.025,
            objective=-0.024,
            solver_api="cvxpy",
            optimization_time=0.5,
        )

        after = datetime.now(UTC)

        assert before <= result.timestamp <= after

    def test_to_dict_conversion(self, sample_optimization_result: OptimizationResult) -> None:
        """Test converting result to dictionary."""
        result_dict = sample_optimization_result.to_dict()

        assert isinstance(result_dict, dict)
        assert "weights" in result_dict
        assert "expected_return" in result_dict
        assert "cvar" in result_dict
        assert "objective" in result_dict
        assert "solver_api" in result_dict
        assert "optimization_time" in result_dict
        assert "timestamp" in result_dict
        assert "constraints_satisfied" in result_dict
        assert "warnings" in result_dict

        assert result_dict["weights"] == sample_optimization_result.weights
        assert result_dict["solver_api"] == "cvxpy"
        assert result_dict["constraints_satisfied"] is True

    def test_to_dict_timestamp_iso_format(
        self, sample_optimization_result: OptimizationResult
    ) -> None:
        """Test timestamp is ISO format in dictionary."""
        result_dict = sample_optimization_result.to_dict()

        timestamp_str = result_dict["timestamp"]
        assert isinstance(timestamp_str, str)
        assert "T" in timestamp_str  # ISO format includes T separator

        # Verify it can be parsed back
        parsed = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        assert isinstance(parsed, datetime)

    def test_result_with_none_metrics(self) -> None:
        """Test result with None metrics."""
        result = OptimizationResult(
            weights={"AAPL": 1.0},
            expected_return=None,
            cvar=None,
            objective=None,
            solver_api="cvxpy",
            optimization_time=0.1,
        )

        assert result.expected_return is None
        assert result.cvar is None
        assert result.objective is None

        result_dict = result.to_dict()
        assert result_dict["expected_return"] is None
        assert result_dict["cvar"] is None

    def test_empty_weights(self) -> None:
        """Test result with empty weights."""
        result = OptimizationResult(
            weights={},
            expected_return=0.0,
            cvar=0.0,
            objective=0.0,
            solver_api="cvxpy",
            optimization_time=0.1,
        )

        assert result.weights == {}
        assert len(result.weights) == 0

    def test_cuopt_solver(self) -> None:
        """Test result from cuOpt solver."""
        result = OptimizationResult(
            weights={"AAPL": 0.5, "MSFT": 0.5},
            expected_return=0.002,
            cvar=0.03,
            objective=-0.028,
            solver_api="cuopt",
            optimization_time=0.2,
        )

        assert result.solver_api == "cuopt"


class TestScenarioResult:
    """Test ScenarioResult dataclass."""

    def test_create_scenario_result(self) -> None:
        """Test creating basic scenario result."""
        paths = np.random.randn(1000, 10, 5)

        result = ScenarioResult(
            simulated_paths=paths,
            n_paths=1000,
            n_assets=5,
            method="log_gbm",
            generation_time=1.5,
        )

        assert result.n_paths == 1000
        assert result.n_assets == 5
        assert result.method == "log_gbm"
        assert result.generation_time == 1.5
        assert result.simulated_paths.shape == (1000, 10, 5)

    def test_scenario_timestamp_auto_generated(self) -> None:
        """Test timestamp is auto-generated."""
        before = datetime.now(UTC)

        result = ScenarioResult(
            simulated_paths=np.zeros((100, 5, 3)),
            n_paths=100,
            n_assets=3,
            method="log_gbm",
            generation_time=0.5,
        )

        after = datetime.now(UTC)

        assert before <= result.timestamp <= after

    def test_to_dict_conversion(self) -> None:
        """Test converting scenario result to dictionary."""
        paths = np.random.randn(500, 20, 4)

        result = ScenarioResult(
            simulated_paths=paths,
            n_paths=500,
            n_assets=4,
            method="geometric_brownian",
            generation_time=2.0,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["n_paths"] == 500
        assert result_dict["n_assets"] == 4
        assert result_dict["method"] == "geometric_brownian"
        assert result_dict["generation_time"] == 2.0
        assert "timestamp" in result_dict

        # Note: simulated_paths not included in dict (too large)
        assert "simulated_paths" not in result_dict

    def test_to_dict_timestamp_iso_format(self) -> None:
        """Test timestamp is ISO format in dictionary."""
        result = ScenarioResult(
            simulated_paths=np.zeros((10, 5, 2)),
            n_paths=10,
            n_assets=2,
            method="log_gbm",
            generation_time=0.1,
        )

        result_dict = result.to_dict()
        timestamp_str = result_dict["timestamp"]

        assert isinstance(timestamp_str, str)
        assert "T" in timestamp_str

        parsed = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        assert isinstance(parsed, datetime)

    def test_different_simulation_methods(self) -> None:
        """Test results with different simulation methods."""
        methods = ["log_gbm", "geometric_brownian", "monte_carlo", "historical"]

        for method in methods:
            result = ScenarioResult(
                simulated_paths=np.zeros((100, 10, 3)),
                n_paths=100,
                n_assets=3,
                method=method,
                generation_time=0.5,
            )

            assert result.method == method

    def test_large_scenario_result(self) -> None:
        """Test scenario result with large arrays."""
        n_paths = 10000
        n_dates = 252
        n_assets = 20

        paths = np.random.randn(n_paths, n_dates, n_assets)

        result = ScenarioResult(
            simulated_paths=paths,
            n_paths=n_paths,
            n_assets=n_assets,
            method="log_gbm",
            generation_time=5.0,
        )

        assert result.simulated_paths.shape == (n_paths, n_dates, n_assets)
        assert result.n_paths == n_paths
        assert result.n_assets == n_assets

    def test_dataframe_paths(self) -> None:
        """Test scenario result with DataFrame paths."""
        import pandas as pd

        df = pd.DataFrame(np.random.randn(100, 5), columns=["A", "B", "C", "D", "E"])

        result = ScenarioResult(
            simulated_paths=df,
            n_paths=100,
            n_assets=5,
            method="log_gbm",
            generation_time=0.3,
        )

        assert isinstance(result.simulated_paths, pd.DataFrame)
        assert result.n_assets == 5
