"""
Adapters for the NVIDIA Quantitative Portfolio Optimization (QPO) blueprint.

These helpers keep the blueprint as an optional dependency and provide
guardrails so the rest of Ordinis can call into QPO when the GPU stack is
available, while failing gracefully when it is not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QPO_SRC = (
    REPO_ROOT / "external" / "nvidia-blueprints" / "quantitative-portfolio-optimization" / "src"
)


class QPOEnvironmentError(RuntimeError):
    """Raised when the QPO blueprint or its dependencies are unavailable."""


def _ensure_in_sys_path(path: Path) -> None:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _import_qpo_module(module_name: str, qpo_src: Path) -> Any:
    """Import a QPO module from the external blueprint path with clear errors."""
    if not qpo_src.exists():
        raise QPOEnvironmentError(
            f"QPO blueprint not found at {qpo_src}. "
            "Clone or add the submodule under external/nvidia-blueprints/quantitative-portfolio-optimization."
        )

    _ensure_in_sys_path(qpo_src)

    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - relies on external deps
        raise QPOEnvironmentError(
            "QPO dependencies missing. Install RAPIDS/cuOpt stack as per the blueprint instructions "
            "(e.g., conda install -c rapidsai -c nvidia -c conda-forge rapids=23.12 cuda-version=12.2 "
            "&& pip install cvxpy)."
        ) from exc


@dataclass
class QPOScenarioGenerator:
    """Wrapper around QPO scenario generation utilities."""

    qpo_src: Path = DEFAULT_QPO_SRC

    def _loader(self) -> Callable[..., Any]:
        module = _import_qpo_module("scenario_generation", self.qpo_src)
        if not hasattr(module, "ForwardPathSimulator"):
            raise QPOEnvironmentError(
                "ForwardPathSimulator not found in scenario_generation module."
            )
        return module.ForwardPathSimulator

    def generate(
        self,
        fitting_data: pd.DataFrame,
        generation_dates: list[pd.Timestamp] | Any,
        n_paths: int,
        method: str = "log_gbm",
        plot_paths: bool = False,
        n_plots: int = 0,
    ) -> Any:
        """Generate synthetic forward paths using QPO's ForwardPathSimulator."""
        simulator_cls = self._loader()
        sim = simulator_cls(
            fitting_data=fitting_data,
            generation_dates=generation_dates,
            n_paths=n_paths,
            method=method,
        )
        sim.generate(plot_paths=plot_paths, n_plots=n_plots)
        return sim.simulated_paths


@dataclass
class QPOPortfolioOptimizer:
    """Wrapper around QPO CVaR optimizer (cuOpt/CVXPY)."""

    qpo_src: Path = DEFAULT_QPO_SRC

    def _load_optimizer(self) -> tuple[Any, Any]:
        cvar_optimizer = _import_qpo_module("cvar_optimizer", self.qpo_src)
        cvar_params = _import_qpo_module("cvar_parameters", self.qpo_src)
        if not hasattr(cvar_optimizer, "CVaR") or not hasattr(cvar_params, "CvarParameters"):
            raise QPOEnvironmentError("CVaR optimizer components not found in QPO blueprint.")
        return cvar_optimizer.CVaR, cvar_params.CvarParameters

    def optimize_from_returns(
        self,
        returns: pd.DataFrame,
        target_return: float = 0.001,
        max_weight: float = 0.2,
        risk_aversion: float = 0.5,
        api: str = "cvxpy",
        solver_settings: dict[str, Any] | None = None,
        execute: bool = True,
    ) -> dict[str, Any]:
        """
        Run Mean-CVaR optimization using the QPO blueprint.

        Args:
            returns: DataFrame of asset returns (columns = assets).
            target_return: Desired target return.
            max_weight: Per-asset weight cap.
            risk_aversion: CVaR penalty inside objective.
            api: Solver choice ("cvxpy" or "cuopt").
            solver_settings: Optional solver overrides.
            execute: If False, performs environment validation only.

        Returns:
            Dict containing weights, metrics, and raw results when execute=True.

        Raises:
            QPOEnvironmentError when dependencies are missing or QPO is absent.
        """
        CVaR, CvarParameters = self._load_optimizer()

        if not execute:
            return {"status": "validated", "api": api, "target_return": target_return}

        returns_dict = {col: returns[col].dropna().values for col in returns.columns}
        params = CvarParameters(
            w_min=0.0,
            w_max=max_weight,
            c_min=0.0,
            c_max=1.0,
            risk_aversion=risk_aversion,
        )

        api_settings = {"api": api}
        optimizer = CVaR(returns_dict, params, api_settings=api_settings)
        result_row, portfolio = optimizer.solve_optimization_problem(
            solver_settings=solver_settings or {}, print_results=False
        )

        weights = getattr(portfolio, "weights", None)
        metrics = {
            "expected_return": getattr(portfolio, "expected_return", None),
            "cvar": getattr(portfolio, "portfolio_cvar", None),
            "objective": getattr(result_row, "objective", None) if result_row is not None else None,
        }

        return {"weights": weights, "metrics": metrics, "result": result_row}
