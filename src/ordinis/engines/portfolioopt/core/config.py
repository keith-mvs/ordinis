"""
PortfolioOpt Engine Configuration.

Defines configuration parameters for the GPU-accelerated portfolio optimization engine
that wraps NVIDIA's Quantitative Portfolio Optimization (QPO) blueprint.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ordinis.engines.base import BaseEngineConfig


@dataclass
class PortfolioOptEngineConfig(BaseEngineConfig):
    """
    Configuration for the PortfolioOptEngine.

    Attributes:
        engine_id: Unique identifier for the engine.
        engine_name: Display name for the engine.
        qpo_src: Path to the QPO blueprint source directory.
        default_api: Default solver API ("cvxpy" or "cuopt").
        target_return: Default target return for optimization.
        max_weight: Default maximum weight per asset.
        risk_aversion: Default CVaR risk aversion parameter.
        n_paths: Default number of simulation paths for scenario generation.
        simulation_method: Default method for path simulation.
        enable_governance: Whether to enable governance hooks.
        require_preflight: Whether to require preflight approval before optimization.
    """

    engine_id: str = "portfolioopt"
    engine_name: str = "PortfolioOpt Engine"
    qpo_src: Path | None = None
    default_api: Literal["cvxpy", "cuopt"] = "cvxpy"
    target_return: float = 0.001
    max_weight: float = 0.20
    risk_aversion: float = 0.5
    n_paths: int = 1000
    simulation_method: str = "log_gbm"
    enable_governance: bool = True
    require_preflight: bool = True

    # Optimization constraints
    min_weight: float = 0.0
    max_concentration: float = 0.25  # Max single-asset concentration
    min_diversification: int = 5  # Minimum number of assets with non-zero weights

    # Risk limits
    max_cvar: float = 0.10  # Maximum acceptable CVaR (10%)
    max_volatility: float = 0.25  # Maximum portfolio volatility (25% annualized)

    # Solver settings
    solver_timeout: float = 60.0  # Seconds
    solver_verbose: bool = False

    def validate(self) -> list[str]:
        """Validate configuration parameters."""
        errors = super().validate()

        if self.target_return < 0:
            errors.append("target_return must be non-negative")
        if not 0 < self.max_weight <= 1:
            errors.append("max_weight must be between 0 and 1")
        if self.risk_aversion < 0:
            errors.append("risk_aversion must be non-negative")
        if self.n_paths < 100:
            errors.append("n_paths should be at least 100 for statistical significance")
        if not 0 < self.max_concentration <= 1:
            errors.append("max_concentration must be between 0 and 1")
        if self.min_diversification < 1:
            errors.append("min_diversification must be at least 1")
        if self.max_cvar <= 0:
            errors.append("max_cvar must be positive")
        if self.solver_timeout <= 0:
            errors.append("solver_timeout must be positive")

        return errors
