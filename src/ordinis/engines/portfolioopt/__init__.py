"""
PortfolioOpt Engine - GPU-Accelerated Portfolio Optimization.

Provides Mean-CVaR portfolio optimization using NVIDIA's Quantitative
Portfolio Optimization (QPO) blueprint with governance integration.

Key Components:
- PortfolioOptEngine: Main engine class with optimize() and generate_scenarios()
- PortfolioOptEngineConfig: Configuration with risk limits and solver settings
- PortfolioOptGovernanceHook: Risk and compliance validation

Example:
    >>> from ordinis.engines.portfolioopt import (
    ...     PortfolioOptEngine,
    ...     PortfolioOptEngineConfig,
    ...     PortfolioOptGovernanceHook,
    ... )
    >>> config = PortfolioOptEngineConfig(default_api="cvxpy")
    >>> hook = PortfolioOptGovernanceHook()
    >>> engine = PortfolioOptEngine(config, governance_hook=hook)
    >>> await engine.initialize()
    >>> result = await engine.optimize(returns_df)
"""

from .core import (
    OptimizationResult,
    PortfolioOptEngine,
    PortfolioOptEngineConfig,
    ScenarioResult,
)
from .hooks import (
    DataQualityRule,
    PortfolioOptGovernanceHook,
    RiskLimitRule,
    SolverValidationRule,
)

__all__ = [
    # Hooks
    "DataQualityRule",
    # Core
    "OptimizationResult",
    "PortfolioOptEngine",
    "PortfolioOptEngineConfig",
    "PortfolioOptGovernanceHook",
    "RiskLimitRule",
    "ScenarioResult",
    "SolverValidationRule",
]
