"""
PortfolioOpt Engine - GPU-Accelerated Portfolio Optimization.

Wraps NVIDIA's Quantitative Portfolio Optimization (QPO) blueprint to provide
Mean-CVaR optimization with governance integration and standardized interfaces.

Capabilities:
- Scenario generation via forward path simulation
- Mean-CVaR portfolio optimization (CVXPY or cuOpt)
- Risk metrics calculation
- Governance preflight/audit integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from ordinis.engines.base import (
    AuditRecord,
    BaseEngine,
    HealthLevel,
    HealthStatus,
)
from ordinis.quant import (
    DEFAULT_QPO_SRC,
    QPOEnvironmentError,
    QPOPortfolioOptimizer,
    QPOScenarioGenerator,
)

from .config import PortfolioOptEngineConfig

if TYPE_CHECKING:
    from ordinis.engines.base import GovernanceHook

_logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a portfolio optimization run."""

    weights: dict[str, float]
    expected_return: float | None
    cvar: float | None
    objective: float | None
    solver_api: str
    optimization_time: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    constraints_satisfied: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "weights": self.weights,
            "expected_return": self.expected_return,
            "cvar": self.cvar,
            "objective": self.objective,
            "solver_api": self.solver_api,
            "optimization_time": self.optimization_time,
            "timestamp": self.timestamp.isoformat(),
            "constraints_satisfied": self.constraints_satisfied,
            "warnings": self.warnings,
        }


@dataclass
class ScenarioResult:
    """Result of scenario generation."""

    simulated_paths: Any  # numpy array or DataFrame
    n_paths: int
    n_assets: int
    method: str
    generation_time: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "n_paths": self.n_paths,
            "n_assets": self.n_assets,
            "method": self.method,
            "generation_time": self.generation_time,
            "timestamp": self.timestamp.isoformat(),
        }


class PortfolioOptEngine(BaseEngine[PortfolioOptEngineConfig]):
    """
    GPU-accelerated portfolio optimization engine.

    Wraps NVIDIA QPO blueprint with governance integration, providing:
    - Mean-CVaR optimization via CVXPY or cuOpt
    - Forward path simulation for scenario generation
    - Risk constraint validation
    - Audit trail for all optimizations

    Example:
        >>> config = PortfolioOptEngineConfig(default_api="cvxpy")
        >>> engine = PortfolioOptEngine(config)
        >>> await engine.initialize()
        >>> result = await engine.optimize(returns_df, target_return=0.001)
        >>> print(result.weights)
    """

    def __init__(
        self,
        config: PortfolioOptEngineConfig | None = None,
        governance_hook: GovernanceHook | None = None,
    ) -> None:
        """Initialize the PortfolioOptEngine."""
        config = config or PortfolioOptEngineConfig()
        super().__init__(config, governance_hook)

        self._qpo_src = config.qpo_src or DEFAULT_QPO_SRC
        self._optimizer: QPOPortfolioOptimizer | None = None
        self._scenario_gen: QPOScenarioGenerator | None = None
        self._qpo_available: bool = False
        self._optimization_history: list[OptimizationResult] = []

    async def _do_initialize(self) -> None:
        """Initialize QPO components."""
        try:
            self._optimizer = QPOPortfolioOptimizer(qpo_src=self._qpo_src)
            self._scenario_gen = QPOScenarioGenerator(qpo_src=self._qpo_src)

            # Validate environment without running optimization
            self._optimizer.optimize_from_returns(
                pd.DataFrame({"test": [0.01, 0.02, -0.01]}),
                execute=False,
            )
            self._qpo_available = True
            _logger.info("QPO environment validated successfully")

        except QPOEnvironmentError as e:
            _logger.warning("QPO environment not available: %s", e)
            self._qpo_available = False

    async def _do_shutdown(self) -> None:
        """Shutdown engine resources."""
        self._optimizer = None
        self._scenario_gen = None
        _logger.info("PortfolioOptEngine shutdown complete")

    async def _do_health_check(self) -> HealthStatus:
        """Check engine health."""
        if not self._qpo_available:
            return HealthStatus(
                level=HealthLevel.DEGRADED,
                message="QPO environment not available - optimization disabled",
                details={"qpo_available": False, "qpo_src": str(self._qpo_src)},
            )

        return HealthStatus(
            level=HealthLevel.HEALTHY,
            message="PortfolioOptEngine operational",
            details={
                "qpo_available": True,
                "default_api": self.config.default_api,
                "optimizations_run": len(self._optimization_history),
            },
        )

    def is_available(self) -> bool:
        """Check if QPO optimization is available."""
        return self._qpo_available and self.is_running

    async def optimize(
        self,
        returns: pd.DataFrame,
        target_return: float | None = None,
        max_weight: float | None = None,
        risk_aversion: float | None = None,
        api: str | None = None,
        solver_settings: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """
        Run Mean-CVaR portfolio optimization.

        Args:
            returns: DataFrame of asset returns (columns = assets).
            target_return: Desired target return (default from config).
            max_weight: Per-asset weight cap (default from config).
            risk_aversion: CVaR penalty (default from config).
            api: Solver choice ("cvxpy" or "cuopt", default from config).
            solver_settings: Optional solver overrides.

        Returns:
            OptimizationResult with weights and metrics.

        Raises:
            RuntimeError: If engine not initialized or QPO unavailable.
            QPOEnvironmentError: If optimization fails.
        """
        if not self.is_available():
            raise RuntimeError("PortfolioOptEngine not available for optimization")

        target_return = target_return if target_return is not None else self.config.target_return
        max_weight = max_weight if max_weight is not None else self.config.max_weight
        risk_aversion = risk_aversion if risk_aversion is not None else self.config.risk_aversion
        api = api or self.config.default_api

        context = {
            "operation": "optimize",
            "n_assets": len(returns.columns),
            "n_periods": len(returns),
            "target_return": target_return,
            "max_weight": max_weight,
            "risk_aversion": risk_aversion,
            "api": api,
        }

        async with self.track_operation("optimize", context):
            # Run governance preflight if enabled
            if self.config.require_preflight and self._governance:
                result = await self._governance.preflight(context)
                if not result.allowed:
                    raise RuntimeError(f"Optimization blocked by governance: {result.reason}")

            start_time = datetime.now(UTC)

            # Run optimization
            if self._optimizer is None:
                raise RuntimeError("Optimizer not initialized")
            raw_result = self._optimizer.optimize_from_returns(
                returns=returns,
                target_return=target_return,
                max_weight=max_weight,
                risk_aversion=risk_aversion,
                api=api,
                solver_settings=solver_settings,
                execute=True,
            )

            end_time = datetime.now(UTC)
            optimization_time = (end_time - start_time).total_seconds()

            # Extract results
            weights = raw_result.get("weights", {})
            metrics = raw_result.get("metrics", {})

            # Validate constraints
            warnings: list[str] = []
            constraints_satisfied = True

            if weights:
                max_actual_weight = max(weights.values()) if weights else 0
                if max_actual_weight > self.config.max_concentration:
                    warnings.append(
                        f"Concentration limit exceeded: {max_actual_weight:.2%} > {self.config.max_concentration:.2%}"
                    )
                    constraints_satisfied = False

                non_zero_count = sum(1 for w in weights.values() if w > 0.001)
                if non_zero_count < self.config.min_diversification:
                    warnings.append(
                        f"Diversification below minimum: {non_zero_count} < {self.config.min_diversification}"
                    )
                    constraints_satisfied = False

            cvar = metrics.get("cvar")
            if cvar is not None and cvar > self.config.max_cvar:
                warnings.append(f"CVaR exceeds limit: {cvar:.2%} > {self.config.max_cvar:.2%}")
                constraints_satisfied = False

            result = OptimizationResult(
                weights=weights if isinstance(weights, dict) else {},
                expected_return=metrics.get("expected_return"),
                cvar=cvar,
                objective=metrics.get("objective"),
                solver_api=api,
                optimization_time=optimization_time,
                constraints_satisfied=constraints_satisfied,
                warnings=warnings,
            )

            self._optimization_history.append(result)

            # Audit the optimization
            if self._governance:
                await self._governance.audit(
                    AuditRecord(
                        engine=self.config.name,
                        action="optimize",
                        inputs=context,
                        outputs=result.to_dict(),
                        latency_ms=optimization_time * 1000,
                    )
                )

            return result

    async def generate_scenarios(
        self,
        fitting_data: pd.DataFrame,
        generation_dates: list[pd.Timestamp],
        n_paths: int | None = None,
        method: str | None = None,
    ) -> ScenarioResult:
        """
        Generate synthetic forward paths for scenario analysis.

        Args:
            fitting_data: Historical data for model fitting.
            generation_dates: Dates for forward simulation.
            n_paths: Number of simulation paths (default from config).
            method: Simulation method (default from config).

        Returns:
            ScenarioResult with simulated paths.
        """
        if not self.is_available():
            raise RuntimeError("PortfolioOptEngine not available for scenario generation")

        n_paths = n_paths or self.config.n_paths
        method = method or self.config.simulation_method

        context = {
            "operation": "generate_scenarios",
            "n_assets": len(fitting_data.columns),
            "n_paths": n_paths,
            "method": method,
            "n_dates": len(generation_dates),
        }

        async with self.track_operation("generate_scenarios", context):
            start_time = datetime.now(UTC)

            if self._scenario_gen is None:
                raise RuntimeError("Scenario generator not initialized")
            paths = self._scenario_gen.generate(
                fitting_data=fitting_data,
                generation_dates=generation_dates,
                n_paths=n_paths,
                method=method,
            )

            end_time = datetime.now(UTC)
            generation_time = (end_time - start_time).total_seconds()

            return ScenarioResult(
                simulated_paths=paths,
                n_paths=n_paths,
                n_assets=len(fitting_data.columns),
                method=method,
                generation_time=generation_time,
            )

    def get_optimization_history(self) -> list[OptimizationResult]:
        """Get history of optimizations run by this engine."""
        return list(self._optimization_history)

    def get_last_optimization(self) -> OptimizationResult | None:
        """Get the most recent optimization result."""
        return self._optimization_history[-1] if self._optimization_history else None
