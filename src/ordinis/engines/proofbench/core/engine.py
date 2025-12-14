"""
ProofBench Backtesting Engine.

Standardized engine extending BaseEngine for backtesting operations.
Wraps SimulationEngine with governance hooks and lifecycle management.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ordinis.engines.base import (
    AuditRecord,
    BaseEngine,
    EngineMetrics,
    GovernanceHook,
    HealthLevel,
    HealthStatus,
    PreflightContext,
)
from ordinis.engines.proofbench.core.config import ProofBenchEngineConfig
from ordinis.engines.proofbench.core.execution import Order
from ordinis.engines.proofbench.core.simulator import (
    SimulationConfig,
    SimulationEngine,
    SimulationResults,
)


class ProofBenchEngine(BaseEngine[ProofBenchEngineConfig]):
    """Unified backtesting engine extending BaseEngine.

    Wraps SimulationEngine with standardized lifecycle management,
    governance hooks, and metrics tracking.

    Example:
        >>> from ordinis.engines.proofbench import (
        ...     ProofBenchEngine,
        ...     ProofBenchEngineConfig,
        ... )
        >>> config = ProofBenchEngineConfig(
        ...     initial_capital=100000.0,
        ...     enable_governance=True,
        ... )
        >>> engine = ProofBenchEngine(config)
        >>> await engine.initialize()
        >>> engine.load_data("AAPL", df)
        >>> engine.set_strategy(my_strategy)
        >>> results = await engine.run_backtest()
    """

    def __init__(
        self,
        config: ProofBenchEngineConfig | None = None,
        governance_hook: GovernanceHook | None = None,
    ) -> None:
        """Initialize the ProofBench engine.

        Args:
            config: Engine configuration (uses defaults if None)
            governance_hook: Optional governance hook for preflight/audit
        """
        super().__init__(config or ProofBenchEngineConfig(), governance_hook)

        self._simulator: SimulationEngine | None = None
        self._last_backtest: datetime | None = None
        self._backtests_run: int = 0
        self._last_results: SimulationResults | None = None

    async def _do_initialize(self) -> None:
        """Initialize ProofBench engine resources."""
        sim_config = SimulationConfig(
            initial_capital=self.config.initial_capital,
            execution_config=self.config.execution_config,
            bar_frequency=self.config.bar_frequency,
            record_equity_frequency=self.config.record_equity_frequency,
            enable_logging=self.config.enable_logging,
            risk_free_rate=self.config.risk_free_rate,
        )
        self._simulator = SimulationEngine(sim_config)
        self._last_backtest = None
        self._backtests_run = 0
        self._last_results = None
        self.config.loaded_symbols.clear()

    async def _do_shutdown(self) -> None:
        """Shutdown ProofBench engine resources."""
        self._simulator = None
        self._last_results = None

    async def _do_health_check(self) -> HealthStatus:
        """Check ProofBench engine health.

        Returns:
            Current health status
        """
        issues: list[str] = []

        if self._simulator is None:
            issues.append("Simulator not initialized")

        if not self.config.loaded_symbols:
            issues.append("No data loaded")

        level = HealthLevel.HEALTHY if not issues else HealthLevel.DEGRADED
        return HealthStatus(
            level=level,
            message="ProofBench engine operational" if not issues else "; ".join(issues),
            details={
                "loaded_symbols": self.config.loaded_symbols,
                "backtests_run": self._backtests_run,
                "last_backtest": (self._last_backtest.isoformat() if self._last_backtest else None),
            },
        )

    def load_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Load historical data for a symbol.

        Args:
            symbol: Stock symbol
            data: DataFrame with OHLCV columns and datetime index

        Raises:
            RuntimeError: If engine not initialized
            ValueError: If data format is invalid
        """
        if self._simulator is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        self._simulator.load_data(symbol, data)
        if symbol not in self.config.loaded_symbols:
            self.config.loaded_symbols.append(symbol)

    def set_strategy(self, strategy_callback: Callable) -> None:
        """Set strategy callback function.

        The callback is called on each bar with (engine, symbol, bar).
        It should submit orders using engine.submit_order().

        Args:
            strategy_callback: Function to call on each bar

        Raises:
            RuntimeError: If engine not initialized
        """
        if self._simulator is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        self._simulator.set_strategy(strategy_callback)

    def submit_order(self, order: Order) -> None:
        """Submit an order for execution.

        Args:
            order: Order to submit

        Raises:
            RuntimeError: If engine not initialized
        """
        if self._simulator is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        self._simulator.submit_order(order)

    async def run_backtest(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> SimulationResults:
        """Run the backtest simulation.

        Includes governance preflight check if enabled.

        Args:
            start: Start date (uses data start if None)
            end: End date (uses data end if None)

        Returns:
            SimulationResults object
        """
        if self._simulator is None:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        timestamp = datetime.now(UTC)

        # Governance preflight check
        if self.config.enable_governance and self._governance_hook:
            context = PreflightContext(
                operation="run_backtest",
                parameters={
                    "start": start.isoformat() if start else None,
                    "end": end.isoformat() if end else None,
                    "symbols": self.config.loaded_symbols,
                    "initial_capital": self.config.initial_capital,
                },
                timestamp=timestamp,
                trace_id=f"proofbench-{timestamp.timestamp()}",
            )
            result = await self.preflight(context)
            if not result.allowed:
                self._audit(
                    AuditRecord(
                        timestamp=timestamp,
                        operation="run_backtest",
                        status="blocked",
                        details={"reason": result.reason},
                    )
                )
                raise PermissionError(f"Backtest blocked by governance: {result.reason}")

        async with self.track_operation("run_backtest"):
            results = self._simulator.run(start, end)
            self._last_backtest = datetime.now(UTC)
            self._backtests_run += 1
            self._last_results = results

            # Governance audit
            if self.config.enable_governance:
                self._audit(
                    AuditRecord(
                        timestamp=self._last_backtest,
                        operation="run_backtest",
                        status="success",
                        details={
                            "total_return": results.metrics.total_return,
                            "sharpe_ratio": results.metrics.sharpe_ratio,
                            "max_drawdown": results.metrics.max_drawdown,
                            "total_trades": results.metrics.total_trades,
                        },
                    )
                )

            return results

    def get_position(self, symbol: str) -> Any:
        """Get current position for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Position object or None
        """
        if self._simulator is None:
            return None
        return self._simulator.get_position(symbol)

    def get_cash(self) -> float:
        """Get current cash balance.

        Returns:
            Cash balance
        """
        if self._simulator is None:
            return 0.0
        return self._simulator.get_cash()

    def get_equity(self) -> float:
        """Get current total equity.

        Returns:
            Total equity
        """
        if self._simulator is None:
            return 0.0
        return self._simulator.get_equity()

    def get_last_results(self) -> SimulationResults | None:
        """Get results from last backtest.

        Returns:
            Last SimulationResults or None
        """
        return self._last_results

    def get_metrics(self) -> EngineMetrics:
        """Get ProofBench engine metrics.

        Returns:
            Current engine metrics including backtest-specific stats
        """
        metrics = super().get_metrics()
        metrics.custom_metrics.update(
            {
                "loaded_symbols": len(self.config.loaded_symbols),
                "backtests_run": self._backtests_run,
                "last_backtest": (self._last_backtest.isoformat() if self._last_backtest else None),
            }
        )

        # Add last results metrics if available
        if self._last_results:
            metrics.custom_metrics.update(
                {
                    "last_total_return": self._last_results.metrics.total_return,
                    "last_sharpe_ratio": self._last_results.metrics.sharpe_ratio,
                    "last_max_drawdown": self._last_results.metrics.max_drawdown,
                }
            )

        return metrics
