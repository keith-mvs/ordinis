"""
Portfolio Rebalancing Engine.

Standardized engine extending BaseEngine for portfolio rebalancing operations.
Orchestrates multiple rebalancing strategies with governance hooks.
"""

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
from ordinis.engines.portfolio.core.config import PortfolioEngineConfig
from ordinis.engines.portfolio.core.models import (
    ExecutionResult,
    RebalancingHistory,
    StrategyType,
)
from ordinis.engines.portfolio.events import (
    EventHooks,
    RebalanceEvent,
    RebalanceEventType,
)


class PortfolioEngine(BaseEngine[PortfolioEngineConfig]):
    """Unified portfolio rebalancing engine extending BaseEngine.

    Orchestrates multiple rebalancing strategies and manages execution history.
    Provides a unified interface for portfolio rebalancing regardless of strategy.

    Example:
        >>> from ordinis.engines.portfolio import (
        ...     PortfolioEngine,
        ...     PortfolioEngineConfig,
        ...     TargetAllocationRebalancer,
        ...     TargetAllocation,
        ...     StrategyType,
        ... )
        >>> config = PortfolioEngineConfig(
        ...     default_strategy=StrategyType.TARGET_ALLOCATION,
        ...     enable_governance=True,
        ... )
        >>> engine = PortfolioEngine(config)
        >>> await engine.initialize()
        >>> targets = [
        ...     TargetAllocation("AAPL", 0.40),
        ...     TargetAllocation("MSFT", 0.30),
        ...     TargetAllocation("GOOGL", 0.30),
        ... ]
        >>> target_strategy = TargetAllocationRebalancer(targets)
        >>> engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)
        >>> positions = {"AAPL": 10, "MSFT": 5, "GOOGL": 3}
        >>> prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 100.0}
        >>> decisions = await engine.generate_rebalancing_decisions(positions, prices)
    """

    def __init__(
        self,
        config: PortfolioEngineConfig | None = None,
        governance_hook: GovernanceHook | None = None,
    ) -> None:
        """Initialize the portfolio rebalancing engine.

        Args:
            config: Engine configuration (uses defaults if None)
            governance_hook: Optional governance hook for preflight/audit
        """
        super().__init__(config or PortfolioEngineConfig(), governance_hook)

        self.strategies: dict[StrategyType, Any] = {}
        self.history: list[RebalancingHistory] = []
        self.last_rebalance_date: datetime | None = None
        self.event_hooks = EventHooks()

    # -------------------------------------------------------------------------
    # Protocol Implementation
    # -------------------------------------------------------------------------

    async def update(self, fills: list[Any]) -> None:
        """
        Update portfolio with trade fills (Protocol implementation).
        """
        # In a real implementation, this would update the internal position state
        # For now, we just log the update
        if fills:
            self.last_rebalance_date = datetime.now(UTC)
            # We could also emit an event here
            # await self.event_hooks.emit(RebalanceEvent(...))

    async def get_state(self) -> Any:
        """
        Get current portfolio state (Protocol implementation).
        """
        # Return a simplified state representation
        return {
            "last_rebalance": self.last_rebalance_date,
            "history_count": len(self.history),
            "strategies": list(self.strategies.keys()),
        }

    async def _do_initialize(self) -> None:
        """Initialize portfolio engine resources."""
        self.strategies.clear()
        self.history.clear()
        self.last_rebalance_date = None
        self.event_hooks = EventHooks()

    async def _do_shutdown(self) -> None:
        """Shutdown portfolio engine resources."""
        self.event_hooks.clear()

    async def _do_health_check(self) -> HealthStatus:
        """Check portfolio engine health.

        Returns:
            Current health status
        """
        issues: list[str] = []

        # Check if any strategies are registered
        if not self.strategies:
            issues.append("No strategies registered")

        # Check history size
        if self.config.max_history_entries > 0:
            if len(self.history) >= self.config.max_history_entries:
                issues.append(
                    f"History at capacity ({len(self.history)}/{self.config.max_history_entries})"
                )

        level = HealthLevel.HEALTHY if not issues else HealthLevel.DEGRADED
        return HealthStatus(
            level=level,
            message="Portfolio engine operational" if not issues else "; ".join(issues),
            details={
                "strategies_registered": len(self.strategies),
                "history_entries": len(self.history),
                "last_rebalance": (
                    self.last_rebalance_date.isoformat() if self.last_rebalance_date else None
                ),
            },
        )

    def register_strategy(
        self,
        strategy_type: StrategyType,
        strategy: Any,
    ) -> None:
        """Register a rebalancing strategy.

        Args:
            strategy_type: Type of strategy being registered
            strategy: Instance of the strategy (must have generate_rebalance_orders method)

        Raises:
            ValueError: If strategy doesn't have required methods
        """
        if not hasattr(strategy, "generate_rebalance_orders"):
            raise ValueError(
                f"Strategy must have 'generate_rebalance_orders' method, got {type(strategy)}"
            )

        self.strategies[strategy_type] = strategy
        if strategy_type not in self.config.registered_strategies:
            self.config.registered_strategies.append(strategy_type)

    def get_strategy(
        self,
        strategy_type: StrategyType | None = None,
    ) -> Any:
        """Get a registered strategy.

        Args:
            strategy_type: Type of strategy to get (None = default strategy)

        Returns:
            The registered strategy instance

        Raises:
            ValueError: If strategy is not registered
        """
        stype = strategy_type if strategy_type is not None else self.config.default_strategy

        if stype not in self.strategies:
            raise ValueError(
                f"Strategy {stype.value} not registered. Available: {list(self.strategies.keys())}"
            )

        return self.strategies[stype]

    async def generate_rebalancing_decisions(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        strategy_type: StrategyType | None = None,
        **strategy_kwargs: Any,
    ) -> list[Any]:
        """Generate rebalancing decisions using the specified strategy.

        Includes governance preflight check if enabled.

        Args:
            positions: Current shares held per symbol
            prices: Current price per symbol
            strategy_type: Strategy to use (None = default)
            **strategy_kwargs: Additional arguments to pass to the strategy

        Returns:
            List of decision objects from the strategy
        """
        stype = strategy_type if strategy_type is not None else self.config.default_strategy
        timestamp = datetime.now(tz=UTC)

        # Governance preflight check
        if self.config.enable_governance and self._governance_hook:
            context = PreflightContext(
                operation="generate_rebalancing_decisions",
                parameters={
                    "positions": positions,
                    "prices": prices,
                    "strategy_type": stype.value,
                    **strategy_kwargs,
                },
                timestamp=timestamp,
                trace_id=f"portfolio-{timestamp.timestamp()}",
            )
            result = await self.preflight(context)
            if not result.allowed:
                self._audit(
                    AuditRecord(
                        timestamp=timestamp,
                        operation="generate_rebalancing_decisions",
                        status="blocked",
                        details={"reason": result.reason},
                    )
                )
                return []

        async with self.track_operation("generate_rebalancing_decisions"):
            strategy = self.get_strategy(stype)

            # Emit rebalance started event
            self.event_hooks.emit(
                RebalanceEvent(
                    timestamp=timestamp,
                    event_type=RebalanceEventType.REBALANCE_STARTED,
                    strategy_type=stype,
                    data={"positions": positions, "prices": prices},
                    metadata=strategy_kwargs,
                )
            )

            # Generate decisions
            decisions = strategy.generate_rebalance_orders(positions, prices, **strategy_kwargs)

            # Emit decisions generated event
            if decisions:
                total_adjustment = sum(getattr(d, "adjustment_value", 0.0) for d in decisions)
                self.event_hooks.emit(
                    RebalanceEvent(
                        timestamp=timestamp,
                        event_type=RebalanceEventType.DECISIONS_GENERATED,
                        strategy_type=stype,
                        data={
                            "decisions_count": len(decisions),
                            "total_adjustment_value": total_adjustment,
                        },
                    )
                )

            # Track in history if enabled
            if self.config.track_history and decisions:
                total_adjustment = sum(getattr(d, "adjustment_value", 0.0) for d in decisions)

                history_entry = RebalancingHistory(
                    timestamp=timestamp,
                    strategy_type=stype,
                    decisions_count=len(decisions),
                    total_adjustment_value=total_adjustment,
                    execution_status="planned",
                    metadata=strategy_kwargs,
                )
                self.history.append(history_entry)
                self._trim_history()

            return decisions

    def should_rebalance(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        strategy_type: StrategyType | None = None,
        **strategy_kwargs: Any,
    ) -> bool:
        """Check if portfolio should be rebalanced using the specified strategy.

        Args:
            positions: Current shares held per symbol
            prices: Current price per symbol
            strategy_type: Strategy to use (None = default)
            **strategy_kwargs: Additional arguments to pass to the strategy

        Returns:
            True if rebalancing is recommended
        """
        strategy = self.get_strategy(strategy_type)

        if not hasattr(strategy, "should_rebalance"):
            decisions = strategy.generate_rebalance_orders(positions, prices, **strategy_kwargs)
            return len(decisions) > 0

        return strategy.should_rebalance(positions, prices, **strategy_kwargs)

    async def execute_rebalancing(
        self,
        decisions: list[Any],
        execution_callback: Any = None,
    ) -> ExecutionResult:
        """Execute rebalancing decisions.

        Includes governance audit logging if enabled.

        Args:
            decisions: List of rebalancing decisions to execute
            execution_callback: Optional callable to execute each decision
                               Should accept (decision) and return (success: bool, error: str | None)

        Returns:
            ExecutionResult with execution summary
        """
        timestamp = datetime.now(tz=UTC)

        if not decisions:
            return ExecutionResult(
                timestamp=timestamp,
                decisions_executed=0,
                decisions_failed=0,
                total_value_traded=0.0,
                success=True,
                errors=[],
            )

        async with self.track_operation("execute_rebalancing"):
            # Emit execution started event
            self.event_hooks.emit(
                RebalanceEvent(
                    timestamp=timestamp,
                    event_type=RebalanceEventType.EXECUTION_STARTED,
                    data={"decisions_count": len(decisions)},
                )
            )

            executed = 0
            failed = 0
            total_traded = 0.0
            errors: list[str] = []

            if execution_callback is None:
                executed = len(decisions)
                total_traded = sum(abs(getattr(d, "adjustment_value", 0.0)) for d in decisions)
            else:
                for decision in decisions:
                    try:
                        success, error = execution_callback(decision)
                        if success:
                            executed += 1
                            total_traded += abs(getattr(decision, "adjustment_value", 0.0))
                            self.event_hooks.emit(
                                RebalanceEvent(
                                    timestamp=datetime.now(tz=UTC),
                                    event_type=RebalanceEventType.ORDER_EXECUTED,
                                    data={
                                        "symbol": decision.symbol,
                                        "adjustment_value": abs(
                                            getattr(decision, "adjustment_value", 0.0)
                                        ),
                                    },
                                )
                            )
                        else:
                            failed += 1
                            if error:
                                errors.append(f"{decision.symbol}: {error}")
                            self.event_hooks.emit(
                                RebalanceEvent(
                                    timestamp=datetime.now(tz=UTC),
                                    event_type=RebalanceEventType.ORDER_FAILED,
                                    data={"symbol": decision.symbol, "error": error},
                                )
                            )
                    except Exception as e:
                        failed += 1
                        errors.append(f"{decision.symbol}: {e!s}")
                        self.event_hooks.emit(
                            RebalanceEvent(
                                timestamp=datetime.now(tz=UTC),
                                event_type=RebalanceEventType.ORDER_FAILED,
                                data={"symbol": decision.symbol, "error": str(e)},
                            )
                        )

            # Update history
            completion_timestamp = datetime.now(tz=UTC)
            if self.config.track_history and self.history:
                self.history[-1].execution_status = "executed" if failed == 0 else "partial"
                self.last_rebalance_date = completion_timestamp

            result = ExecutionResult(
                timestamp=completion_timestamp,
                decisions_executed=executed,
                decisions_failed=failed,
                total_value_traded=total_traded,
                success=(failed == 0),
                errors=errors,
            )

            # Emit rebalance completed event
            self.event_hooks.emit(
                RebalanceEvent(
                    timestamp=completion_timestamp,
                    event_type=RebalanceEventType.REBALANCE_COMPLETED,
                    data={
                        "executed": executed,
                        "failed": failed,
                        "total_traded": total_traded,
                        "success": failed == 0,
                    },
                )
            )

            # Governance audit
            if self.config.enable_governance:
                self._audit(
                    AuditRecord(
                        timestamp=completion_timestamp,
                        operation="execute_rebalancing",
                        status="success" if failed == 0 else "partial",
                        details={
                            "executed": executed,
                            "failed": failed,
                            "total_traded": total_traded,
                            "errors": errors,
                        },
                    )
                )

            return result

    def get_history_summary(
        self,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Get summary of rebalancing history.

        Args:
            limit: Maximum number of history entries to return (None = all)

        Returns:
            DataFrame with history summary
        """
        if not self.history:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "strategy_type",
                    "decisions_count",
                    "total_adjustment_value",
                    "execution_status",
                ]
            )

        history_subset = self.history[-limit:] if limit else self.history

        data = [
            {
                "timestamp": h.timestamp,
                "strategy_type": h.strategy_type.value,
                "decisions_count": h.decisions_count,
                "total_adjustment_value": h.total_adjustment_value,
                "execution_status": h.execution_status,
            }
            for h in history_subset
        ]

        return pd.DataFrame(data)

    def clear_history(self) -> None:
        """Clear rebalancing history."""
        self.history = []
        self.last_rebalance_date = None

    def get_registered_strategies(self) -> list[StrategyType]:
        """Get list of registered strategies.

        Returns:
            List of StrategyType values for registered strategies
        """
        return list(self.strategies.keys())

    def get_metrics(self) -> EngineMetrics:
        """Get portfolio engine metrics.

        Returns:
            Current engine metrics including portfolio-specific stats
        """
        metrics = super().get_metrics()
        metrics.custom_metrics.update(
            {
                "strategies_registered": len(self.strategies),
                "history_entries": len(self.history),
                "last_rebalance": (
                    self.last_rebalance_date.isoformat() if self.last_rebalance_date else None
                ),
            }
        )
        return metrics

    def _trim_history(self) -> None:
        """Trim history to max entries if configured."""
        if self.config.max_history_entries > 0:
            if len(self.history) > self.config.max_history_entries:
                self.history = self.history[-self.config.max_history_entries :]
