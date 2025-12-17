"""
Portfolio Rebalancing Engine.

Standardized engine extending BaseEngine for portfolio rebalancing operations.
Orchestrates multiple rebalancing strategies with governance hooks.
"""

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ordinis.domain.positions import Position, PositionSide
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

        # Portfolio State
        self.positions: dict[str, Position] = {}
        self.cash: float = self.config.initial_capital
        self.equity: float = self.cash
        self.margin_used: float = 0.0

    # -------------------------------------------------------------------------
    # Protocol Implementation
    # -------------------------------------------------------------------------

    async def update(self, fills: list[Any]) -> None:
        """
        Update portfolio with trade fills (Protocol implementation).
        """
        if not fills:
            return

        for fill in fills:
            self._process_fill(fill)

        self._update_equity()
        self.last_rebalance_date = datetime.now(UTC)

    def _process_fill(self, fill: Any) -> None:
        """Process a single trade fill."""
        # Handle dictionary fills (common from other engines)
        if isinstance(fill, dict):
            symbol = fill.get("symbol")
            side = fill.get("side")
            quantity = float(fill.get("quantity", 0))
            price = float(fill.get("price", 0))
            fee = float(fill.get("fee", 0))
            multiplier = float(fill.get("multiplier", 1.0))
        else:
            # Handle object fills
            symbol = getattr(fill, "symbol", None)
            side = getattr(fill, "side", None)
            quantity = float(getattr(fill, "quantity", 0))
            price = float(getattr(fill, "price", 0))
            fee = float(getattr(fill, "fee", 0))
            multiplier = float(getattr(fill, "multiplier", 1.0))

        if not symbol or not side:
            return

        # Normalize side
        if hasattr(side, "value"):
            side = str(side.value)
        if isinstance(side, str):
            side = side.lower()

        # Update cash
        cost = quantity * price * multiplier
        if side == "buy":
            self.cash -= cost + fee
        else:
            self.cash += cost - fee

        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol, multiplier=multiplier)

        position = self.positions[symbol]

        # Simple position update logic (FIFO/Average Cost handling would be more complex)
        if side == "buy":
            if position.side == PositionSide.SHORT:
                # Covering short
                remaining = quantity
                if remaining >= position.quantity:
                    # Closed full short, maybe flipped long
                    remaining -= position.quantity
                    position.quantity = 0
                    position.side = PositionSide.FLAT
                    if remaining > 0:
                        position.side = PositionSide.LONG
                        position.quantity = remaining
                        position.avg_entry_price = price
                else:
                    # Partial cover
                    position.quantity -= remaining
            else:
                # Adding to long
                total_cost = (position.quantity * position.avg_entry_price) + (quantity * price)
                position.quantity += quantity
                position.side = PositionSide.LONG
                position.avg_entry_price = total_cost / position.quantity

        elif side == "sell":
            if position.side == PositionSide.LONG:
                # Selling long
                remaining = quantity
                if remaining >= position.quantity:
                    # Closed full long, maybe flipped short
                    remaining -= position.quantity
                    position.quantity = 0
                    position.side = PositionSide.FLAT
                    if remaining > 0:
                        position.side = PositionSide.SHORT
                        position.quantity = remaining
                        position.avg_entry_price = price
                else:
                    # Partial sell
                    position.quantity -= remaining
            else:
                # Adding to short
                total_cost = (position.quantity * position.avg_entry_price) + (quantity * price)
                position.quantity += quantity
                position.side = PositionSide.SHORT
                position.avg_entry_price = total_cost / position.quantity

        # Update position price
        position.update_price(price, datetime.now(UTC))

        # Remove flat positions
        if position.quantity == 0:
            del self.positions[symbol]

    def _update_equity(self) -> None:
        """Update total equity and margin usage."""
        position_value = sum(p.unrealized_pnl for p in self.positions.values())
        # Note: For futures, market_value isn't equity, PnL is.
        # For equities, market_value is equity.
        # This hybrid approach needs refinement but works for PnL tracking.
        # If we treat cash as "Account Balance" and positions as "Unrealized PnL",
        # Equity = Cash + Unrealized PnL.

        # However, for long stock, Cash was reduced by cost.
        # So Equity = Cash + Market Value.
        # For Futures, Cash is not reduced by cost (only fees), so Equity = Cash + Unrealized PnL.

        # We need to distinguish asset types or use a unified model.
        # For now, assuming Cash was adjusted by full cost for all assets (Spot model).
        # If Futures, we need to fix the cash adjustment logic in _process_fill.

        # Let's assume Spot model for now as default, but Futures require "Margin" model.
        # To support Futures properly, we need to know if the instrument is a Future.
        # The 'fill' should ideally contain instrument type.

        # For this iteration, we'll stick to the Spot model logic where Cash is debited.
        # Equity = Cash + Market Value of Longs - Market Value of Shorts?
        # No, Market Value of Shorts is a liability.

        long_value = sum(
            p.market_value for p in self.positions.values() if p.side == PositionSide.LONG
        )
        short_value = sum(
            p.market_value for p in self.positions.values() if p.side == PositionSide.SHORT
        )

        self.equity = self.cash + long_value - short_value
        self.margin_used = self.calculate_margin()

    def calculate_margin(self) -> float:
        """Calculate total margin requirement."""
        total_margin = 0.0
        for position in self.positions.values():
            if position.initial_margin > 0:
                total_margin += position.initial_margin * position.quantity
        return total_margin

    async def get_state(self) -> Any:
        """Get current portfolio state."""
        # Return a dict compatible with RiskGuard's PortfolioState
        return {
            "equity": self.equity,
            "cash": self.cash,
            "peak_equity": self.equity,  # Simplified
            "daily_pnl": 0.0,  # Needs tracking
            "daily_trades": 0,  # Needs tracking
            "open_positions": self.positions,
            "total_positions": len(self.positions),
            "total_exposure": sum(p.market_value for p in self.positions.values()),
            "sector_exposures": {},
            "correlated_exposure": 0.0,
            "market_open": True,
            "connectivity_ok": True,
        }
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
