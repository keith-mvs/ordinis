"""
Portfolio Rebalancing Engine.

Standardized engine extending BaseEngine for portfolio rebalancing operations.
Orchestrates multiple rebalancing strategies with governance hooks.

Phase 2 enhancements (2025-12-17):
- FeedbackCollector integration for portfolio state snapshots
- Periodic state recording to LearningEngine

Phase 3 enhancements (2025-12-19):
- PortfolioOptAdapter integration for GPU-optimized weight rebalancing
- ExecutionFeedbackCollector for closed-loop execution quality tracking
- Drift-band and calendar-based rebalancing triggers
- Transaction cost awareness in rebalancing decisions
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

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
from ordinis.engines.portfolio.core.config import (
    PortfolioEngineConfig,
)
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

if TYPE_CHECKING:
    from ordinis.engines.learning.collectors.feedback import FeedbackCollector
    from ordinis.engines.portfolio.adapters.portfolioopt_adapter import (
        DriftAnalysis,
        PortfolioOptAdapter,
        PortfolioWeight,
        RebalanceCondition,
    )
    from ordinis.engines.portfolio.feedback.execution_feedback import (
        ExecutionFeedbackCollector,
    )
    from ordinis.engines.portfolioopt.core.engine import OptimizationResult


class PortfolioEngine(BaseEngine[PortfolioEngineConfig]):
    """Unified portfolio rebalancing engine extending BaseEngine.

    Orchestrates multiple rebalancing strategies and manages execution history.
    Provides a unified interface for portfolio rebalancing regardless of strategy.

    Phase 3 Features:
    - PortfolioOptAdapter: Bridges GPU-optimized weights to rebalancing
    - ExecutionFeedbackCollector: Tracks execution quality for learning
    - Drift-band rebalancing: Triggers when positions deviate from targets
    - Calendar rebalancing: Scheduled rebalancing (daily/weekly/monthly)

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
        feedback_collector: "FeedbackCollector | None" = None,
        portfolioopt_adapter: "PortfolioOptAdapter | None" = None,
        execution_feedback: "ExecutionFeedbackCollector | None" = None,
    ) -> None:
        """Initialize the portfolio rebalancing engine.

        Args:
            config: Engine configuration (uses defaults if None)
            governance_hook: Optional governance hook for preflight/audit
            feedback_collector: Feedback collector for portfolio state snapshots
            portfolioopt_adapter: Adapter for GPU-optimized weight rebalancing
            execution_feedback: Collector for execution quality tracking
        """
        super().__init__(config or PortfolioEngineConfig(), governance_hook)

        self.strategies: dict[StrategyType, Any] = {}
        self.history: list[RebalancingHistory] = []
        self.last_rebalance_date: datetime | None = None
        self.event_hooks = EventHooks()
        self._feedback = feedback_collector

        # Phase 3: PortfolioOpt integration
        self._portfolioopt_adapter = portfolioopt_adapter
        self._execution_feedback = execution_feedback
        self._target_weights: list[PortfolioWeight] = []
        self._last_optimization_result: OptimizationResult | None = None

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

        # Phase 2: Record portfolio state snapshot after update
        if self._feedback:
            total_exposure = sum(
                abs(p.quantity * p.current_price)
                for p in self.positions.values()
                if hasattr(p, "current_price")
            )
            unrealized_pnl = sum(
                p.unrealized_pnl for p in self.positions.values() if hasattr(p, "unrealized_pnl")
            )
            try:
                await self._feedback.record_portfolio_state_snapshot(
                    equity=self.equity,
                    cash=self.cash,
                    buying_power=self.cash,  # Simplified - real would come from broker
                    position_count=len(self.positions),
                    total_exposure=total_exposure,
                    margin_used=self.margin_used,
                    unrealized_pnl=unrealized_pnl,
                    daily_pnl=0.0,  # Would need daily tracking
                )
            except Exception as e:
                # Don't fail update if feedback fails
                pass

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

    # -------------------------------------------------------------------------
    # PortfolioOpt Integration (Phase 3)
    # -------------------------------------------------------------------------

    def set_portfolioopt_adapter(self, adapter: "PortfolioOptAdapter") -> None:
        """Set the PortfolioOpt adapter for GPU-optimized rebalancing.

        Args:
            adapter: Configured PortfolioOptAdapter instance
        """
        self._portfolioopt_adapter = adapter

    def set_execution_feedback(self, collector: "ExecutionFeedbackCollector") -> None:
        """Set the execution feedback collector.

        Args:
            collector: ExecutionFeedbackCollector instance
        """
        self._execution_feedback = collector

    def apply_optimization_result(
        self,
        optimization_result: "OptimizationResult",
    ) -> list["PortfolioWeight"]:
        """Apply optimization result from PortfolioOptEngine.

        Converts GPU-optimized weights to target allocations using the
        configured adapter settings.

        Args:
            optimization_result: Result from PortfolioOptEngine.optimize()

        Returns:
            List of PortfolioWeight targets

        Raises:
            RuntimeError: If PortfolioOptAdapter is not configured
        """
        if not self._portfolioopt_adapter:
            raise RuntimeError(
                "PortfolioOptAdapter not configured. " "Call set_portfolioopt_adapter() first."
            )

        self._last_optimization_result = optimization_result
        self._target_weights = self._portfolioopt_adapter.convert_to_targets(
            optimization_result, total_equity=self.equity
        )
        return self._target_weights

    def get_current_weights(
        self,
        prices: dict[str, float],
    ) -> dict[str, float]:
        """Calculate current portfolio weights.

        Args:
            prices: Current prices by symbol

        Returns:
            Weight percentages by symbol (including 'CASH')
        """
        if not self._portfolioopt_adapter:
            # Fallback: simple calculation without adapter
            positions = {s: p.quantity for s, p in self.positions.items()}
            position_values = {s: qty * prices.get(s, 0.0) for s, qty in positions.items()}
            total = sum(position_values.values()) + self.cash
            if total <= 0:
                return {"CASH": 100.0}
            weights = {s: (v / total) * 100 for s, v in position_values.items()}
            weights["CASH"] = (self.cash / total) * 100
            return weights

        positions = {s: p.quantity for s, p in self.positions.items()}
        return self._portfolioopt_adapter.calculate_current_weights(positions, prices, self.cash)

    def analyze_drift(
        self,
        prices: dict[str, float],
        target_weights: list["PortfolioWeight"] | None = None,
    ) -> "DriftAnalysis":
        """Analyze portfolio drift from target weights.

        Uses configured drift-band settings to determine if rebalancing
        is warranted.

        Args:
            prices: Current prices by symbol
            target_weights: Override target weights (default: use stored)

        Returns:
            DriftAnalysis with drift metrics and recommendations

        Raises:
            RuntimeError: If adapter not configured or no targets set
        """
        if not self._portfolioopt_adapter:
            raise RuntimeError("PortfolioOptAdapter not configured")

        targets = target_weights or self._target_weights
        if not targets:
            raise RuntimeError("No target weights set. Call apply_optimization_result() first.")

        current_weights = self.get_current_weights(prices)
        return self._portfolioopt_adapter.analyze_drift(current_weights, targets, self.cash)

    def should_rebalance_drift(
        self,
        prices: dict[str, float],
    ) -> bool:
        """Check if drift-based rebalancing should occur.

        Evaluates current drift against configured threshold and cooldown.

        Args:
            prices: Current prices by symbol

        Returns:
            True if drift threshold exceeded and not in cooldown
        """
        if not self.config.drift_band.enabled:
            return False

        if not self._portfolioopt_adapter or not self._target_weights:
            return False

        try:
            drift_analysis = self.analyze_drift(prices)
            return drift_analysis.drift_triggered
        except RuntimeError:
            return False

    def should_rebalance_calendar(
        self,
        now: datetime | None = None,
    ) -> bool:
        """Check if calendar-based rebalancing should occur.

        Args:
            now: Current timestamp (default: now)

        Returns:
            True if calendar rebalance is due
        """
        if not self.config.calendar.enabled:
            return False

        if not self._portfolioopt_adapter:
            return False

        return self._portfolioopt_adapter.should_calendar_rebalance(now)

    async def generate_rebalance_trades(
        self,
        prices: dict[str, float],
        target_weights: list["PortfolioWeight"] | None = None,
        condition: "RebalanceCondition | None" = None,
    ) -> list[dict[str, Any]]:
        """Generate trades to rebalance portfolio to target weights.

        Integrates with governance preflight and execution feedback.

        Args:
            prices: Current prices by symbol
            target_weights: Override targets (default: use stored)
            condition: What triggered this rebalance

        Returns:
            List of trade instructions ready for execution

        Raises:
            RuntimeError: If adapter not configured
        """
        if not self._portfolioopt_adapter:
            raise RuntimeError("PortfolioOptAdapter not configured")

        targets = target_weights or self._target_weights
        if not targets:
            raise RuntimeError("No target weights set")

        # Governance preflight
        timestamp = datetime.now(tz=UTC)
        if self.config.enable_governance and self._governance_hook:
            context = PreflightContext(
                operation="generate_rebalance_trades",
                parameters={
                    "target_count": len(targets),
                    "equity": self.equity,
                    "condition": condition.name if condition else "MANUAL",
                },
                timestamp=timestamp,
                trace_id=f"rebalance-{timestamp.timestamp()}",
            )
            result = await self.preflight(context)
            if not result.allowed:
                self._audit(
                    AuditRecord(
                        timestamp=timestamp,
                        operation="generate_rebalance_trades",
                        status="blocked",
                        details={"reason": result.reason},
                    )
                )
                return []

        current_weights = self.get_current_weights(prices)

        # Generate trades
        trades = self._portfolioopt_adapter.calculate_rebalance_trades(
            current_weights=current_weights,
            target_weights=targets,
            total_equity=self.equity,
            prices=prices,
            min_order_value=self.config.min_trade_value,
        )

        # Apply execution feedback adjustments if available
        if self._execution_feedback and self.config.execution_feedback.enabled:
            trades = self._apply_execution_feedback(trades)

        # Emit event
        self.event_hooks.emit(
            RebalanceEvent(
                timestamp=timestamp,
                event_type=RebalanceEventType.DECISIONS_GENERATED,
                data={
                    "trade_count": len(trades),
                    "total_notional": sum(t.get("notional", 0) for t in trades),
                },
            )
        )

        return trades

    def _apply_execution_feedback(
        self,
        trades: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply execution feedback adjustments to trades.

        Reduces sizing for symbols with poor execution quality.

        Args:
            trades: Original trade list

        Returns:
            Adjusted trade list
        """
        if not self._execution_feedback:
            return trades

        adjusted = []
        for trade in trades:
            symbol = trade.get("symbol")
            if not symbol:
                adjusted.append(trade)
                continue

            # Check if sizing should be reduced for this symbol
            recommendation = self._execution_feedback.should_adjust_sizing(symbol)

            if recommendation and recommendation.should_reduce:
                factor = self.config.execution_feedback.sizing_reduction_factor
                trade = dict(trade)  # Copy
                trade["notional"] = trade.get("notional", 0) * factor
                trade["shares"] = trade.get("shares", 0) * factor
                trade["sizing_adjusted"] = True
                trade["adjustment_reason"] = recommendation.reason

            adjusted.append(trade)

        return adjusted

    async def invest_cash(
        self,
        prices: dict[str, float],
        cash_amount: float | None = None,
    ) -> list[dict[str, Any]]:
        """Create trades to invest available cash proportionally.

        Alpaca-style invest_cash: Only buys, no sells. Requires minimum $10.

        Args:
            prices: Current prices by symbol
            cash_amount: Override cash amount (default: available cash)

        Returns:
            List of buy orders to invest cash
        """
        if not self._portfolioopt_adapter:
            raise RuntimeError("PortfolioOptAdapter not configured")

        if not self._target_weights:
            raise RuntimeError("No target weights set")

        amount = cash_amount if cash_amount is not None else self.cash

        return self._portfolioopt_adapter.create_invest_cash_trades(
            cash_amount=amount,
            target_weights=self._target_weights,
            prices=prices,
        )

    def record_execution(
        self,
        order_id: str,
        symbol: str,
        side: str,
        expected_price: float,
        filled_price: float,
        expected_qty: float,
        filled_qty: float,
        estimated_cost_bps: float = 0.0,
        execution_time_ms: float = 0.0,
    ) -> None:
        """Record trade execution for feedback learning.

        Args:
            order_id: Unique order identifier
            symbol: Traded symbol
            side: 'buy' or 'sell'
            expected_price: Price at order submission
            filled_price: Actual fill price
            expected_qty: Requested quantity
            filled_qty: Actual filled quantity
            estimated_cost_bps: Estimated transaction cost
            execution_time_ms: Time to fill
        """
        if not self._execution_feedback:
            return

        self._execution_feedback.record_order_submission(
            order_id=order_id,
            symbol=symbol,
            side=side,
            expected_price=expected_price,
            expected_qty=expected_qty,
            estimated_cost_bps=estimated_cost_bps,
        )

        self._execution_feedback.record_fill(
            order_id=order_id,
            filled_avg_price=filled_price,
            filled_qty=filled_qty,
            execution_time_ms=execution_time_ms,
        )

    def get_execution_metrics(
        self,
        lookback_hours: int = 24,
    ) -> Any:
        """Get execution quality metrics.

        Args:
            lookback_hours: Hours of history to analyze

        Returns:
            ExecutionQualityMetrics or None if not available
        """
        if not self._execution_feedback:
            return None

        return self._execution_feedback.get_quality_metrics(lookback_hours=lookback_hours)

    def estimate_rebalance_cost(
        self,
        prices: dict[str, float],
    ) -> float:
        """Estimate total transaction cost for rebalancing to targets.

        Args:
            prices: Current prices by symbol

        Returns:
            Estimated cost in dollars
        """
        if not self._portfolioopt_adapter or not self._target_weights:
            return 0.0

        current_weights = self.get_current_weights(prices)
        trades = self._portfolioopt_adapter.calculate_rebalance_trades(
            current_weights=current_weights,
            target_weights=self._target_weights,
            total_equity=self.equity,
            prices=prices,
        )

        return self._portfolioopt_adapter.estimate_rebalance_cost(trades)
