"""
Unified Portfolio Rebalancing Engine.

Orchestrates multiple rebalancing strategies and manages execution.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pandas as pd

from ordinis.engines.portfolio.events import EventHooks, RebalanceEvent, RebalanceEventType


class StrategyType(Enum):
    """Types of rebalancing strategies."""

    TARGET_ALLOCATION = "target_allocation"
    RISK_PARITY = "risk_parity"
    SIGNAL_DRIVEN = "signal_driven"
    THRESHOLD_BASED = "threshold_based"


@dataclass
class RebalancingHistory:
    """Historical record of a rebalancing event.

    Attributes:
        timestamp: When the rebalancing occurred
        strategy_type: Which strategy was used
        decisions_count: Number of rebalancing decisions generated
        total_adjustment_value: Total dollar value of adjustments
        execution_status: Status of execution (planned, executed, failed)
        metadata: Additional strategy-specific information
    """

    timestamp: datetime
    strategy_type: StrategyType
    decisions_count: int
    total_adjustment_value: float
    execution_status: str
    metadata: dict[str, Any]


@dataclass
class ExecutionResult:
    """Result of executing rebalancing decisions.

    Attributes:
        timestamp: When execution completed
        decisions_executed: Number of decisions successfully executed
        decisions_failed: Number of decisions that failed
        total_value_traded: Total dollar value traded
        success: True if all decisions executed successfully
        errors: List of error messages if any
    """

    timestamp: datetime
    decisions_executed: int
    decisions_failed: int
    total_value_traded: float
    success: bool
    errors: list[str]


class RebalancingEngine:
    """Unified portfolio rebalancing engine.

    Orchestrates multiple rebalancing strategies and manages execution history.
    Provides a unified interface for portfolio rebalancing regardless of strategy.

    Example:
        >>> from ordinis.engines.portfolio import (
        ...     TargetAllocationRebalancer,
        ...     TargetAllocation,
        ...     RebalancingEngine,
        ...     StrategyType,
        ... )
        >>> targets = [
        ...     TargetAllocation("AAPL", 0.40),
        ...     TargetAllocation("MSFT", 0.30),
        ...     TargetAllocation("GOOGL", 0.30),
        ... ]
        >>> target_strategy = TargetAllocationRebalancer(targets)
        >>> engine = RebalancingEngine(default_strategy=StrategyType.TARGET_ALLOCATION)
        >>> engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)
        >>> positions = {"AAPL": 10, "MSFT": 5, "GOOGL": 3}
        >>> prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 100.0}
        >>> decisions = engine.generate_rebalancing_decisions(positions, prices)
    """

    def __init__(
        self,
        default_strategy: StrategyType = StrategyType.TARGET_ALLOCATION,
        track_history: bool = True,
        event_hooks: EventHooks | None = None,
    ) -> None:
        """Initialize the rebalancing engine.

        Args:
            default_strategy: Default strategy to use for rebalancing
            track_history: Whether to track rebalancing history
            event_hooks: Event hooks manager (None = create new instance)
        """
        self.default_strategy = default_strategy
        self.track_history = track_history
        self.event_hooks = event_hooks if event_hooks is not None else EventHooks()
        self.strategies: dict[StrategyType, Any] = {}
        self.history: list[RebalancingHistory] = []
        self.last_rebalance_date: datetime | None = None

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
        # Validate strategy has required methods
        if not hasattr(strategy, "generate_rebalance_orders"):
            raise ValueError(
                f"Strategy must have 'generate_rebalance_orders' method, got {type(strategy)}"
            )

        self.strategies[strategy_type] = strategy

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
        stype = strategy_type if strategy_type is not None else self.default_strategy

        if stype not in self.strategies:
            raise ValueError(
                f"Strategy {stype.value} not registered. Available: {list(self.strategies.keys())}"
            )

        return self.strategies[stype]

    def generate_rebalancing_decisions(
        self,
        positions: dict[str, float],
        prices: dict[str, float],
        strategy_type: StrategyType | None = None,
        **strategy_kwargs: Any,
    ) -> list[Any]:
        """Generate rebalancing decisions using the specified strategy.

        Args:
            positions: Current shares held per symbol
            prices: Current price per symbol
            strategy_type: Strategy to use (None = default)
            **strategy_kwargs: Additional arguments to pass to the strategy

        Returns:
            List of decision objects from the strategy
        """
        strategy = self.get_strategy(strategy_type)
        stype = strategy_type if strategy_type is not None else self.default_strategy

        # Emit rebalance started event
        timestamp = datetime.now(tz=UTC)
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
        if self.track_history and decisions:
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

        # Check if strategy has should_rebalance method
        if not hasattr(strategy, "should_rebalance"):
            # Fall back to checking if generate_rebalance_orders returns any decisions
            decisions = strategy.generate_rebalance_orders(positions, prices, **strategy_kwargs)
            return len(decisions) > 0

        return strategy.should_rebalance(positions, prices, **strategy_kwargs)

    def execute_rebalancing(
        self,
        decisions: list[Any],
        execution_callback: Any = None,
    ) -> ExecutionResult:
        """Execute rebalancing decisions.

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

        # If no callback provided, just simulate execution
        if execution_callback is None:
            executed = len(decisions)
            total_traded = sum(abs(getattr(d, "adjustment_value", 0.0)) for d in decisions)
        else:
            # Execute each decision using the callback
            for decision in decisions:
                try:
                    success, error = execution_callback(decision)
                    if success:
                        executed += 1
                        total_traded += abs(getattr(decision, "adjustment_value", 0.0))
                        # Emit order executed event
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
                        # Emit order failed event
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
                    # Emit order failed event
                    self.event_hooks.emit(
                        RebalanceEvent(
                            timestamp=datetime.now(tz=UTC),
                            event_type=RebalanceEventType.ORDER_FAILED,
                            data={"symbol": decision.symbol, "error": str(e)},
                        )
                    )

        # Update history if tracked
        completion_timestamp = datetime.now(tz=UTC)
        if self.track_history and self.history:
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

        # Limit history if requested
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
