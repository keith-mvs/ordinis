"""
Integration adapters for portfolio rebalancing.

Bridges between portfolio rebalancing and other Ordinis engines:
- SignalCore signals -> Signal-driven rebalancing
- ProofBench events -> Rebalancing events
- FlowRoute orders -> Rebalancing execution
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ordinis.engines.portfolio.events import EventHooks, RebalanceEvent, RebalanceEventType
from ordinis.engines.portfolio.strategies.signal_driven import SignalInput

# Check if SignalCore is available
try:
    from ordinis.engines.signalcore.core.signal import Signal, SignalBatch

    SIGNALCORE_AVAILABLE = True
except ImportError:
    SIGNALCORE_AVAILABLE = False


# Check if ProofBench is available
try:
    from ordinis.engines.proofbench.core.events import Event, EventType

    PROOFBENCH_AVAILABLE = True
except ImportError:
    PROOFBENCH_AVAILABLE = False


class SignalCoreAdapter:
    """Adapter between SignalCore signals and signal-driven rebalancing.

    Converts SignalCore Signal/SignalBatch objects to SignalInput for rebalancing.

    Example:
        >>> from ordinis.engines.signalcore.core.signal import Signal, Direction, SignalType
        >>> signal = Signal(
        ...     symbol="AAPL",
        ...     timestamp=datetime.now(tz=UTC),
        ...     signal_type=SignalType.ENTRY,
        ...     direction=Direction.LONG,
        ...     probability=0.75,
        ...     expected_return=0.05,
        ...     confidence_interval=(0.02, 0.08),
        ...     score=0.6,
        ...     model_id="momentum_v1",
        ...     model_version="1.0",
        ... )
        >>> adapter = SignalCoreAdapter()
        >>> signal_input = adapter.convert_signal(signal)
        >>> print(signal_input.signal)  # Composite signal value
    """

    def __init__(
        self,
        probability_weight: float = 0.5,
        scale_to_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        """Initialize SignalCore adapter.

        Args:
            probability_weight: Weight for probability vs score in composite (0-1)
            scale_to_range: Output signal range (default: -1 to +1)

        Raises:
            ImportError: If SignalCore is not available
        """
        if not SIGNALCORE_AVAILABLE:
            raise ImportError("SignalCore not available. Install ordinis[signalcore]")

        self.probability_weight = probability_weight
        self.scale_to_range = scale_to_range

    def convert_signal(self, signal: "Signal") -> SignalInput:
        """Convert SignalCore Signal to SignalInput for rebalancing.

        Args:
            signal: SignalCore signal object

        Returns:
            SignalInput for signal-driven rebalancing
        """
        # Composite signal combining score and probability
        # score is already in [-1, 1], probability in [0, 1]
        # Normalize probability to [-1, 1] for combining
        normalized_prob = 2.0 * signal.probability - 1.0

        # Weighted combination
        composite = (
            self.probability_weight * normalized_prob + (1 - self.probability_weight) * signal.score
        )

        # Scale to desired range
        min_val, max_val = self.scale_to_range
        scaled_signal = min_val + (composite + 1.0) / 2.0 * (max_val - min_val)

        return SignalInput(
            symbol=signal.symbol,
            signal=scaled_signal,
            confidence=signal.probability,
            source=signal.model_id,
        )

    def convert_batch(self, batch: "SignalBatch") -> list[SignalInput]:
        """Convert SignalCore SignalBatch to list of SignalInput.

        Args:
            batch: SignalCore signal batch

        Returns:
            List of SignalInput objects
        """
        return [self.convert_signal(signal) for signal in batch.signals]

    def filter_actionable(
        self,
        batch: "SignalBatch",
        min_probability: float = 0.6,
        min_score: float = 0.3,
    ) -> list[SignalInput]:
        """Convert only actionable signals from batch.

        Args:
            batch: SignalCore signal batch
            min_probability: Minimum probability threshold
            min_score: Minimum absolute score threshold

        Returns:
            List of SignalInput objects for actionable signals
        """
        actionable = batch.filter_actionable(min_probability, min_score)
        return [self.convert_signal(signal) for signal in actionable]


class ProofBenchAdapter:
    """Adapter between ProofBench events and rebalancing events.

    Bridges backtesting events with rebalancing hooks for integrated testing.

    Example:
        >>> from ordinis.engines.portfolio.events import EventHooks
        >>> hooks = EventHooks()
        >>> adapter = ProofBenchAdapter(hooks)
        >>> adapter.register_handlers()  # Auto-emit rebalancing events during backtest
    """

    def __init__(self, event_hooks: EventHooks) -> None:
        """Initialize ProofBench adapter.

        Args:
            event_hooks: Rebalancing event hooks manager

        Raises:
            ImportError: If ProofBench is not available
        """
        if not PROOFBENCH_AVAILABLE:
            raise ImportError("ProofBench not available. Install ordinis[proofbench]")

        self.event_hooks = event_hooks

    def convert_to_rebalance_event(
        self,
        proofbench_event: "Event",
        strategy_type: Any = None,
    ) -> RebalanceEvent | None:
        """Convert ProofBench event to rebalancing event.

        Args:
            proofbench_event: ProofBench event
            strategy_type: Rebalancing strategy type (optional)

        Returns:
            RebalanceEvent or None if not convertible
        """
        # Map ProofBench event types to rebalancing event types
        event_mapping = {
            EventType.SIGNAL: RebalanceEventType.CHECK_TRIGGERED,
            EventType.ORDER_SUBMIT: RebalanceEventType.EXECUTION_STARTED,
            EventType.ORDER_FILL: RebalanceEventType.ORDER_EXECUTED,
            EventType.ORDER_CANCEL: RebalanceEventType.ORDER_FAILED,
        }

        if proofbench_event.event_type not in event_mapping:
            return None

        return RebalanceEvent(
            timestamp=proofbench_event.timestamp,
            event_type=event_mapping[proofbench_event.event_type],
            strategy_type=strategy_type,
            data=proofbench_event.data,
            metadata={"source": "proofbench", "priority": proofbench_event.priority},
        )

    def emit_rebalance_started(
        self,
        timestamp: datetime,
        strategy_type: Any,
        positions: dict[str, float],
        prices: dict[str, float],
    ) -> None:
        """Emit rebalance started event.

        Args:
            timestamp: Event timestamp
            strategy_type: Rebalancing strategy
            positions: Current positions
            prices: Current prices
        """
        event = RebalanceEvent(
            timestamp=timestamp,
            event_type=RebalanceEventType.REBALANCE_STARTED,
            strategy_type=strategy_type,
            data={
                "positions": positions,
                "prices": prices,
            },
        )
        self.event_hooks.emit(event)

    def emit_decisions_generated(
        self,
        timestamp: datetime,
        strategy_type: Any,
        decisions: list[Any],
    ) -> None:
        """Emit decisions generated event.

        Args:
            timestamp: Event timestamp
            strategy_type: Rebalancing strategy
            decisions: Rebalancing decisions
        """
        event = RebalanceEvent(
            timestamp=timestamp,
            event_type=RebalanceEventType.DECISIONS_GENERATED,
            strategy_type=strategy_type,
            data={
                "decisions_count": len(decisions),
                "total_adjustment_value": sum(
                    abs(getattr(d, "adjustment_value", 0.0)) for d in decisions
                ),
            },
        )
        self.event_hooks.emit(event)

    def emit_rebalance_completed(
        self,
        timestamp: datetime,
        strategy_type: Any,
        executed: int,
        failed: int,
        total_traded: float,
    ) -> None:
        """Emit rebalance completed event.

        Args:
            timestamp: Event timestamp
            strategy_type: Rebalancing strategy
            executed: Number of decisions executed
            failed: Number of decisions failed
            total_traded: Total value traded
        """
        event = RebalanceEvent(
            timestamp=timestamp,
            event_type=RebalanceEventType.REBALANCE_COMPLETED,
            strategy_type=strategy_type,
            data={
                "executed": executed,
                "failed": failed,
                "total_traded": total_traded,
                "success": failed == 0,
            },
        )
        self.event_hooks.emit(event)


@dataclass
class FlowRouteOrderRequest:
    """Order request for FlowRoute integration.

    Attributes:
        symbol: Ticker symbol
        shares: Number of shares (positive = buy, negative = sell)
        order_type: Market or limit order
        price: Limit price (None for market orders)
        timestamp: Request timestamp
        metadata: Additional order metadata
    """

    symbol: str
    shares: float
    order_type: str = "market"
    price: float | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(tz=UTC)
        if self.metadata is None:
            self.metadata = {}


class FlowRouteAdapter:
    """Adapter for FlowRoute live trading integration.

    Converts rebalancing decisions to FlowRoute order requests.

    Example:
        >>> adapter = FlowRouteAdapter(account_id="TRADING_001")
        >>> decisions = [ThresholdDecision(...), ThresholdDecision(...)]
        >>> orders = adapter.create_orders(decisions)
        >>> for order in orders:
        ...     flowroute_engine.submit_order(order)
    """

    def __init__(
        self,
        account_id: str | None = None,
        order_type: str = "market",
        include_metadata: bool = True,
    ) -> None:
        """Initialize FlowRoute adapter.

        Args:
            account_id: Trading account identifier
            order_type: Default order type (market or limit)
            include_metadata: Include rebalancing metadata in orders
        """
        self.account_id = account_id
        self.order_type = order_type
        self.include_metadata = include_metadata

    def create_order(
        self,
        decision: Any,
        timestamp: datetime | None = None,
    ) -> FlowRouteOrderRequest:
        """Convert rebalancing decision to FlowRoute order request.

        Args:
            decision: Rebalancing decision (any strategy type)
            timestamp: Order timestamp (default: now)

        Returns:
            FlowRoute order request
        """
        metadata = {}
        if self.include_metadata:
            metadata["source"] = "portfolio_rebalancing"
            if hasattr(decision, "trigger_reason"):
                metadata["trigger"] = decision.trigger_reason
            if hasattr(decision, "drift"):
                metadata["drift"] = decision.drift

        if self.account_id:
            metadata["account_id"] = self.account_id

        return FlowRouteOrderRequest(
            symbol=decision.symbol,
            shares=decision.adjustment_shares,
            order_type=self.order_type,
            timestamp=timestamp or datetime.now(tz=UTC),
            metadata=metadata,
        )

    def create_orders(
        self,
        decisions: list[Any],
        timestamp: datetime | None = None,
    ) -> list[FlowRouteOrderRequest]:
        """Convert multiple decisions to order requests.

        Args:
            decisions: List of rebalancing decisions
            timestamp: Order timestamp (default: now)

        Returns:
            List of FlowRoute order requests
        """
        return [self.create_order(decision, timestamp) for decision in decisions]

    def filter_min_value(
        self,
        orders: list[FlowRouteOrderRequest],
        min_value: float,
        prices: dict[str, float],
    ) -> list[FlowRouteOrderRequest]:
        """Filter orders below minimum dollar value.

        Args:
            orders: Order requests to filter
            min_value: Minimum dollar value threshold
            prices: Current prices per symbol

        Returns:
            Filtered order requests
        """
        filtered = []
        for order in orders:
            if order.symbol in prices:
                value = abs(order.shares) * prices[order.symbol]
                if value >= min_value:
                    filtered.append(order)

        return filtered
