"""
Orchestration pipeline: Signal → RiskGuard → FlowRoute → Portfolio.

End-to-end signal-to-execution workflow with governance.
"""

from dataclasses import dataclass
from datetime import datetime

from ordinis.engines.flowroute.core.engine import FlowRouteEngine
from ordinis.engines.flowroute.core.orders import Order, OrderIntent, OrderSide
from ordinis.engines.portfolio.engine import RebalancingEngine
from ordinis.engines.riskguard.core.engine import (
    PortfolioState,
    ProposedTrade,
    RiskGuardEngine,
)
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalBatch


@dataclass
class PipelineConfig:
    """Configuration for the orchestration pipeline.

    Attributes:
        position_size_pct: Default % of equity per position
        max_position_size_pct: Maximum % per symbol
        max_portfolio_exposure_pct: Maximum total exposure
        risk_per_trade_pct: Max risk as % of equity
        stop_loss_pct: Default stop loss (e.g., 0.02 = 2%)
        enable_governance: Enable governance checks
    """

    position_size_pct: float = 0.05  # 5% default
    max_position_size_pct: float = 0.15  # 15% max
    max_portfolio_exposure_pct: float = 1.0  # 100% max total
    risk_per_trade_pct: float = 0.01  # 1% risk per trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    enable_governance: bool = True


class OrchestrationPipeline:
    """Orchestrates full signal-to-execution workflow."""

    def __init__(
        self,
        risk_engine: RiskGuardEngine,
        execution_engine: FlowRouteEngine,
        portfolio_engine: RebalancingEngine,
        config: PipelineConfig | None = None,
    ):
        """Initialize pipeline.

        Args:
            risk_engine: RiskGuard engine for validation
            execution_engine: FlowRoute engine for order execution
            portfolio_engine: Portfolio rebalancing engine
            config: Pipeline configuration
        """
        self.risk_engine = risk_engine
        self.execution_engine = execution_engine
        self.portfolio_engine = portfolio_engine
        self.config = config or PipelineConfig()

        # Track execution history
        self.executed_orders: list[Order] = []
        self.rejected_signals: list[Signal] = []
        self.pipeline_metrics = {
            "signals_received": 0,
            "signals_passed_risk": 0,
            "signals_rejected": 0,
            "orders_submitted": 0,
            "orders_failed": 0,
        }

    async def process_signal_batch(
        self,
        batch: SignalBatch,
        portfolio_state: PortfolioState,
        prices: dict[str, float],
    ) -> list[Order]:
        """Process a batch of signals through full pipeline.

        Flow:
        1. Filter actionable signals
        2. Validate through RiskGuard
        3. Create intents and orders
        4. Submit through FlowRoute
        5. Update portfolio

        Args:
            batch: SignalBatch from SignalCore
            portfolio_state: Current portfolio state
            prices: Current market prices

        Returns:
            List of successfully submitted orders
        """
        submitted_orders = []

        # Filter actionable signals
        actionable = batch.filter_actionable()
        self.pipeline_metrics["signals_received"] += len(batch.signals)

        for signal in actionable:
            # Process through pipeline
            order = await self._process_signal(signal, portfolio_state, prices)

            if order:
                submitted_orders.append(order)
                self.pipeline_metrics["orders_submitted"] += 1
            else:
                self.rejected_signals.append(signal)
                self.pipeline_metrics["signals_rejected"] += 1

        return submitted_orders

    async def _process_signal(
        self,
        signal: Signal,
        portfolio_state: PortfolioState,
        prices: dict[str, float],
    ) -> Order | None:
        """Process single signal through full pipeline.

        Args:
            signal: Trading signal
            portfolio_state: Current portfolio state
            prices: Current prices

        Returns:
            Submitted Order or None if rejected
        """
        # Step 1: Convert signal to trade intent
        trade_intent = self._signal_to_trade(signal, prices, portfolio_state)

        # Step 2: Validate through RiskGuard
        proposed_trade = ProposedTrade(
            symbol=signal.symbol,
            direction=signal.direction.value,
            quantity=trade_intent.get("quantity", 0),
            entry_price=prices.get(signal.symbol, 0.0),
            stop_price=trade_intent.get("stop_price"),
        )

        passed, checks, adjusted_signal = self.risk_engine.evaluate_signal(
            signal, proposed_trade, portfolio_state
        )

        if not passed:
            self.pipeline_metrics["signals_rejected"] += 1
            return None

        self.pipeline_metrics["signals_passed_risk"] += 1

        # Step 3: Create OrderIntent
        intent = OrderIntent(
            symbol=signal.symbol,
            side=OrderSide.BUY if signal.direction == Direction.LONG else OrderSide.SELL,
            quantity=int(trade_intent.get("quantity", 0)),
            order_type=trade_intent.get("order_type", "MARKET"),
            signal_id=id(signal),
            metadata={
                "signal_score": signal.score,
                "signal_probability": signal.probability,
                "risk_checks": len(checks),
            },
        )

        # Step 4: Create and submit order
        order = self.execution_engine.create_order_from_intent(intent)

        success, msg = await self.execution_engine.submit_order(order)

        if not success:
            self.pipeline_metrics["orders_failed"] += 1
            return None

        self.executed_orders.append(order)
        return order

    def _signal_to_trade(
        self,
        signal: Signal,
        prices: dict[str, float],
        portfolio: PortfolioState,
    ) -> dict:
        """Convert signal to trade parameters.

        Args:
            signal: Trading signal
            prices: Current prices
            portfolio: Portfolio state

        Returns:
            Dict with quantity, stop_price, etc.
        """
        current_price = prices.get(signal.symbol, 0.0)

        if current_price <= 0:
            return {"quantity": 0}

        # Position sizing: use signal probability and score
        base_position_size = portfolio.equity * self.config.position_size_pct

        # Adjust based on signal confidence
        confidence_adjustment = signal.probability * abs(signal.score)
        position_value = base_position_size * confidence_adjustment

        # Cap at max position size
        max_position_value = portfolio.equity * self.config.max_position_size_pct
        position_value = min(position_value, max_position_value)

        quantity = int(position_value / current_price)

        if quantity <= 0:
            return {"quantity": 0}

        # Calculate stop loss
        stop_pct = self.config.stop_loss_pct
        if signal.direction == Direction.LONG:
            stop_price = current_price * (1 - stop_pct)
        else:
            stop_price = current_price * (1 + stop_pct)

        return {
            "quantity": quantity,
            "stop_price": stop_price,
            "order_type": "MARKET",
            "position_value": position_value,
        }

    def get_metrics(self) -> dict:
        """Get pipeline execution metrics.

        Returns:
            Dict with metrics
        """
        return {
            **self.pipeline_metrics,
            "executed_orders": len(self.executed_orders),
            "rejected_signals": len(self.rejected_signals),
            "success_rate": (
                self.pipeline_metrics["signals_passed_risk"]
                / max(self.pipeline_metrics["signals_received"], 1)
            ),
        }


class OrderIntent:
    """Intent to execute a trade.

    Bridge between RiskGuard decisions and FlowRoute orders.
    """

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: str = "MARKET",
        signal_id: int | None = None,
        strategy_id: str | None = None,
        metadata: dict | None = None,
    ):
        """Initialize intent.

        Args:
            symbol: Stock ticker
            side: BUY or SELL
            quantity: Number of shares
            order_type: MARKET, LIMIT, etc.
            signal_id: Associated signal ID
            strategy_id: Associated strategy ID
            metadata: Additional context
        """
        self.intent_id = id(self)
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.order_type = order_type
        self.signal_id = signal_id
        self.strategy_id = strategy_id
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.limit_price: float | None = None
        self.stop_price: float | None = None
        self.time_in_force = "GTC"  # Good-til-canceled
