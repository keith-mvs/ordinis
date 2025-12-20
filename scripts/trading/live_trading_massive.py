"""
Production live paper trading using Massive real-time data and Alpaca broker.

This script combines the best of both:
- Massive for high-quality real-time market data
- Alpaca for paper trading order execution

Features:
- Multi-strategy portfolio management
- Real-time minute bars from Massive
- 4 strategies: MA Crossover, RSI, Breakout, VWAP
- Signal aggregation (consensus, majority, weighted, any)
- Risk management integration
- Market hours validation
- Comprehensive logging

Usage:
    python scripts/trading/live_trading_massive.py
"""

import asyncio
from datetime import UTC, datetime
import logging
import os
import sys
from typing import Any

# Add src to path
sys.path.insert(0, "src")

# Third-party imports
from prometheus_client import start_http_server

# Local imports
from ordinis.bus.engine import StreamingBus
from ordinis.engines.flowroute.adapters.alpaca import AlpacaBrokerAdapter
from ordinis.engines.flowroute.adapters.massive_data import MassiveMarketDataAdapter
from ordinis.engines.flowroute.core.orders import Order, OrderType
from ordinis.engines.flowroute.portfolio_manager import PortfolioManager
from ordinis.engines.flowroute.strategies import (
    BreakoutStrategy,
    MACrossoverStrategy,
    RSIMeanReversionStrategy,
    VWAPStrategy,
)
from ordinis.engines.governance.core.config import GovernanceEngineConfig
from ordinis.engines.governance.core.engine import UnifiedGovernanceEngine
from ordinis.engines.orchestration.core.config import OrchestrationEngineConfig
from ordinis.engines.orchestration.core.engine import (
    AnalyticsEngineProtocol,
    DataSourceProtocol,
    ExecutionEngineProtocol,
    OrchestrationEngine,
    RiskEngineProtocol,
    SignalEngineProtocol,
)
from ordinis.engines.riskguard.core.engine import PortfolioState, ProposedTrade, RiskGuardEngine
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType
from ordinis.utils.metrics import metrics, start_health_server
from ordinis.utils.structured_logging import slog

# Use UTC timezone
UTC = UTC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("live_trading")


def get_env_key(key: str) -> str:
    """Get environment variable or return masked string."""
    val = os.getenv(key)
    if val:
        return f"{val[:4]}...{val[-4:]}"
    return val or "<not set>"


ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
MASSIVE_API_KEY = os.getenv("MASSIVE_API_KEY", "")

print("\n[ENV] Alpaca Key:", get_env_key("ALPACA_API_KEY"))
print("[ENV] Alpaca Secret:", get_env_key("ALPACA_SECRET_KEY"))
print("[ENV] Massive Key:", get_env_key("MASSIVE_API_KEY"))


class MassiveDataStream(DataSourceProtocol):
    """
    Streaming data source using Massive WebSocket/Polling.
    Publishes market events to StreamingBus.
    """

    def __init__(
        self, bus: StreamingBus, massive_adapter: MassiveMarketDataAdapter, symbols: list[str]
    ):
        self.bus = bus
        self.massive = massive_adapter
        self.symbols = symbols
        self._latest_data: dict[str, Any] = {}
        self._streaming = False

    async def start(self):
        """Start streaming and publish to bus."""
        logger.info(f"[DataStream] Starting Massive stream for {self.symbols}")

        # Initialize with historical data
        for symbol in self.symbols:
            prices = await self.massive.get_price_history(symbol, periods=100, timeframe="1Min")
            if prices:
                for price in prices:
                    self._update_cache(symbol, {"price": price, "volume": 1000})

        self._streaming = True
        # Start background task to poll/stream
        asyncio.create_task(self._stream_loop())

    async def _stream_loop(self):
        """Continuous streaming loop."""
        from ordinis.bus.models import BusEvent

        while self._streaming:
            for symbol in self.symbols:
                trade = await self.massive.get_latest_trade(symbol)
                if trade:
                    data = {
                        "price": trade["price"],
                        "volume": trade.get("size", 1000),
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                    self._update_cache(symbol, data)

                    # Publish BusEvent to bus
                    event = BusEvent(
                        event_type="market_data",
                        payload={"data": data},
                        source="MassiveDataStream",
                        symbol=symbol,
                    )
                    await self.bus.publish(event)
            await asyncio.sleep(5)  # Poll every 5 seconds

    def _update_cache(self, symbol: str, data: dict):
        """Update latest data cache."""
        self._latest_data[symbol] = data

    async def get_latest(self, symbols: list[str] | None = None) -> dict[str, Any]:
        """Get latest market data for symbols."""
        if symbols:
            return {s: self._latest_data.get(s, {}) for s in symbols if s in self._latest_data}
        return self._latest_data

    async def stop(self):
        """Stop streaming."""
        self._streaming = False
        logger.info("[DataStream] Stopped")


class MultiStrategySignalEngine(SignalEngineProtocol):
    """
    Signal engine using portfolio of strategies.
    Generates standardized Signal objects for OrchestrationEngine.
    """

    def __init__(self, portfolio_manager: PortfolioManager):
        self.portfolio_manager = portfolio_manager
        self.price_history: dict[str, list[float]] = {}

    async def generate_signals(self, data: dict[str, Any]) -> list[Signal]:
        """Generate signals from market data."""
        signals = []

        for symbol, market_data in data.items():
            if not market_data or "price" not in market_data:
                continue

            price = market_data["price"]
            volume = market_data.get("volume", 1000)

            # Update portfolio manager
            portfolio_signal = self.portfolio_manager.update(price, volume=volume)

            if portfolio_signal:
                # Convert PortfolioSignal to Ordinis Signal
                direction = (
                    Direction.LONG if portfolio_signal.direction == "buy" else Direction.SHORT
                )

                signal = Signal(
                    symbol=symbol,
                    timestamp=datetime.now(UTC),
                    signal_type=SignalType.ENTRY
                    if portfolio_signal.direction == "buy"
                    else SignalType.EXIT,
                    direction=direction,
                    score=abs(portfolio_signal.strength),
                    probability=portfolio_signal.confidence,
                    expected_return=0.01,  # Placeholder
                    confidence_interval=(0.005, 0.015),
                    model_id="multi_strategy_portfolio",
                    model_version="1.0.0",
                    metadata={
                        "strategies": portfolio_signal.contributing_strategies,
                        "reasons": portfolio_signal.reasons,
                        "consensus": portfolio_signal.consensus,
                        "aggregation_mode": self.portfolio_manager.mode,
                    },
                )
                signals.append(signal)

                slog.info(
                    f"[Signal] {signal.direction.name} {symbol} | "
                    f"Score: {signal.score:.2f} | Prob: {signal.probability:.2%} | "
                    f"Strategies: {', '.join(portfolio_signal.contributing_strategies)}"
                )

                # Record signal metric
                metrics.record_signal(symbol, "multi_strategy", signal.direction.name)

        return signals


class RiskGuardAdapter(RiskEngineProtocol):
    """
    Adapter for RiskGuardEngine with proper portfolio state.
    """

    def __init__(self, engine: RiskGuardEngine, broker: AlpacaBrokerAdapter):
        self.engine = engine
        self.broker = broker

    async def evaluate(self, signals: list[Any]) -> tuple[list[Any], list[str]]:
        """Evaluate signals through risk engine."""
        # Get current portfolio state
        account = await self.broker.get_account()
        positions = await self.broker.get_positions()

        # Build open_positions dict[str, Position]
        open_positions = {}
        for pos in positions:
            symbol = pos.get("symbol")
            if symbol:
                # Create a simple Position object or dict as expected by RiskGuard
                # Assuming RiskGuard expects a dict or object with these fields
                open_positions[symbol] = {
                    "symbol": symbol,
                    "quantity": int(pos.get("quantity", 0)),
                    "entry_price": float(pos.get("avg_entry_price", 0)),
                    "current_price": float(pos.get("current_price", pos.get("avg_entry_price", 0))),
                    "market_value": float(pos.get("market_value", 0)),
                    "unrealized_pnl": float(pos.get("unrealized_pl", 0)),
                }

        portfolio_state = PortfolioState(
            equity=float(account.get("equity", 100000)),
            cash=float(account.get("cash", 100000)),
            peak_equity=float(account.get("equity", 100000)),
            daily_pnl=0.0,
            daily_trades=0,
            open_positions=open_positions,
            total_positions=len(open_positions),
            total_exposure=0.0,
            sector_exposures={},
            correlated_exposure=0.0,
            market_open=True,
            connectivity_ok=True,
        )

        approved = []
        rejected_reasons = []

        for signal in signals:
            # Convert signal to proposed trade
            trade = ProposedTrade(
                symbol=signal.symbol,
                side="buy" if signal.direction == Direction.LONG else "sell",
                quantity=1,  # Will be sized properly
                price=100.0,  # Placeholder
            )

            # Evaluate through RiskGuard
            is_approved, reason = await self.engine.evaluate_trade(trade, portfolio_state)

            if is_approved:
                approved.append(signal)
            else:
                rejected_reasons.append(f"{signal.symbol}: {reason}")
                slog.warning(f"[Risk] REJECTED {signal.symbol} - {reason}")
                metrics.record_rejection(signal.symbol, reason)

        return approved, rejected_reasons


class AlpacaExecutionEngine(ExecutionEngineProtocol):
    """
    Execution engine using Alpaca broker.
    Converts signals to orders and executes.
    """

    def __init__(self, broker: AlpacaBrokerAdapter, position_size_usd: float = 1000):
        self.broker = broker
        self.position_size_usd = position_size_usd
        self.order_count = 0

    async def execute(self, signals: list[Signal]) -> list[dict]:
        """Execute approved signals."""
        results = []

        for signal in signals:
            # Determine order side and quantity
            side = "buy" if signal.direction == Direction.LONG else "sell"

            # Get current price for position sizing
            # (In production, use signal metadata or latest quote)
            qty = max(1, int(self.position_size_usd / 680))  # Placeholder

            order = Order(
                order_id=f"ORD-{self.order_count:06d}",
                symbol=signal.symbol,
                side=side,
                quantity=qty,
                order_type=OrderType.MARKET,
            )

            # Submit to Alpaca
            result = await self.broker.submit_order(order)

            if result["success"]:
                slog.info(
                    f"[Execution] ✓ {side.upper()} {qty} {signal.symbol} | "
                    f"Order ID: {result.get('broker_order_id')}"
                )
                self.order_count += 1

                # Record order metric
                metrics.record_order(signal.symbol, side, "filled")
            else:
                slog.error(f"[Execution] ✗ FAILED {signal.symbol} - {result.get('error')}")

                # Record error metric
                metrics.record_trading_error("execution_failed", str(result.get("error")))

            results.append(result)

        return results


class AnalyticsLogger(AnalyticsEngineProtocol):
    """Analytics and reporting."""

    async def record(self, results: list[Any]) -> None:
        """Record execution results."""
        logger.info(f"[Analytics] Recorded {len(results)} execution results")


async def main():
    """Main entry point."""
    # Start Prometheus metrics server
    start_http_server(3005)
    slog.info("[Prometheus] Metrics available at http://localhost:3005/metrics")

    # Start Health Check server
    start_health_server(3006, metrics.get_health_status)
    slog.info("[Health] Health check available at http://localhost:3006/health")

    print("\n" + "=" * 70)
    print("ORDINIS - PRODUCTION LIVE TRADING")
    print("Massive Real-time Data + Alpaca Execution + Full Architecture")
    print("=" * 70 + "\n")

    # Verify credentials
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    massive_key = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")

    if not alpaca_key or not alpaca_secret:
        slog.error("[ERROR] Missing Alpaca credentials!")
        return

    if not massive_key:
        slog.error("[ERROR] Missing Massive credentials!")
        return

    # Configuration
    symbols = ["SPY"]  # Can add more symbols

    # 1. Initialize Core Components
    slog.info("[INIT] Initializing components...")

    bus = StreamingBus()

    # Data source
    massive_adapter = MassiveMarketDataAdapter(api_key=massive_key)
    data_stream = MassiveDataStream(bus, massive_adapter, symbols)

    # Broker and execution
    broker = AlpacaBrokerAdapter(api_key=alpaca_key, api_secret=alpaca_secret, paper=True)
    await broker.connect()

    # Update initial account metrics
    account = await broker.get_account()
    metrics.update_account_metrics(account)

    execution_engine = AlpacaExecutionEngine(broker, position_size_usd=1000)

    # Strategies
    strategies = [
        MACrossoverStrategy(fast_period=5, slow_period=15, name="MA_5/15"),
        RSIMeanReversionStrategy(period=7, oversold=40, overbought=60, name="RSI_7"),
        BreakoutStrategy(lookback_period=10, breakout_threshold=0.002, name="Breakout_10"),
        VWAPStrategy(deviation_threshold=0.001, name="VWAP"),
    ]

    portfolio_manager = PortfolioManager(
        strategies=strategies, mode="weighted", min_strategies_ready=2
    )

    signal_engine = MultiStrategySignalEngine(portfolio_manager)

    # Risk management
    risk_guard = RiskGuardEngine()
    risk_adapter = RiskGuardAdapter(risk_guard, broker)

    # Analytics
    analytics = AnalyticsLogger()

    # 2. Initialize Governance
    gov_config = GovernanceEngineConfig(enable_audit=True)
    governance = UnifiedGovernanceEngine(gov_config)
    await governance.initialize()

    # 3. Initialize Orchestration Engine
    orch_config = OrchestrationEngineConfig(
        mode="live",
        cycle_interval_ms=60000,  # 1 minute cycles
        enable_governance=True,
    )

    orchestrator = OrchestrationEngine(orch_config, governance_hook=governance)
    await orchestrator.initialize()

    orchestrator.register_engines(
        signal_engine=signal_engine,
        risk_engine=risk_adapter,
        execution_engine=execution_engine,
        analytics_engine=analytics,
        data_source=data_stream,
    )

    # 4. Start Data Stream
    await data_stream.start()

    # 5. Run Orchestration Loop
    slog.info("\n[GO] Starting live trading with orchestration engine\n")

    try:
        await orchestrator.run_loop(symbols=symbols)
    except KeyboardInterrupt:
        slog.info("\n[STOP] Shutdown requested")
    finally:
        await data_stream.stop()
        account = await broker.get_account()
        positions = await broker.get_positions()

        # Update final metrics
        metrics.update_account_metrics(account)
        metrics.update_position_metrics(positions)

        print("\n" + "=" * 70)
        print("SESSION COMPLETE")
        print("=" * 70)
        print(f"Final Equity: ${float(account.get('equity', 0)):,.2f}")
        print(f"Orders Executed: {execution_engine.order_count}")
        print(f"Open Positions: {len(positions)}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
