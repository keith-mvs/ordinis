"""
Paper Trading Integration Script.

Wires together the Ordinis components for paper trading:
- StreamingBus (Data Source)
- SignalCore (Signal Generation)
- RiskGuard (Risk Management)
- FlowRoute (Execution)
- OrchestrationEngine (Coordinator)

Implements necessary adapters to satisfy OrchestrationEngine protocols.
"""

import asyncio
from datetime import UTC, datetime
import logging
import os
import sys
from typing import Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ordinis.bus.engine import StreamingBus
from ordinis.engines.flowroute.adapters.paper import PaperBrokerAdapter
from ordinis.engines.flowroute.core.engine import FlowRouteEngine
from ordinis.engines.governance.core.config import GovernanceEngineConfig
from ordinis.engines.governance.core.engine import UnifiedGovernanceEngine
from ordinis.engines.orchestration.core.config import OrchestrationEngineConfig
from ordinis.engines.orchestration.core.engine import (
    AnalyticsEngineProtocol,
    DataSourceProtocol,
    ExecutionEngineProtocol,
    OrchestrationEngine,
    PipelineEngines,
    RiskEngineProtocol,
    SignalEngineProtocol,
)
from ordinis.engines.riskguard.core.engine import RiskGuardEngine
from ordinis.engines.signalcore.core.config import SignalCoreEngineConfig
from ordinis.engines.signalcore.core.engine import SignalCoreEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("paper_trading")


class DataSourceAdapter(DataSourceProtocol):
    """Adapter for StreamingBus to satisfy DataSourceProtocol."""

    def __init__(self, bus: StreamingBus):
        self.bus = bus
        self._latest_data: dict[str, Any] = {}

    async def start(self):
        """Start the bus and subscribe to market data."""
        # In a real implementation, we would subscribe to the bus here
        # and update self._latest_data on incoming events.
        # For now, we'll simulate data availability or assume the bus is populated elsewhere.

    async def get_latest(self, symbols: list[str] | None = None) -> dict[str, Any]:
        """Get latest market data."""
        # TODO: Implement actual data retrieval from bus cache
        # For now, return dummy data for testing if empty
        if not self._latest_data and symbols:
            return {
                symbol: {"price": 150.0, "volume": 1000, "timestamp": datetime.now(UTC)}
                for symbol in symbols
            }
        return self._latest_data


class SignalAdapter(SignalEngineProtocol):
    """Adapter for SignalCoreEngine."""

    def __init__(self, engine: SignalCoreEngine):
        self.engine = engine

    async def generate_signals(self, data: Any) -> list[Any]:
        """Generate signals from market data."""
        signals = []
        # data is expected to be dict[symbol, market_data]
        for symbol, market_data in data.items():
            # Convert market_data to DataFrame if needed by SignalCore
            # For now, we'll assume SignalCore can handle what we pass or we mock it
            # signal = await self.engine.generate_signal(symbol, market_data)
            # if signal:
            #     signals.append(signal)
            pass

        # Return dummy signal for testing
        if not signals and data:
            from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType

            for symbol in data:
                signals.append(
                    Signal(
                        symbol=symbol,
                        timestamp=datetime.now(UTC),
                        signal_type=SignalType.ENTRY,
                        direction=Direction.LONG,
                        score=0.8,
                        probability=0.8,
                        expected_return=0.05,
                        confidence_interval=(0.02, 0.08),
                        model_id="test_model",
                        model_version="1.0.0",
                        metadata={"reason": "test"},
                    )
                )
        return signals


class RiskAdapter(RiskEngineProtocol):
    """Adapter for RiskGuardEngine."""

    def __init__(self, engine: RiskGuardEngine):
        self.engine = engine

    async def evaluate(self, signals: list[Any]) -> tuple[list[Any], list[str]]:
        """Evaluate signals."""
        # In a real implementation:
        # approved, rejected = await self.engine.evaluate_batch(signals)
        # return approved, rejected

        # For now, approve all
        return signals, []


class ExecutionAdapter(ExecutionEngineProtocol):
    """Adapter for FlowRouteEngine."""

    def __init__(self, engine: FlowRouteEngine):
        self.engine = engine

    async def execute(self, orders: list[Any]) -> list[Any]:
        """Execute orders."""
        results = []
        for order in orders:
            # Convert signal/intent to Order if needed, or assume 'orders' are Intents
            # FlowRoute expects OrderIntent to create Order, then submit Order

            # Mocking execution for now
            # success, msg = await self.engine.submit_order(order)
            # results.append({"order_id": order.order_id, "success": success, "msg": msg})
            results.append({"order_id": "mock_id", "success": True, "msg": "Simulated execution"})
        return results


class AnalyticsAdapter(AnalyticsEngineProtocol):
    """Simple analytics logger."""

    async def record(self, results: list[Any]) -> None:
        """Record results."""
        logger.info(f"Analytics recorded: {len(results)} execution results")
        for res in results:
            logger.info(f"  - {res}")


async def main():
    logger.info("Initializing Paper Trading System...")

    # 1. Initialize Components
    bus = StreamingBus()

    signal_config = SignalCoreEngineConfig(enable_governance=False)  # Disable governance for test
    signal_engine = SignalCoreEngine(signal_config)

    # RiskGuard config?
    risk_engine = RiskGuardEngine()  # Assuming default config works

    # FlowRoute with Paper Broker
    broker = PaperBrokerAdapter()
    execution_engine = FlowRouteEngine(broker_adapter=broker)

    # 2. Initialize Adapters
    data_source = DataSourceAdapter(bus)
    signal_adapter = SignalAdapter(signal_engine)
    risk_adapter = RiskAdapter(risk_engine)
    execution_adapter = ExecutionAdapter(execution_engine)
    analytics_adapter = AnalyticsAdapter()

    # 3. Initialize Governance Engine
    gov_config = GovernanceEngineConfig(enable_audit=True)
    governance_engine = UnifiedGovernanceEngine(gov_config)
    await governance_engine.initialize()

    # 4. Initialize Orchestration Engine
    pipeline = PipelineEngines(
        signal_engine=signal_adapter,
        risk_engine=risk_adapter,
        execution_engine=execution_adapter,
        analytics_engine=analytics_adapter,
        data_source=data_source,
    )

    orch_config = OrchestrationEngineConfig(
        mode="paper", cycle_interval_ms=5000, enable_governance=True
    )

    orchestrator = OrchestrationEngine(orch_config, governance_hook=governance_engine)
    await orchestrator.initialize()

    orchestrator.register_engines(
        signal_engine=signal_adapter,
        risk_engine=risk_adapter,
        execution_engine=execution_adapter,
        analytics_engine=analytics_adapter,
        data_source=data_source,
    )

    # 5. Run
    logger.info("Starting Orchestration Engine loop...")
    try:
        # In a real scenario, we'd run this indefinitely
        # For this script, maybe run one cycle or a few
        await orchestrator.run_cycle(symbols=["AAPL"])
        logger.info("Cycle completed successfully.")
    except Exception as e:
        logger.error(f"Error in orchestration loop: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
