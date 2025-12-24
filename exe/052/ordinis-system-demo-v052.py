#!/usr/bin/env python
"""
Ordinis System Integration Demo v0.52

Full system architecture test with all engines working together.
Tests the SYSTEM, not just a strategy.

Architecture:
    Orchestration → SignalCore → RiskGuard → Portfolio → Analytics
    All connected via Message Bus for event-driven communication

Version History:
    v0.50 - Basic consolidated trading script
    v0.51 - Added regime detection and configurable parameters
    v0.52 - FULL SYSTEM INTEGRATION with all engines
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import Ordinis engines
from ordinis.engines.orchestration import (
    OrchestrationEngine,
    OrchestrationEngineConfig,
)
from ordinis.engines.signalcore.core.engine import (
    SignalCoreEngine,
    SignalCoreEngineConfig,
)
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models import (
    ATROptimizedRSIModel,
    BollingerBandsModel,
    MACDModel,
    RSIMeanReversionModel,
    HMMRegimeModel,
)
from ordinis.engines.riskguard import RiskGuardEngine, RiskGuardConfig
from ordinis.engines.portfolio import PortfolioEngine, PortfolioConfig
from ordinis.engines.analytics.core import AnalyticsEngine, AnalyticsConfig
from ordinis.message_bus import MessageBus, Event, EventType

# Import adapters
from ordinis.adapters.brokers.alpaca_broker import AlpacaBroker
from ordinis.adapters.streaming.massive_stream import MassiveStreamManager

# Import monitoring
from ordinis.monitoring.health import HealthMonitor, ComponentHealth


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("system_demo")


# =============================================================================
# MESSAGE BUS IMPLEMENTATION
# =============================================================================

class SystemMessageBus:
    """Central message bus for inter-engine communication."""

    def __init__(self):
        """Initialize the message bus."""
        self.subscribers: Dict[str, List[callable]] = {}
        self.event_log: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("message_bus")

    def publish(self, event_type: str, data: Any, source: str) -> None:
        """Publish an event to all subscribers."""
        event = {
            "type": event_type,
            "data": data,
            "source": source,
            "timestamp": datetime.now(UTC),
        }

        self.event_log.append(event)
        self.logger.info(f"EVENT | {event_type} from {source}")

        # Notify subscribers
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    asyncio.create_task(callback(event))
                except Exception as e:
                    self.logger.error(f"Subscriber error: {e}")

    def subscribe(self, event_type: str, callback: callable) -> None:
        """Subscribe to an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        self.logger.debug(f"Subscribed to {event_type}")

    def get_event_stats(self) -> Dict[str, int]:
        """Get event statistics."""
        stats = {}
        for event in self.event_log:
            event_type = event["type"]
            stats[event_type] = stats.get(event_type, 0) + 1
        return stats


# =============================================================================
# SYSTEM ORCHESTRATOR
# =============================================================================

class SystemOrchestrator:
    """Main orchestrator for the full system integration."""

    def __init__(self):
        """Initialize the system orchestrator."""
        self.logger = logger
        self.message_bus = SystemMessageBus()

        # Engine instances (to be initialized)
        self.orchestration_engine: Optional[OrchestrationEngine] = None
        self.signalcore_engine: Optional[SignalCoreEngine] = None
        self.riskguard_engine: Optional[RiskGuardEngine] = None
        self.portfolio_engine: Optional[PortfolioEngine] = None
        self.analytics_engine: Optional[AnalyticsEngine] = None

        # Market data
        self.broker: Optional[AlpacaBroker] = None
        self.stream_manager: Optional[MassiveStreamManager] = None

        # Health monitoring
        self.health_monitor = HealthMonitor()

        # State
        self._running = False
        self.cycle_count = 0
        self.start_time = datetime.now(UTC)

    async def initialize(self) -> bool:
        """Initialize all system components."""
        self.logger.info("="*80)
        self.logger.info("INITIALIZING ORDINIS SYSTEM v0.52")
        self.logger.info("="*80)

        try:
            # 1. Initialize SignalCore Engine
            await self._init_signalcore()

            # 2. Initialize RiskGuard Engine
            await self._init_riskguard()

            # 3. Initialize Portfolio Engine
            await self._init_portfolio()

            # 4. Initialize Analytics Engine
            await self._init_analytics()

            # 5. Initialize Orchestration Engine
            await self._init_orchestration()

            # 6. Set up message bus subscriptions
            self._setup_subscriptions()

            # 7. Initialize market data connections
            await self._init_market_data()

            # 8. Register health checks
            self._register_health_checks()

            self.logger.info("System initialization complete")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def _init_signalcore(self) -> None:
        """Initialize SignalCore engine with multiple models."""
        self.logger.info("Initializing SignalCore engine...")

        config = SignalCoreEngineConfig(
            min_probability=0.6,
            min_score=0.3,
            enable_governance=False,  # Simplified for demo
            enable_ensemble=True,
        )

        self.signalcore_engine = SignalCoreEngine(config)
        await self.signalcore_engine.initialize()

        # Register models
        models = [
            (ATROptimizedRSIModel, {"rsi_oversold": 30, "rsi_overbought": 70}),
            (BollingerBandsModel, {"period": 20, "std_dev": 2.0}),
            (MACDModel, {"fast_period": 12, "slow_period": 26}),
            (RSIMeanReversionModel, {"oversold": 30, "overbought": 70}),
        ]

        for i, (model_class, params) in enumerate(models):
            model_config = ModelConfig(
                model_id=f"{model_class.__name__}_{i}",
                model_type="technical",
                parameters=params,
                weight=1.0
            )
            model = model_class(model_config)
            self.signalcore_engine.register_model(model)
            self.logger.info(f"  Registered model: {model_class.__name__}")

        # Also try to register HMM model if available
        try:
            hmm_config = ModelConfig(
                model_id="hmm_regime",
                model_type="ml",
                parameters={"n_states": 3, "min_history": 50}
            )
            hmm_model = HMMRegimeModel(hmm_config)
            self.signalcore_engine.register_model(hmm_model)
            self.logger.info("  Registered model: HMMRegimeModel (ML)")
        except Exception as e:
            self.logger.warning(f"  Could not register HMM model: {e}")

    async def _init_riskguard(self) -> None:
        """Initialize RiskGuard engine."""
        self.logger.info("Initializing RiskGuard engine...")

        config = RiskGuardConfig(
            max_position_size=0.05,  # 5% max per position
            max_portfolio_var=0.02,   # 2% portfolio VaR
            max_correlation=0.7,      # 70% correlation limit
            max_drawdown=0.1,         # 10% max drawdown
        )

        self.riskguard_engine = RiskGuardEngine(config)
        await self.riskguard_engine.initialize()

    async def _init_portfolio(self) -> None:
        """Initialize Portfolio engine."""
        self.logger.info("Initializing Portfolio engine...")

        config = PortfolioConfig(
            initial_capital=100000,
            max_positions=20,
            rebalance_threshold=0.1,
            commission=0.001,  # 0.1% commission
        )

        self.portfolio_engine = PortfolioEngine(config)
        await self.portfolio_engine.initialize()

    async def _init_analytics(self) -> None:
        """Initialize Analytics engine."""
        self.logger.info("Initializing Analytics engine...")

        config = AnalyticsConfig(
            metrics_interval=30,  # Calculate metrics every 30 seconds
            performance_window=86400,  # 24 hour window
            enable_ml_attribution=True,
        )

        self.analytics_engine = AnalyticsEngine(config)
        await self.analytics_engine.initialize()

    async def _init_orchestration(self) -> None:
        """Initialize Orchestration engine."""
        self.logger.info("Initializing Orchestration engine...")

        config = OrchestrationEngineConfig(
            mode="paper",  # Paper trading mode
            cycle_interval=60,  # 60 second cycles
            enable_monitoring=True,
        )

        self.orchestration_engine = OrchestrationEngine(config)
        await self.orchestration_engine.initialize()

        # Register engines with orchestrator
        self.orchestration_engine.register_engine("signalcore", self.signalcore_engine)
        self.orchestration_engine.register_engine("riskguard", self.riskguard_engine)
        self.orchestration_engine.register_engine("portfolio", self.portfolio_engine)
        self.orchestration_engine.register_engine("analytics", self.analytics_engine)

    async def _init_market_data(self) -> None:
        """Initialize market data connections."""
        self.logger.info("Initializing market data connections...")

        # Initialize Alpaca broker
        self.broker = AlpacaBroker()
        await self.broker.initialize()

        # Initialize Massive stream
        massive_key = os.environ.get("MASSIVE_API_KEY")
        if massive_key:
            self.stream_manager = MassiveStreamManager(
                api_key=massive_key,
                stream_name="system-demo",
                is_sandbox=False,
            )
            await self.stream_manager.connect()

            # Subscribe to symbols
            symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMD", "NVDA", "SPY", "QQQ"]
            await self.stream_manager.subscribe(symbols)
            self.logger.info(f"  Subscribed to {len(symbols)} symbols")

    def _setup_subscriptions(self) -> None:
        """Set up message bus subscriptions."""
        self.logger.info("Setting up message bus subscriptions...")

        # SignalCore publishes signals
        self.message_bus.subscribe("signal_generated", self._on_signal_generated)

        # RiskGuard evaluates signals
        self.message_bus.subscribe("risk_evaluation", self._on_risk_evaluation)

        # Portfolio executes orders
        self.message_bus.subscribe("order_executed", self._on_order_executed)

        # Analytics tracks performance
        self.message_bus.subscribe("performance_update", self._on_performance_update)

    def _register_health_checks(self) -> None:
        """Register health checks for all components."""
        self.health_monitor.register_component("signalcore", self.signalcore_engine)
        self.health_monitor.register_component("riskguard", self.riskguard_engine)
        self.health_monitor.register_component("portfolio", self.portfolio_engine)
        self.health_monitor.register_component("analytics", self.analytics_engine)
        self.health_monitor.register_component("orchestration", self.orchestration_engine)
        self.health_monitor.register_component("message_bus", self.message_bus)

    async def _on_signal_generated(self, event: Dict) -> None:
        """Handle signal generated event."""
        signal = event["data"]
        self.logger.info(f"SIGNAL | {signal.symbol} | {signal.signal_type} | Confidence: {signal.confidence:.2%}")

        # Forward to RiskGuard for evaluation
        self.message_bus.publish("evaluate_signal", signal, "system")

    async def _on_risk_evaluation(self, event: Dict) -> None:
        """Handle risk evaluation event."""
        evaluation = event["data"]
        self.logger.info(f"RISK | {evaluation['status']} | {evaluation['reason']}")

        if evaluation["status"] == "approved":
            # Forward to Portfolio for execution
            self.message_bus.publish("execute_order", evaluation["signal"], "system")

    async def _on_order_executed(self, event: Dict) -> None:
        """Handle order execution event."""
        order = event["data"]
        self.logger.info(f"EXECUTED | {order['symbol']} | {order['side']} | {order['qty']} shares")

    async def _on_performance_update(self, event: Dict) -> None:
        """Handle performance update event."""
        metrics = event["data"]
        self.logger.info(
            f"PERFORMANCE | Sharpe: {metrics.get('sharpe', 0):.2f} | "
            f"Win Rate: {metrics.get('win_rate', 0):.1%} | "
            f"Drawdown: {metrics.get('drawdown', 0):.1%}"
        )

    async def run_cycle(self) -> None:
        """Run a single trading cycle."""
        self.cycle_count += 1
        cycle_start = datetime.now(UTC)

        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"TRADING CYCLE #{self.cycle_count}")
        self.logger.info(f"{'='*80}")

        try:
            # 1. Orchestration coordinates the cycle
            result = await self.orchestration_engine.run_cycle(
                symbols=["AAPL", "MSFT", "GOOGL", "TSLA", "AMD"]
            )

            # 2. Log cycle results
            cycle_time = (datetime.now(UTC) - cycle_start).total_seconds()
            self.logger.info(f"Cycle complete in {cycle_time:.1f}s")

            # 3. Publish cycle metrics
            self.message_bus.publish(
                "cycle_complete",
                {
                    "cycle": self.cycle_count,
                    "duration": cycle_time,
                    "signals": result.signals_generated,
                    "orders": result.orders_executed,
                },
                "orchestrator"
            )

        except Exception as e:
            self.logger.error(f"Cycle failed: {e}")

    async def run(self) -> None:
        """Run the main system loop."""
        self._running = True
        self.logger.info("\nSystem running - Press Ctrl+C to stop")

        try:
            while self._running:
                await self.run_cycle()

                # Check system health
                health = self.health_monitor.check_health()
                if not health.is_healthy:
                    self.logger.warning(f"System health degraded: {health.issues}")

                # Wait for next cycle
                await asyncio.sleep(60)  # 60 second cycles

        except KeyboardInterrupt:
            self.logger.info("Shutdown requested")
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Shutdown all system components."""
        self.logger.info("\nShutting down system...")
        self._running = False

        # Generate final report
        self._generate_report()

        # Shutdown engines
        if self.orchestration_engine:
            await self.orchestration_engine.shutdown()
        if self.signalcore_engine:
            await self.signalcore_engine.shutdown()
        if self.riskguard_engine:
            await self.riskguard_engine.shutdown()
        if self.portfolio_engine:
            await self.portfolio_engine.shutdown()
        if self.analytics_engine:
            await self.analytics_engine.shutdown()

        # Close connections
        if self.stream_manager:
            await self.stream_manager.disconnect()
        if self.broker:
            await self.broker.close()

        self.logger.info("System shutdown complete")

    def _generate_report(self) -> None:
        """Generate final system report."""
        runtime = (datetime.now(UTC) - self.start_time).total_seconds() / 60
        event_stats = self.message_bus.get_event_stats()

        report = f"""
{'='*80}
ORDINIS SYSTEM v0.52 - FINAL REPORT
{'='*80}
Runtime: {runtime:.1f} minutes
Cycles Completed: {self.cycle_count}

Event Statistics:
"""
        for event_type, count in event_stats.items():
            report += f"  {event_type}: {count}\n"

        report += f"""
System Components:
  SignalCore: {self.signalcore_engine.get_metrics() if self.signalcore_engine else 'N/A'}
  RiskGuard: Active
  Portfolio: Active
  Analytics: Active
  Orchestration: {self.cycle_count} cycles

{'='*80}
"""
        print(report)

        # Save report to file
        with open("exe/052/system_report.txt", "w") as f:
            f.write(report)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def main():
    """Main entry point for the system demo."""
    # Load environment variables
    load_dotenv()

    # Print banner
    print()
    print("="*80)
    print("ORDINIS SYSTEM INTEGRATION DEMO v0.52")
    print("Testing FULL SYSTEM Architecture")
    print("="*80)
    print()
    print("Components:")
    print("  ✓ Orchestration Engine (Coordinator)")
    print("  ✓ SignalCore Engine (Multiple Models)")
    print("  ✓ RiskGuard Engine (Risk Management)")
    print("  ✓ Portfolio Engine (Position Management)")
    print("  ✓ Analytics Engine (Performance Tracking)")
    print("  ✓ Message Bus (Event Distribution)")
    print()

    # Create and run system
    system = SystemOrchestrator()

    if await system.initialize():
        await system.run()
    else:
        logger.error("System initialization failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())