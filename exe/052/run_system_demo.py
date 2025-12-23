#!/usr/bin/env python
"""
Ordinis System Demo v0.52 - Simplified Runner

Tests the system architecture with available engines.
Creates stubs for missing components.
"""

import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("system_demo")


# =============================================================================
# STUB IMPLEMENTATIONS FOR MISSING ENGINES
# =============================================================================

class StubRiskGuardEngine:
    """Stub RiskGuard engine for demo."""

    def __init__(self):
        self.logger = logging.getLogger("riskguard")
        self.max_position_size = 0.05
        self.max_drawdown = 0.1

    async def initialize(self):
        self.logger.info("RiskGuard initialized (stub)")
        return True

    async def evaluate_signal(self, signal):
        """Evaluate a signal for risk."""
        # Simple risk check
        if signal.get("confidence", 0) < 0.5:
            return {"status": "rejected", "reason": "Low confidence"}

        return {"status": "approved", "reason": "Risk acceptable"}

    async def shutdown(self):
        self.logger.info("RiskGuard shutdown")


class StubPortfolioEngine:
    """Stub Portfolio engine for demo."""

    def __init__(self):
        self.logger = logging.getLogger("portfolio")
        self.positions = {}
        self.cash = 100000
        self.equity = 100000

    async def initialize(self):
        self.logger.info("Portfolio initialized (stub)")
        return True

    async def execute_order(self, signal):
        """Execute an order based on signal."""
        symbol = signal.get("symbol")
        self.positions[symbol] = self.positions.get(symbol, 0) + 100
        self.logger.info(f"Executed: {symbol} - 100 shares")
        return {"status": "filled", "symbol": symbol, "qty": 100}

    def get_metrics(self):
        return {
            "positions": len(self.positions),
            "equity": self.equity,
            "cash": self.cash
        }

    async def shutdown(self):
        self.logger.info("Portfolio shutdown")


class StubAnalyticsEngine:
    """Stub Analytics engine for demo."""

    def __init__(self):
        self.logger = logging.getLogger("analytics")
        self.trades = []

    async def initialize(self):
        self.logger.info("Analytics initialized (stub)")
        return True

    def calculate_metrics(self):
        """Calculate performance metrics."""
        return {
            "sharpe": np.random.uniform(0.5, 2.0),
            "win_rate": np.random.uniform(0.4, 0.7),
            "drawdown": np.random.uniform(0.01, 0.1),
            "trades": len(self.trades)
        }

    async def shutdown(self):
        self.logger.info("Analytics shutdown")


# =============================================================================
# SIMPLIFIED MESSAGE BUS
# =============================================================================

class MessageBus:
    """Simple message bus for demo."""

    def __init__(self):
        self.logger = logging.getLogger("message_bus")
        self.events = []

    def publish(self, event_type, data):
        """Publish an event."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(UTC)
        }
        self.events.append(event)
        self.logger.info(f"EVENT: {event_type}")
        return event


# =============================================================================
# SYSTEM DEMO WITH AVAILABLE ENGINES
# =============================================================================

async def run_system_demo():
    """Run the system demo with available components."""

    logger.info("="*80)
    logger.info("ORDINIS SYSTEM DEMO v0.52")
    logger.info("Testing System Architecture")
    logger.info("="*80)

    # Initialize components
    message_bus = MessageBus()
    riskguard = StubRiskGuardEngine()
    portfolio = StubPortfolioEngine()
    analytics = StubAnalyticsEngine()

    # Try to import real engines
    signalcore_available = False
    orchestration_available = False

    try:
        from ordinis.engines.signalcore.core.engine import SignalCoreEngine
        from ordinis.engines.signalcore.core.config import SignalCoreEngineConfig
        signalcore_available = True
        logger.info("✓ SignalCore engine available")
    except ImportError as e:
        logger.warning(f"✗ SignalCore not available: {e}")

    try:
        from ordinis.engines.orchestration.core import OrchestrationEngine
        orchestration_available = True
        logger.info("✓ Orchestration engine available")
    except ImportError as e:
        logger.warning(f"✗ Orchestration not available: {e}")

    # Initialize engines
    await riskguard.initialize()
    await portfolio.initialize()
    await analytics.initialize()

    logger.info("")
    logger.info("Starting demo cycles...")
    logger.info("")

    # Run demo cycles
    for cycle in range(3):
        logger.info(f"{'='*60}")
        logger.info(f"CYCLE #{cycle + 1}")
        logger.info(f"{'='*60}")

        # 1. Generate signals (stub if SignalCore not available)
        if signalcore_available:
            # Use real SignalCore
            try:
                config = SignalCoreEngineConfig(
                    min_probability=0.6,
                    min_score=0.3
                )
                signalcore = SignalCoreEngine(config)
                await signalcore.initialize()

                # Generate test data
                data = pd.DataFrame({
                    'close': np.random.uniform(100, 110, 100),
                    'volume': np.random.uniform(1000000, 2000000, 100)
                })

                signal = await signalcore.generate_signal("AAPL", data)
                if signal:
                    logger.info(f"SignalCore: Generated {signal.signal_type}")
                    message_bus.publish("signal_generated", signal.__dict__)
                else:
                    logger.info("SignalCore: No signal generated")

            except Exception as e:
                logger.error(f"SignalCore error: {e}")
                # Fall back to stub
                signal = {"symbol": "AAPL", "confidence": 0.7, "type": "BUY"}
                logger.info("SignalCore (stub): Generated BUY signal")
                message_bus.publish("signal_generated", signal)
        else:
            # Use stub signal
            signals = [
                {"symbol": "AAPL", "confidence": 0.75, "type": "BUY"},
                {"symbol": "MSFT", "confidence": 0.65, "type": "SELL"},
                {"symbol": "GOOGL", "confidence": 0.45, "type": "BUY"},
            ]
            for signal in signals:
                logger.info(f"Signal (stub): {signal['symbol']} {signal['type']} (conf: {signal['confidence']})")
                message_bus.publish("signal_generated", signal)

                # 2. Risk evaluation
                risk_result = await riskguard.evaluate_signal(signal)
                logger.info(f"RiskGuard: {risk_result['status']} - {risk_result['reason']}")
                message_bus.publish("risk_evaluation", risk_result)

                # 3. Execute if approved
                if risk_result['status'] == 'approved':
                    order = await portfolio.execute_order(signal)
                    message_bus.publish("order_executed", order)

        # 4. Calculate metrics
        metrics = analytics.calculate_metrics()
        logger.info(f"Analytics: Sharpe={metrics['sharpe']:.2f}, Win={metrics['win_rate']:.1%}, DD={metrics['drawdown']:.1%}")
        message_bus.publish("metrics_calculated", metrics)

        # 5. Portfolio status
        portfolio_metrics = portfolio.get_metrics()
        logger.info(f"Portfolio: {portfolio_metrics['positions']} positions, ${portfolio_metrics['equity']:,.0f} equity")

        logger.info("")
        await asyncio.sleep(2)

    # Final report
    logger.info("="*80)
    logger.info("DEMO COMPLETE")
    logger.info("="*80)
    logger.info(f"Total events: {len(message_bus.events)}")

    event_types = {}
    for event in message_bus.events:
        event_type = event['type']
        event_types[event_type] = event_types.get(event_type, 0) + 1

    logger.info("Event breakdown:")
    for event_type, count in event_types.items():
        logger.info(f"  {event_type}: {count}")

    # Shutdown
    await riskguard.shutdown()
    await portfolio.shutdown()
    await analytics.shutdown()

    logger.info("")
    logger.info("System architecture demonstration complete!")
    logger.info("This shows how engines communicate via message bus")
    logger.info("Full implementation would use real engines")


if __name__ == "__main__":
    load_dotenv()
    asyncio.run(run_system_demo())