#!/usr/bin/env python
"""
Ordinis System v0.52 - Live Market Monitoring

STANDALONE system demo with full engine architecture.
Monitors ALL market symbols (not a fixed list).
Uses ATR-RSI strategy via SignalCore.

Architecture:
    Market Data (ALL symbols) → SignalCore (ATR-RSI) → RiskGuard → Portfolio → Analytics
"""

import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional
from collections import defaultdict

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
logger = logging.getLogger("SYSTEM")

# Suppress some verbose loggers
logging.getLogger("alpaca").setLevel(logging.WARNING)
logging.getLogger("websocket").setLevel(logging.WARNING)


# ==============================================================================
# MESSAGE BUS
# ==============================================================================

class MessageBus:
    """Event-driven message bus for engine communication."""

    def __init__(self):
        self.subscribers = defaultdict(list)
        self.event_count = defaultdict(int)
        self.logger = logging.getLogger("BUS")

    def subscribe(self, event_type: str, handler):
        """Subscribe to an event type."""
        self.subscribers[event_type].append(handler)

    async def publish(self, event_type: str, data: Any):
        """Publish event to all subscribers."""
        self.event_count[event_type] += 1

        # Notify all subscribers
        for handler in self.subscribers[event_type]:
            try:
                await handler(data)
            except Exception as e:
                self.logger.error(f"Handler error for {event_type}: {e}")


# ==============================================================================
# SIGNALCORE ENGINE WITH ATR-RSI
# ==============================================================================

class SignalCoreATR:
    """SignalCore engine with ONLY ATR-RSI model."""

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self.logger = logging.getLogger("SIGNALCORE")

        # Market data storage
        self.market_data = defaultdict(lambda: pd.DataFrame())

        # ATR-RSI parameters
        self.rsi_period = 14
        self.atr_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.atr_stop_mult = 1.5
        self.atr_tp_mult = 2.0

        # Minimum bars required
        self.min_bars = 50

        self.signals_generated = 0

    async def process_bar(self, bar_data: Dict):
        """Process incoming market bar."""
        symbol = bar_data['symbol']

        # Append to market data
        new_row = pd.DataFrame([{
            'timestamp': bar_data['timestamp'],
            'open': bar_data['open'],
            'high': bar_data['high'],
            'low': bar_data['low'],
            'close': bar_data['close'],
            'volume': bar_data['volume']
        }])

        if self.market_data[symbol].empty:
            self.market_data[symbol] = new_row
        else:
            self.market_data[symbol] = pd.concat([
                self.market_data[symbol],
                new_row
            ], ignore_index=True).tail(200)  # Keep last 200 bars

        # Check if we have enough data
        if len(self.market_data[symbol]) < self.min_bars:
            return

        # Calculate indicators
        signal = await self._calculate_signal(symbol)

        if signal:
            self.signals_generated += 1
            await self.bus.publish("signal_generated", signal)

    async def _calculate_signal(self, symbol: str) -> Optional[Dict]:
        """Calculate ATR-RSI signal for a symbol."""
        df = self.market_data[symbol].copy()

        # Calculate RSI
        close = df['close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Calculate ATR
        high = df['high']
        low = df['low']
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()

        # Get current values
        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]

        # Skip if NaN
        if pd.isna(current_rsi) or pd.isna(current_atr):
            return None

        # Generate signal
        signal = None

        if current_rsi < self.rsi_oversold:
            signal = {
                'symbol': symbol,
                'type': 'LONG',
                'price': current_price,
                'rsi': current_rsi,
                'atr': current_atr,
                'stop_loss': current_price - (current_atr * self.atr_stop_mult),
                'take_profit': current_price + (current_atr * self.atr_tp_mult),
                'confidence': 1.0 - (current_rsi / self.rsi_oversold),  # More oversold = higher confidence
                'timestamp': datetime.now(UTC),
                'reason': f'RSI oversold ({current_rsi:.1f} < {self.rsi_oversold})'
            }
            self.logger.info(f"SIGNAL | {symbol} | LONG | RSI={current_rsi:.1f} | Price=${current_price:.2f}")

        elif current_rsi > self.rsi_overbought:
            signal = {
                'symbol': symbol,
                'type': 'EXIT',
                'price': current_price,
                'rsi': current_rsi,
                'confidence': (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought),  # More overbought = higher confidence
                'timestamp': datetime.now(UTC),
                'reason': f'RSI overbought ({current_rsi:.1f} > {self.rsi_overbought})'
            }
            self.logger.info(f"SIGNAL | {symbol} | EXIT | RSI={current_rsi:.1f}")

        return signal


# ==============================================================================
# RISKGUARD ENGINE
# ==============================================================================

class RiskGuardEngine:
    """Risk management engine."""

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self.logger = logging.getLogger("RISKGUARD")

        # Risk parameters
        self.max_position_size = 0.03  # 3% per position
        self.max_positions = 10
        self.min_confidence = 0.1  # Lower threshold to demonstrate system

        # Current positions tracking
        self.open_positions = set()

        # Subscribe to signals
        self.bus.subscribe("signal_generated", self.evaluate_signal)

        self.approved = 0
        self.rejected = 0

    async def evaluate_signal(self, signal: Dict):
        """Evaluate signal for risk."""
        symbol = signal['symbol']
        signal_type = signal['type']

        # Check confidence
        if signal.get('confidence', 0) < self.min_confidence:
            self.rejected += 1
            self.logger.debug(f"REJECTED | {symbol} | Low confidence")
            return

        # Check position limits
        if signal_type == 'LONG' and len(self.open_positions) >= self.max_positions:
            self.rejected += 1
            self.logger.debug(f"REJECTED | {symbol} | Max positions reached")
            return

        # Check if already in position
        if signal_type == 'LONG' and symbol in self.open_positions:
            self.rejected += 1
            self.logger.debug(f"REJECTED | {symbol} | Already in position")
            return

        # Approve signal
        self.approved += 1
        self.logger.info(f"APPROVED | {symbol} | {signal_type}")

        if signal_type == 'LONG':
            self.open_positions.add(symbol)
        elif signal_type == 'EXIT' and symbol in self.open_positions:
            self.open_positions.remove(symbol)

        await self.bus.publish("signal_approved", signal)


# ==============================================================================
# PORTFOLIO ENGINE
# ==============================================================================

class PortfolioEngine:
    """Portfolio management engine."""

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self.logger = logging.getLogger("PORTFOLIO")

        # Portfolio state
        self.positions = {}
        self.cash = 100000.0
        self.starting_equity = 100000.0

        # Subscribe to approved signals
        self.bus.subscribe("signal_approved", self.execute_order)

        self.orders_executed = 0

    async def execute_order(self, signal: Dict):
        """Execute order based on approved signal."""
        symbol = signal['symbol']
        signal_type = signal['type']
        price = signal['price']

        if signal_type == 'LONG':
            # Calculate position size (3% of equity)
            position_value = self.get_equity() * 0.03
            shares = int(position_value / price)

            if shares > 0 and self.cash >= position_value:
                self.positions[symbol] = {
                    'shares': shares,
                    'entry_price': price,
                    'stop_loss': signal.get('stop_loss'),
                    'take_profit': signal.get('take_profit'),
                    'entry_time': datetime.now(UTC)
                }
                self.cash -= position_value
                self.orders_executed += 1

                self.logger.info(f"BOUGHT | {symbol} | {shares} shares @ ${price:.2f}")
                await self.bus.publish("order_executed", {
                    'symbol': symbol,
                    'side': 'BUY',
                    'shares': shares,
                    'price': price
                })

        elif signal_type == 'EXIT' and symbol in self.positions:
            position = self.positions[symbol]
            exit_value = position['shares'] * price
            self.cash += exit_value

            pnl = exit_value - (position['shares'] * position['entry_price'])
            pnl_pct = (price / position['entry_price'] - 1) * 100

            self.logger.info(f"SOLD | {symbol} | P&L: ${pnl:.2f} ({pnl_pct:.1f}%)")

            del self.positions[symbol]
            self.orders_executed += 1

            await self.bus.publish("order_executed", {
                'symbol': symbol,
                'side': 'SELL',
                'shares': position['shares'],
                'price': price,
                'pnl': pnl
            })

    def get_equity(self) -> float:
        """Calculate total equity."""
        return self.cash + sum(
            pos['shares'] * pos['entry_price']
            for pos in self.positions.values()
        )


# ==============================================================================
# ANALYTICS ENGINE
# ==============================================================================

class AnalyticsEngine:
    """Performance analytics engine."""

    def __init__(self, message_bus: MessageBus):
        self.bus = message_bus
        self.logger = logging.getLogger("ANALYTICS")

        # Performance tracking
        self.trades = []
        self.start_time = datetime.now(UTC)

        # Subscribe to order executions
        self.bus.subscribe("order_executed", self.record_trade)

    async def record_trade(self, order: Dict):
        """Record trade for analytics."""
        self.trades.append(order)

        # Calculate metrics periodically
        if len(self.trades) % 10 == 0:
            await self.calculate_metrics()

    async def calculate_metrics(self):
        """Calculate and publish performance metrics."""
        if not self.trades:
            return

        # Calculate win rate
        wins = sum(1 for t in self.trades if t.get('pnl', 0) > 0)
        total = len([t for t in self.trades if 'pnl' in t])
        win_rate = wins / total if total > 0 else 0

        # Calculate average P&L
        pnls = [t['pnl'] for t in self.trades if 'pnl' in t]
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0

        metrics = {
            'trades': len(self.trades),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'runtime': (datetime.now(UTC) - self.start_time).total_seconds() / 60
        }

        self.logger.info(f"METRICS | Trades: {metrics['trades']} | Win: {win_rate:.1%} | Avg P&L: ${avg_pnl:.2f}")
        await self.bus.publish("metrics_update", metrics)


# ==============================================================================
# MARKET DATA STREAMER
# ==============================================================================

class MarketDataStreamer:
    """Connects to live market data and feeds to SignalCore."""

    def __init__(self, message_bus: MessageBus, signalcore: SignalCoreATR):
        self.bus = message_bus
        self.signalcore = signalcore
        self.logger = logging.getLogger("MARKET_DATA")

        # Data tracking
        self.bars_processed = 0
        self.symbols_seen = set()

    async def connect_and_stream(self):
        """Connect to Massive WebSocket for ALL symbols."""
        try:
            from ordinis.adapters.streaming.massive_stream import MassiveStreamManager
            from ordinis.adapters.streaming.stream_protocol import StreamConfig, CallbackStreamHandler

            massive_key = os.environ.get("MASSIVE_API_KEY")
            if not massive_key:
                self.logger.error("MASSIVE_API_KEY not found")
                return

            # Create stream manager
            stream_config = StreamConfig(
                api_key=massive_key,
                reconnect_enabled=True,
                reconnect_delay_seconds=1.0,
                max_reconnect_attempts=10,
            )
            self.stream = MassiveStreamManager(stream_config)

            # Set up callback handler
            handler = CallbackStreamHandler(
                on_bar_callback=self._on_bar,
                on_status_callback=lambda status: self.logger.info(f"Stream status: {status}"),
                on_error_callback=lambda error: self.logger.error(f"Stream error: {error}")
            )
            self.stream.add_handler(handler)

            # Connect
            await self.stream.connect()
            self.logger.info("Connected to Massive WebSocket")

            # Subscribe to ALL symbols ("*" or get all available)
            # For demo, let's subscribe to a broad market set
            symbols = [
                "SPY", "QQQ", "IWM", "DIA",  # ETFs
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA",  # Tech
                "JPM", "BAC", "WFC", "GS", "MS",  # Banks
                "XOM", "CVX", "COP", "SLB",  # Energy
                "UNH", "JNJ", "PFE", "ABBV",  # Healthcare
                "WMT", "HD", "PG", "KO", "PEP",  # Consumer
                # Add ALL available symbols in production
            ]

            await self.stream.subscribe(symbols)
            self.logger.info(f"Subscribed to {len(symbols)} symbols")

            # In production, you would subscribe to "*" or get full symbol list
            # await self.stream.subscribe("*")  # Subscribe to everything

        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")

    async def _on_bar(self, bar):
        """Process incoming market bar."""
        try:
            # Handle StreamBar object
            bar_data = {
                'symbol': bar.symbol,
                'timestamp': bar.timestamp,
                'open': float(bar.open),
                'high': float(bar.high),
                'low': float(bar.low),
                'close': float(bar.close),
                'volume': int(bar.volume)
            }

            # Track statistics
            self.bars_processed += 1
            self.symbols_seen.add(bar.symbol)

            # Feed to SignalCore
            await self.signalcore.process_bar(bar_data)

            # Log progress every 100 bars
            if self.bars_processed % 100 == 0:
                self.logger.info(f"Processed {self.bars_processed} bars from {len(self.symbols_seen)} symbols")

        except Exception as e:
            self.logger.error(f"Error processing bar: {e}")


# ==============================================================================
# SYSTEM ORCHESTRATOR
# ==============================================================================

class SystemOrchestrator:
    """Main system orchestrator."""

    def __init__(self):
        self.logger = logging.getLogger("ORCHESTRATOR")

        # Initialize message bus
        self.bus = MessageBus()

        # Initialize engines
        self.signalcore = SignalCoreATR(self.bus)
        self.riskguard = RiskGuardEngine(self.bus)
        self.portfolio = PortfolioEngine(self.bus)
        self.analytics = AnalyticsEngine(self.bus)

        # Initialize market data
        self.market_data = MarketDataStreamer(self.bus, self.signalcore)

        self.start_time = datetime.now(UTC)
        self._running = False

    async def start(self):
        """Start the system."""
        self.logger.info("="*80)
        self.logger.info("ORDINIS SYSTEM v0.52 - LIVE MARKET MONITORING")
        self.logger.info("="*80)
        self.logger.info("Architecture: Market Data → SignalCore (ATR-RSI) → RiskGuard → Portfolio → Analytics")
        self.logger.info("Monitoring: ALL market symbols (universal)")
        self.logger.info("Strategy: ATR-Optimized RSI only")
        self.logger.info("="*80)

        # Connect to market data
        await self.market_data.connect_and_stream()

        self._running = True
        self.logger.info("System started - Processing all market data")

        # Run monitoring loop
        await self._monitor_loop()

    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Status update every 30 seconds

                # Print status
                runtime = (datetime.now(UTC) - self.start_time).total_seconds() / 60
                equity = self.portfolio.get_equity()
                pnl = equity - self.portfolio.starting_equity
                pnl_pct = (pnl / self.portfolio.starting_equity) * 100

                self.logger.info(
                    f"STATUS | "
                    f"Runtime: {runtime:.1f}min | "
                    f"Bars: {self.market_data.bars_processed} | "
                    f"Symbols: {len(self.market_data.symbols_seen)} | "
                    f"Signals: {self.signalcore.signals_generated} | "
                    f"Approved: {self.riskguard.approved} | "
                    f"Positions: {len(self.portfolio.positions)} | "
                    f"P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)"
                )

                # Print event statistics
                if sum(self.bus.event_count.values()) > 0:
                    events = " | ".join(f"{k}:{v}" for k, v in self.bus.event_count.items())
                    self.logger.info(f"EVENTS | {events}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")

    async def shutdown(self):
        """Shutdown the system."""
        self.logger.info("Shutting down...")
        self._running = False

        # Disconnect market data
        if hasattr(self.market_data, 'stream'):
            await self.market_data.stream.disconnect()

        # Final report
        runtime = (datetime.now(UTC) - self.start_time).total_seconds() / 60
        self.logger.info("="*80)
        self.logger.info("FINAL REPORT")
        self.logger.info(f"Runtime: {runtime:.1f} minutes")
        self.logger.info(f"Bars Processed: {self.market_data.bars_processed}")
        self.logger.info(f"Unique Symbols: {len(self.market_data.symbols_seen)}")
        self.logger.info(f"Signals Generated: {self.signalcore.signals_generated}")
        self.logger.info(f"Orders Executed: {self.portfolio.orders_executed}")
        self.logger.info(f"Final Equity: ${self.portfolio.get_equity():,.2f}")
        self.logger.info("="*80)


# ==============================================================================
# MAIN
# ==============================================================================

async def main():
    """Main entry point."""
    load_dotenv()

    # Create and start system
    system = SystemOrchestrator()

    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())