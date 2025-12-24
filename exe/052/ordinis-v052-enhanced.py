#!/usr/bin/env python
"""
Ordinis System v0.52 Enhanced - Advanced Traceable Logging
Full system architecture with comprehensive event tracking and performance metrics.

Features:
- Correlation IDs for event tracking across engines
- Performance metrics at each pipeline stage
- Detailed latency measurements
- Event flow visualization
- System health monitoring
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
from collections import defaultdict, deque

# Add parent directory to path
sys.path.insert(0, os.path.abspath("../.."))


@dataclass
class EventMetrics:
    """Track metrics for each event through the pipeline."""
    event_id: str
    correlation_id: str
    symbol: str
    event_type: str
    created_at: float = field(default_factory=time.time)
    stages: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_stage(self, stage: str, metadata: Optional[Dict] = None):
        """Record when event reaches a stage."""
        self.stages[stage] = time.time()
        if metadata:
            self.metadata[stage] = metadata

    def get_latency(self, from_stage: str, to_stage: str) -> Optional[float]:
        """Calculate latency between stages."""
        if from_stage in self.stages and to_stage in self.stages:
            return self.stages[to_stage] - self.stages[from_stage]
        return None

    def total_latency(self) -> float:
        """Total processing time."""
        if self.stages:
            return max(self.stages.values()) - self.created_at
        return 0.0


class MetricsCollector:
    """Centralized metrics collection and reporting."""

    def __init__(self):
        self.events: Dict[str, EventMetrics] = {}
        self.stage_latencies: defaultdict = defaultdict(deque)
        self.signal_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.last_report_time = time.time()

    def track_event(self, event_id: str, correlation_id: str, symbol: str, event_type: str) -> EventMetrics:
        """Start tracking a new event."""
        metrics = EventMetrics(event_id, correlation_id, symbol, event_type)
        self.events[event_id] = metrics
        return metrics

    def record_stage(self, event_id: str, stage: str, metadata: Optional[Dict] = None):
        """Record event reaching a stage."""
        if event_id in self.events:
            self.events[event_id].add_stage(stage, metadata)

            # Track stage latencies
            if len(self.events[event_id].stages) > 1:
                prev_stages = list(self.events[event_id].stages.keys())
                if len(prev_stages) >= 2:
                    prev_stage = prev_stages[-2]
                    latency = self.events[event_id].get_latency(prev_stage, stage)
                    if latency:
                        key = f"{prev_stage}->{stage}"
                        self.stage_latencies[key].append(latency * 1000)  # Convert to ms
                        if len(self.stage_latencies[key]) > 100:
                            self.stage_latencies[key].popleft()

    def generate_report(self) -> Dict:
        """Generate performance metrics report."""
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_events": len(self.events),
            "signal_counts": dict(self.signal_counts),
            "error_counts": dict(self.error_counts),
            "stage_latencies_ms": {}
        }

        # Calculate average latencies
        for stage_pair, latencies in self.stage_latencies.items():
            if latencies:
                report["stage_latencies_ms"][stage_pair] = {
                    "avg": sum(latencies) / len(latencies),
                    "min": min(latencies),
                    "max": max(latencies),
                    "count": len(latencies)
                }

        return report


class EnhancedLogManager:
    """Enhanced logging with correlation IDs and structured output."""

    def __init__(self, name: str, metrics: MetricsCollector):
        self.name = name
        self.metrics = metrics
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup structured logger with custom formatting."""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)

        # Console handler with detailed format
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        fmt = logging.Formatter(
            '[%(asctime)s.%(msecs)03d] %(levelname)8s | %(name)12s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console.setFormatter(fmt)
        logger.addHandler(console)

        # File handler with JSON format for analysis
        os.makedirs('logs/v052', exist_ok=True)
        file_handler = logging.FileHandler(f'logs/v052/{self.name.lower()}.jsonl')
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        return logger

    def log_event(self, level: str, event_id: str, correlation_id: str, message: str, **kwargs):
        """Log with correlation tracking."""
        log_entry = {
            "timestamp": time.time(),
            "event_id": event_id,
            "correlation_id": correlation_id,
            "message": message,
            **kwargs
        }

        # Log to file as JSON
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.stream.write(json.dumps(log_entry) + '\n')
                handler.stream.flush()

        # Log to console in readable format
        formatted_msg = f"[{correlation_id[:8]}] {message}"
        if kwargs:
            formatted_msg += f" | {kwargs}"
        getattr(self.logger, level.lower())(formatted_msg)


class SignalCoreEngine:
    """Signal generation with ATR-RSI strategy and event tracking."""

    def __init__(self, logger: EnhancedLogManager, metrics: MetricsCollector):
        self.logger = logger
        self.metrics = metrics
        self.history = defaultdict(deque)

        # ATR-RSI parameters
        self.rsi_period = 14
        self.atr_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.atr_stop_mult = 2.0
        self.atr_tp_mult = 3.0

        self.signal_count = 0

    def calculate_rsi(self, prices: List[float]) -> float:
        """Calculate RSI."""
        if len(prices) < self.rsi_period + 1:
            return 50.0

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-self.rsi_period:]) / self.rsi_period
        avg_loss = sum(losses[-self.rsi_period:]) / self.rsi_period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float]) -> float:
        """Calculate ATR."""
        if len(highs) < self.atr_period + 1:
            return 0.0

        trs = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)

        return sum(trs[-self.atr_period:]) / self.atr_period if trs else 0.0

    async def process_bar(self, bar_data: Dict) -> Optional[Dict]:
        """Process market bar and generate signal with tracking."""
        event_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())

        # Start tracking
        event_metrics = self.metrics.track_event(
            event_id, correlation_id, bar_data['symbol'], 'BAR_PROCESS'
        )

        symbol = bar_data['symbol']

        # Update history
        self.history[symbol].append({
            'open': bar_data['open'],
            'high': bar_data['high'],
            'low': bar_data['low'],
            'close': bar_data['close'],
            'volume': bar_data['volume'],
            'timestamp': bar_data['timestamp']
        })

        # Keep only recent history
        if len(self.history[symbol]) > 100:
            self.history[symbol].popleft()

        # Need minimum bars
        if len(self.history[symbol]) < self.rsi_period + 1:
            return None

        # Track calculation stage
        self.metrics.record_stage(event_id, 'CALCULATION_START')

        # Calculate indicators
        closes = [bar['close'] for bar in self.history[symbol]]
        highs = [bar['high'] for bar in self.history[symbol]]
        lows = [bar['low'] for bar in self.history[symbol]]

        current_rsi = self.calculate_rsi(closes)
        current_atr = self.calculate_atr(highs, lows, closes)
        current_price = bar_data['close']

        self.metrics.record_stage(event_id, 'CALCULATION_COMPLETE', {
            'rsi': current_rsi,
            'atr': current_atr,
            'price': current_price
        })

        # Generate signal
        signal = None

        if current_rsi < self.rsi_oversold:
            signal = {
                'id': event_id,
                'correlation_id': correlation_id,
                'symbol': symbol,
                'type': 'LONG',
                'price': current_price,
                'rsi': current_rsi,
                'atr': current_atr,
                'stop_loss': current_price - (current_atr * self.atr_stop_mult),
                'take_profit': current_price + (current_atr * self.atr_tp_mult),
                'confidence': 1.0 - (current_rsi / self.rsi_oversold),
                'timestamp': datetime.now(UTC),
                'reason': f'RSI oversold ({current_rsi:.1f} < {self.rsi_oversold})'
            }
        elif current_rsi > self.rsi_overbought:
            signal = {
                'id': event_id,
                'correlation_id': correlation_id,
                'symbol': symbol,
                'type': 'EXIT',
                'price': current_price,
                'rsi': current_rsi,
                'confidence': (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought),
                'timestamp': datetime.now(UTC),
                'reason': f'RSI overbought ({current_rsi:.1f} > {self.rsi_overbought})'
            }

        if signal:
            self.signal_count += 1
            self.metrics.record_stage(event_id, 'SIGNAL_GENERATED')
            self.metrics.signal_counts[signal['type']] += 1

            self.logger.log_event(
                'info', event_id, correlation_id,
                f"SIGNAL | {symbol} | {signal['type']}",
                rsi=f"{current_rsi:.1f}",
                price=f"${current_price:.2f}",
                confidence=f"{signal['confidence']:.2%}"
            )

            return signal

        return None


class MessageBus:
    """Event bus with correlation tracking and metrics."""

    def __init__(self, logger: EnhancedLogManager, metrics: MetricsCollector):
        self.logger = logger
        self.metrics = metrics
        self.subscribers = defaultdict(list)
        self.event_count = 0

    async def publish(self, event_type: str, event_data: Dict):
        """Publish event with tracking."""
        self.event_count += 1
        event_id = event_data.get('id', str(uuid.uuid4()))
        correlation_id = event_data.get('correlation_id', str(uuid.uuid4()))

        self.metrics.record_stage(event_id, f'BUS_{event_type}')

        self.logger.log_event(
            'debug', event_id, correlation_id,
            f"EVENT | {event_type}",
            symbol=event_data.get('symbol'),
            subscribers=len(self.subscribers[event_type])
        )

        # Notify subscribers
        for callback in self.subscribers[event_type]:
            try:
                await callback(event_data)
            except Exception as e:
                self.metrics.error_counts[event_type] += 1
                self.logger.log_event(
                    'error', event_id, correlation_id,
                    f"Subscriber error: {e}",
                    event_type=event_type
                )

    def subscribe(self, event_type: str, callback):
        """Subscribe to event type."""
        self.subscribers[event_type].append(callback)


class RiskGuardEngine:
    """Risk evaluation with detailed tracking."""

    def __init__(self, bus: MessageBus, logger: EnhancedLogManager, metrics: MetricsCollector):
        self.bus = bus
        self.logger = logger
        self.metrics = metrics

        # Risk parameters
        self.max_position_size = 0.03
        self.max_positions = 10
        self.min_confidence = 0.1

        # Track positions
        self.open_positions = set()

        # Subscribe to signals
        bus.subscribe('SIGNAL', self.evaluate_signal)

    async def evaluate_signal(self, signal: Dict):
        """Evaluate signal with tracking."""
        event_id = signal['id']
        correlation_id = signal['correlation_id']

        self.metrics.record_stage(event_id, 'RISK_EVALUATION_START')

        # Evaluation logic
        approved = True
        reasons = []

        # Check confidence
        if signal.get('confidence', 0) < self.min_confidence:
            approved = False
            reasons.append(f"Low confidence: {signal.get('confidence', 0):.2%}")

        # Check position limits
        if signal['type'] == 'LONG' and len(self.open_positions) >= self.max_positions:
            approved = False
            reasons.append(f"Max positions reached: {len(self.open_positions)}/{self.max_positions}")

        # Check if already have position
        if signal['type'] == 'LONG' and signal['symbol'] in self.open_positions:
            approved = False
            reasons.append("Already have position")

        self.metrics.record_stage(event_id, 'RISK_EVALUATION_COMPLETE', {
            'approved': approved,
            'reasons': reasons
        })

        status = "APPROVED" if approved else "REJECTED"
        self.logger.log_event(
            'info', event_id, correlation_id,
            f"{status} | {signal['symbol']} | {signal['type']}",
            confidence=f"{signal.get('confidence', 0):.2%}",
            reasons=reasons if reasons else None
        )

        if approved:
            # Track position
            if signal['type'] == 'LONG':
                self.open_positions.add(signal['symbol'])
            elif signal['type'] == 'EXIT' and signal['symbol'] in self.open_positions:
                self.open_positions.discard(signal['symbol'])

            # Forward to portfolio
            await self.bus.publish('RISK_APPROVED', signal)


class PortfolioEngine:
    """Portfolio management with execution tracking."""

    def __init__(self, bus: MessageBus, logger: EnhancedLogManager, metrics: MetricsCollector):
        self.bus = bus
        self.logger = logger
        self.metrics = metrics

        self.positions = {}
        self.cash = 100000
        self.position_size = 0.03

        # Subscribe to approved signals
        bus.subscribe('RISK_APPROVED', self.execute_trade)

    async def execute_trade(self, signal: Dict):
        """Execute trade with tracking."""
        event_id = signal['id']
        correlation_id = signal['correlation_id']

        self.metrics.record_stage(event_id, 'EXECUTION_START')

        symbol = signal['symbol']

        if signal['type'] == 'LONG':
            # Calculate position size
            position_value = self.cash * self.position_size
            shares = int(position_value / signal['price'])

            if shares > 0 and position_value <= self.cash:
                # Execute buy
                self.positions[symbol] = {
                    'shares': shares,
                    'entry_price': signal['price'],
                    'entry_time': time.time()
                }
                self.cash -= shares * signal['price']

                self.metrics.record_stage(event_id, 'EXECUTION_COMPLETE', {
                    'action': 'BUY',
                    'shares': shares,
                    'price': signal['price']
                })

                self.logger.log_event(
                    'info', event_id, correlation_id,
                    f"BOUGHT | {symbol}",
                    shares=shares,
                    price=f"${signal['price']:.2f}",
                    value=f"${shares * signal['price']:.2f}"
                )

                # Publish execution event
                await self.bus.publish('EXECUTION', {
                    'id': event_id,
                    'correlation_id': correlation_id,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': signal['price']
                })

        elif signal['type'] == 'EXIT' and symbol in self.positions:
            # Execute sell
            position = self.positions[symbol]
            shares = position['shares']
            entry_price = position['entry_price']
            exit_price = signal['price']

            # Calculate P&L
            pnl = (exit_price - entry_price) * shares
            pnl_pct = (exit_price - entry_price) / entry_price * 100

            # Update cash
            self.cash += shares * exit_price
            del self.positions[symbol]

            self.metrics.record_stage(event_id, 'EXECUTION_COMPLETE', {
                'action': 'SELL',
                'shares': shares,
                'price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })

            self.logger.log_event(
                'info', event_id, correlation_id,
                f"SOLD | {symbol}",
                shares=shares,
                price=f"${exit_price:.2f}",
                pnl=f"${pnl:+.2f}",
                pnl_pct=f"{pnl_pct:+.2f}%"
            )

            # Publish execution event
            await self.bus.publish('EXECUTION', {
                'id': event_id,
                'correlation_id': correlation_id,
                'symbol': symbol,
                'action': 'SELL',
                'shares': shares,
                'price': exit_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })


class SystemOrchestrator:
    """Main orchestrator with comprehensive monitoring."""

    def __init__(self):
        # Setup metrics collector
        self.metrics = MetricsCollector()

        # Setup enhanced loggers
        self.log_orchestrator = EnhancedLogManager('ORCHESTRATOR', self.metrics)
        self.log_signalcore = EnhancedLogManager('SIGNALCORE', self.metrics)
        self.log_riskguard = EnhancedLogManager('RISKGUARD', self.metrics)
        self.log_portfolio = EnhancedLogManager('PORTFOLIO', self.metrics)
        self.log_bus = EnhancedLogManager('MESSAGEBUS', self.metrics)

        # Initialize engines
        self.signal_engine = SignalCoreEngine(self.log_signalcore, self.metrics)
        self.bus = MessageBus(self.log_bus, self.metrics)
        self.risk_engine = RiskGuardEngine(self.bus, self.log_riskguard, self.metrics)
        self.portfolio_engine = PortfolioEngine(self.bus, self.log_portfolio, self.metrics)

        # Streaming
        self.stream = None
        self.start_time = time.time()
        self.bar_count = 0

        # List of symbols to monitor
        self.symbols = [
            "SPY", "QQQ", "IWM", "DIA", "AAPL", "MSFT", "NVDA", "TSLA",
            "AMD", "META", "GOOGL", "AMZN", "NFLX", "JPM", "BAC", "GS",
            "XOM", "CVX", "COP", "SLB", "BA", "CAT", "DE", "HON",
            "UNH", "PFE", "JNJ", "ABBV", "MRK"
        ]

    async def connect_market_data(self):
        """Connect to market data with error handling."""
        try:
            from ordinis.adapters.streaming.massive_stream import MassiveStreamManager
            from ordinis.adapters.streaming.stream_protocol import StreamConfig, CallbackStreamHandler

            massive_key = os.environ.get("MASSIVE_API_KEY")
            if not massive_key:
                self.log_orchestrator.logger.warning("MASSIVE_API_KEY not set, running without market data")
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
                on_status_callback=lambda status: self.log_orchestrator.logger.info(f"Stream status: {status}"),
                on_error_callback=lambda error: self.log_orchestrator.logger.error(f"Stream error: {error}")
            )
            self.stream.add_handler(handler)

            # Connect
            await self.stream.connect()
            self.log_orchestrator.logger.info("Connected to Massive WebSocket")

            # Subscribe to symbols
            await self.stream.subscribe(self.symbols)
            self.log_orchestrator.logger.info(f"Subscribed to {len(self.symbols)} symbols")

        except Exception as e:
            self.log_orchestrator.logger.error(f"Failed to connect market data: {e}")

    async def _on_bar(self, bar):
        """Process incoming market bar."""
        try:
            self.bar_count += 1

            # Convert bar to dict
            bar_data = {
                'symbol': bar.symbol,
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            }

            # Process through signal engine
            signal = await self.signal_engine.process_bar(bar_data)

            if signal:
                # Publish to message bus
                await self.bus.publish('SIGNAL', signal)

        except Exception as e:
            self.log_orchestrator.logger.error(f"Bar processing error: {e}")

    async def report_metrics(self):
        """Periodically report system metrics."""
        while True:
            await asyncio.sleep(60)  # Report every minute

            try:
                report = self.metrics.generate_report()

                runtime = time.time() - self.start_time
                runtime_mins = runtime / 60

                # Calculate portfolio value
                portfolio_value = self.portfolio_engine.cash
                for symbol, position in self.portfolio_engine.positions.items():
                    # Use last known price (would need market data for real value)
                    portfolio_value += position['shares'] * position['entry_price']

                # Log comprehensive status
                self.log_orchestrator.logger.info(
                    f"METRICS | Runtime: {runtime_mins:.1f}min | "
                    f"Bars: {self.bar_count} | "
                    f"Signals: {self.signal_engine.signal_count} | "
                    f"Positions: {len(self.portfolio_engine.positions)} | "
                    f"Portfolio: ${portfolio_value:,.2f} | "
                    f"Events: {self.bus.event_count}"
                )

                # Log latency metrics
                if report['stage_latencies_ms']:
                    self.log_orchestrator.logger.info("LATENCIES (ms):")
                    for stage_pair, metrics in report['stage_latencies_ms'].items():
                        self.log_orchestrator.logger.info(
                            f"  {stage_pair}: avg={metrics['avg']:.1f} "
                            f"min={metrics['min']:.1f} max={metrics['max']:.1f}"
                        )

                # Save detailed report
                with open('logs/v052/metrics_report.json', 'w') as f:
                    json.dump(report, f, indent=2, default=str)

            except Exception as e:
                self.log_orchestrator.logger.error(f"Metrics reporting error: {e}")

    async def run(self):
        """Run the system with enhanced monitoring."""
        self.log_orchestrator.logger.info("="*70)
        self.log_orchestrator.logger.info("ORDINIS SYSTEM v0.52 ENHANCED - TRACEABLE LOGGING")
        self.log_orchestrator.logger.info("="*70)
        self.log_orchestrator.logger.info("Architecture: Market -> SignalCore -> RiskGuard -> Portfolio")
        self.log_orchestrator.logger.info("Features: Correlation IDs | Performance Metrics | Event Tracking")
        self.log_orchestrator.logger.info("="*70)

        # Connect to market data
        await self.connect_market_data()

        # Start metrics reporting
        metrics_task = asyncio.create_task(self.report_metrics())

        # Main loop
        self.log_orchestrator.logger.info("System running with advanced traceability...")
        self.log_orchestrator.logger.info(f"Monitoring {len(self.symbols)} symbols")
        self.log_orchestrator.logger.info("Check logs/v052/ for detailed JSON logs")

        try:
            while True:
                await asyncio.sleep(10)

                # Quick status
                positions_str = ', '.join(self.portfolio_engine.positions.keys()) if self.portfolio_engine.positions else 'None'
                self.log_orchestrator.logger.info(
                    f"STATUS | Bars: {self.bar_count} | "
                    f"Signals: {self.signal_engine.signal_count} | "
                    f"Positions: {positions_str}"
                )

        except KeyboardInterrupt:
            self.log_orchestrator.logger.info("Shutting down...")
            if self.stream:
                await self.stream.disconnect()
            metrics_task.cancel()


async def main():
    """Main entry point."""
    orchestrator = SystemOrchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    # Set up environment
    from dotenv import load_dotenv
    load_dotenv()

    # Run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSystem stopped by user")