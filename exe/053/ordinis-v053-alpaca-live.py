#!/usr/bin/env python
"""
Ordinis System v0.53 - Real Alpaca Integration
Connects to real Alpaca paper trading account instead of simulated portfolio.
Uses actual $10k account balance and executes real trades.

Features:
- Real broker connection (Alpaca paper trading)
- Actual account balance and positions from broker
- Real order execution with confirmation
- Correlation IDs for event tracking
- Performance metrics at each pipeline stage
"""

import asyncio
import logging
import os
import sys
import time
import uuid
import subprocess
from datetime import datetime, UTC
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
from collections import defaultdict, deque
from decimal import Decimal

# Add parent directory to path
sys.path.insert(0, os.path.abspath("../.."))

# Import Alpaca SDK
from alpaca_trade_api import REST


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
        self.order_counts = defaultdict(int)
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
        """Generate comprehensive performance metrics report."""
        report = {
            "timestamp": datetime.now(UTC).isoformat(),
            "total_events": len(self.events),
            "signal_counts": dict(self.signal_counts),
            "order_counts": dict(self.order_counts),
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

    def __init__(self, name: str, metrics: MetricsCollector, session_dir: str = None):
        self.name = name
        self.metrics = metrics
        self.session_dir = session_dir or 'logs/v053'
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
        os.makedirs(self.session_dir, exist_ok=True)
        file_handler = logging.FileHandler(f'{self.session_dir}/{self.name.lower()}.jsonl')
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


class AlpacaConnection:
    """Manages Alpaca API connection."""

    def __init__(self, logger: EnhancedLogManager):
        self.logger = logger
        self.api = None
        self.api_key = None
        self.api_secret = None

    def load_credentials(self):
        """Load credentials from Windows User environment."""
        # Try to get from Windows User environment variables
        result = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             '[System.Environment]::GetEnvironmentVariable("APCA_API_KEY_ID", "User")'],
            capture_output=True, text=True
        )
        self.api_key = result.stdout.strip() if result.returncode == 0 else None

        result = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             '[System.Environment]::GetEnvironmentVariable("APCA_API_SECRET_KEY", "User")'],
            capture_output=True, text=True
        )
        self.api_secret = result.stdout.strip() if result.returncode == 0 else None

        # Fallback to process environment
        if not self.api_key:
            self.api_key = os.environ.get("APCA_API_KEY_ID")
        if not self.api_secret:
            self.api_secret = os.environ.get("APCA_API_SECRET_KEY")

        return self.api_key and self.api_secret

    def connect(self) -> bool:
        """Connect to Alpaca API."""
        if not self.load_credentials():
            self.logger.logger.error("Missing Alpaca API credentials")
            return False

        try:
            self.api = REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url='https://paper-api.alpaca.markets'
            )

            # Test connection
            account = self.api.get_account()
            self.logger.logger.info(
                f"Connected to Alpaca | Status: {account.status} | "
                f"Equity: ${float(account.equity):,.2f} | "
                f"Cash: ${float(account.cash):,.2f}"
            )
            return True

        except Exception as e:
            self.logger.logger.error(f"Failed to connect to Alpaca: {e}")
            return False

    def get_account(self):
        """Get current account information."""
        return self.api.get_account() if self.api else None

    def get_positions(self):
        """Get current positions."""
        return self.api.list_positions() if self.api else []

    def submit_order(self, symbol: str, qty: int, side: str, order_type: str = 'market'):
        """Submit order to Alpaca."""
        if not self.api:
            return None

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force='day'
            )
            return order
        except Exception as e:
            self.logger.logger.error(f"Order submission failed: {e}")
            return None

    def get_order(self, order_id: str):
        """Get order status."""
        if not self.api:
            return None

        try:
            return self.api.get_order(order_id)
        except:
            return None


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

    def __init__(self, bus: MessageBus, logger: EnhancedLogManager, metrics: MetricsCollector, alpaca: AlpacaConnection):
        self.bus = bus
        self.logger = logger
        self.metrics = metrics
        self.alpaca = alpaca

        # Risk parameters
        self.max_position_size_pct = 0.03  # 3% of equity per position
        self.max_positions = 10
        self.min_confidence = 0.3
        self.min_buying_power = 500.0

        # Subscribe to signals
        bus.subscribe('SIGNAL', self.evaluate_signal)

    async def evaluate_signal(self, signal: Dict):
        """Evaluate signal with tracking."""
        event_id = signal['id']
        correlation_id = signal['correlation_id']

        self.metrics.record_stage(event_id, 'RISK_EVALUATION_START')

        # Get current account state
        account = self.alpaca.get_account()
        positions = self.alpaca.get_positions()

        # Evaluation logic
        approved = True
        reasons = []

        # Check confidence
        if signal.get('confidence', 0) < self.min_confidence:
            approved = False
            reasons.append(f"Low confidence: {signal.get('confidence', 0):.2%}")

        # Check position limits
        if signal['type'] == 'LONG' and len(positions) >= self.max_positions:
            approved = False
            reasons.append(f"Max positions reached: {len(positions)}/{self.max_positions}")

        # Check if already have position
        has_position = any(p.symbol == signal['symbol'] for p in positions)
        if signal['type'] == 'LONG' and has_position:
            approved = False
            reasons.append("Already have position")

        # Check buying power for LONG signals
        if signal['type'] == 'LONG' and account:
            position_size = float(account.equity) * self.max_position_size_pct
            if float(account.buying_power) < position_size + self.min_buying_power:
                approved = False
                reasons.append(f"Insufficient buying power: ${float(account.buying_power):.2f}")

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
            # Forward to portfolio
            await self.bus.publish('RISK_APPROVED', signal)


class AlpacaPortfolioEngine:
    """Portfolio management with real Alpaca trading."""

    def __init__(self, bus: MessageBus, logger: EnhancedLogManager, metrics: MetricsCollector, alpaca: AlpacaConnection):
        self.bus = bus
        self.logger = logger
        self.metrics = metrics
        self.alpaca = alpaca

        self.position_size_pct = 0.03  # 3% of equity per position
        self.pending_orders = {}

        # Subscribe to approved signals
        bus.subscribe('RISK_APPROVED', self.execute_trade)

    async def execute_trade(self, signal: Dict):
        """Execute trade with real Alpaca API."""
        event_id = signal['id']
        correlation_id = signal['correlation_id']

        self.metrics.record_stage(event_id, 'ALPACA_EXECUTION_START')

        symbol = signal['symbol']

        # Get current account and positions
        account = self.alpaca.get_account()
        positions = self.alpaca.get_positions()

        if not account:
            self.logger.log_event(
                'error', event_id, correlation_id,
                "Failed to get account info"
            )
            return

        if signal['type'] == 'LONG':
            # Calculate position size
            position_value = float(account.equity) * self.position_size_pct
            shares = int(position_value / signal['price'])

            if shares > 0:
                # Submit buy order
                order = self.alpaca.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='buy',
                    order_type='market'
                )

                if order:
                    self.pending_orders[correlation_id] = order.id
                    self.metrics.order_counts['BUY'] += 1

                    self.metrics.record_stage(event_id, 'ALPACA_ORDER_SUBMITTED', {
                        'order_id': order.id,
                        'shares': shares,
                        'price': signal['price']
                    })

                    self.logger.log_event(
                        'info', event_id, correlation_id,
                        f"ORDER SUBMITTED | BUY {symbol}",
                        shares=shares,
                        estimated_value=f"${shares * signal['price']:.2f}",
                        order_id=order.id
                    )

                    # Monitor order status
                    asyncio.create_task(
                        self.monitor_order(order.id, event_id, correlation_id, symbol, 'BUY', shares)
                    )

        elif signal['type'] == 'EXIT':
            # Find position
            position = next((p for p in positions if p.symbol == symbol), None)

            if position and int(position.qty) > 0:
                # Submit sell order
                order = self.alpaca.submit_order(
                    symbol=symbol,
                    qty=int(position.qty),
                    side='sell',
                    order_type='market'
                )

                if order:
                    self.pending_orders[correlation_id] = order.id
                    self.metrics.order_counts['SELL'] += 1

                    self.metrics.record_stage(event_id, 'ALPACA_ORDER_SUBMITTED', {
                        'order_id': order.id,
                        'shares': int(position.qty),
                        'entry_price': float(position.avg_entry_price)
                    })

                    self.logger.log_event(
                        'info', event_id, correlation_id,
                        f"ORDER SUBMITTED | SELL {symbol}",
                        shares=int(position.qty),
                        entry_price=f"${float(position.avg_entry_price):.2f}",
                        order_id=order.id
                    )

                    # Monitor order status
                    asyncio.create_task(
                        self.monitor_order(order.id, event_id, correlation_id, symbol, 'SELL', int(position.qty))
                    )

    async def monitor_order(self, order_id: str, event_id: str, correlation_id: str, symbol: str, side: str, qty: int):
        """Monitor order until filled."""
        max_wait = 30  # seconds
        start_time = time.time()

        while time.time() - start_time < max_wait:
            order = self.alpaca.get_order(order_id)

            if not order:
                break

            if order.status == 'filled':
                self.metrics.record_stage(event_id, 'ALPACA_ORDER_FILLED')
                self.metrics.order_counts[f'{side}_FILLED'] += 1

                self.logger.log_event(
                    'info', event_id, correlation_id,
                    f"ORDER FILLED | {side} {symbol}",
                    shares=qty,
                    filled_price=f"${float(order.filled_avg_price or 0):.2f}",
                    value=f"${qty * float(order.filled_avg_price or 0):.2f}"
                )

                # Publish execution event
                await self.bus.publish('EXECUTION', {
                    'id': event_id,
                    'correlation_id': correlation_id,
                    'symbol': symbol,
                    'side': side,
                    'shares': qty,
                    'price': float(order.filled_avg_price or 0),
                    'order_id': order_id
                })
                break

            elif order.status in ['cancelled', 'rejected', 'expired']:
                self.metrics.record_stage(event_id, f'ALPACA_ORDER_{order.status.upper()}')
                self.metrics.order_counts[f'{side}_{order.status.upper()}'] += 1

                self.logger.log_event(
                    'warning', event_id, correlation_id,
                    f"ORDER {order.status.upper()} | {side} {symbol}",
                    order_id=order_id
                )
                break

            await asyncio.sleep(1.0)


class SystemOrchestrator:
    """Main orchestrator with real Alpaca integration."""

    def __init__(self):
        # Create session directory with version number
        self.version = "053"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = f"logs/session_{timestamp}_{self.version}"
        os.makedirs(self.session_dir, exist_ok=True)

        # Setup metrics collector
        self.metrics = MetricsCollector()

        # Setup enhanced loggers with session directory
        self.log_orchestrator = EnhancedLogManager('ORCHESTRATOR', self.metrics, self.session_dir)
        self.log_signalcore = EnhancedLogManager('SIGNALCORE', self.metrics, self.session_dir)
        self.log_riskguard = EnhancedLogManager('RISKGUARD', self.metrics, self.session_dir)
        self.log_portfolio = EnhancedLogManager('ALPACA', self.metrics, self.session_dir)
        self.log_bus = EnhancedLogManager('MESSAGEBUS', self.metrics, self.session_dir)

        # Alpaca connection
        self.alpaca = AlpacaConnection(self.log_portfolio)

        # Initialize engines
        self.signal_engine = SignalCoreEngine(self.log_signalcore, self.metrics)
        self.bus = MessageBus(self.log_bus, self.metrics)
        self.risk_engine = RiskGuardEngine(self.bus, self.log_riskguard, self.metrics, self.alpaca)
        self.portfolio_engine = AlpacaPortfolioEngine(self.bus, self.log_portfolio, self.metrics, self.alpaca)

        # Streaming
        self.stream = None
        self.start_time = time.time()
        self.bar_count = 0

        # List of symbols to monitor
        self.symbols = [
            "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "AMD",
            "META", "GOOGL", "AMZN", "NFLX", "JPM", "BAC", "XOM", "CVX"
        ]

    async def connect_market_data(self):
        """Connect to market data with error handling."""
        try:
            from ordinis.adapters.streaming.massive_stream import MassiveStreamManager
            from ordinis.adapters.streaming.stream_protocol import StreamConfig, CallbackStreamHandler

            massive_key = os.environ.get("MASSIVE_API_KEY")
            if not massive_key:
                # Try Windows User environment
                result = subprocess.run(
                    ['powershell', '-NoProfile', '-Command',
                     '[System.Environment]::GetEnvironmentVariable("MASSIVE_API_KEY", "User")'],
                    capture_output=True, text=True
                )
                massive_key = result.stdout.strip() if result.returncode == 0 else None

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
        """Periodically generate and save comprehensive metrics reports."""
        while True:
            await asyncio.sleep(60)  # Report every minute

            try:
                # Generate metrics report
                report = self.metrics.generate_report()

                runtime = time.time() - self.start_time
                runtime_mins = runtime / 60

                # Get account info
                account = self.alpaca.get_account()
                positions = self.alpaca.get_positions()

                if account:
                    # Add account info to report
                    report["account"] = {
                        "equity": float(account.equity),
                        "cash": float(account.cash),
                        "buying_power": float(account.buying_power),
                        "positions_count": len(positions)
                    }

                # Add system info to report
                report["system"] = {
                    "runtime_minutes": runtime_mins,
                    "bars_processed": self.bar_count,
                    "signals_generated": self.signal_engine.signal_count,
                    "events_published": self.bus.event_count
                }

                # Log comprehensive metrics
                self.log_orchestrator.logger.info(
                    f"METRICS | Runtime: {runtime_mins:.1f}min | "
                    f"Bars: {self.bar_count} | "
                    f"Signals: {self.signal_engine.signal_count} | "
                    f"Orders: {sum(self.metrics.order_counts.values())} | "
                    f"Equity: ${float(account.equity) if account else 0:,.2f} | "
                    f"Events: {self.bus.event_count}"
                )

                # Log order counts
                if self.metrics.order_counts:
                    order_summary = []
                    for order_type, count in self.metrics.order_counts.items():
                        order_summary.append(f"{order_type}: {count}")
                    self.log_orchestrator.logger.info(f"ORDER COUNTS | {' | '.join(order_summary)}")

                # Log latency metrics
                if report['stage_latencies_ms']:
                    self.log_orchestrator.logger.info("LATENCIES (ms):")
                    for stage_pair, metrics in report['stage_latencies_ms'].items():
                        self.log_orchestrator.logger.info(
                            f"  {stage_pair}: avg={metrics['avg']:.1f} "
                            f"min={metrics['min']:.1f} max={metrics['max']:.1f}"
                        )

                # Save detailed report to JSON file
                with open(f'{self.session_dir}/metrics_report.json', 'w') as f:
                    json.dump(report, f, indent=2, default=str)

            except Exception as e:
                self.log_orchestrator.logger.error(f"Metrics reporting error: {e}")

    async def report_status(self):
        """Periodically report account and position status."""
        while True:
            await asyncio.sleep(30)  # Every 30 seconds

            try:
                # Get account info
                account = self.alpaca.get_account()
                positions = self.alpaca.get_positions()

                if account:
                    # Log account status
                    self.log_orchestrator.logger.info(
                        f"ACCOUNT | Equity: ${float(account.equity):,.2f} | "
                        f"Cash: ${float(account.cash):,.2f} | "
                        f"Positions: {len(positions)} | "
                        f"Buying Power: ${float(account.buying_power):,.2f}"
                    )

                    # Log positions
                    if positions:
                        for pos in positions:
                            pnl = float(pos.unrealized_pl or 0)
                            pnl_pct = float(pos.unrealized_plpc or 0) * 100
                            self.log_orchestrator.logger.info(
                                f"POSITION | {pos.symbol} | "
                                f"Qty: {pos.qty} | "
                                f"Entry: ${float(pos.avg_entry_price):.2f} | "
                                f"Current: ${float(pos.current_price or 0):.2f} | "
                                f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)"
                            )

                    # Quick status line
                    positions_str = ', '.join([p.symbol for p in positions]) if positions else 'None'
                    self.log_orchestrator.logger.info(
                        f"STATUS | Bars: {self.bar_count} | "
                        f"Signals: {self.signal_engine.signal_count} | "
                        f"Positions: {positions_str}"
                    )

            except Exception as e:
                self.log_orchestrator.logger.error(f"Status reporting error: {e}")

    async def run(self):
        """Run the system with real Alpaca trading."""
        self.log_orchestrator.logger.info("="*70)
        self.log_orchestrator.logger.info("ORDINIS SYSTEM v0.53 - REAL ALPACA INTEGRATION")
        self.log_orchestrator.logger.info("="*70)
        self.log_orchestrator.logger.info("Architecture: Market -> SignalCore -> RiskGuard -> Alpaca")
        self.log_orchestrator.logger.info("Features: Real Trading | Live Account | Correlation Tracking")
        self.log_orchestrator.logger.info("="*70)

        # Connect to Alpaca
        if not self.alpaca.connect():
            self.log_orchestrator.logger.error("Failed to connect to Alpaca. Exiting.")
            return

        # Verify account
        account = self.alpaca.get_account()
        if account:
            if float(account.equity) > 50000:
                self.log_orchestrator.logger.warning(
                    f"WARNING: High equity account (${float(account.equity):,.2f}). "
                    "Please verify this is the correct account."
                )

        # Connect to market data
        await self.connect_market_data()

        # Start status reporting
        status_task = asyncio.create_task(self.report_status())

        # Start metrics reporting
        metrics_task = asyncio.create_task(self.report_metrics())

        # Main loop
        self.log_orchestrator.logger.info("System running with real Alpaca integration...")
        self.log_orchestrator.logger.info(f"Monitoring {len(self.symbols)} symbols")
        self.log_orchestrator.logger.info("Check logs/v053/ for detailed JSON logs")

        try:
            while True:
                await asyncio.sleep(10)

        except KeyboardInterrupt:
            self.log_orchestrator.logger.info("Shutting down...")
            if self.stream:
                await self.stream.disconnect()
            status_task.cancel()


async def main():
    """Main entry point."""
    orchestrator = SystemOrchestrator()
    await orchestrator.run()


if __name__ == "__main__":
    # Run
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSystem stopped by user")