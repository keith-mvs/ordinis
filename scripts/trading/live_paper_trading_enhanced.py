#!/usr/bin/env python
"""
Enhanced Live Paper Trading with Advanced Logging and Position Management.

Features:
- Full position management for existing holdings
- Advanced multi-file logging system
- Real-time P&L tracking per position
- Position health monitoring
- Trade execution logging
- Performance metrics tracking

Usage:
    python scripts/trading/live_paper_trading_enhanced.py

Environment Variables:
    MASSIVE_API_KEY: Massive/Polygon API key for market data
    APCA_API_KEY_ID: Alpaca API key
    APCA_API_SECRET_KEY: Alpaca secret key
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ordinis.adapters.broker import AlpacaBroker, OrderSide, OrderType
from ordinis.adapters.streaming.massive_stream import MassiveStreamManager
from ordinis.adapters.streaming.stream_protocol import (
    CallbackStreamHandler,
    StreamBar,
    StreamConfig,
    StreamQuote,
    StreamStatus,
)
from ordinis.engines.signalcore.features.technical import TechnicalIndicators


# ============================================================================
# ADVANCED LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Set up comprehensive logging system with multiple specialized loggers."""

    # Create logs directory structure
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Archive previous session logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = log_dir / f"session_{timestamp}"
    archive_dir.mkdir(exist_ok=True)

    # Main logger - general activity
    main_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "live_trading_main.log"),
            logging.FileHandler(archive_dir / "main.log"),
        ],
    )

    # 1. TRADE LOGGER - All trade execution details
    trade_logger = logging.getLogger("trades")
    trade_handler = logging.FileHandler(log_dir / "trades.log")
    trade_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s"
    ))
    trade_logger.addHandler(trade_handler)
    trade_logger.addHandler(logging.FileHandler(archive_dir / "trades.log"))
    trade_logger.setLevel(logging.INFO)

    # 2. SIGNAL LOGGER - All generated signals
    signal_logger = logging.getLogger("signals")
    signal_handler = logging.FileHandler(log_dir / "signals.log")
    signal_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s"
    ))
    signal_logger.addHandler(signal_handler)
    signal_logger.addHandler(logging.FileHandler(archive_dir / "signals.log"))
    signal_logger.setLevel(logging.INFO)

    # 3. POSITION LOGGER - Position status and P&L
    position_logger = logging.getLogger("positions")
    position_handler = logging.FileHandler(log_dir / "positions.log")
    position_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s"
    ))
    position_logger.addHandler(position_handler)
    position_logger.addHandler(logging.FileHandler(archive_dir / "positions.log"))
    position_logger.setLevel(logging.INFO)

    # 4. PERFORMANCE LOGGER - Metrics and statistics
    perf_logger = logging.getLogger("performance")
    perf_handler = logging.FileHandler(log_dir / "performance.log")
    perf_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s"
    ))
    perf_logger.addHandler(perf_handler)
    perf_logger.addHandler(logging.FileHandler(archive_dir / "performance.log"))
    perf_logger.setLevel(logging.INFO)

    # 5. DATA LOGGER - Market data flow
    data_logger = logging.getLogger("market_data")
    data_handler = logging.FileHandler(log_dir / "market_data.log")
    data_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s"
    ))
    data_logger.addHandler(data_handler)
    data_logger.addHandler(logging.FileHandler(archive_dir / "market_data.log"))
    data_logger.setLevel(logging.INFO)

    # 6. ERROR LOGGER - All errors and exceptions
    error_logger = logging.getLogger("errors")
    error_handler = logging.FileHandler(log_dir / "errors.log")
    error_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s\n%(exc_info)s"
    ))
    error_logger.addHandler(error_handler)
    error_logger.addHandler(logging.FileHandler(archive_dir / "errors.log"))
    error_logger.setLevel(logging.ERROR)

    # 7. AUDIT LOGGER - Detailed execution audit trail
    audit_logger = logging.getLogger("audit")
    audit_handler = logging.FileHandler(log_dir / "audit.log")
    audit_handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    audit_logger.addHandler(audit_handler)
    audit_logger.addHandler(logging.FileHandler(archive_dir / "audit.log"))
    audit_logger.setLevel(logging.DEBUG)

    return {
        "main": logging.getLogger(__name__),
        "trades": trade_logger,
        "signals": signal_logger,
        "positions": position_logger,
        "performance": perf_logger,
        "market_data": data_logger,
        "errors": error_logger,
        "audit": audit_logger,
        "archive_dir": archive_dir,
    }


# ============================================================================
# POSITION MANAGER CLASS
# ============================================================================

class PositionManager:
    """Manages and tracks all open positions with detailed analytics."""

    def __init__(self, loggers: dict):
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.closed_positions: List[Dict[str, Any]] = []
        self.loggers = loggers
        self.entry_prices: Dict[str, float] = {}

    def add_position(self, symbol: str, quantity: int, entry_price: float,
                    stop_loss: float, take_profit: float, reason: str) -> None:
        """Add a new position to tracking."""
        self.positions[symbol] = {
            "symbol": symbol,
            "quantity": quantity,
            "entry_price": entry_price,
            "entry_time": datetime.now(UTC),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reason": reason,
            "current_price": entry_price,
            "unrealized_pnl": 0.0,
            "unrealized_pnl_pct": 0.0,
            "high_water_mark": entry_price,
            "max_drawdown": 0.0,
            "bars_held": 0,
        }
        self.entry_prices[symbol] = entry_price

        self.loggers["positions"].info(
            f"NEW POSITION | {symbol} | Qty: {quantity} | Entry: ${entry_price:.2f} | "
            f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | Reason: {reason}"
        )

    def update_position(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Update position with current price and calculate metrics."""
        if symbol not in self.positions:
            return {}

        pos = self.positions[symbol]
        pos["current_price"] = current_price
        pos["bars_held"] += 1

        # Calculate P&L
        entry = pos["entry_price"]
        qty = pos["quantity"]
        pos["unrealized_pnl"] = (current_price - entry) * qty
        pos["unrealized_pnl_pct"] = ((current_price - entry) / entry) * 100

        # Track high water mark and drawdown
        if current_price > pos["high_water_mark"]:
            pos["high_water_mark"] = current_price
        pos["max_drawdown"] = ((pos["high_water_mark"] - current_price) / pos["high_water_mark"]) * 100

        # Check for exit conditions
        exit_signal = None
        if current_price <= pos["stop_loss"]:
            exit_signal = "STOP_LOSS"
        elif current_price >= pos["take_profit"]:
            exit_signal = "TAKE_PROFIT"
        elif pos["max_drawdown"] > 5.0:  # Trailing stop at 5% from high
            exit_signal = "TRAILING_STOP"

        if exit_signal:
            self.loggers["signals"].info(
                f"EXIT SIGNAL | {symbol} | Type: {exit_signal} | "
                f"Current: ${current_price:.2f} | P&L: ${pos['unrealized_pnl']:.2f} ({pos['unrealized_pnl_pct']:.1f}%)"
            )

        return {"position": pos, "exit_signal": exit_signal}

    def close_position(self, symbol: str, exit_price: float, exit_reason: str) -> Dict[str, Any]:
        """Close a position and record final metrics."""
        if symbol not in self.positions:
            return {}

        pos = self.positions[symbol]
        pos["exit_price"] = exit_price
        pos["exit_time"] = datetime.now(UTC)
        pos["exit_reason"] = exit_reason

        # Final P&L
        pos["realized_pnl"] = (exit_price - pos["entry_price"]) * pos["quantity"]
        pos["realized_pnl_pct"] = ((exit_price - pos["entry_price"]) / pos["entry_price"]) * 100
        pos["hold_duration"] = (pos["exit_time"] - pos["entry_time"]).total_seconds() / 60  # minutes

        self.closed_positions.append(pos)
        del self.positions[symbol]

        self.loggers["trades"].info(
            f"CLOSED | {symbol} | Exit: ${exit_price:.2f} | "
            f"P&L: ${pos['realized_pnl']:.2f} ({pos['realized_pnl_pct']:.1f}%) | "
            f"Duration: {pos['hold_duration']:.1f} min | Reason: {exit_reason}"
        )

        return pos

    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate overall portfolio metrics."""
        total_unrealized = sum(p["unrealized_pnl"] for p in self.positions.values())
        total_realized = sum(p["realized_pnl"] for p in self.closed_positions)

        wins = [p for p in self.closed_positions if p["realized_pnl"] > 0]
        losses = [p for p in self.closed_positions if p["realized_pnl"] <= 0]

        win_rate = (len(wins) / len(self.closed_positions) * 100) if self.closed_positions else 0

        avg_win = sum(p["realized_pnl"] for p in wins) / len(wins) if wins else 0
        avg_loss = sum(p["realized_pnl"] for p in losses) / len(losses) if losses else 0

        profit_factor = abs(sum(p["realized_pnl"] for p in wins) / sum(p["realized_pnl"] for p in losses)) if losses else 0

        return {
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed_positions),
            "total_unrealized_pnl": total_unrealized,
            "total_realized_pnl": total_realized,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "winning_trades": len(wins),
            "losing_trades": len(losses),
        }


# ============================================================================
# ENHANCED LIVE TRADING RUNNER
# ============================================================================

class EnhancedLivePaperTradingRunner:
    """Enhanced live paper trading with comprehensive position management."""

    def __init__(self, config_path: str = "configs/strategies/atr_optimized_rsi.yaml"):
        """Initialize the enhanced trading runner."""
        self.loggers = setup_logging()
        self.logger = self.loggers["main"]

        self.logger.info("=" * 80)
        self.logger.info("ENHANCED LIVE PAPER TRADING SYSTEM")
        self.logger.info("=" * 80)

        self.config = self._load_config(config_path)

        # Get existing positions symbols
        self.existing_symbols = ["MP", "RUN", "PATH", "UPST", "BAC", "FSLR", "AMD"]

        # Combine with config symbols
        config_symbols = list(self.config.get("symbols", {}).keys())
        self.symbols = list(set(self.existing_symbols + config_symbols))

        self.logger.info(f"Monitoring {len(self.symbols)} symbols (including {len(self.existing_symbols)} existing positions)")

        # Data storage
        self.bars: Dict[str, List[Dict]] = defaultdict(list)
        self.latest_quotes: Dict[str, StreamQuote] = {}

        # Position management
        self.position_manager = PositionManager(self.loggers)
        self.existing_positions: Dict[str, Dict] = {}

        # Order management
        self._last_order_time: Dict[str, datetime] = {}
        self._order_cooldown_seconds = 60
        self._last_global_order: Optional[datetime] = None
        self._global_order_cooldown = 5

        # Risk parameters
        risk_config = self.config.get("risk_management", {})
        self.max_position_pct = risk_config.get("max_position_size_pct", 3.0)
        self.max_daily_loss = risk_config.get("max_daily_loss_pct", 2.0)
        self.max_positions = risk_config.get("max_concurrent_positions", 10)

        # Performance tracking
        self.session_start = datetime.now(UTC)
        self.bars_processed = 0
        self.signals_generated = 0
        self.orders_submitted = 0

        # Components
        self.broker: Optional[AlpacaBroker] = None
        self.stream: Optional[MassiveStreamManager] = None
        self._running = False

    def _load_config(self, path: str) -> dict:
        """Load strategy configuration."""
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        self.logger.warning(f"Config not found: {path}, using defaults")
        return {}

    async def _load_existing_positions(self) -> None:
        """Load and track existing positions from broker."""
        if not self.broker:
            return

        positions = await self.broker.get_positions()

        for pos in positions:
            symbol = pos.symbol
            if symbol in self.existing_symbols:
                # Get current price
                current_price = float(pos.current_price) if hasattr(pos, 'current_price') else float(pos.market_value) / float(pos.quantity)

                # Estimate entry price (avg cost basis if available)
                entry_price = float(pos.avg_entry_price) if hasattr(pos, 'avg_entry_price') else current_price

                # Calculate ATR-based stops (estimated)
                atr_estimate = current_price * 0.02  # Estimate 2% ATR
                stop_loss = entry_price - (atr_estimate * 1.5)
                take_profit = entry_price + (atr_estimate * 2.0)

                self.position_manager.add_position(
                    symbol=symbol,
                    quantity=int(pos.quantity),
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason="EXISTING_POSITION"
                )

                self.existing_positions[symbol] = {
                    "quantity": int(pos.quantity),
                    "market_value": float(pos.market_value),
                    "entry_price": entry_price,
                    "current_price": current_price,
                }

        self.loggers["positions"].info(f"Loaded {len(self.existing_positions)} existing positions")
        for symbol, pos in self.existing_positions.items():
            self.loggers["positions"].info(
                f"  {symbol}: {pos['quantity']} shares @ ${pos['entry_price']:.2f} (current: ${pos['current_price']:.2f})"
            )

    async def start(self) -> None:
        """Start the enhanced trading system."""
        try:
            # Initialize broker
            self.broker = AlpacaBroker(paper=True)
            if not await self.broker.connect():
                self.logger.error("Failed to connect to Alpaca broker")
                return

            account = await self.broker.get_account()
            self.logger.info(f"Connected to Alpaca - Equity: ${account.equity:,.2f} | Cash: ${account.cash:,.2f}")

            # Load existing positions
            await self._load_existing_positions()

            # Initialize Massive stream
            massive_key = os.environ.get("MASSIVE_API_KEY", "")
            if not massive_key:
                self.logger.error("MASSIVE_API_KEY not set")
                return

            stream_config = StreamConfig(
                api_key=massive_key,
                reconnect_enabled=True,
                reconnect_delay_seconds=1.0,
                max_reconnect_attempts=10,
            )

            self.stream = MassiveStreamManager(stream_config)

            # Set up event handlers
            handler = CallbackStreamHandler(
                on_bar_callback=self._on_bar,
                on_quote_callback=self._on_quote,
                on_status_callback=self._on_status,
                on_error_callback=self._on_error,
            )
            self.stream.add_handler(handler)

            # Connect and subscribe
            await self.stream.connect()
            await self.stream.subscribe(self.symbols)

            self.logger.info(f"Subscribed to {len(self.symbols)} symbols for real-time data")
            self._running = True

            # Performance monitoring task
            asyncio.create_task(self._performance_monitor())

            # Main loop
            while self._running:
                await asyncio.sleep(1)

                # Periodic status update every minute
                if datetime.now().second == 0:
                    await self._print_status()

        except KeyboardInterrupt:
            self.logger.info("Shutdown signal received")
        except Exception as e:
            self.loggers["errors"].error(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the trading system and generate final reports."""
        self._running = False

        if self.stream:
            await self.stream.disconnect()

        # Generate final report
        self._generate_session_report()

        self.logger.info("Enhanced live trading stopped")

    async def _on_bar(self, bar: StreamBar) -> None:
        """Handle incoming minute bar with comprehensive logging."""
        self.bars_processed += 1
        symbol = bar.symbol

        # Log to market data logger (only for tracked symbols)
        if symbol in self.symbols:
            self.loggers["market_data"].info(
                f"BAR | {symbol} | O={bar.open:.2f} H={bar.high:.2f} L={bar.low:.2f} C={bar.close:.2f} V={bar.volume}"
            )

        # Store bar data
        bar_data = {
            "timestamp": bar.timestamp,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        self.bars[symbol].append(bar_data)

        # Keep last 100 bars
        if len(self.bars[symbol]) > 100:
            self.bars[symbol] = self.bars[symbol][-100:]

        # Update existing positions
        if symbol in self.position_manager.positions:
            result = self.position_manager.update_position(symbol, bar.close)
            if result.get("exit_signal"):
                await self._execute_exit(symbol, result["exit_signal"], bar.close)

        # Check for new signals if we have enough data
        if len(self.bars[symbol]) >= 20:
            await self._check_signal(symbol)

    async def _on_quote(self, quote: StreamQuote) -> None:
        """Handle incoming quote."""
        self.latest_quotes[quote.symbol] = quote

    async def _on_status(self, status: StreamStatus, message: str) -> None:
        """Handle stream status change."""
        self.loggers["audit"].info(f"Stream status: {status.name} - {message}")

    async def _on_error(self, error: Exception) -> None:
        """Handle stream error."""
        self.loggers["errors"].error(f"Stream error: {error}", exc_info=True)

    async def _check_signal(self, symbol: str) -> None:
        """Check for trading signals with comprehensive analysis."""
        if not self.broker:
            return

        # Build DataFrame
        df = pd.DataFrame(self.bars[symbol])
        if len(df) < 20:
            return

        # Calculate indicators
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Technical indicators
        rsi = TechnicalIndicators.rsi(close, 14)
        tr = pd.concat(
            [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()

        # Volume analysis
        avg_volume = volume.rolling(20).mean()
        volume_ratio = volume.iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1

        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]

        # Get symbol config
        symbol_config = self.config.get("symbols", {}).get(symbol, {})
        rsi_oversold = symbol_config.get("rsi_oversold", 30)
        rsi_exit = symbol_config.get("rsi_exit", 50)
        rsi_overbought = symbol_config.get("rsi_overbought", 70)

        # Generate signal with multi-factor analysis
        signal = None
        signal_reason = None
        signal_strength = 0

        if symbol in self.position_manager.positions:
            # Check exit conditions for existing position
            pos = self.position_manager.positions[symbol]

            if current_rsi > rsi_exit:
                signal = "exit_long"
                signal_reason = f"RSI exit ({current_rsi:.1f} > {rsi_exit})"
                signal_strength = min((current_rsi - rsi_exit) / 10, 1.0)

            elif current_rsi > rsi_overbought:
                signal = "exit_long"
                signal_reason = f"RSI overbought ({current_rsi:.1f} > {rsi_overbought})"
                signal_strength = 1.0

        elif symbol not in self.position_manager.positions:
            # Check entry conditions
            if current_rsi < rsi_oversold:
                # Additional filters
                if volume_ratio > 1.5:  # Volume spike
                    signal_strength += 0.3

                if len(self.position_manager.positions) < self.max_positions:
                    signal = "long"
                    signal_reason = f"RSI oversold ({current_rsi:.1f} < {rsi_oversold}) + Vol ratio {volume_ratio:.1f}"
                    signal_strength = min(1.0 - (current_rsi / rsi_oversold), 1.0) + (signal_strength * 0.3)
                else:
                    self.loggers["signals"].info(
                        f"SIGNAL SKIPPED | {symbol} | RSI={current_rsi:.1f} | Max positions reached ({self.max_positions})"
                    )

        # Log and execute signals
        if signal:
            self.signals_generated += 1

            self.loggers["signals"].info(
                f"SIGNAL | {symbol} | {signal.upper()} | Price=${current_price:.2f} | "
                f"RSI={current_rsi:.1f} | ATR={current_atr:.2f} | Strength={signal_strength:.2f} | {signal_reason}"
            )

            self.loggers["audit"].debug(
                f"Signal Analysis | {symbol} | Signal: {signal} | "
                f"RSI: {current_rsi:.2f} | ATR: {current_atr:.2f} | "
                f"Volume Ratio: {volume_ratio:.2f} | Bars: {len(df)}"
            )

            await self._execute_signal(symbol, signal, current_price, current_atr, signal_strength)

    async def _execute_signal(self, symbol: str, signal: str, price: float, atr: float, strength: float) -> None:
        """Execute trading signal with comprehensive order management."""
        if not self.broker:
            return

        now = datetime.now(UTC)

        # Check cooldowns
        if self._last_global_order:
            elapsed = (now - self._last_global_order).total_seconds()
            if elapsed < self._global_order_cooldown:
                self.loggers["audit"].debug(
                    f"Global cooldown active: {self._global_order_cooldown - elapsed:.1f}s remaining"
                )
                return

        if symbol in self._last_order_time:
            elapsed = (now - self._last_order_time[symbol]).total_seconds()
            if elapsed < self._order_cooldown_seconds:
                self.loggers["audit"].debug(
                    f"Symbol {symbol} cooldown: {self._order_cooldown_seconds - elapsed:.1f}s remaining"
                )
                return

        try:
            account = await self.broker.get_account()
            symbol_config = self.config.get("symbols", {}).get(symbol, {})

            if signal == "long":
                # Position sizing based on signal strength and account equity
                equity = float(account.equity)
                base_position_value = equity * (self.max_position_pct / 100)

                # Adjust for signal strength
                position_value = base_position_value * min(strength + 0.5, 1.0)
                quantity = int(position_value / price)

                if quantity <= 0:
                    self.loggers["trades"].warning(
                        f"Skipping {symbol}: Insufficient capital for position"
                    )
                    return

                # Submit order
                order = await self.broker.submit_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=Decimal(str(quantity)),
                    order_type=OrderType.MARKET,
                )

                self.orders_submitted += 1

                # Calculate stops
                scaled_atr = atr * 20  # Scale for minute bars
                atr_stop = symbol_config.get("atr_stop_mult", 1.5)
                atr_tp = symbol_config.get("atr_tp_mult", 2.0)

                stop_loss = price - (scaled_atr * atr_stop)
                take_profit = price + (scaled_atr * atr_tp)

                # Track position
                self.position_manager.add_position(
                    symbol=symbol,
                    quantity=quantity,
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reason=f"RSI_OVERSOLD_STRENGTH_{strength:.2f}"
                )

                # Update cooldowns
                self._last_order_time[symbol] = now
                self._last_global_order = now

                self.loggers["trades"].info(
                    f"BUY ORDER | {symbol} | Qty: {quantity} | Price: ${price:.2f} | "
                    f"Value: ${position_value:.2f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | "
                    f"Order ID: {order.id if order else 'N/A'}"
                )

            elif signal.startswith("exit") and symbol in self.position_manager.positions:
                await self._execute_exit(symbol, signal.upper(), price)

        except Exception as e:
            self.loggers["errors"].error(
                f"Error executing {signal} for {symbol}: {e}", exc_info=True
            )

    async def _execute_exit(self, symbol: str, exit_reason: str, price: float) -> None:
        """Execute position exit."""
        if symbol not in self.position_manager.positions:
            return

        pos = self.position_manager.positions[symbol]
        quantity = pos["quantity"]

        try:
            # Submit exit order
            order = await self.broker.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=Decimal(str(quantity)),
                order_type=OrderType.MARKET,
            )

            self.orders_submitted += 1

            # Close position tracking
            closed = self.position_manager.close_position(symbol, price, exit_reason)

            # Update cooldowns
            now = datetime.now(UTC)
            self._last_order_time[symbol] = now
            self._last_global_order = now

            self.loggers["trades"].info(
                f"SELL ORDER | {symbol} | Qty: {quantity} | Price: ${price:.2f} | "
                f"P&L: ${closed['realized_pnl']:.2f} ({closed['realized_pnl_pct']:.1f}%) | "
                f"Reason: {exit_reason} | Order ID: {order.id if order else 'N/A'}"
            )

        except Exception as e:
            self.loggers["errors"].error(
                f"Error executing exit for {symbol}: {e}", exc_info=True
            )

    async def _performance_monitor(self) -> None:
        """Background task to monitor and log performance metrics."""
        while self._running:
            await asyncio.sleep(60)  # Every minute

            metrics = self.position_manager.get_portfolio_metrics()

            # Add session metrics
            session_duration = (datetime.now(UTC) - self.session_start).total_seconds() / 60
            metrics["session_duration_min"] = session_duration
            metrics["bars_processed"] = self.bars_processed
            metrics["signals_generated"] = self.signals_generated
            metrics["orders_submitted"] = self.orders_submitted
            metrics["bars_per_minute"] = self.bars_processed / session_duration if session_duration > 0 else 0

            self.loggers["performance"].info(
                f"METRICS | Open: {metrics['open_positions']} | Closed: {metrics['closed_positions']} | "
                f"Unrealized: ${metrics['total_unrealized_pnl']:.2f} | Realized: ${metrics['total_realized_pnl']:.2f} | "
                f"Win Rate: {metrics['win_rate']:.1f}% | PF: {metrics['profit_factor']:.2f} | "
                f"Signals: {self.signals_generated} | Orders: {self.orders_submitted}"
            )

    async def _print_status(self) -> None:
        """Print comprehensive status update."""
        if not self.broker:
            return

        account = await self.broker.get_account()
        metrics = self.position_manager.get_portfolio_metrics()

        self.logger.info("-" * 70)
        self.logger.info(
            f"STATUS | Equity: ${account.equity:,.2f} | Cash: ${account.cash:,.2f} | "
            f"Positions: {metrics['open_positions']} | "
            f"P&L: ${metrics['total_unrealized_pnl']:.2f} (U) ${metrics['total_realized_pnl']:.2f} (R)"
        )

        if self.position_manager.positions:
            positions_str = ", ".join([
                f"{sym}({p['unrealized_pnl_pct']:.1f}%)"
                for sym, p in self.position_manager.positions.items()
            ])
            self.logger.info(f"Open: {positions_str}")

    def _generate_session_report(self) -> None:
        """Generate comprehensive session report."""
        metrics = self.position_manager.get_portfolio_metrics()
        session_duration = (datetime.now(UTC) - self.session_start).total_seconds() / 60

        report = f"""
{'=' * 80}
SESSION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}

Session Duration: {session_duration:.1f} minutes
Archive Directory: {self.loggers['archive_dir']}

ACTIVITY SUMMARY:
  Bars Processed: {self.bars_processed:,}
  Signals Generated: {self.signals_generated}
  Orders Submitted: {self.orders_submitted}

POSITION SUMMARY:
  Open Positions: {metrics['open_positions']}
  Closed Positions: {metrics['closed_positions']}

PERFORMANCE METRICS:
  Total Realized P&L: ${metrics['total_realized_pnl']:.2f}
  Total Unrealized P&L: ${metrics['total_unrealized_pnl']:.2f}
  Win Rate: {metrics['win_rate']:.1f}%
  Winning Trades: {metrics['winning_trades']}
  Losing Trades: {metrics['losing_trades']}
  Average Win: ${metrics['avg_win']:.2f}
  Average Loss: ${metrics['avg_loss']:.2f}
  Profit Factor: {metrics['profit_factor']:.2f}

LOG FILES:
  Main: logs/live_trading_main.log
  Trades: logs/trades.log
  Signals: logs/signals.log
  Positions: logs/positions.log
  Performance: logs/performance.log
  Market Data: logs/market_data.log
  Errors: logs/errors.log
  Audit: logs/audit.log

Archived to: {self.loggers['archive_dir']}
{'=' * 80}
"""

        print(report)

        # Save report to file
        report_path = self.loggers["archive_dir"] / "session_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        # Also save as JSON for programmatic access
        report_data = {
            "session_start": self.session_start.isoformat(),
            "session_duration_minutes": session_duration,
            "activity": {
                "bars_processed": self.bars_processed,
                "signals_generated": self.signals_generated,
                "orders_submitted": self.orders_submitted,
            },
            "performance": metrics,
            "archive_dir": str(self.loggers["archive_dir"]),
        }

        json_path = self.loggers["archive_dir"] / "session_report.json"
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)


async def main() -> None:
    """Main entry point."""
    runner = EnhancedLivePaperTradingRunner()
    await runner.start()


if __name__ == "__main__":
    asyncio.run(main())