#!/usr/bin/env python
"""
Live Paper Trading with Massive WebSocket Streaming.

Connects to Massive/Polygon real-time data feed and executes
ATR-Optimized RSI strategy on paper trading account.

Usage:
    python scripts/trading/live_paper_trading.py

Environment Variables:
    MASSIVE_API_KEY: Massive/Polygon API key for market data
    APCA_API_KEY_ID: Alpaca API key
    APCA_API_SECRET_KEY: Alpaca secret key
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal
import logging
import os
from pathlib import Path
import sys

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/live_paper_trading.log"),
    ],
)
logger = logging.getLogger(__name__)

# Dedicated signal logger for easy filtering
signal_logger = logging.getLogger("signals")
signal_handler = logging.FileHandler("logs/signals.log")
signal_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
signal_logger.addHandler(signal_handler)
signal_logger.setLevel(logging.INFO)


class LivePaperTradingRunner:
    """
    Live paper trading runner with real-time Massive data feed.

    Features:
    - Real-time minute bars from Massive WebSocket
    - ATR-Optimized RSI signal generation
    - Paper trading execution via Alpaca
    - Position tracking and P&L monitoring
    """

    def __init__(self, config_path: str = "configs/strategies/atr_optimized_rsi.yaml"):
        """Initialize the live trading runner."""
        self.config = self._load_config(config_path)
        self.symbols = list(self.config.get("symbols", {}).keys())

        # Data storage - bars per symbol
        self.bars: dict[str, list[dict]] = defaultdict(list)
        self.latest_quotes: dict[str, StreamQuote] = {}

        # Position tracking
        self.positions: dict[str, dict] = {}
        self.daily_pnl = Decimal("0")
        self.total_trades = 0
        self.winning_trades = 0

        # Order throttling - prevent rapid-fire orders
        self._last_order_time: dict[str, datetime] = {}
        self._order_cooldown_seconds = 60  # Minimum seconds between orders per symbol
        self._global_order_cooldown = 5  # Minimum seconds between any orders
        self._last_global_order: datetime | None = None

        # Risk parameters
        risk_config = self.config.get("risk_management", {})
        self.max_position_pct = risk_config.get("max_position_size_pct", 3.0)
        self.max_daily_loss = risk_config.get("max_daily_loss_pct", 2.0)
        self.max_positions = risk_config.get("max_concurrent_positions", 10)

        # Components (initialized in start())
        self.broker: AlpacaBroker | None = None
        self.stream: MassiveStreamManager | None = None
        self._running = False

    def _load_config(self, path: str) -> dict:
        """Load strategy configuration."""
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        logger.warning("Config not found: %s, using defaults", path)
        return {}

    async def start(self) -> None:
        """Start the live trading system."""
        logger.info("=" * 60)
        logger.info("LIVE PAPER TRADING RUNNER")
        logger.info("=" * 60)

        # Initialize broker
        self.broker = AlpacaBroker(paper=True)
        if not await self.broker.connect():
            logger.error("Failed to connect to Alpaca broker")
            return

        account = await self.broker.get_account()
        logger.info("Connected to Alpaca - Equity: $%.2f", account.equity)

        # Initialize Massive stream
        massive_key = os.environ.get("MASSIVE_API_KEY", "")
        if not massive_key:
            logger.error("MASSIVE_API_KEY not set")
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

        logger.info("Subscribed to %d symbols", len(self.symbols))
        logger.info(
            "Symbols: %s", ", ".join(self.symbols[:10]) + ("..." if len(self.symbols) > 10 else "")
        )

        self._running = True

        # Main loop
        try:
            while self._running:
                await asyncio.sleep(1)

                # Periodic status update
                if datetime.now().second == 0:
                    await self._print_status()

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop the trading system."""
        self._running = False

        if self.stream:
            await self.stream.disconnect()

        logger.info("Live trading stopped")
        await self._print_status()

    async def _on_bar(self, bar: StreamBar) -> None:
        """Handle incoming minute bar."""
        symbol = bar.symbol

        # Log bar reception
        logger.info(
            "BAR | %s | O=%.2f H=%.2f L=%.2f C=%.2f V=%d",
            symbol,
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
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

        # Check for trading signals
        if len(self.bars[symbol]) >= 20:
            await self._check_signal(symbol)

    async def _on_quote(self, quote: StreamQuote) -> None:
        """Handle incoming quote."""
        self.latest_quotes[quote.symbol] = quote

    async def _on_status(self, status: StreamStatus, message: str) -> None:
        """Handle stream status change."""
        logger.info("Stream status: %s - %s", status.name, message)

    async def _on_error(self, error: Exception) -> None:
        """Handle stream error."""
        logger.error("Stream error: %s", error)

    async def _check_signal(self, symbol: str) -> None:
        """Check for trading signals on a symbol."""
        if not self.broker:
            return

        # Build DataFrame from bars
        df = pd.DataFrame(self.bars[symbol])
        if len(df) < 20:
            return

        # Calculate indicators
        close = df["close"]
        high = df["high"]
        low = df["low"]

        rsi = TechnicalIndicators.rsi(close, 14)
        tr = pd.concat(
            [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()

        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]

        # Get symbol config
        symbol_config = self.config.get("symbols", {}).get(symbol, {})
        rsi_oversold = symbol_config.get("rsi_oversold", 35)
        rsi_exit = symbol_config.get("rsi_exit", 50)

        # Generate signal
        signal = None
        signal_reason = None

        if symbol in self.positions:
            # Check for exit
            pos = self.positions[symbol]
            if pos["direction"] == "long" and current_rsi > rsi_exit:
                signal = "exit_long"
                signal_reason = f"RSI exit ({current_rsi:.1f} > {rsi_exit})"
            # Check stop loss
            elif pos["direction"] == "long" and current_price <= pos["stop_loss"]:
                signal = "exit_long"
                signal_reason = f"Stop loss hit (${current_price:.2f} <= ${pos['stop_loss']:.2f})"
            # Check take profit
            elif pos["direction"] == "long" and current_price >= pos["take_profit"]:
                signal = "exit_long"
                signal_reason = (
                    f"Take profit hit (${current_price:.2f} >= ${pos['take_profit']:.2f})"
                )
        elif current_rsi < rsi_oversold:
            if len(self.positions) < self.max_positions:
                signal = "long"
                signal_reason = f"RSI oversold ({current_rsi:.1f} < {rsi_oversold})"
            else:
                # Log skipped signal due to max positions
                logger.info(
                    "SIGNAL SKIPPED | %s | RSI=%.1f | Reason: Max positions (%d/%d)",
                    symbol,
                    current_rsi,
                    len(self.positions),
                    self.max_positions,
                )

        # Log all generated signals
        if signal:
            log_msg = f"SIGNAL | {symbol} | {signal.upper()} | Price=${current_price:.2f} | RSI={current_rsi:.1f} | ATR={current_atr:.2f} | {signal_reason}"
            logger.info(log_msg)
            signal_logger.info(log_msg)
            await self._execute_signal(symbol, signal, current_price, current_atr)

    async def _execute_signal(self, symbol: str, signal: str, price: float, atr: float) -> None:
        """Execute a trading signal."""
        if not self.broker:
            return

        now = datetime.now(UTC)

        # Check global order cooldown
        if self._last_global_order:
            elapsed = (now - self._last_global_order).total_seconds()
            if elapsed < self._global_order_cooldown:
                logger.debug(
                    "Global cooldown: %.1fs remaining", self._global_order_cooldown - elapsed
                )
                return

        # Check per-symbol cooldown
        if symbol in self._last_order_time:
            elapsed = (now - self._last_order_time[symbol]).total_seconds()
            if elapsed < self._order_cooldown_seconds:
                logger.debug(
                    "Symbol %s cooldown: %.1fs remaining",
                    symbol,
                    self._order_cooldown_seconds - elapsed,
                )
                return

        try:
            account = await self.broker.get_account()
            symbol_config = self.config.get("symbols", {}).get(symbol, {})

            if signal == "long":
                # Calculate position size based on max position percentage of equity
                equity = float(account.equity)
                max_position_value = equity * (self.max_position_pct / 100)

                # Calculate quantity
                quantity = int(max_position_value / price)

                # Minimum 1 share if we can afford it
                if quantity <= 0 and price < max_position_value:
                    quantity = 1

                if quantity <= 0:
                    logger.warning(
                        "Skipping %s: price $%.2f exceeds max position $%.2f",
                        symbol,
                        price,
                        max_position_value,
                    )
                    return

                # Cap at reasonable size
                order_value = quantity * price
                if order_value > equity * 0.1:  # Max 10% of equity per position
                    quantity = int((equity * 0.1) / price)
                    if quantity <= 0:
                        logger.warning(
                            "Skipping %s: price $%.2f too high for 10%% equity cap ($%.2f)",
                            symbol,
                            price,
                            equity * 0.1,
                        )
                        return

                # Submit order
                order = await self.broker.submit_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=Decimal(str(quantity)),
                    order_type=OrderType.MARKET,
                )

                # Track position with ATR-based stops
                # Scale ATR for minute bars (multiply by sqrt of minutes in trading day)
                # ~390 minutes, sqrt(390) ≈ 20
                scaled_atr = atr * 20  # Approximate daily ATR from minute ATR
                atr_stop = symbol_config.get("atr_stop_mult", 1.5)
                atr_tp = symbol_config.get("atr_tp_mult", 2.0)

                self.positions[symbol] = {
                    "direction": "long",
                    "entry_price": price,
                    "quantity": quantity,
                    "stop_loss": price - (scaled_atr * atr_stop),
                    "take_profit": price + (scaled_atr * atr_tp),
                    "entry_time": datetime.now(UTC),
                    "order_id": order.id if order else None,
                }

                # Update order timestamps
                self._last_order_time[symbol] = now
                self._last_global_order = now

                logger.info(
                    "LONG %s: %d shares @ $%.2f (SL: $%.2f, TP: $%.2f)",
                    symbol,
                    quantity,
                    price,
                    self.positions[symbol]["stop_loss"],
                    self.positions[symbol]["take_profit"],
                )

            elif signal == "exit_long" and symbol in self.positions:
                pos = self.positions[symbol]
                quantity = pos["quantity"]

                # Submit exit order
                await self.broker.submit_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=Decimal(str(quantity)),
                    order_type=OrderType.MARKET,
                )

                # Calculate P&L
                pnl_pct = ((price - pos["entry_price"]) / pos["entry_price"]) * 100
                pnl_abs = (price - pos["entry_price"]) * quantity

                self.total_trades += 1
                if pnl_pct > 0:
                    self.winning_trades += 1
                self.daily_pnl += Decimal(str(pnl_abs))

                # Update order timestamps
                self._last_order_time[symbol] = now
                self._last_global_order = now

                logger.info(
                    "EXIT %s: %d shares @ $%.2f | P&L: %+.2f%% ($%+.2f)",
                    symbol,
                    quantity,
                    price,
                    pnl_pct,
                    pnl_abs,
                )

                del self.positions[symbol]

        except Exception as e:
            logger.error("Error executing %s for %s: %s", signal, symbol, e)

    async def _print_status(self) -> None:
        """Print current status."""
        if not self.broker:
            return

        account = await self.broker.get_account()
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0

        logger.info("-" * 50)
        logger.info(
            "Status: Equity=$%.2f | Positions=%d | Trades=%d | WinRate=%.1f%%",
            account.equity,
            len(self.positions),
            self.total_trades,
            win_rate,
        )
        if self.positions:
            logger.info("Open: %s", ", ".join(self.positions.keys()))


async def main() -> None:
    """Main entry point."""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    runner = LivePaperTradingRunner()
    await runner.start()


if __name__ == "__main__":
    asyncio.run(main())
