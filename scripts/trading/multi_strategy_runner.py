#!/usr/bin/env python
"""
Multi-Strategy Live Paper Trading Runner.

Orchestrates multiple trading strategies with:
- Broker position sync on startup
- Volatility-adjusted position sizing (1% base)
- Capital allocation per strategy
- Symbol exclusion across strategies
- Strategy-level P&L tracking

Usage:
    python scripts/trading/multi_strategy_runner.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum, auto
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
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
from ordinis.engines.portfolio.sizing import (
    PositionSizer,
    SizingConfig,
    SizingMethod,
)
from ordinis.engines.signalcore.features.technical import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/multi_strategy.log"),
    ],
)
logger = logging.getLogger(__name__)

# Signal logger
signal_logger = logging.getLogger("signals")
signal_handler = logging.FileHandler("logs/signals.log")
signal_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
signal_logger.addHandler(signal_handler)
signal_logger.setLevel(logging.INFO)

# Trade logger (JSON Lines format for easy parsing)
trade_logger = logging.getLogger("trades")
trade_handler = logging.FileHandler("logs/trades.jsonl")
trade_handler.setFormatter(logging.Formatter("%(message)s"))
trade_logger.addHandler(trade_handler)
trade_logger.setLevel(logging.INFO)

# Position snapshot logger (JSON Lines)
position_logger = logging.getLogger("positions")
position_handler = logging.FileHandler("logs/positions.jsonl")
position_handler.setFormatter(logging.Formatter("%(message)s"))
position_logger.addHandler(position_handler)
position_logger.setLevel(logging.INFO)

# Event logger for all significant events
event_logger = logging.getLogger("events")
event_handler = logging.FileHandler("logs/events.jsonl")
event_handler.setFormatter(logging.Formatter("%(message)s"))
event_logger.addHandler(event_handler)
event_logger.setLevel(logging.INFO)


def log_event(event_type: str, data: dict) -> None:
    """Log an event in JSON format."""
    try:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event_type,
            **data,
        }
        event_logger.info(json.dumps(record, default=str))
    except Exception as e:
        logger.error("Failed to log event %s: %s", event_type, e)


def log_trade(trade_data: dict) -> None:
    """Log a trade in JSON format."""
    try:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            **trade_data,
        }
        trade_logger.info(json.dumps(record, default=str))
    except Exception as e:
        logger.error("Failed to log trade: %s", e)


def log_positions(positions: dict, equity: float) -> None:
    """Log position snapshot in JSON format."""
    try:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "equity": equity,
            "positions": positions,
        }
        position_logger.info(json.dumps(record, default=str))
    except Exception as e:
        logger.error("Failed to log positions: %s", e)


class SignalType(Enum):
    """Trading signal types."""

    LONG = auto()
    SHORT = auto()
    EXIT_LONG = auto()
    EXIT_SHORT = auto()
    HOLD = auto()


@dataclass
class TradingSignal:
    """Trading signal from a strategy."""

    symbol: str
    signal_type: SignalType
    price: float
    strength: float  # 0-1 confidence
    strategy_name: str
    reason: str
    stop_loss: float | None = None
    take_profit: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPosition:
    """Position tracked by a strategy."""

    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    entry_time: datetime
    strategy_name: str
    order_id: str | None = None


@dataclass
class StrategyConfig:
    """Configuration for a strategy."""

    name: str
    config_path: str
    capital_allocation_pct: float  # % of total equity for this strategy
    max_positions: int
    position_size_pct: float  # Base position size (before vol adjustment)
    enabled: bool = True


@dataclass
class StrategyState:
    """Runtime state for a strategy."""

    config: StrategyConfig
    positions: dict[str, StrategyPosition] = field(default_factory=dict)
    allocated_capital: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    total_trades: int = 0
    winning_trades: int = 0

    @property
    def win_rate(self) -> float:
        return (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0.0


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.symbols = self._extract_symbols(config)

    def _extract_symbols(self, config: dict) -> list[str]:
        """Extract symbol list from config."""
        symbols = config.get("symbols", {})
        if isinstance(symbols, dict):
            return list(symbols.keys())
        if isinstance(symbols, list):
            return symbols
        return []

    @abstractmethod
    def check_signal(
        self,
        symbol: str,
        bars: list[dict],
        current_position: StrategyPosition | None,
    ) -> TradingSignal | None:
        """Check for trading signal on a symbol."""
        ...


class ATRRSIStrategy(Strategy):
    """ATR-Optimized RSI Mean Reversion Strategy."""

    def __init__(self, config: dict):
        super().__init__("ATR-RSI", config)
        self.global_params = config.get("global_params", {})
        self.symbol_configs = config.get("symbols", {})

    def check_signal(
        self,
        symbol: str,
        bars: list[dict],
        current_position: StrategyPosition | None,
    ) -> TradingSignal | None:
        """Check for RSI-based mean reversion signals."""
        if len(bars) < 20:
            return None

        df = pd.DataFrame(bars)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Calculate indicators
        rsi = TechnicalIndicators.rsi(close, 14)
        tr = pd.concat(
            [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()

        current_rsi = rsi.iloc[-1]
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]

        if pd.isna(current_rsi) or pd.isna(current_atr):
            return None

        # Get symbol-specific params
        sym_config = self.symbol_configs.get(symbol, {})
        rsi_oversold = sym_config.get(
            "rsi_oversold", self.global_params.get("default_rsi_oversold", 30)
        )
        rsi_exit = sym_config.get("rsi_exit", self.global_params.get("default_rsi_exit", 50))
        atr_stop_mult = sym_config.get(
            "atr_stop_mult", self.global_params.get("default_atr_stop_mult", 1.5)
        )
        atr_tp_mult = sym_config.get(
            "atr_tp_mult", self.global_params.get("default_atr_tp_mult", 2.0)
        )

        # Scale ATR for minute bars
        scaled_atr = current_atr * 20

        # Check for exit signals first
        if current_position:
            if current_position.direction == "long":
                # RSI exit
                if current_rsi > rsi_exit:
                    return TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        price=current_price,
                        strength=1.0,
                        strategy_name=self.name,
                        reason=f"RSI exit ({current_rsi:.1f} > {rsi_exit})",
                    )
                # Stop loss
                if current_price <= current_position.stop_loss:
                    return TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        price=current_price,
                        strength=1.0,
                        strategy_name=self.name,
                        reason=f"Stop loss hit (${current_price:.2f} <= ${current_position.stop_loss:.2f})",
                    )
                # Take profit
                if current_price >= current_position.take_profit:
                    return TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        price=current_price,
                        strength=1.0,
                        strategy_name=self.name,
                        reason=f"Take profit hit (${current_price:.2f} >= ${current_position.take_profit:.2f})",
                    )
        elif current_rsi < rsi_oversold:
            # Signal strength based on how oversold
            strength = min(1.0, (rsi_oversold - current_rsi) / 10)

            return TradingSignal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                price=current_price,
                strength=strength,
                strategy_name=self.name,
                reason=f"RSI oversold ({current_rsi:.1f} < {rsi_oversold})",
                stop_loss=current_price - (scaled_atr * atr_stop_mult),
                take_profit=current_price + (scaled_atr * atr_tp_mult),
                metadata={"rsi": current_rsi, "atr": current_atr},
            )

        return None


class TrendFollowingStrategy(Strategy):
    """ADX + Moving Average Trend Following Strategy."""

    def __init__(self, config: dict):
        super().__init__("Trend", config)
        self.global_params = config.get("global_params", {})
        self.symbol_configs = config.get("symbols", {})

    def check_signal(
        self,
        symbol: str,
        bars: list[dict],
        current_position: StrategyPosition | None,
    ) -> TradingSignal | None:
        """Check for trend following signals."""
        if len(bars) < 50:
            return None

        df = pd.DataFrame(bars)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Calculate indicators
        fast_ma = close.rolling(self.global_params.get("fast_ma_period", 20)).mean()
        slow_ma = close.rolling(self.global_params.get("slow_ma_period", 50)).mean()

        # ADX calculation
        adx = self._calculate_adx(high, low, close, self.global_params.get("adx_period", 14))

        # ATR for stops
        tr = pd.concat(
            [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()

        current_fast_ma = fast_ma.iloc[-1]
        current_slow_ma = slow_ma.iloc[-1]
        current_adx = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        current_atr = atr.iloc[-1]
        current_price = close.iloc[-1]

        if pd.isna(current_fast_ma) or pd.isna(current_slow_ma):
            return None

        # Get symbol-specific params
        sym_config = self.symbol_configs.get(symbol, {})
        adx_threshold = sym_config.get("adx_threshold", self.global_params.get("adx_threshold", 25))
        atr_stop_mult = sym_config.get(
            "atr_stop_mult", self.global_params.get("atr_stop_mult", 2.0)
        )
        atr_tp_mult = sym_config.get("atr_tp_mult", self.global_params.get("atr_tp_mult", 3.0))

        # Scale ATR
        scaled_atr = current_atr * 20

        # Check for exits first
        if current_position:
            if current_position.direction == "long":
                # MA crossover exit
                if current_fast_ma < current_slow_ma:
                    return TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        price=current_price,
                        strength=1.0,
                        strategy_name=self.name,
                        reason="MA crossover bearish",
                    )
                # Stop loss
                if current_price <= current_position.stop_loss:
                    return TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        price=current_price,
                        strength=1.0,
                        strategy_name=self.name,
                        reason=f"Stop loss hit",
                    )
                # Take profit
                if current_price >= current_position.take_profit:
                    return TradingSignal(
                        symbol=symbol,
                        signal_type=SignalType.EXIT_LONG,
                        price=current_price,
                        strength=1.0,
                        strategy_name=self.name,
                        reason=f"Take profit hit",
                    )
        else:
            # Entry: Bullish MA crossover with strong trend (ADX > threshold)
            prev_fast = fast_ma.iloc[-2] if len(fast_ma) > 1 else 0
            prev_slow = slow_ma.iloc[-2] if len(slow_ma) > 1 else 0

            # Fresh crossover
            crossover = (current_fast_ma > current_slow_ma) and (prev_fast <= prev_slow)
            # Already in uptrend
            uptrend = current_fast_ma > current_slow_ma and current_adx > adx_threshold

            if crossover or (uptrend and current_adx > adx_threshold * 1.2):
                strength = min(1.0, current_adx / 50)

                return TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    price=current_price,
                    strength=strength,
                    strategy_name=self.name,
                    reason=f"{'MA crossover' if crossover else 'Strong uptrend'} (ADX={current_adx:.1f})",
                    stop_loss=current_price - (scaled_atr * atr_stop_mult),
                    take_profit=current_price + (scaled_atr * atr_tp_mult),
                    metadata={
                        "adx": current_adx,
                        "fast_ma": current_fast_ma,
                        "slow_ma": current_slow_ma,
                    },
                )

        return None

    def _calculate_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Calculate ADX indicator."""
        # True Range
        tr = pd.concat(
            [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
        ).max(axis=1)

        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed averages
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx


class MultiStrategyOrchestrator:
    """
    Orchestrates multiple trading strategies with proper risk management.

    Features:
    - Broker position sync on startup
    - Volatility-adjusted position sizing
    - Capital allocation per strategy
    - Symbol exclusion across strategies
    - Strategy-level P&L tracking
    """

    def __init__(self, strategy_configs: list[StrategyConfig]):
        """Initialize the orchestrator."""
        self.strategy_configs = strategy_configs
        self.strategies: dict[str, Strategy] = {}
        self.strategy_states: dict[str, StrategyState] = {}

        # Shared data
        self.bars: dict[str, list[dict]] = defaultdict(list)
        self.latest_quotes: dict[str, StreamQuote] = {}

        # Position sizer
        self.sizer = PositionSizer(
            SizingConfig(
                method=SizingMethod.VOLATILITY_ADJUSTED,
                max_position_pct=0.03,  # Max 3% per position (even with vol adjustment)
                min_position_pct=0.005,  # Min 0.5%
                vol_target=0.15,  # Target 15% annualized vol
                vol_lookback_days=20,
            )
        )

        # Global tracking
        self.global_positions: dict[str, str] = {}  # symbol -> strategy_name
        self._last_order_time: dict[str, datetime] = {}
        self._order_cooldown_seconds = 60
        self._global_order_cooldown = 5
        self._last_global_order: datetime | None = None

        # Components
        self.broker: AlpacaBroker | None = None
        self.stream: MassiveStreamManager | None = None
        self._running = False

    def _load_strategy_config(self, path: str) -> dict:
        """Load strategy configuration from YAML."""
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        logger.warning("Config not found: %s", path)
        return {}

    def _initialize_strategies(self) -> None:
        """Initialize all configured strategies."""
        for config in self.strategy_configs:
            if not config.enabled:
                continue

            strategy_config = self._load_strategy_config(config.config_path)

            # Create strategy instance based on name
            if "rsi" in config.name.lower() or "atr" in config.name.lower():
                strategy = ATRRSIStrategy(strategy_config)
            elif "trend" in config.name.lower():
                strategy = TrendFollowingStrategy(strategy_config)
            else:
                logger.warning("Unknown strategy type: %s", config.name)
                continue

            self.strategies[config.name] = strategy
            self.strategy_states[config.name] = StrategyState(config=config)

            logger.info(
                "Initialized strategy: %s with %d symbols, %.0f%% capital",
                config.name,
                len(strategy.symbols),
                config.capital_allocation_pct * 100,
            )

    async def _sync_broker_positions(self) -> None:
        """Sync positions from broker on startup."""
        if not self.broker:
            return

        broker_positions = await self.broker.get_positions()

        if not broker_positions:
            logger.info("No existing broker positions to sync")
            return

        logger.info("Syncing %d broker positions...", len(broker_positions))

        for pos in broker_positions:
            symbol = pos.symbol

            # CRITICAL: Skip short positions - our strategies only go long
            if pos.quantity <= 0:
                logger.warning(
                    "SKIP SYNC | %s | Short position (%d shares) - not tracking",
                    symbol,
                    pos.quantity,
                )
                continue

            # Try to assign to a strategy that trades this symbol
            assigned = False
            for name, strategy in self.strategies.items():
                if symbol in strategy.symbols:
                    state = self.strategy_states[name]

                    # Estimate stop/TP from current price (conservative)
                    current_price = float(pos.current_price)
                    entry_price = float(pos.avg_entry_price)

                    # Use 8% stop, 12% TP as defaults for synced positions
                    stop_loss = entry_price * 0.92
                    take_profit = entry_price * 1.12

                    state.positions[symbol] = StrategyPosition(
                        symbol=symbol,
                        direction="long",
                        entry_price=entry_price,
                        quantity=pos.quantity,  # Already validated > 0
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        entry_time=datetime.now(UTC),
                        strategy_name=name,
                    )

                    self.global_positions[symbol] = name
                    assigned = True

                    logger.info(
                        "Synced %s to %s: %d shares @ $%.2f",
                        symbol,
                        name,
                        pos.quantity,
                        entry_price,
                    )

                    # Log position sync event
                    log_event(
                        "POSITION_SYNCED",
                        {
                            "symbol": symbol,
                            "strategy": name,
                            "quantity": int(pos.quantity),
                            "entry_price": float(entry_price),
                            "current_price": float(current_price),
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "unrealized_pnl": float(pos.unrealized_pnl)
                            if hasattr(pos, "unrealized_pnl")
                            else None,
                        },
                    )
                    break

            if not assigned:
                logger.warning(
                    "Position %s not assigned to any strategy (not in symbol lists)", symbol
                )
                log_event(
                    "POSITION_UNASSIGNED",
                    {
                        "symbol": symbol,
                        "quantity": int(pos.quantity),
                        "reason": "not in any strategy symbol list",
                    },
                )

    async def _allocate_capital(self) -> None:
        """Allocate capital to each strategy."""
        if not self.broker:
            return

        account = await self.broker.get_account()
        total_equity = float(account.equity)

        for name, state in self.strategy_states.items():
            allocation = total_equity * state.config.capital_allocation_pct
            state.allocated_capital = Decimal(str(allocation))

            logger.info(
                "Allocated $%.2f to %s (%.0f%%)",
                allocation,
                name,
                state.config.capital_allocation_pct * 100,
            )

        self.sizer.set_portfolio_value(total_equity)

    def _get_all_symbols(self) -> list[str]:
        """Get all unique symbols across all strategies."""
        symbols = set()
        for strategy in self.strategies.values():
            symbols.update(strategy.symbols)
        return list(symbols)

    async def start(self) -> None:
        """Start the multi-strategy system."""
        logger.info("=" * 60)
        logger.info("MULTI-STRATEGY PAPER TRADING RUNNER")
        logger.info("=" * 60)

        # Log startup event
        log_event(
            "SYSTEM_START",
            {
                "strategies": [c.name for c in self.strategy_configs],
            },
        )

        # Initialize strategies
        self._initialize_strategies()

        if not self.strategies:
            logger.error("No strategies initialized")
            return

        # Log strategy initialization
        for name, strategy in self.strategies.items():
            state = self.strategy_states[name]
            log_event(
                "STRATEGY_INIT",
                {
                    "strategy": name,
                    "symbols": len(strategy.symbols),
                    "capital_allocation_pct": state.config.capital_allocation_pct,
                    "max_positions": state.config.max_positions,
                    "position_size_pct": state.config.position_size_pct,
                },
            )

        # Initialize broker
        self.broker = AlpacaBroker(paper=True)
        if not await self.broker.connect():
            logger.error("Failed to connect to Alpaca broker")
            return

        account = await self.broker.get_account()
        logger.info("Connected to Alpaca - Equity: $%.2f", account.equity)

        log_event(
            "BROKER_CONNECTED",
            {
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
            },
        )

        # Sync existing positions
        await self._sync_broker_positions()

        # Allocate capital
        await self._allocate_capital()

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

        # Subscribe to all symbols
        all_symbols = self._get_all_symbols()
        await self.stream.connect()
        await self.stream.subscribe(all_symbols)

        logger.info(
            "Subscribed to %d symbols across %d strategies", len(all_symbols), len(self.strategies)
        )

        self._running = True

        # Main loop
        try:
            while self._running:
                await asyncio.sleep(1)

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

        logger.info("Multi-strategy trading stopped")
        await self._print_status()

    async def _on_bar(self, bar: StreamBar) -> None:
        """Handle incoming minute bar."""
        symbol = bar.symbol

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

        # Update sizer with returns data
        if len(self.bars[symbol]) >= 2:
            df = pd.DataFrame(self.bars[symbol])
            returns = df["close"].pct_change().dropna()
            self.sizer.set_returns(symbol, returns)

        # Check signals for each strategy that trades this symbol
        for name, strategy in self.strategies.items():
            if symbol not in strategy.symbols:
                continue

            state = self.strategy_states[name]
            current_position = state.positions.get(symbol)

            # Check signal
            signal = strategy.check_signal(symbol, self.bars[symbol], current_position)

            if signal:
                await self._handle_signal(signal, state)

    async def _on_quote(self, quote: StreamQuote) -> None:
        """Handle incoming quote."""
        self.latest_quotes[quote.symbol] = quote

    async def _on_status(self, status: StreamStatus, message: str) -> None:
        """Handle stream status change."""
        logger.info("Stream status: %s - %s", status.name, message)

    async def _on_error(self, error: Exception) -> None:
        """Handle stream error."""
        logger.error("Stream error: %s", error)

    async def _handle_signal(self, signal: TradingSignal, state: StrategyState) -> None:
        """Handle a trading signal with proper risk management."""
        symbol = signal.symbol
        now = datetime.now(UTC)

        # Log signal
        log_msg = (
            f"SIGNAL | {signal.strategy_name} | {symbol} | {signal.signal_type.name} | "
            f"Price=${signal.price:.2f} | Strength={signal.strength:.2f} | {signal.reason}"
        )
        logger.info(log_msg)
        signal_logger.info(log_msg)

        # Check cooldowns
        if self._last_global_order:
            elapsed = (now - self._last_global_order).total_seconds()
            if elapsed < self._global_order_cooldown:
                return

        if symbol in self._last_order_time:
            elapsed = (now - self._last_order_time[symbol]).total_seconds()
            if elapsed < self._order_cooldown_seconds:
                return

        # Execute based on signal type
        if signal.signal_type == SignalType.LONG:
            await self._execute_entry(signal, state)
        elif signal.signal_type == SignalType.EXIT_LONG:
            await self._execute_exit(signal, state)

    async def _execute_entry(self, signal: TradingSignal, state: StrategyState) -> None:
        """Execute a long entry with volatility-adjusted sizing."""
        symbol = signal.symbol

        # Check if symbol already owned by another strategy
        if symbol in self.global_positions:
            logger.info("SKIP | %s already owned by %s", symbol, self.global_positions[symbol])
            return

        # Check max positions for this strategy
        if len(state.positions) >= state.config.max_positions:
            logger.info(
                "SKIP | %s at max positions (%d/%d)",
                state.config.name,
                len(state.positions),
                state.config.max_positions,
            )
            return

        if not self.broker:
            return

        try:
            # Calculate position size using volatility-adjusted sizer
            # Override base percentage with strategy config
            size_result = self.sizer.calculate_size(
                symbol=symbol,
                current_price=signal.price,
                signal_strength=signal.strength,
            )

            # Apply strategy-specific position limit
            max_position_value = float(state.allocated_capital) * state.config.position_size_pct
            if float(size_result.notional_value) > max_position_value:
                quantity = int(max_position_value / signal.price)
            else:
                quantity = size_result.shares

            if quantity <= 0:
                logger.warning("Position size too small for %s", symbol)
                return

            # Submit order
            order = await self.broker.submit_order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=Decimal(str(quantity)),
                order_type=OrderType.MARKET,
            )

            # Track position
            state.positions[symbol] = StrategyPosition(
                symbol=symbol,
                direction="long",
                entry_price=signal.price,
                quantity=quantity,
                stop_loss=signal.stop_loss or signal.price * 0.92,
                take_profit=signal.take_profit or signal.price * 1.12,
                entry_time=datetime.now(UTC),
                strategy_name=signal.strategy_name,
                order_id=order.id if order else None,
            )

            self.global_positions[symbol] = signal.strategy_name

            # Update timestamps
            now = datetime.now(UTC)
            self._last_order_time[symbol] = now
            self._last_global_order = now

            position_value = quantity * signal.price
            pct_of_allocation = (position_value / float(state.allocated_capital)) * 100

            logger.info(
                "ENTRY | %s | %s | %d shares @ $%.2f ($%.0f, %.1f%% of allocation) | SL=$%.2f TP=$%.2f",
                signal.strategy_name,
                symbol,
                quantity,
                signal.price,
                position_value,
                pct_of_allocation,
                state.positions[symbol].stop_loss,
                state.positions[symbol].take_profit,
            )

            # Log trade for post-market analysis
            log_trade(
                {
                    "action": "ENTRY",
                    "strategy": signal.strategy_name,
                    "symbol": symbol,
                    "side": "long",
                    "quantity": int(quantity),
                    "price": float(signal.price),
                    "value": float(position_value),
                    "pct_of_allocation": float(pct_of_allocation),
                    "stop_loss": float(state.positions[symbol].stop_loss),
                    "take_profit": float(state.positions[symbol].take_profit),
                    "signal_strength": float(signal.strength) if signal.strength else 0.0,
                    "reason": signal.reason,
                    "metadata": signal.metadata,
                }
            )

        except Exception as e:
            logger.error("Error executing entry for %s: %s", symbol, e)

    async def _execute_exit(self, signal: TradingSignal, state: StrategyState) -> None:
        """Execute an exit order."""
        symbol = signal.symbol

        if symbol not in state.positions:
            return

        if not self.broker:
            return

        try:
            pos = state.positions[symbol]
            quantity = pos.quantity

            # CRITICAL: Validate position exists on broker before selling
            # This prevents accidental short selling
            broker_pos = await self.broker.get_position(symbol)
            if broker_pos is None:
                logger.warning(
                    "SKIP EXIT | %s | Position not found on broker - removing from tracking", symbol
                )
                # Clean up stale position from tracking
                del state.positions[symbol]
                if symbol in self.global_positions:
                    del self.global_positions[symbol]
                return

            # Only sell what we actually own (prevent shorts)
            broker_qty = broker_pos.quantity
            if broker_qty <= 0:
                logger.warning(
                    "SKIP EXIT | %s | Broker shows %d shares (not long) - removing from tracking",
                    symbol,
                    broker_qty,
                )
                del state.positions[symbol]
                if symbol in self.global_positions:
                    del self.global_positions[symbol]
                return

            # Use the smaller of tracked qty and broker qty
            sell_qty = min(quantity, broker_qty)
            if sell_qty != quantity:
                logger.warning(
                    "EXIT QTY ADJUSTED | %s | Tracked=%d, Broker=%d, Selling=%d",
                    symbol,
                    quantity,
                    broker_qty,
                    sell_qty,
                )

            # Submit exit order
            await self.broker.submit_order(
                symbol=symbol,
                side=OrderSide.SELL,
                quantity=Decimal(str(sell_qty)),
                order_type=OrderType.MARKET,
            )

            # Update quantity for P&L calculation
            quantity = sell_qty

            # Calculate P&L
            pnl_pct = ((signal.price - pos.entry_price) / pos.entry_price) * 100
            pnl_abs = (signal.price - pos.entry_price) * quantity

            state.total_trades += 1
            if pnl_pct > 0:
                state.winning_trades += 1
            state.realized_pnl += Decimal(str(pnl_abs))

            # Update timestamps
            now = datetime.now(UTC)
            self._last_order_time[symbol] = now
            self._last_global_order = now

            logger.info(
                "EXIT | %s | %s | %d shares @ $%.2f | P&L: %+.2f%% ($%+.2f) | %s",
                signal.strategy_name,
                symbol,
                quantity,
                signal.price,
                pnl_pct,
                pnl_abs,
                signal.reason,
            )

            # Log trade for post-market analysis
            hold_time_minutes = (
                (datetime.now(UTC) - pos.entry_time).total_seconds() / 60 if pos.entry_time else 0
            )
            log_trade(
                {
                    "action": "EXIT",
                    "strategy": signal.strategy_name,
                    "symbol": symbol,
                    "side": "long",
                    "quantity": int(quantity),
                    "entry_price": float(pos.entry_price),
                    "exit_price": float(signal.price),
                    "pnl_pct": float(pnl_pct),
                    "pnl_abs": float(pnl_abs),
                    "hold_time_minutes": float(hold_time_minutes),
                    "reason": signal.reason,
                    "win": pnl_pct > 0,
                    "strategy_total_trades": int(state.total_trades),
                    "strategy_win_rate": float(state.win_rate),
                    "strategy_realized_pnl": float(state.realized_pnl),
                }
            )

            # Remove from tracking
            del state.positions[symbol]
            if symbol in self.global_positions:
                del self.global_positions[symbol]

        except Exception as e:
            logger.error("Error executing exit for %s: %s", symbol, e)

    async def _print_status(self) -> None:
        """Print status for all strategies."""
        if not self.broker:
            return

        account = await self.broker.get_account()

        logger.info("-" * 70)
        logger.info("PORTFOLIO STATUS | Equity: $%.2f | Cash: $%.2f", account.equity, account.cash)
        logger.info("-" * 70)

        # Build position snapshot for logging
        all_positions = {}
        for name, state in self.strategy_states.items():
            positions_str = ", ".join(state.positions.keys()) if state.positions else "None"
            logger.info(
                "%s | Positions: %d/%d | Trades: %d | WinRate: %.1f%% | P&L: $%+.2f | Open: %s",
                name,
                len(state.positions),
                state.config.max_positions,
                state.total_trades,
                state.win_rate,
                state.realized_pnl,
                positions_str,
            )

            # Collect positions for snapshot
            for symbol, pos in state.positions.items():
                all_positions[symbol] = {
                    "strategy": name,
                    "quantity": int(pos.quantity) if pos.quantity else 0,
                    "entry_price": float(pos.entry_price) if pos.entry_price else 0.0,
                    "stop_loss": float(pos.stop_loss) if pos.stop_loss else 0.0,
                    "take_profit": float(pos.take_profit) if pos.take_profit else 0.0,
                    "entry_time": pos.entry_time.isoformat() if pos.entry_time else None,
                }

        # Log position snapshot every 5 minutes
        if datetime.now().minute % 5 == 0:
            log_positions(all_positions, float(account.equity))


async def main() -> None:
    """Main entry point."""
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Configure strategies
    strategies = [
        StrategyConfig(
            name="ATR-RSI",
            config_path="configs/strategies/atr_optimized_rsi.yaml",
            capital_allocation_pct=0.50,  # 50% of capital
            max_positions=15,
            position_size_pct=0.05,  # 5% base position size ($500 on $10k)
        ),
        StrategyConfig(
            name="Trend",
            config_path="configs/strategies/trend_following.yaml",
            capital_allocation_pct=0.50,  # 50% of capital
            max_positions=10,
            position_size_pct=0.08,  # 8% base for trend (fewer positions, larger size)
        ),
    ]

    orchestrator = MultiStrategyOrchestrator(strategies)
    await orchestrator.start()


if __name__ == "__main__":
    asyncio.run(main())
