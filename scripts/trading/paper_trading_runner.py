#!/usr/bin/env python
"""
Paper Trading Runner - Runs ATR-Optimized RSI strategy in paper mode.

Connects:
- RegimeDetector for stock filtering
- ATR-Optimized RSI strategy for signals
- Alpaca or Simulated broker for execution

Usage:
    # With Alpaca paper trading
    export APCA_API_KEY_ID=your_key
    export APCA_API_SECRET_KEY=your_secret
    python scripts/trading/paper_trading_runner.py --broker alpaca

    # With simulated broker (no API needed)
    python scripts/trading/paper_trading_runner.py --broker simulated
"""

import argparse
import asyncio
from decimal import Decimal
import logging
from pathlib import Path
import sys

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ordinis.adapters.broker import (
    AlpacaBroker,
    BrokerAdapter,
    OrderSide,
    OrderType,
    SimulatedBroker,
)
from ordinis.engines.signalcore.features.technical import TechnicalIndicators
from ordinis.engines.signalcore.regime_detector import RegimeDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/paper_trading.log"),
    ],
)
logger = logging.getLogger(__name__)


class PaperTradingRunner:
    """
    Paper trading runner for ATR-Optimized RSI strategy.

    Features:
    - Regime filtering before trading
    - ATR-based position sizing
    - Stop loss and take profit management
    - Daily P&L tracking
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        config_path: str = "configs/strategies/atr_optimized_rsi.yaml",
    ):
        """Initialize runner."""
        self.broker = broker
        self.config = self._load_config(config_path)
        self.detector = RegimeDetector()

        # State
        self.positions: dict[str, dict] = {}  # symbol -> position info
        self.daily_pnl = Decimal("0")
        self.total_trades = 0
        self.winning_trades = 0

        # Risk limits
        self.max_position_pct = self.config.get("risk_management", {}).get(
            "max_position_size_pct", 5.0
        )
        self.max_daily_loss = self.config.get("risk_management", {}).get("max_daily_loss_pct", 2.0)
        self.max_positions = self.config.get("risk_management", {}).get(
            "max_concurrent_positions", 5
        )

    def _load_config(self, path: str) -> dict:
        """Load strategy config."""
        config_path = Path(path)
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}

    async def connect(self) -> bool:
        """Connect to broker."""
        connected = await self.broker.connect()
        if connected:
            account = await self.broker.get_account()
            logger.info(f"Connected to broker. Equity: ${account.equity:,.2f}")
            logger.info(f"Paper mode: {account.is_paper}")
        return connected

    async def get_tradeable_symbols(self) -> list[str]:
        """Get symbols that pass regime filter."""
        symbols = list(self.config.get("symbols", {}).keys())
        tradeable = []

        for symbol in symbols:
            # Would need real-time data here
            # For now, use config-based filtering
            symbol_config = self.config.get("symbols", {}).get(symbol, {})
            if symbol_config.get("backtest_profit_factor", 0) >= 1.2:
                tradeable.append(symbol)

        return tradeable

    def compute_indicators(self, df: pd.DataFrame) -> dict:
        """Compute RSI and ATR from price data."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        rsi = TechnicalIndicators.rsi(close, 14)

        tr = pd.concat(
            [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1
        ).max(axis=1)
        atr = tr.rolling(14).mean()

        return {
            "rsi": rsi.iloc[-1],
            "atr": atr.iloc[-1],
            "close": close.iloc[-1],
        }

    def get_signal(self, symbol: str, indicators: dict) -> str | None:
        """
        Generate trading signal.

        Returns: "long", "short", "exit_long", "exit_short", or None
        """
        symbol_config = self.config.get("symbols", {}).get(symbol, {})
        rsi_oversold = symbol_config.get("rsi_oversold", 35)
        rsi_overbought = 65  # For shorts
        rsi_exit = symbol_config.get("rsi_exit", 50)

        rsi = indicators["rsi"]

        # Check if we have a position
        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos["direction"] == "long" and rsi > rsi_exit:
                return "exit_long"
            if pos["direction"] == "short" and rsi < rsi_exit:
                return "exit_short"
        elif rsi < rsi_oversold:
            return "long"
        elif rsi > rsi_overbought:
            return "short"

        return None

    def calculate_position_size(
        self,
        symbol: str,
        price: Decimal,
        atr: float,
        equity: Decimal,
    ) -> Decimal:
        """
        Calculate position size based on ATR and risk limits.

        Position sized so that 1.5×ATR stop = max_position_pct of equity.
        """
        symbol_config = self.config.get("symbols", {}).get(symbol, {})
        atr_stop_mult = symbol_config.get("atr_stop_mult", 1.5)

        # Risk per trade
        risk_amount = equity * Decimal(str(self.max_position_pct / 100))

        # Stop distance
        stop_distance = Decimal(str(atr * atr_stop_mult))

        # Position size
        if stop_distance > 0:
            shares = risk_amount / stop_distance
            # Round down to whole shares
            shares = Decimal(int(shares))
        else:
            shares = Decimal("0")

        # Cap at max position value
        max_value = equity * Decimal(str(self.max_position_pct / 100))
        max_shares = max_value / price if price > 0 else Decimal("0")

        return min(shares, Decimal(int(max_shares)))

    async def execute_signal(
        self,
        symbol: str,
        signal: str,
        indicators: dict,
    ) -> bool:
        """Execute a trading signal."""
        try:
            account = await self.broker.get_account()
            price = Decimal(str(indicators["close"]))
            atr = indicators["atr"]

            if signal == "long":
                if len(self.positions) >= self.max_positions:
                    logger.warning(f"Max positions reached, skipping {symbol}")
                    return False

                quantity = self.calculate_position_size(symbol, price, atr, account.equity)
                if quantity <= 0:
                    return False

                order = await self.broker.submit_order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                )

                symbol_config = self.config.get("symbols", {}).get(symbol, {})
                atr_stop = symbol_config.get("atr_stop_mult", 1.5)
                atr_tp = symbol_config.get("atr_tp_mult", 2.0)

                self.positions[symbol] = {
                    "direction": "long",
                    "entry_price": float(price),
                    "quantity": float(quantity),
                    "stop_loss": float(price) - (atr * atr_stop),
                    "take_profit": float(price) + (atr * atr_tp),
                    "order_id": order.id,
                }

                logger.info(f"LONG {symbol}: {quantity} shares @ ${price:.2f}")
                return True

            if signal == "short":
                if len(self.positions) >= self.max_positions:
                    return False

                quantity = self.calculate_position_size(symbol, price, atr, account.equity)
                if quantity <= 0:
                    return False

                order = await self.broker.submit_order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                )

                symbol_config = self.config.get("symbols", {}).get(symbol, {})
                atr_stop = symbol_config.get("atr_stop_mult", 1.5)
                atr_tp = symbol_config.get("atr_tp_mult", 2.0)

                self.positions[symbol] = {
                    "direction": "short",
                    "entry_price": float(price),
                    "quantity": float(quantity),
                    "stop_loss": float(price) + (atr * atr_stop),
                    "take_profit": float(price) - (atr * atr_tp),
                    "order_id": order.id,
                }

                logger.info(f"SHORT {symbol}: {quantity} shares @ ${price:.2f}")
                return True

            if signal in ("exit_long", "exit_short"):
                if symbol not in self.positions:
                    return False

                pos = self.positions[symbol]
                quantity = Decimal(str(pos["quantity"]))

                side = OrderSide.SELL if signal == "exit_long" else OrderSide.BUY

                order = await self.broker.submit_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                )

                # Calculate P&L
                entry = pos["entry_price"]
                exit_price = float(price)

                if pos["direction"] == "long":
                    pnl = (exit_price - entry) / entry * 100
                else:
                    pnl = (entry - exit_price) / entry * 100

                self.total_trades += 1
                if pnl > 0:
                    self.winning_trades += 1
                self.daily_pnl += Decimal(str(pnl * pos["quantity"]))

                logger.info(f"EXIT {symbol}: {pnl:+.2f}% P&L")

                del self.positions[symbol]
                return True

        except Exception as e:
            logger.error(f"Error executing {signal} for {symbol}: {e}")
            return False

    async def check_stops(self, symbol: str, current_price: float) -> str | None:
        """Check if stop loss or take profit hit."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]

        if pos["direction"] == "long":
            if current_price <= pos["stop_loss"]:
                return "exit_long"
            if current_price >= pos["take_profit"]:
                return "exit_long"
        else:  # short
            if current_price >= pos["stop_loss"]:
                return "exit_short"
            if current_price <= pos["take_profit"]:
                return "exit_short"

        return None

    async def run_once(self, price_data: dict[str, pd.DataFrame]) -> None:
        """Run one iteration of the strategy."""
        tradeable = await self.get_tradeable_symbols()

        for symbol in tradeable:
            if symbol not in price_data:
                continue

            df = price_data[symbol]
            if len(df) < 50:
                continue

            indicators = self.compute_indicators(df)

            # Check stops first
            stop_signal = await self.check_stops(symbol, indicators["close"])
            if stop_signal:
                await self.execute_signal(symbol, stop_signal, indicators)
                continue

            # Check for new signals
            signal = self.get_signal(symbol, indicators)
            if signal:
                await self.execute_signal(symbol, signal, indicators)

    async def get_status(self) -> dict:
        """Get current trading status."""
        account = await self.broker.get_account()

        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "positions": len(self.positions),
            "daily_pnl": float(self.daily_pnl),
            "total_trades": self.total_trades,
            "win_rate": (self.winning_trades / self.total_trades * 100)
            if self.total_trades > 0
            else 0,
            "open_positions": list(self.positions.keys()),
        }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Paper Trading Runner")
    parser.add_argument(
        "--broker",
        choices=["alpaca", "simulated"],
        default="simulated",
        help="Broker to use",
    )
    parser.add_argument(
        "--config",
        default="configs/strategies/atr_optimized_rsi.yaml",
        help="Strategy config path",
    )
    args = parser.parse_args()

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    # Initialize broker
    if args.broker == "alpaca":
        broker = AlpacaBroker(paper=True)
    else:
        broker = SimulatedBroker(initial_cash=Decimal("100000"))

    # Initialize runner
    runner = PaperTradingRunner(broker, args.config)

    # Connect
    if not await runner.connect():
        logger.error("Failed to connect to broker")
        return

    # Print config
    print("\n" + "=" * 60)
    print("PAPER TRADING RUNNER")
    print("=" * 60)
    print(f"Broker: {args.broker}")
    print(f"Config: {args.config}")

    tradeable = await runner.get_tradeable_symbols()
    print(f"Tradeable symbols: {tradeable}")

    status = await runner.get_status()
    print(f"Starting equity: ${status['equity']:,.2f}")
    print("=" * 60)

    # In a real implementation, this would:
    # 1. Subscribe to real-time data
    # 2. Run on each bar/tick
    # 3. Manage positions continuously

    print("\n[OK] Paper trading runner initialized successfully!")
    print("   In production, connect to real-time data feed.")
    print("   Use `runner.run_once(price_data)` on each bar.")


if __name__ == "__main__":
    asyncio.run(main())
