"""
Enhanced live paper trading with Alpaca market data streaming.

Uses Alpaca's market data feed for real-time bars and historical data.
Supports multiple strategies and proper risk management.

Usage:
    python scripts/trading/run_live_alpaca_enhanced.py --symbol SPY --strategy ma_cross
"""

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, "src")

from ordinis.engines.flowroute.adapters.alpaca import AlpacaBrokerAdapter
from ordinis.engines.flowroute.adapters.alpaca_data import AlpacaMarketDataAdapter
from ordinis.engines.flowroute.core.orders import Order, OrderType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MACrossoverStrategy:
    """Moving average crossover strategy."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prices: list[float] = []
        self.prev_signal: str | None = None

    def update(self, price: float) -> str | None:
        """Update with new price and generate signal."""
        self.prices.append(price)

        # Keep only necessary data
        if len(self.prices) > self.slow_period * 2:
            self.prices = self.prices[-self.slow_period * 2 :]

        if len(self.prices) < self.slow_period:
            return None

        fast_ma = sum(self.prices[-self.fast_period :]) / self.fast_period
        slow_ma = sum(self.prices[-self.slow_period :]) / self.slow_period

        # Generate signal on crossover
        if fast_ma > slow_ma and self.prev_signal != "bullish":
            self.prev_signal = "bullish"
            logger.info(f"[SIGNAL] BUY - Fast MA ({fast_ma:.2f}) > Slow MA ({slow_ma:.2f})")
            return "buy"
        if fast_ma < slow_ma and self.prev_signal != "bearish":
            self.prev_signal = "bearish"
            logger.info(f"[SIGNAL] SELL - Fast MA ({fast_ma:.2f}) < Slow MA ({slow_ma:.2f})")
            return "sell"

        return None


class LiveTradingEngine:
    """Live trading engine with Alpaca."""

    def __init__(
        self,
        broker: AlpacaBrokerAdapter,
        market_data: AlpacaMarketDataAdapter,
        strategy: MACrossoverStrategy,
        symbol: str = "SPY",
        position_size_usd: float = 10000.0,
    ):
        self.broker = broker
        self.market_data = market_data
        self.strategy = strategy
        self.symbol = symbol
        self.position_size_usd = position_size_usd
        self.order_count = 0
        self.running = False

    async def get_current_position(self) -> int:
        """Get current position quantity."""
        positions = await self.broker.get_positions()
        for pos in positions:
            if pos.get("symbol") == self.symbol:
                return int(pos.get("quantity", 0))
        return 0

    async def execute_signal(self, signal: str, current_price: float) -> None:
        """Execute trading signal."""
        current_qty = await self.get_current_position()

        if signal == "buy" and current_qty == 0:
            # Calculate quantity
            qty = int(self.position_size_usd / current_price)
            if qty > 0:
                order = Order(
                    order_id=f"ORDER-{self.order_count:04d}",
                    symbol=self.symbol,
                    side="buy",
                    quantity=qty,
                    order_type=OrderType.MARKET,
                )
                result = await self.broker.submit_order(order)
                if result["success"]:
                    logger.info(
                        f"[ORDER] BUY {qty} {self.symbol} @ ${current_price:.2f} | Order ID: {result.get('broker_order_id')}"
                    )
                    self.order_count += 1
                else:
                    logger.error(f"[ERROR] BUY order failed: {result.get('error')}")

        elif signal == "sell" and current_qty > 0:
            order = Order(
                order_id=f"ORDER-{self.order_count:04d}",
                symbol=self.symbol,
                side="sell",
                quantity=current_qty,
                order_type=OrderType.MARKET,
            )
            result = await self.broker.submit_order(order)
            if result["success"]:
                logger.info(
                    f"[ORDER] SELL {current_qty} {self.symbol} @ ${current_price:.2f} | Order ID: {result.get('broker_order_id')}"
                )
                self.order_count += 1
            else:
                logger.error(f"[ERROR] SELL order failed: {result.get('error')}")

    async def run_polling_mode(self, interval: int = 60) -> None:
        """Run in polling mode (checks price every N seconds)."""
        logger.info(f"[MODE] Polling mode - checking every {interval}s")

        try:
            while self.running:
                # Check if market is open
                if not self.market_data.is_market_open():
                    logger.info("[MARKET] Outside market hours, waiting...")
                    await asyncio.sleep(300)  # Check every 5 min
                    continue

                # Get latest bar
                bar = self.market_data.get_latest_bar(self.symbol)
                if bar:
                    price = bar["close"]
                    logger.info(f"[PRICE] {self.symbol}: ${price:.2f} | Vol: {bar['volume']:,}")

                    # Update strategy
                    signal = self.strategy.update(price)
                    if signal:
                        await self.execute_signal(signal, price)

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            logger.info("\n[STOP] Trading stopped by user")
        except Exception as e:
            logger.error(f"[ERROR] Trading loop error: {e}", exc_info=True)

    async def run_streaming_mode(self) -> None:
        """Run in streaming mode (real-time bar updates)."""
        logger.info("[MODE] Streaming mode - real-time bars")

        def on_bar(bar: dict):
            """Handle incoming bar."""
            price = bar["close"]
            logger.info(
                f"[BAR] {bar['symbol']}: ${price:.2f} @ {bar['timestamp']} | Vol: {bar['volume']:,}"
            )

            # Update strategy
            signal = self.strategy.update(price)
            if signal:
                # Execute signal in async context
                asyncio.create_task(self.execute_signal(signal, price))

        try:
            await self.market_data.stream_bars([self.symbol], on_bar)
        except KeyboardInterrupt:
            logger.info("\n[STOP] Trading stopped by user")
        except Exception as e:
            logger.error(f"[ERROR] Streaming error: {e}", exc_info=True)

    async def start(self, mode: str = "polling", interval: int = 60) -> None:
        """Start the trading engine."""
        self.running = True

        # Load historical data to initialize strategy
        logger.info("[INIT] Loading historical data...")
        prices = self.market_data.get_price_history(
            symbol=self.symbol,
            periods=self.strategy.slow_period + 10,
            timeframe="1Min",
        )

        if prices:
            logger.info(f"[INIT] Loaded {len(prices)} historical prices")
            for price in prices:
                self.strategy.update(price)
        else:
            logger.warning("[INIT] No historical data available, starting fresh")

        # Start trading loop
        if mode == "streaming":
            await self.run_streaming_mode()
        else:
            await self.run_polling_mode(interval)

    async def stop(self) -> None:
        """Stop the trading engine."""
        self.running = False
        await self.market_data.stop_stream()
        logger.info("[ENGINE] Stopped")


async def main(args) -> None:
    """Main entry point."""
    print("\n" + "=" * 70)
    print("ORDINIS - LIVE ALPACA PAPER TRADING (ENHANCED)")
    print("=" * 70 + "\n")

    # Verify credentials
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not api_secret:
        logger.error("[ERROR] Missing Alpaca credentials!")
        logger.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        return

    # Initialize components
    logger.info("[INIT] Initializing Alpaca components...")

    broker = AlpacaBrokerAdapter(
        api_key=api_key,
        api_secret=api_secret,
        paper=True,
    )

    market_data = AlpacaMarketDataAdapter(
        api_key=api_key,
        api_secret=api_secret,
    )

    # Test connection
    connected = await broker.connect()
    if not connected:
        logger.error("[ERROR] Failed to connect to Alpaca")
        return

    # Get account info
    account = await broker.get_account()
    logger.info(f"[ACCOUNT] Equity: ${float(account.get('equity', 0)):,.2f}")
    logger.info(f"[ACCOUNT] Buying Power: ${float(account.get('buying_power', 0)):,.2f}")

    # Initialize strategy
    strategy = MACrossoverStrategy(fast_period=args.fast_ma, slow_period=args.slow_ma)
    logger.info(f"[STRATEGY] MA Crossover ({args.fast_ma}/{args.slow_ma})")

    # Create trading engine
    engine = LiveTradingEngine(
        broker=broker,
        market_data=market_data,
        strategy=strategy,
        symbol=args.symbol,
        position_size_usd=args.position_size,
    )

    # Start trading
    logger.info(f"\n[START] Trading {args.symbol} with ${args.position_size:,.0f} positions")
    logger.info(f"[START] Mode: {args.mode}")
    logger.info("Press Ctrl+C to stop\n")

    await engine.start(mode=args.mode, interval=args.interval)

    # Final summary
    final_account = await broker.get_account()
    positions = await broker.get_positions()

    print("\n" + "=" * 70)
    print("TRADING SESSION COMPLETE")
    print("=" * 70)
    print(f"Final Equity: ${float(final_account.get('equity', 0)):,.2f}")
    print(f"Open Positions: {len(positions)}")
    print(f"Orders Placed: {engine.order_count}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live paper trading with Alpaca")
    parser.add_argument("--symbol", default="SPY", help="Symbol to trade")
    parser.add_argument(
        "--mode", default="polling", choices=["polling", "streaming"], help="Trading mode"
    )
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds")
    parser.add_argument("--position-size", type=float, default=10000.0, help="Position size in USD")
    parser.add_argument("--fast-ma", type=int, default=20, help="Fast MA period")
    parser.add_argument("--slow-ma", type=int, default=50, help="Slow MA period")

    args = parser.parse_args()
    asyncio.run(main(args))
