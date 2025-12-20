"""
Run live paper trading with Alpaca broker.

This script connects to Alpaca's paper trading API and executes
a simple MA crossover strategy in real-time during market hours.

Prerequisites:
- Set ALPACA_API_KEY and ALPACA_SECRET_KEY in environment variables
- Install: pip install alpaca-py

Usage:
    python scripts/trading/run_live_alpaca.py
"""

import asyncio
from datetime import datetime, time
import logging
import os
import sys

sys.path.insert(0, "src")

from ordinis.engines.flowroute.adapters.alpaca import AlpacaBrokerAdapter
from ordinis.engines.flowroute.core.orders import Order, OrderType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SimpleMAStrategy:
    """Simple moving average crossover strategy."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prices: list[float] = []
        self.prev_signal: str | None = None

    def update(self, price: float) -> str | None:
        """Update with new price and generate signal."""
        self.prices.append(price)

        if len(self.prices) < self.slow_period:
            return None

        # Keep only necessary data
        if len(self.prices) > self.slow_period * 2:
            self.prices = self.prices[-self.slow_period * 2 :]

        fast_ma = sum(self.prices[-self.fast_period :]) / self.fast_period
        slow_ma = sum(self.prices[-self.slow_period :]) / self.slow_period

        # Generate signal on crossover
        if fast_ma > slow_ma and self.prev_signal != "bullish":
            self.prev_signal = "bullish"
            return "buy"
        if fast_ma < slow_ma and self.prev_signal != "bearish":
            self.prev_signal = "bearish"
            return "sell"

        return None


def is_market_hours() -> bool:
    """Check if current time is within market hours (9:30 AM - 4:00 PM ET)."""
    now = datetime.now().time()
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= now <= market_close


async def get_current_position(broker: AlpacaBrokerAdapter, symbol: str) -> int:
    """Get current position quantity for symbol."""
    positions = await broker.get_positions()
    for pos in positions:
        if pos.get("symbol") == symbol:
            return int(pos.get("quantity", 0))
    return 0


async def trading_loop(
    broker: AlpacaBrokerAdapter,
    symbol: str,
    strategy: SimpleMAStrategy,
    position_size_usd: float = 10000.0,
    check_interval: int = 60,
) -> None:
    """Main trading loop."""
    logger.info(f"Starting trading loop for {symbol}")
    logger.info(f"Position size: ${position_size_usd:,.2f}")
    logger.info(f"Check interval: {check_interval}s")

    order_count = 0

    try:
        while True:
            # Check market hours
            if not is_market_hours():
                logger.info("Outside market hours, waiting...")
                await asyncio.sleep(300)  # Check every 5 min
                continue

            # Get current quote
            quote = await broker.get_quote(symbol)
            if not quote:
                logger.warning(f"No quote data for {symbol}")
                await asyncio.sleep(check_interval)
                continue

            current_price = quote.get("last", quote.get("bid", 0))
            logger.info(f"{symbol} price: ${current_price:.2f}")

            # Update strategy
            signal = strategy.update(current_price)

            if signal:
                # Get current position
                current_qty = await get_current_position(broker, symbol)
                logger.info(f"Signal: {signal.upper()} | Current position: {current_qty} shares")

                # Execute signal
                if signal == "buy" and current_qty == 0:
                    # Calculate quantity
                    qty = int(position_size_usd / current_price)
                    if qty > 0:
                        order = Order(
                            order_id=f"ORDER-{order_count:04d}",
                            symbol=symbol,
                            side="buy",
                            quantity=qty,
                            order_type=OrderType.MARKET,
                        )
                        result = await broker.submit_order(order)
                        if result["success"]:
                            logger.info(
                                f"✅ BUY order submitted: {qty} shares @ ${current_price:.2f}"
                            )
                            order_count += 1
                        else:
                            logger.error(f"❌ BUY order failed: {result.get('error')}")

                elif signal == "sell" and current_qty > 0:
                    order = Order(
                        order_id=f"ORDER-{order_count:04d}",
                        symbol=symbol,
                        side="sell",
                        quantity=current_qty,
                        order_type=OrderType.MARKET,
                    )
                    result = await broker.submit_order(order)
                    if result["success"]:
                        logger.info(
                            f"✅ SELL order submitted: {current_qty} shares @ ${current_price:.2f}"
                        )
                        order_count += 1
                    else:
                        logger.error(f"❌ SELL order failed: {result.get('error')}")

            await asyncio.sleep(check_interval)

    except KeyboardInterrupt:
        logger.info("\n⛔ Trading loop stopped by user")
    except Exception as e:
        logger.error(f"❌ Trading loop error: {e}", exc_info=True)


async def main() -> None:
    """Main entry point."""
    print("\n" + "=" * 70)
    print("ORDINIS - LIVE ALPACA PAPER TRADING")
    print("=" * 70 + "\n")

    # Verify environment variables
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not api_secret:
        logger.error("❌ Missing Alpaca credentials!")
        logger.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        return

    # Initialize broker
    logger.info("🔌 Connecting to Alpaca paper trading...")
    broker = AlpacaBrokerAdapter(
        api_key=api_key,
        api_secret=api_secret,
        paper=True,  # Paper trading mode
    )

    # Test connection
    connected = await broker.connect()
    if not connected:
        logger.error("❌ Failed to connect to Alpaca")
        return

    # Get account info
    account = await broker.get_account()
    logger.info(f"💰 Account Equity: ${float(account.get('equity', 0)):,.2f}")
    logger.info(f"💵 Buying Power: ${float(account.get('buying_power', 0)):,.2f}")

    # Initialize strategy
    strategy = SimpleMAStrategy(fast_period=20, slow_period=50)
    logger.info("📊 Strategy: MA Crossover (20/50)")

    # Run trading loop
    logger.info("\n🚀 Starting live trading loop...")
    logger.info("Press Ctrl+C to stop\n")

    await trading_loop(
        broker=broker,
        symbol="SPY",
        strategy=strategy,
        position_size_usd=10000.0,
        check_interval=60,  # Check every 60 seconds
    )

    # Final account summary
    final_account = await broker.get_account()
    positions = await broker.get_positions()

    print("\n" + "=" * 70)
    print("TRADING SESSION COMPLETE")
    print("=" * 70)
    print(f"Final Equity: ${float(final_account.get('equity', 0)):,.2f}")
    print(f"Open Positions: {len(positions)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
