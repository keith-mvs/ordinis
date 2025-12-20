"""
Production live paper trading using Massive real-time data and Alpaca broker.

This script combines the best of both:
- Massive for high-quality real-time market data
- Alpaca for paper trading order execution

Features:
- Multi-strategy portfolio management
- Real-time minute bars from Massive
- 4 strategies: MA Crossover, RSI, Breakout, VWAP
- Signal aggregation (consensus, majority, weighted, any)
- Risk management integration
- Market hours validation
- Comprehensive logging

Usage:
    # Run with default weighted aggregation
    python scripts/trading/run_live_massive_alpaca.py --symbol SPY

    # Require all strategies to agree (conservative)
    python scripts/trading/run_live_massive_alpaca.py --mode consensus

    # Majority vote (balanced)
    python scripts/trading/run_live_massive_alpaca.py --mode majority

    # Any strategy can trigger (aggressive)
    python scripts/trading/run_live_massive_alpaca.py --mode any
"""

import argparse
import asyncio
import logging
import os
import signal
import sys

sys.path.insert(0, "src")

from ordinis.engines.flowroute.adapters.alpaca import AlpacaBrokerAdapter
from ordinis.engines.flowroute.adapters.massive_data import MassiveMarketDataAdapter
from ordinis.engines.flowroute.core.orders import Order, OrderType
from ordinis.engines.flowroute.portfolio_manager import PortfolioManager
from ordinis.engines.flowroute.strategies import (
    BreakoutStrategy,
    MACrossoverStrategy,
    RSIMeanReversionStrategy,
    VWAPStrategy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class LiveTradingEngine:
    """Multi-strategy live trading engine with Massive data and Alpaca execution."""

    def __init__(
        self,
        broker: AlpacaBrokerAdapter,
        market_data: MassiveMarketDataAdapter,
        portfolio_manager: PortfolioManager,
        symbol: str = "SPY",
        position_size_usd: float = 10000.0,
        check_interval: int = 60,
    ):
        self.broker = broker
        self.market_data = market_data
        self.portfolio_manager = portfolio_manager
        self.symbol = symbol
        self.position_size_usd = position_size_usd
        self.check_interval = check_interval
        self.order_count = 0
        self.running = False
        self.last_price = 0.0
        self.signal_count = 0

    async def get_current_position(self) -> int:
        """Get current position quantity."""
        positions = await self.broker.get_positions()
        for pos in positions:
            if pos.get("symbol") == self.symbol:
                return int(pos.get("quantity", 0))
        return 0

    async def execute_signal(self, signal, current_price: float) -> None:
        """
        Execute trading signal from portfolio manager.

        Args:
            signal: PortfolioSignal with direction, strength, confidence, etc.
            current_price: Current market price
        """
        current_qty = await self.get_current_position()

        # Log detailed signal information
        logger.info(
            f"[PORTFOLIO SIGNAL] {signal.direction.upper()} | "
            f"Strength: {signal.strength:.2f} | "
            f"Confidence: {signal.confidence:.2%} | "
            f"Consensus: {signal.consensus:.1%}"
        )
        logger.info(f"[CONTRIBUTING] {', '.join(signal.contributing_strategies)}")
        for reason in signal.reasons:
            logger.info(f"  • {reason}")

        if signal.direction == "buy" and current_qty == 0:
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
                        f"[✓ BUY] {qty} {self.symbol} @ ${current_price:.2f} | "
                        f"Order ID: {result.get('broker_order_id')} | "
                        f"Confidence: {signal.confidence:.1%}"
                    )
                    self.order_count += 1
                    self.signal_count += 1
                else:
                    logger.error(f"[ERROR] BUY failed: {result.get('error')}")

        elif signal.direction == "sell" and current_qty > 0:
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
                    f"[✓ SELL] {current_qty} {self.symbol} @ ${current_price:.2f} | "
                    f"Order ID: {result.get('broker_order_id')} | "
                    f"Confidence: {signal.confidence:.1%}"
                )
                self.order_count += 1
                self.signal_count += 1
            else:
                logger.error(f"[ERROR] SELL failed: {result.get('error')}")

    async def run(self) -> None:
        """Run the multi-strategy trading engine."""
        self.running = True
        logger.info(f"[START] Multi-strategy trading on {self.symbol}")
        logger.info(f"[CONFIG] Check interval: {self.check_interval}s")
        logger.info(f"[CONFIG] Position size: ${self.position_size_usd:,.0f}")

        # Display portfolio status
        status = self.portfolio_manager.get_status()
        logger.info(f"[PORTFOLIO] {status['total_strategies']} strategies loaded")
        logger.info(f"[PORTFOLIO] Aggregation mode: {status['aggregation_mode']}")

        # Load historical data to initialize all strategies
        logger.info("[INIT] Loading historical data for strategies...")
        max_lookback = 100  # Sufficient for all strategies
        prices = await self.market_data.get_price_history(
            symbol=self.symbol,
            periods=max_lookback,
            timeframe="1Min",
        )

        if prices:
            logger.info(f"[INIT] Loaded {len(prices)} historical bars")
            for price in prices:
                self.portfolio_manager.update(price, volume=1000.0)  # Dummy volume for init
            logger.info(
                f"[INIT] {self.portfolio_manager.get_status()['ready_strategies']}/"
                f"{status['total_strategies']} strategies ready"
            )
        else:
            logger.warning("[INIT] No historical data, strategies will warm up live")

        try:
            while self.running:
                # Check market hours
                if not self.market_data.is_market_open():
                    logger.info("[MARKET] Closed - waiting...")
                    await asyncio.sleep(300)  # Check every 5 min
                    continue

                # Get latest trade price
                trade = await self.market_data.get_latest_trade(self.symbol)
                if not trade:
                    logger.warning(f"[DATA] No trade data for {self.symbol}")
                    await asyncio.sleep(self.check_interval)
                    continue

                price = trade["price"]
                self.last_price = price

                # Log price update
                price_change = ""
                if (
                    self.portfolio_manager.strategies
                    and len(self.portfolio_manager.strategies[0].prices) > 0
                ):
                    # Get price from first strategy's history
                    try:
                        prev_price = self.portfolio_manager.strategies[0].prices[-1]
                        change = ((price - prev_price) / prev_price) * 100
                        price_change = f" ({change:+.2f}%)"
                    except:
                        pass

                logger.info(f"[PRICE] {self.symbol}: ${price:.2f}{price_change}")

                # Get volume if available
                volume = trade.get("size", 1000.0)

                # Update portfolio manager (all strategies)
                signal = self.portfolio_manager.update(price, volume=volume)
                if signal:
                    await self.execute_signal(signal, price)

                await asyncio.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\n[STOP] Trading stopped by user")
        except Exception as e:
            logger.error(f"[ERROR] Trading loop error: {e}", exc_info=True)
        finally:
            self.running = False

    async def stop(self) -> None:
        """Stop the trading engine."""
        self.running = False
        logger.info("[ENGINE] Stopped")


async def main(args) -> None:
    """Main entry point."""
    print("\n" + "=" * 70)
    print("ORDINIS - MULTI-STRATEGY LIVE PAPER TRADING")
    print("Massive Market Data + Alpaca Broker")
    print("=" * 70 + "\n")

    # Verify credentials
    alpaca_key = os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
    massive_key = os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")

    if not alpaca_key or not alpaca_secret:
        logger.error("[ERROR] Missing Alpaca credentials!")
        logger.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        return

    if not massive_key:
        logger.error("[ERROR] Missing Massive credentials!")
        logger.error("Set MASSIVE_API_KEY environment variable")
        return

    # Initialize components
    logger.info("[INIT] Initializing components...")

    broker = AlpacaBrokerAdapter(
        api_key=alpaca_key,
        api_secret=alpaca_secret,
        paper=True,
    )

    market_data = MassiveMarketDataAdapter(api_key=massive_key)

    # Test connections
    logger.info("[TEST] Testing Alpaca connection...")
    connected = await broker.connect()
    if not connected:
        logger.error("[ERROR] Failed to connect to Alpaca")
        return

    logger.info("[TEST] Testing Massive market data...")
    quote = await market_data.get_latest_quote(args.symbol)
    if not quote:
        logger.error(f"[ERROR] Failed to get quote for {args.symbol}")
        return
    logger.info(f"[TEST] Massive OK - {args.symbol} @ ${quote['bid']:.2f}/{quote['ask']:.2f}")

    # Get account info
    account = await broker.get_account()
    logger.info(f"[ACCOUNT] Equity: ${float(account.get('equity', 0)):,.2f}")
    logger.info(f"[ACCOUNT] Buying Power: ${float(account.get('buying_power', 0)):,.2f}")

    # Check market status
    if market_data.is_market_open():
        logger.info("[MARKET] Status: OPEN")
    else:
        logger.warning("[MARKET] Status: CLOSED - will wait for open")

    # Initialize all strategies (aggressive parameters for more signals)
    logger.info(f"\n[STRATEGIES] Initializing {args.mode} portfolio...")

    strategies = [
        MACrossoverStrategy(fast_period=5, slow_period=15, name="MA_5/15"),
        RSIMeanReversionStrategy(period=7, oversold=40, overbought=60, name="RSI_7"),
        BreakoutStrategy(lookback_period=10, breakout_threshold=0.002, name="Breakout_10"),
        VWAPStrategy(deviation_threshold=0.001, name="VWAP"),
    ]

    portfolio_manager = PortfolioManager(
        strategies=strategies,
        mode=args.mode,
        min_strategies_ready=2,  # At least 2 strategies must be initialized
    )

    # Create multi-strategy trading engine
    engine = LiveTradingEngine(
        broker=broker,
        market_data=market_data,
        portfolio_manager=portfolio_manager,
        symbol=args.symbol,
        position_size_usd=args.position_size,
        check_interval=args.interval,
    )

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("\n[SIGNAL] Shutdown signal received")
        loop.create_task(engine.stop())

    if sys.platform != "win32":
        loop.add_signal_handler(signal.SIGINT, signal_handler)
        loop.add_signal_handler(signal.SIGTERM, signal_handler)

    # Start trading
    logger.info("\n[GO] Starting multi-strategy live trading")
    logger.info("Press Ctrl+C to stop\n")

    await engine.run()

    # Final summary
    final_account = await broker.get_account()
    positions = await broker.get_positions()

    # Get portfolio stats
    portfolio_status = engine.portfolio_manager.get_status()

    print("\n" + "=" * 70)
    print("MULTI-STRATEGY TRADING SESSION COMPLETE")
    print("=" * 70)
    print(f"Final Equity: ${float(final_account.get('equity', 0)):,.2f}")
    print(f"Last Price: ${engine.last_price:.2f}")
    print(f"Portfolio Signals: {engine.signal_count}")
    print(f"Orders Executed: {engine.order_count}")
    print(f"Open Positions: {len(positions)}")

    print("\nStrategy Performance:")
    for strat_status in portfolio_status["strategy_status"]:
        signal_dir = strat_status["last_signal"] or "None"
        print(f"  {strat_status['name']:20} - Last Signal: {signal_dir}")

    if positions:
        print("\nOpen Positions:")
        for pos in positions:
            symbol = pos.get("symbol")
            qty = pos.get("quantity")
            pnl = pos.get("unrealized_pnl", 0)
            print(f"  {symbol}: {qty} shares | P&L: ${pnl:,.2f}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-strategy live paper trading with Massive + Alpaca",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Aggregation Modes:
  consensus  - All strategies must agree (most conservative)
  majority   - >50% of strategies must agree (balanced)
  weighted   - Weight by confidence scores (recommended)
  any        - Any single strategy can trigger (most aggressive)

Strategies Included (aggressive parameters for testing):
  • MA Crossover (5/15)     - Fast trend following
  • RSI (7)                 - Mean reversion (40/60)
  • Breakout (10 period)    - Momentum (0.2% threshold)
  • VWAP                    - Institutional reference (0.1% deviation)
        """,
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol to trade")
    parser.add_argument(
        "--interval", type=int, default=60, help="Polling interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--position-size", type=float, default=1000.0, help="Position size in USD (default: 1000)"
    )
    parser.add_argument(
        "--mode",
        default="weighted",
        choices=["consensus", "majority", "weighted", "any"],
        help="Signal aggregation mode (default: weighted)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\n[EXIT] Program terminated")
