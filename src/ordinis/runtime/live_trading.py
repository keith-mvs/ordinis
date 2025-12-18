#!/usr/bin/env python3
"""
Live Trading Runtime.

Complete integration of ATR-Optimized RSI strategy with Ordinis runtime.
Supports paper trading and live trading modes.

Usage:
    # Paper trading
    python -m ordinis.runtime.live_trading --mode paper --config configs/strategies/atr_optimized_rsi.yaml

    # Simulated trading (no broker, for testing)
    python -m ordinis.runtime.live_trading --mode simulated --config configs/strategies/atr_optimized_rsi.yaml
"""

import argparse
import asyncio
from datetime import UTC, datetime
import logging
from pathlib import Path
import sys

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ordinis.adapters.broker.broker import (
    AlpacaBroker,
    BrokerAdapter,
    Order,
    OrderSide,
    OrderType,
    SimulatedBroker,
)
from ordinis.engines.flowroute.adapters.alpaca_data import AlpacaMarketDataAdapter
from ordinis.engines.signalcore.core.signal import Direction, Signal, SignalType
from ordinis.engines.signalcore.strategy_loader import StrategyLoader

logger = logging.getLogger(__name__)


class LiveTradingRuntime:
    """
    Live trading runtime that connects strategy to broker.

    Handles:
    - Real-time signal generation
    - Order execution
    - Position management
    - Risk management
    - Logging and monitoring
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        strategy_loader: StrategyLoader,
        data_adapter: AlpacaMarketDataAdapter | None = None,
        mode: str = "paper",
        poll_interval: int = 60,
    ):
        """
        Initialize live trading runtime.

        Args:
            broker: Broker implementation.
            strategy_loader: Strategy loader with models.
            data_adapter: Market data adapter (created automatically if None).
            mode: Trading mode (paper, simulated, live).
            poll_interval: Seconds between data polls.
        """
        self.broker = broker
        self.loader = strategy_loader
        self.mode = mode
        self.poll_interval = poll_interval
        self._data_adapter = data_adapter

        self.running = False
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.start_equity: float | None = None

        # Risk limits
        self.max_daily_loss_pct = 2.0
        self.max_position_pct = 5.0
        self.max_concurrent = 5

    async def connect(self) -> bool:
        """Connect to broker and initialize data adapter."""
        if await self.broker.connect():
            account = await self.broker.get_account()
            self.start_equity = account.equity
            logger.info(f"Connected to broker. Equity: ${account.equity:,.2f}")

            # Initialize data adapter if not provided
            if self._data_adapter is None and self.mode != "simulated":
                try:
                    self._data_adapter = AlpacaMarketDataAdapter()
                    logger.info("Initialized Alpaca market data adapter")
                except Exception as e:
                    logger.warning(f"Could not initialize data adapter: {e}")

            return True
        return False

    async def disconnect(self):
        """Disconnect from broker."""
        await self.broker.disconnect()
        logger.info("Disconnected from broker")

    async def get_market_data(self, symbol: str, bars: int = 100) -> pd.DataFrame | None:
        """
        Fetch market data for a symbol using Alpaca data adapter.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            bars: Number of bars to fetch

        Returns:
            DataFrame with OHLCV data or None if unavailable
        """
        if self._data_adapter is None:
            logger.warning(f"No data adapter available for {symbol}")
            return None

        try:
            logger.debug(f"Fetching {bars} bars for {symbol}...")

            # Get historical bars (5-minute timeframe to match backtest)
            df = self._data_adapter.get_historical_bars(
                symbol=symbol,
                timeframe="5Min",
                limit=bars,
            )

            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            logger.info(f"Got {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")
            return None

    async def process_symbol(self, symbol: str) -> Signal | None:
        """
        Process a single symbol and generate signal.

        Args:
            symbol: Stock symbol.

        Returns:
            Signal if generated, None otherwise.
        """
        model = self.loader.get_model(symbol)
        if not model:
            return None

        # Get market data
        df = await self.get_market_data(symbol)
        if df is None or len(df) < 50:
            logger.debug(f"Insufficient data for {symbol}")
            return None

        # Check regime
        should_trade, reason = self.loader.should_trade(symbol, df)
        if not should_trade:
            logger.info(f"Skipping {symbol}: {reason}")
            return None

        # Generate signal
        try:
            signal = await model.generate(symbol, df, datetime.now(UTC))

            if signal and signal.signal_type != SignalType.HOLD:
                logger.info(
                    f"Signal for {symbol}: {signal.signal_type.value} "
                    f"confidence={signal.confidence:.2f}"
                )
                return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")

        return None

    async def execute_signal(self, symbol: str, signal: Signal) -> bool:
        """
        Execute a trading signal.

        Args:
            symbol: Stock symbol.
            signal: Generated signal.

        Returns:
            True if order placed successfully.
        """
        # Check risk limits
        if not await self.check_risk_limits(symbol, signal):
            return False

        # Calculate position size
        account = await self.broker.get_account()
        risk_params = self.loader.get_risk_params(symbol)
        max_position = account.equity * (risk_params["max_position_pct"] / 100)

        # Get current price (from metadata or estimate)
        price = signal.metadata.get("entry_price", 0)
        if price <= 0:
            logger.warning(f"No price available for {symbol}")
            return False

        # Calculate shares
        shares = int(max_position / price)
        if shares <= 0:
            logger.warning(f"Position size too small for {symbol}")
            return False

        # Determine order side based on signal direction
        # ENTRY signals use direction (LONG=BUY, SHORT=SELL)
        if signal.signal_type == SignalType.ENTRY:
            if signal.direction == Direction.LONG:
                side = OrderSide.BUY
            elif signal.direction == Direction.SHORT:
                side = OrderSide.SELL
            else:
                logger.warning(f"Unknown direction for {symbol}")
                return False
        else:
            # EXIT signals handled in manage_positions
            return False

        # Create order
        order = Order(
            symbol=symbol,
            side=side,
            quantity=shares,
            order_type=OrderType.MARKET,
            limit_price=None,
        )

        # Submit order
        try:
            result = await self.broker.submit_order(order)
            if result:
                self.trade_count += 1
                logger.info(
                    f"Order placed: {side.value} {shares} {symbol} "
                    f"(order_id: {result.order_id})"
                )
                return True

        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")

        return False

    async def check_risk_limits(self, symbol: str, signal: Signal) -> bool:
        """Check if trade passes risk limits."""
        account = await self.broker.get_account()
        positions = await self.broker.get_positions()

        # Check max concurrent positions
        if len(positions) >= self.max_concurrent:
            logger.warning(f"Max concurrent positions ({self.max_concurrent}) reached")
            return False

        # Check daily loss limit
        if self.start_equity:
            daily_change = (account.equity - self.start_equity) / self.start_equity * 100
            if daily_change < -self.max_daily_loss_pct:
                logger.warning(
                    f"Daily loss limit ({self.max_daily_loss_pct}%) reached: {daily_change:.2f}%"
                )
                return False

        # Check if we already have a position in this symbol
        if symbol in positions:
            logger.info(f"Already have position in {symbol}")
            return False

        return True

    async def manage_positions(self):
        """Manage existing positions (check stops, exits)."""
        positions_list = await self.broker.get_positions()

        for position in positions_list:
            symbol = position.symbol
            model = self.loader.get_model(symbol)
            if not model:
                continue

            # Get current data for exit signals
            df = await self.get_market_data(symbol)
            if df is None:
                continue

            # Generate signal to check for exit
            try:
                signal = await model.generate(symbol, df, datetime.now(UTC))

                # Check for exit conditions (EXIT signal or SELL for short positions)
                if signal and signal.signal_type == SignalType.EXIT and position.quantity > 0:
                    # Exit long position
                    exit_order = Order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=int(position.quantity),
                        order_type=OrderType.MARKET,
                        limit_price=None,
                    )
                    result = await self.broker.submit_order(exit_order)
                    if result:
                        logger.info(f"Exit signal: Sold {position.quantity} {symbol}")
                        self.trade_count += 1

            except Exception as e:
                logger.error(f"Error managing position {symbol}: {e}", exc_info=True)

    async def run_loop(self):
        """Main trading loop."""
        logger.info(f"Starting {self.mode} trading loop...")
        self.running = True

        symbols = self.loader.get_symbols()
        logger.info(f"Trading {len(symbols)} symbols: {symbols}")

        while self.running:
            try:
                # Get account status
                account = await self.broker.get_account()
                logger.info(
                    f"Account: Equity=${account.equity:,.2f} "
                    f"Cash=${account.cash:,.2f} "
                    f"BP=${account.buying_power:,.2f}"
                )

                # Manage existing positions
                await self.manage_positions()

                # Process each symbol
                for symbol in symbols:
                    signal = await self.process_symbol(symbol)

                    if signal and signal.signal_type != SignalType.HOLD:
                        await self.execute_signal(symbol, signal)

                    # Small delay between symbols
                    await asyncio.sleep(1)

                # Log summary
                positions = await self.broker.get_positions()
                logger.info(
                    f"Loop complete. Trades: {self.trade_count}, Positions: {len(positions)}"
                )

                # Wait for next poll
                await asyncio.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)

    def stop(self):
        """Stop the trading loop."""
        self.running = False
        logger.info("Stopping trading loop...")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Live Trading Runtime")
    parser.add_argument(
        "--mode",
        choices=["paper", "simulated", "live"],
        default="simulated",
        help="Trading mode",
    )
    parser.add_argument(
        "--config",
        default="configs/strategies/atr_optimized_rsi.yaml",
        help="Strategy config path",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between data polls",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load strategy
    loader = StrategyLoader()
    if not loader.load_strategy(args.config):
        logger.error(f"Failed to load strategy from {args.config}")
        return 1

    # Create broker
    if args.mode == "simulated":
        broker = SimulatedBroker(initial_cash=100_000)
        logger.info("Using simulated broker (no real trades)")
    elif args.mode == "paper":
        # AlpacaBroker handles credentials internally via ordinis.utils.env
        # which reads from Windows User environment (source of truth)
        broker = AlpacaBroker(paper=True)
        logger.info("Using Alpaca paper trading")
    elif args.mode == "live":
        logger.error("Live trading not enabled. Use paper mode for testing.")
        return 1
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1

    # Create runtime
    runtime = LiveTradingRuntime(
        broker=broker,
        strategy_loader=loader,
        mode=args.mode,
        poll_interval=args.poll_interval,
    )

    # Connect and run
    if not await runtime.connect():
        logger.error("Failed to connect to broker")
        return 1

    try:
        await runtime.run_loop()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await runtime.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
