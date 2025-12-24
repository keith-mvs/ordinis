#!/usr/bin/env python3
"""
Ordinis v0.54 ATR-RSI - Live trading with ATR-Optimized RSI SignalCore model.

Features:
- Uses actual SignalCore ATR-Optimized RSI model
- Executes orders on Alpaca paper trading
- Dynamic position sizing based on account equity
- ATR-based stop loss and take profit
- Position tracking and management
"""

import asyncio
import logging
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add ordinis to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class V054AtrRsi:
    """V0.54 using ATR-Optimized RSI SignalCore model with order execution."""

    def __init__(self):
        """Initialize system."""
        self.session_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.bars_processed = 0
        self.signals_generated = 0
        self.orders_placed = 0
        self.orders_filled = 0

        # Position tracking
        self.positions = {}  # symbol -> position info
        self.pending_orders = {}  # order_id -> order info

        # Risk parameters (derived from account, not hardcoded)
        self.max_position_pct = None  # Set during initialization
        self.max_positions = None  # Set during initialization

        # Load credentials
        from load_alpaca_env_v2 import load_alpaca_credentials
        self.api_key, self.api_secret = load_alpaca_credentials()
        self.alpaca = None
        self.model = None
        self.account = None

    async def initialize(self):
        """Initialize system and load SignalCore model."""
        logger.info("="*60)
        logger.info(f"Ordinis v0.54 ATR-RSI Strategy - LIVE TRADING")
        logger.info(f"Session: {self.session_id}")
        logger.info("="*60)

        # Connect to Alpaca
        if self.api_key and self.api_secret:
            try:
                from alpaca_trade_api import REST

                self.alpaca = REST(
                    key_id=self.api_key,
                    secret_key=self.api_secret,
                    base_url='https://paper-api.alpaca.markets'
                )

                self.account = self.alpaca.get_account()
                clock = self.alpaca.get_clock()

                equity = float(self.account.equity)
                buying_power = float(self.account.buying_power)

                logger.info(f"[OK] Connected to Alpaca Paper Trading")
                logger.info(f"  Equity: ${equity:,.2f}")
                logger.info(f"  Buying Power: ${buying_power:,.2f}")
                logger.info(f"  Market Open: {clock.is_open}")

                # Calculate dynamic risk parameters from account
                # Max positions = sqrt(equity / 1000) - portfolio theory
                self.max_positions = max(3, int(math.sqrt(equity / 1000)))
                # Max position size = equity / max_positions * 0.9 (90% allocation)
                self.max_position_pct = 0.9 / self.max_positions

                logger.info(f"\nRisk Parameters (derived from account):")
                logger.info(f"  Max Positions: {self.max_positions}")
                logger.info(f"  Max Position Size: {self.max_position_pct*100:.1f}% of equity")
                logger.info(f"  Max $ Per Position: ${equity * self.max_position_pct:,.2f}")

                # Load existing positions
                await self.sync_positions()

            except Exception as e:
                logger.error(f"Alpaca connection failed: {e}")
                return False

        # Load ATR-Optimized RSI model from SignalCore
        try:
            from ordinis.engines.signalcore.models.atr_optimized_rsi import ATROptimizedRSIModel
            from ordinis.engines.signalcore.core.model import ModelConfig

            config = ModelConfig(
                model_id="atr_rsi_v054",
                model_type="technical",
                parameters={
                    "rsi_period": 14,
                    "atr_period": 14,
                    "use_optimized": True,
                }
            )

            self.model = ATROptimizedRSIModel(config)
            logger.info(f"\n[OK] Loaded SignalCore model: ATROptimizedRSI")

        except ImportError as e:
            logger.error(f"Failed to load ATROptimizedRSI model: {e}")
            return False

        return True

    async def sync_positions(self):
        """Sync positions from Alpaca."""
        try:
            alpaca_positions = self.alpaca.list_positions()
            self.positions = {}

            for pos in alpaca_positions:
                self.positions[pos.symbol] = {
                    'qty': int(pos.qty),
                    'avg_entry': float(pos.avg_entry_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                }

            if self.positions:
                logger.info(f"\nCurrent Positions ({len(self.positions)}):")
                for sym, pos in self.positions.items():
                    pl_str = f"+${pos['unrealized_pl']:.2f}" if pos['unrealized_pl'] >= 0 else f"-${abs(pos['unrealized_pl']):.2f}"
                    logger.info(f"  {sym}: {pos['qty']} shares @ ${pos['avg_entry']:.2f} ({pl_str})")

        except Exception as e:
            logger.error(f"Error syncing positions: {e}")

    def calculate_position_size(self, price, stop_loss):
        """Calculate position size based on risk."""
        try:
            equity = float(self.account.equity)
            buying_power = float(self.account.buying_power)

            # Max dollars for this position
            max_dollars = equity * self.max_position_pct

            # Risk per share (distance to stop loss)
            risk_per_share = abs(price - stop_loss)

            # Risk 1% of equity per trade
            risk_dollars = equity * 0.01

            # Shares based on risk
            if risk_per_share > 0:
                shares_by_risk = int(risk_dollars / risk_per_share)
            else:
                shares_by_risk = int(max_dollars / price)

            # Shares based on max position size
            shares_by_size = int(max_dollars / price)

            # Shares based on buying power
            shares_by_bp = int(buying_power * 0.9 / price)

            # Take minimum of all constraints
            shares = min(shares_by_risk, shares_by_size, shares_by_bp)

            # Ensure at least 1 share
            shares = max(1, shares)

            logger.info(f"  Position sizing: risk={shares_by_risk}, size={shares_by_size}, bp={shares_by_bp} -> {shares} shares")

            return shares

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1

    async def execute_entry(self, signal):
        """Execute entry order."""
        symbol = signal.symbol
        price = signal.price
        stop_loss = signal.metadata.get('stop_loss', price * 0.98)
        take_profit = signal.metadata.get('take_profit', price * 1.04)

        # Check if already in position
        if symbol in self.positions:
            logger.info(f"  Already in position for {symbol}, skipping entry")
            return None

        # Check max positions
        if len(self.positions) >= self.max_positions:
            logger.info(f"  Max positions ({self.max_positions}) reached, skipping entry")
            return None

        # Calculate position size
        shares = self.calculate_position_size(price, stop_loss)

        try:
            # Submit market order
            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=shares,
                side='buy',
                type='market',
                time_in_force='day'
            )

            self.orders_placed += 1

            logger.info(f"\n  [ORDER PLACED] BUY {shares} {symbol}")
            logger.info(f"    Order ID: {order.id}")
            logger.info(f"    Type: Market")
            logger.info(f"    Status: {order.status}")

            # Track pending order
            self.pending_orders[order.id] = {
                'symbol': symbol,
                'side': 'buy',
                'qty': shares,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'signal': signal,
            }

            # Submit bracket orders for stop loss and take profit
            try:
                # Stop loss order
                stop_order = self.alpaca.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='sell',
                    type='stop',
                    stop_price=round(stop_loss, 2),
                    time_in_force='gtc'
                )
                logger.info(f"    Stop Loss: ${stop_loss:.2f} (Order: {stop_order.id})")

                # Take profit order
                tp_order = self.alpaca.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side='sell',
                    type='limit',
                    limit_price=round(take_profit, 2),
                    time_in_force='gtc'
                )
                logger.info(f"    Take Profit: ${take_profit:.2f} (Order: {tp_order.id})")

            except Exception as bracket_err:
                logger.warning(f"    Could not place bracket orders: {bracket_err}")

            return order

        except Exception as e:
            logger.error(f"  [ORDER FAILED] {symbol}: {e}")
            return None

    async def execute_exit(self, signal):
        """Execute exit order."""
        symbol = signal.symbol

        # Check if in position
        if symbol not in self.positions:
            logger.info(f"  No position in {symbol}, skipping exit")
            return None

        position = self.positions[symbol]
        shares = position['qty']

        try:
            # Cancel any existing orders for this symbol
            try:
                orders = self.alpaca.list_orders(status='open')
                for order in orders:
                    if order.symbol == symbol:
                        self.alpaca.cancel_order(order.id)
                        logger.info(f"  Cancelled pending order: {order.id}")
            except Exception as cancel_err:
                logger.debug(f"  Error cancelling orders: {cancel_err}")

            # Submit market sell order
            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=shares,
                side='sell',
                type='market',
                time_in_force='day'
            )

            self.orders_placed += 1

            # Calculate P&L
            entry_price = position['avg_entry']
            exit_price = signal.price
            pnl = (exit_price - entry_price) * shares
            pnl_pct = (exit_price - entry_price) / entry_price * 100

            logger.info(f"\n  [ORDER PLACED] SELL {shares} {symbol}")
            logger.info(f"    Order ID: {order.id}")
            logger.info(f"    Entry: ${entry_price:.2f} -> Exit: ${exit_price:.2f}")
            logger.info(f"    P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%)")
            logger.info(f"    Reason: {signal.metadata.get('reason', 'signal')}")

            return order

        except Exception as e:
            logger.error(f"  [ORDER FAILED] {symbol}: {e}")
            return None

    async def execute_signal(self, signal):
        """Execute a trading signal."""
        from ordinis.engines.signalcore.core.signal import SignalType

        logger.info(f"\nExecuting signal for {signal.symbol}...")

        if signal.signal_type == SignalType.ENTRY:
            return await self.execute_entry(signal)
        elif signal.signal_type == SignalType.EXIT:
            return await self.execute_exit(signal)
        else:
            logger.info(f"  Unknown signal type: {signal.signal_type}")
            return None

    def bars_to_dataframe(self, bars):
        """Convert Alpaca bars to pandas DataFrame."""
        if not bars:
            return None

        data = []
        for bar in bars:
            data.append({
                'timestamp': bar['t'],
                'open': float(bar['o']),
                'high': float(bar['h']),
                'low': float(bar['l']),
                'close': float(bar['c']),
                'volume': float(bar['v']),
            })

        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df

    async def process_symbol(self, symbol):
        """Process a symbol using the ATR-RSI model."""
        if not self.alpaca or not self.model:
            return None

        try:
            bars = self.alpaca.get_bars(
                symbol,
                '5Min',
                limit=50
            )._raw

            if not bars or len(bars) < 30:
                return None

            self.bars_processed += len(bars)

            df = self.bars_to_dataframe(bars)
            if df is None:
                return None

            timestamp = datetime.now(UTC)
            signal = await self.model.generate(symbol, df, timestamp)

            if signal:
                self.signals_generated += 1
                logger.info(f"\n{'*'*50}")
                logger.info(f"SIGNAL: {signal.direction.name} {symbol}")
                logger.info(f"  Type: {signal.signal_type.name}")
                logger.info(f"  Price: ${signal.price:.2f}")
                logger.info(f"  Confidence: {signal.confidence:.2f}")

                if signal.metadata:
                    if 'rsi' in signal.metadata:
                        logger.info(f"  RSI: {signal.metadata['rsi']:.1f}")
                    if 'stop_loss' in signal.metadata:
                        logger.info(f"  Stop Loss: ${signal.metadata['stop_loss']:.2f}")
                    if 'take_profit' in signal.metadata:
                        logger.info(f"  Take Profit: ${signal.metadata['take_profit']:.2f}")
                    if 'reason' in signal.metadata:
                        logger.info(f"  Reason: {signal.metadata['reason']}")

                logger.info(f"{'*'*50}")

                # Execute the signal
                order = await self.execute_signal(signal)
                if order:
                    self.orders_filled += 1

                return signal

            return None

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return None

    def load_universe_from_config(self):
        """Load symbol universe from strategy config."""
        import yaml
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'strategies' / 'atr_optimized_rsi.yaml'

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            symbols = list(config.get('symbols', {}).keys())
            logger.info(f"Loaded {len(symbols)} symbols from {config_path.name}")
            return symbols

        except Exception as e:
            logger.warning(f"Could not load config: {e}, using default universe")
            return [
                'AMD', 'NVDA', 'TSLA', 'META', 'GOOGL', 'AMZN', 'AAPL', 'MSFT',
                'COIN', 'DKNG', 'CRWD', 'NET', 'DDOG', 'ZS', 'OKTA', 'TWLO',
                'SQ', 'PYPL', 'SHOP', 'SNOW', 'PLTR', 'U', 'RBLX', 'ABNB',
                'SPY', 'QQQ', 'IWM'
            ]

    async def scan_universe(self):
        """Scan stock universe for signals."""
        universe = self.load_universe_from_config()

        logger.info(f"\nScanning {len(universe)} symbols...")

        signals = []
        for symbol in universe:
            signal = await self.process_symbol(symbol)
            if signal:
                signals.append(signal)
            await asyncio.sleep(0.2)  # Rate limiting

        return signals

    async def run(self):
        """Main run loop."""
        if not await self.initialize():
            logger.error("Initialization failed")
            return

        logger.info("\n" + "-"*60)
        logger.info("Starting ATR-RSI LIVE TRADING...")
        logger.info("Press Ctrl+C to stop")
        logger.info("-"*60)

        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"Cycle {cycle} - {datetime.now(UTC).strftime('%H:%M:%S UTC')}")
                logger.info(f"{'='*60}")

                # Refresh account and positions
                self.account = self.alpaca.get_account()
                await self.sync_positions()

                # Scan for signals and execute
                signals = await self.scan_universe()

                # Summary
                logger.info(f"\nCycle {cycle} Summary:")
                logger.info(f"  Account Equity: ${float(self.account.equity):,.2f}")
                logger.info(f"  Positions: {len(self.positions)}/{self.max_positions}")
                logger.info(f"  Bars Processed: {self.bars_processed}")
                logger.info(f"  Total Signals: {self.signals_generated}")
                logger.info(f"  Orders Placed: {self.orders_placed}")

                wait_time = 60
                logger.info(f"\nNext scan in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

            except KeyboardInterrupt:
                logger.info("\nShutdown requested")
                break
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                await asyncio.sleep(60)

        # Final summary
        logger.info("\n" + "="*60)
        logger.info("SESSION COMPLETE")
        logger.info(f"  Total Signals: {self.signals_generated}")
        logger.info(f"  Orders Placed: {self.orders_placed}")
        logger.info(f"  Final Equity: ${float(self.account.equity):,.2f}")
        logger.info("="*60)


async def main():
    """Entry point."""
    system = V054AtrRsi()
    await system.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)