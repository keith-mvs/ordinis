"""
Run paper trading with strategy integration.

This script demonstrates the full workflow:
1. Initialize paper broker with market data
2. Load strategy
3. Process market data bars
4. Generate signals and submit orders
5. Track fills and P&L

Can be run with:
- Mock market data (default, no API key needed)
- Sample CSV data (historical simulation)
- Live market data (requires IEX/Polygon API key)
"""

import asyncio
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, "src")

from engines.flowroute.adapters.paper import PaperBrokerAdapter
from engines.flowroute.core.orders import Fill, Order, OrderType


class MockMarketData:
    """Mock market data plugin using sample CSV data."""

    def __init__(self, data_path: str = "data/sample_spy_trending_up.csv"):
        """Load sample data."""
        self.data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.current_idx = 0
        self.symbol = "SPY"

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Return current bar as quote."""
        if self.current_idx >= len(self.data):
            self.current_idx = 0

        row = self.data.iloc[self.current_idx]
        return {
            "symbol": symbol,
            "bid": row["close"] * 0.9999,
            "ask": row["close"] * 1.0001,
            "last": row["close"],
        }

    def advance(self) -> bool:
        """Move to next bar."""
        self.current_idx += 1
        return self.current_idx < len(self.data)

    def get_current_bar(self) -> dict[str, Any] | None:
        """Get current bar data."""
        if self.current_idx >= len(self.data):
            return None
        row = self.data.iloc[self.current_idx]
        return {
            "timestamp": self.data.index[self.current_idx],
            "close": row["close"],
        }


class MACrossoverStrategy:
    """Simple MA crossover strategy for paper trading."""

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.prices: list[float] = []
        self.prev_fast_ma: float | None = None
        self.prev_slow_ma: float | None = None

    def update(self, close_price: float) -> str | None:
        """Update with new price, return signal if any."""
        self.prices.append(close_price)

        if len(self.prices) < self.slow_period:
            return None

        fast_ma = sum(self.prices[-self.fast_period :]) / self.fast_period
        slow_ma = sum(self.prices[-self.slow_period :]) / self.slow_period

        signal = self._check_crossover(fast_ma, slow_ma)
        self.prev_fast_ma = fast_ma
        self.prev_slow_ma = slow_ma

        return signal

    def _check_crossover(self, fast_ma: float, slow_ma: float) -> str | None:
        """Check for MA crossover signal."""
        if self.prev_fast_ma is None or self.prev_slow_ma is None:
            return None

        if self.prev_fast_ma <= self.prev_slow_ma and fast_ma > slow_ma:
            return "buy"
        if self.prev_fast_ma >= self.prev_slow_ma and fast_ma < slow_ma:
            return "sell"
        return None


def print_session_summary(
    bar_num: int,
    order_count: int,
    fills: list[Fill],
    account: dict[str, Any],
    positions: list[dict[str, Any]],
    initial_capital: float,
) -> dict[str, Any]:
    """Print session summary and return results."""
    print("\n[SESSION SUMMARY]")
    print(f"  Bars Processed: {bar_num}")
    print(f"  Orders Placed: {order_count}")
    print(f"  Fills Executed: {len(fills)}")

    print("\n[ACCOUNT]")
    print(f"  Starting Capital: ${initial_capital:,.2f}")
    print(f"  Final Cash: ${account['cash']:,.2f}")
    print(f"  Position Value: ${account['total_position_value']:,.2f}")
    print(f"  Total Equity: ${account['total_equity']:,.2f}")

    pnl = account["total_equity"] - initial_capital
    pnl_pct = (pnl / initial_capital) * 100
    print("\n[P&L]")
    print(f"  Absolute: ${pnl:,.2f}")
    print(f"  Percent: {pnl_pct:.2f}%")

    _print_positions(positions)
    _print_fills(fills)
    _print_trade_analysis(fills)

    print("\n" + "=" * 70)
    print("[SESSION COMPLETE]")
    print("=" * 70 + "\n")

    return {"final_equity": account["total_equity"], "pnl": pnl, "pnl_pct": pnl_pct}


def _print_positions(positions: list[dict[str, Any]]) -> None:
    """Print open positions."""
    if not positions:
        return
    print("\n[OPEN POSITIONS]")
    for pos in positions:
        print(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_price']:.2f}")
        print(f"    Unrealized P&L: ${pos['unrealized_pnl']:.2f}")


def _print_fills(fills: list[Fill]) -> None:
    """Print fill history."""
    print("\n[FILL HISTORY]")
    for fill in fills:
        print(f"  {fill.side.upper():4s} {fill.quantity:4d} {fill.symbol} @ ${fill.price:.2f}")


def _print_trade_analysis(fills: list[Fill]) -> None:
    """Print trade analysis."""
    if len(fills) < 2:
        return

    print("\n[TRADE ANALYSIS]")
    buy_fills = [f for f in fills if f.side == "buy"]
    sell_fills = [f for f in fills if f.side == "sell"]

    trades_completed = min(len(buy_fills), len(sell_fills))
    winning_trades = sum(
        1 for i in range(trades_completed) if (sell_fills[i].price - buy_fills[i].price) > 0
    )

    print(f"  Round-Trip Trades: {trades_completed}")
    if trades_completed > 0:
        print(f"  Win Rate: {(winning_trades / trades_completed) * 100:.1f}%")


async def run_trading_loop(
    broker: PaperBrokerAdapter,
    market_data: MockMarketData,
    strategy: MACrossoverStrategy,
    max_bars: int,
    position_size_pct: float,
) -> tuple[int, int, int]:
    """Run the main trading loop."""
    order_count = 0
    bar_num = 0
    current_position = 0

    print("\n[TRADING LOG]")
    print("-" * 70)

    while bar_num < max_bars:
        bar = market_data.get_current_bar()
        if bar is None:
            break

        signal = strategy.update(bar["close"])
        result = await _process_signal(
            signal, current_position, broker, bar, order_count, position_size_pct
        )

        if result:
            order_count, current_position = result
            _log_trade(bar_num, bar, signal, broker.get_fills()[-1])

        market_data.advance()
        bar_num += 1

    print("-" * 70)
    return bar_num, order_count, current_position


async def _process_signal(
    signal: str | None,
    current_position: int,
    broker: PaperBrokerAdapter,
    bar: dict[str, Any],
    order_count: int,
    position_size_pct: float,
) -> tuple[int, int] | None:
    """Process a trading signal."""
    if signal == "buy" and current_position == 0:
        account = await broker.get_account()
        qty = int(account["cash"] * position_size_pct / bar["close"])
        if qty > 0:
            order = Order(
                order_id=f"ORDER-{order_count:04d}",
                symbol="SPY",
                side="buy",
                quantity=qty,
                order_type=OrderType.MARKET,
            )
            result = await broker.submit_order(order)
            if result["status"] == "filled":
                return order_count + 1, qty

    elif signal == "sell" and current_position > 0:
        order = Order(
            order_id=f"ORDER-{order_count:04d}",
            symbol="SPY",
            side="sell",
            quantity=current_position,
            order_type=OrderType.MARKET,
        )
        result = await broker.submit_order(order)
        if result["status"] == "filled":
            return order_count + 1, 0

    return None


def _log_trade(bar_num: int, bar: dict[str, Any], signal: str | None, fill: Fill) -> None:
    """Log a trade execution."""
    side = "BUY " if signal == "buy" else "SELL"
    ts = bar["timestamp"].strftime("%Y-%m-%d")
    print(f"  Bar {bar_num:3d} | {ts} | {side} {fill.quantity:4d} SPY @ ${fill.price:.2f}")


async def run_paper_trading_session(
    max_bars: int = 100,
    initial_capital: float = 100000.0,
    position_size_pct: float = 0.5,
) -> dict[str, Any]:
    """Run a paper trading session."""
    print("\n" + "=" * 70)
    print("PAPER TRADING SESSION")
    print("=" * 70)

    market_data = MockMarketData()
    broker = PaperBrokerAdapter(
        market_data_plugin=market_data,
        slippage_bps=5.0,
        commission_per_share=0.005,
    )
    broker.reset(initial_capital)

    strategy = MACrossoverStrategy(fast_period=20, slow_period=50)

    print("\n[CONFIG]")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Position Size: {position_size_pct * 100:.0f}% of capital")
    print("  Strategy: MA Crossover (20/50)")

    bar_num, order_count, _ = await run_trading_loop(
        broker, market_data, strategy, max_bars, position_size_pct
    )

    account = await broker.get_account()
    positions = await broker.get_positions()
    fills = broker.get_fills()

    result = print_session_summary(bar_num, order_count, fills, account, positions, initial_capital)
    result["orders"] = order_count
    result["fills"] = len(fills)

    return result


async def main() -> None:
    """Run paper trading demo."""
    result = await run_paper_trading_session(
        max_bars=500,
        initial_capital=100000.0,
        position_size_pct=0.5,
    )
    print(f"Final Result: {result['pnl_pct']:.2f}% return")


if __name__ == "__main__":
    asyncio.run(main())
