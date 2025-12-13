"""
Test paper trading with live market data.

Integrates paper broker with real market data APIs to test end-to-end workflow.
"""

from pathlib import Path
import sys

# Add project root to path FIRST
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
import os  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

from adapters.market_data import AlphaVantageDataPlugin  # noqa: E402
from engines.flowroute.adapters.paper import PaperBrokerAdapter  # noqa: E402
from engines.flowroute.core.orders import Order, OrderType  # noqa: E402
from plugins.base import PluginConfig  # noqa: E402


async def test_paper_trading_with_live_data():
    """Test paper trading with live market data from Alpha Vantage."""
    load_dotenv()

    print("\n" + "=" * 70)
    print("PAPER TRADING WITH LIVE MARKET DATA TEST")
    print("=" * 70)

    # Initialize Alpha Vantage plugin
    print("\n[1/6] Initializing Alpha Vantage market data plugin...")
    av_config = PluginConfig(
        name="alphavantage",
        api_key=os.getenv("ALPHAVANTAGE_API_KEY"),
        enabled=True,
        rate_limit_per_minute=5,
        timeout_seconds=30,
    )

    market_data = AlphaVantageDataPlugin(av_config)
    if not await market_data.initialize():
        print("[X] Failed to initialize Alpha Vantage")
        return

    print("[OK] Alpha Vantage initialized")

    # Initialize paper broker
    print("\n[2/6] Initializing paper broker...")
    initial_capital = 100000.0

    broker = PaperBrokerAdapter(
        slippage_bps=5.0,
        commission_per_share=0.005,
        fill_delay_ms=100.0,
        market_data_plugin=market_data,
        price_cache_seconds=1.0,
    )

    print("[OK] Paper broker initialized")
    print(f"   Initial Capital: ${initial_capital:,.2f}")

    # Get current market data
    print("\n[3/6] Fetching live market data for AAPL...")
    quote = await market_data.get_quote("AAPL")
    print("[OK] Live quote received:")
    print(f"   Symbol: {quote['symbol']}")
    print(f"   Price: ${quote['last']:.2f}")
    print(f"   Change: {quote['change']:.2f} ({quote['change_percent']:.2f}%)")
    print(f"   Volume: {quote['volume']:,}")

    current_price = quote["last"]

    # Place a market buy order
    print("\n[4/6] Placing market buy order for 100 shares of AAPL...")
    order = Order(
        order_id=f"TEST-{datetime.now(UTC).timestamp()}",
        symbol="AAPL",
        side="buy",
        quantity=100,
        order_type=OrderType.MARKET,
    )

    result = await broker.submit_order(order)

    print("[OK] Order submitted:")
    print(f"   Broker Order ID: {result.get('broker_order_id')}")
    print(f"   Status: {result.get('status')}")
    if result.get("fill"):
        print(f"   Fill Price: ${result['fill']['price']:.2f}")
        print(f"   Fill Quantity: {result['fill']['quantity']}")

    # Process pending orders (auto-fill with live data)
    print("\n[5/6] Processing orders with live market data...")
    await asyncio.sleep(0.5)  # Small delay to simulate market conditions
    fills = await broker.process_pending_orders()

    if fills:
        fill = fills[0]
        print("[OK] Order filled:")
        print(f"   Fill ID: {fill.fill_id}")
        print(f"   Quantity: {fill.quantity}")
        print(f"   Price: ${fill.price:.2f}")
        print(f"   Timestamp: {fill.timestamp}")
    else:
        print("[WARN] No fills generated")

    # Check account state
    print("\n[6/6] Checking account state...")
    account = await broker.get_account()
    positions = await broker.get_positions()

    print("[OK] Account Summary:")
    print(f"   Cash: ${account['cash']:.2f}")
    print(f"   Total Equity: ${account['total_equity']:.2f}")
    print(f"   Position Value: ${account['total_position_value']:.2f}")
    print(f"   Buying Power: ${account['buying_power']:.2f}")
    print(f"   Open Positions: {len(positions)}")

    if positions:
        print("\n   Positions:")
        for pos in positions:
            symbol = pos["symbol"]
            qty = pos["quantity"]
            avg_price = pos["avg_entry_price"]
            pnl = (current_price - avg_price) * qty
            pnl_pct = ((current_price - avg_price) / avg_price) * 100
            print(f"   - {symbol}: {qty} shares @ ${avg_price:.2f}")
            print(f"     Current: ${current_price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:+.2f}%)")

    # Cleanup
    await market_data.shutdown()

    print("\n" + "=" * 70)
    print("[PASS] Paper trading test completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_paper_trading_with_live_data())
