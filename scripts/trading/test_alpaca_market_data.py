"""Test Alpaca market data functionality."""

import asyncio
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, "src")

from ordinis.engines.flowroute.adapters.alpaca_data import AlpacaMarketDataAdapter


async def test_market_data():
    """Test all market data features."""
    print("\n" + "=" * 70)
    print("ALPACA MARKET DATA TEST")
    print("=" * 70 + "\n")

    # Check credentials
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not api_secret:
        print("[ERROR] Alpaca credentials not found in environment")
        return

    print(f"[INFO] Using API Key: {api_key[:10]}...")

    adapter = AlpacaMarketDataAdapter(api_key=api_key, api_secret=api_secret)

    # 1. Test latest quote
    print("1. Testing Latest Quote...")
    quote = adapter.get_latest_quote("SPY")
    if quote:
        print("   [OK] SPY Quote:")
        print(f"        Bid: ${quote['bid']:.2f} ({quote['bid_size']} shares)")
        print(f"        Ask: ${quote['ask']:.2f} ({quote['ask_size']} shares)")
        print(f"        Time: {quote['timestamp']}")
    else:
        print("   [FAILED] Could not get quote")

    # 2. Test latest bar
    print("\n2. Testing Latest Bar...")
    bar = adapter.get_latest_bar("SPY")
    if bar:
        print("   [OK] SPY Bar:")
        print(f"        Open:   ${bar['open']:.2f}")
        print(f"        High:   ${bar['high']:.2f}")
        print(f"        Low:    ${bar['low']:.2f}")
        print(f"        Close:  ${bar['close']:.2f}")
        print(f"        Volume: {bar['volume']:,}")
        print(f"        Time:   {bar['timestamp']}")
    else:
        print("   [FAILED] Could not get bar")

    # 3. Test historical bars
    print("\n3. Testing Historical Bars...")
    df = adapter.get_historical_bars(
        symbol="SPY",
        timeframe="1Min",
        start=datetime.now() - timedelta(hours=2),
        limit=10,
    )
    if not df.empty:
        print(f"   [OK] Retrieved {len(df)} bars")
        print("\n   Last 5 bars:")
        print(df.tail(5).to_string(index=False))
    else:
        print("   [FAILED] Could not get historical bars")

    # 4. Test price history
    print("\n4. Testing Price History...")
    prices = adapter.get_price_history("SPY", periods=20, timeframe="1Min")
    if prices:
        print(f"   [OK] Retrieved {len(prices)} prices")
        print(f"        Last 5: {[f'${p:.2f}' for p in prices[-5:]]}")
    else:
        print("   [FAILED] Could not get price history")

    # 5. Test market open check
    print("\n5. Testing Market Hours Check...")
    is_open = adapter.is_market_open()
    print(f"   Market is: {'OPEN' if is_open else 'CLOSED'}")

    # 6. Test streaming (optional - only run for a few seconds)
    print("\n6. Testing Bar Streaming (5 seconds)...")
    print("   Starting stream...")

    bar_count = [0]  # Use list to modify in nested function

    def on_bar(bar_data):
        bar_count[0] += 1
        print(
            f"   [STREAM] {bar_data['symbol']}: ${bar_data['close']:.2f} @ {bar_data['timestamp']}"
        )

    try:
        # Run stream for 5 seconds
        stream_task = asyncio.create_task(adapter.stream_bars(["SPY"], on_bar))
        await asyncio.sleep(5)
        await adapter.stop_stream()
        stream_task.cancel()
        print(f"   [OK] Received {bar_count[0]} bars in 5 seconds")
    except Exception as e:
        print(f"   [INFO] Streaming test skipped or failed: {e}")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_market_data())
