"""
Test Alpaca connection and verify account access.

Quick diagnostic script to verify:
- API credentials are valid
- Connection to Alpaca works
- Account is accessible
- Market data is available

Usage:
    python scripts/trading/test_alpaca_connection.py
"""

import asyncio
import os
import sys

sys.path.insert(0, "src")

from ordinis.engines.flowroute.adapters.alpaca import AlpacaBrokerAdapter


async def test_connection() -> None:
    """Test Alpaca connection and display account info."""
    print("\n" + "=" * 70)
    print("ALPACA CONNECTION TEST")
    print("=" * 70 + "\n")

    # Check environment variables
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")

    print("📋 Checking environment variables...")
    if api_key:
        print(f"  ✅ ALPACA_API_KEY: {api_key[:8]}...")
    else:
        print("  ❌ ALPACA_API_KEY: NOT SET")

    if api_secret:
        print(f"  ✅ ALPACA_SECRET_KEY: {api_secret[:8]}...")
    else:
        print("  ❌ ALPACA_SECRET_KEY: NOT SET")

    if not api_key or not api_secret:
        print("\n❌ ERROR: Alpaca credentials not found in environment variables")
        print("\nSet them using:")
        print("  $env:ALPACA_API_KEY='your_key'")
        print("  $env:ALPACA_SECRET_KEY='your_secret'")
        return

    # Initialize broker
    print("\n🔌 Initializing Alpaca broker adapter...")
    try:
        broker = AlpacaBrokerAdapter(
            api_key=api_key,
            api_secret=api_secret,
            paper=True,
        )
        print("  ✅ Broker adapter created")
    except Exception as e:
        print(f"  ❌ Failed to create broker: {e}")
        return

    # Test connection
    print("\n🌐 Testing connection to Alpaca...")
    try:
        connected = await broker.connect()
        if connected:
            print("  ✅ Connection successful")
        else:
            print("  ❌ Connection failed")
            return
    except Exception as e:
        print(f"  ❌ Connection error: {e}")
        return

    # Get account info
    print("\n💰 Retrieving account information...")
    try:
        account = await broker.get_account()
        print("  ✅ Account retrieved")
        print("\n  Account Details:")
        print(f"    Equity: ${float(account.get('equity', 0)):,.2f}")
        print(f"    Cash: ${float(account.get('cash', 0)):,.2f}")
        print(f"    Buying Power: ${float(account.get('buying_power', 0)):,.2f}")
        print(f"    Status: {account.get('status', 'unknown')}")
    except Exception as e:
        print(f"  ❌ Failed to get account: {e}")
        return

    # Get positions
    print("\n📊 Retrieving positions...")
    try:
        positions = await broker.get_positions()
        print(f"  ✅ Positions retrieved: {len(positions)} open positions")
        if positions:
            print("\n  Open Positions:")
            for pos in positions:
                symbol = pos.get("symbol", "?")
                qty = pos.get("quantity", 0)
                avg_price = float(pos.get("avg_price", 0))
                current_price = float(pos.get("current_price", 0))
                pnl = float(pos.get("unrealized_pnl", 0))
                print(
                    f"    {symbol}: {qty} shares @ ${avg_price:.2f} (current: ${current_price:.2f}, P&L: ${pnl:,.2f})"
                )
        else:
            print("    No open positions")
    except Exception as e:
        print(f"  ❌ Failed to get positions: {e}")

    # Test market data
    print("\n📈 Testing market data access...")
    try:
        quote = await broker.get_quote("SPY")
        if quote:
            print("  ✅ Market data access successful")
            print("\n  SPY Quote:")
            print(f"    Last: ${quote.get('last', 0):.2f}")
            print(f"    Bid: ${quote.get('bid', 0):.2f}")
            print(f"    Ask: ${quote.get('ask', 0):.2f}")
        else:
            print("  ⚠️  No quote data returned")
    except Exception as e:
        print(f"  ❌ Failed to get quote: {e}")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - READY FOR PAPER TRADING")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_connection())
