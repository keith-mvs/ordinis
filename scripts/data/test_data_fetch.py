"""
Test data fetching from market data plugins.

This script verifies that we can fetch real historical data
for backtesting.
"""

import asyncio
from datetime import UTC, datetime, timedelta
import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def test_iex():
    """Test IEX Cloud data fetch."""
    print("\n" + "=" * 60)
    print("Testing IEX Cloud Data Fetch")
    print("=" * 60)

    api_key = os.getenv("IEX_API_KEY")
    if not api_key or api_key == "your_iex_api_key_here":
        print("[X] IEX_API_KEY not set in .env")
        print("    Sign up at https://iexcloud.io (free tier available)")
        return False

    try:
        from plugins.market_data.iex import IEXDataPlugin

        plugin = IEXDataPlugin(config={"api_key": api_key, "sandbox": False})

        print("\n[OK] IEX plugin initialized")

        # Test 1: Get quote
        print("\n[Test 1] Fetching current quote for SPY...")
        quote = await plugin.fetch_quote("SPY")
        print(f"[OK] Quote received: ${quote.get('latestPrice', 'N/A')}")

        # Test 2: Get historical data
        print("\n[Test 2] Fetching historical data (last 30 days)...")
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=30)

        hist_data = await plugin.fetch_historical_data(
            symbol="SPY",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            timeframe="1d",
        )

        if hist_data and len(hist_data) > 0:
            print(f"[OK] Historical data received: {len(hist_data)} bars")
            print(f"     Date range: {hist_data[0].get('date')} to {hist_data[-1].get('date')}")
            print(
                f"     Sample: Close=${hist_data[-1].get('close')}, Volume={hist_data[-1].get('volume'):,}"
            )
        else:
            print("[!] No historical data received")

        print("\n[SUCCESS] IEX plugin working!")
        return True

    except ImportError as e:
        print(f"[X] Import error: {e}")
        return False
    except Exception as e:
        print(f"[X] Error: {e}")
        return False


async def test_massive():
    """Test Massive data fetch."""
    print("\n" + "=" * 60)
    print("Testing Massive Data Fetch")
    print("=" * 60)

    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key or api_key == "your_massive_api_key_here":
        print("[X] MASSIVE_API_KEY not set in .env")
        print("    Sign up at https://massive.com")
        return False

    try:
        from ordinis.adapters.market_data.massive import MassiveDataPlugin
        from ordinis.plugins.base import PluginConfig

        config = PluginConfig(name="massive", options={"api_key": api_key})
        plugin = MassiveDataPlugin(config)

        print("\n[OK] Massive plugin initialized")

        # Test 1: Get quote
        print("\n[Test 1] Fetching current quote for SPY...")
        quote = await plugin.get_quote("SPY")
        print(f"[OK] Quote received: {quote}")

        # Test 2: Get historical data
        print("\n[Test 2] Fetching historical data (last 30 days)...")
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=30)

        hist_data = await plugin.fetch_historical_data(
            symbol="SPY",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            timeframe="1d",
        )

        if hist_data and len(hist_data) > 0:
            print(f"[OK] Historical data received: {len(hist_data)} bars")
            print(
                f"     Sample: Close=${hist_data[-1].get('close')}, Volume={hist_data[-1].get('volume'):,}"
            )
        else:
            print("[!] No historical data received")

        print("\n[SUCCESS] Massive plugin working!")
        return True

    except ImportError as e:
        print(f"[X] Import error: {e}")
        return False
    except Exception as e:
        print(f"[X] Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MARKET DATA FETCH TEST")
    print("=" * 60)

    results = {"iex": await test_iex(), "massive": await test_massive()}

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"IEX Cloud:  {'[OK] Working' if results['iex'] else '[X] Failed'}")
    print(f"Massive:    {'[OK] Working' if results['massive'] else '[X] Failed'}")

    if results["iex"] or results["massive"]:
        print("\n[SUCCESS] At least one data provider working - ready for backtests!")
        return True
    print("\n[X] No data providers working - need API keys")
    print("\nNext steps:")
    print("1. Sign up at https://iexcloud.io (free tier)")
    print("2. Add API key to .env file")
    print("3. Run this script again")
    return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
