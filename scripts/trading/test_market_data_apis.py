"""
Test market data API integrations.

Verifies that all configured market data APIs are working correctly.
"""

from pathlib import Path
import sys

# Add project root to path FIRST
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from datetime import UTC, datetime, timedelta
import os
import traceback

from adapters.market_data import (
    AlphaVantageDataPlugin,
    FinnhubDataPlugin,
    MassiveDataPlugin,
    TwelveDataPlugin,
)
from dotenv import load_dotenv
from plugins.base import PluginConfig


async def test_alphavantage():
    """Test Alpha Vantage API."""
    print("\n" + "=" * 60)
    print("Testing Alpha Vantage API")
    print("=" * 60)

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        print("[X] ALPHAVANTAGE_API_KEY not found in .env")
        return False

    config = PluginConfig(
        name="alphavantage_test",
        api_key=api_key,
        enabled=True,
        rate_limit_per_minute=5,
        timeout_seconds=30,
    )

    plugin = AlphaVantageDataPlugin(config)

    try:
        # Initialize
        print("\n[1/3] Initializing...")
        if not await plugin.initialize():
            print("[X] Failed to initialize Alpha Vantage plugin")
            return False
        print("[OK] Initialized successfully")

        # Get quote
        print("\n[2/3] Fetching quote for AAPL...")
        quote = await plugin.get_quote("AAPL")
        print("[OK] Quote received:")
        print(f"   Symbol: {quote['symbol']}")
        print(f"   Price: ${quote['last']:.2f}")
        print(f"   Change: {quote['change']:.2f} ({quote['change_percent']:.2f}%)")
        print(f"   Volume: {quote['volume']:,}")

        # Get company info
        print("\n[3/3] Fetching company info...")
        company = await plugin.get_company("AAPL")
        print("[OK] Company info received:")
        print(f"   Name: {company['name']}")
        print(f"   Sector: {company['sector']}")
        print(f"   Market Cap: ${company['market_cap']}")

        return True

    except Exception as e:
        print(f"[X] Error: {e}")
        traceback.print_exc()
        return False

    finally:
        await plugin.shutdown()


async def test_finnhub():
    """Test Finnhub API."""
    print("\n" + "=" * 60)
    print("Testing Finnhub API")
    print("=" * 60)

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        print("[X] FINNHUB_API_KEY not found in .env")
        return False

    config = PluginConfig(
        name="finnhub_test",
        api_key=api_key,
        enabled=True,
        rate_limit_per_minute=60,
        timeout_seconds=30,
    )

    plugin = FinnhubDataPlugin(config)

    try:
        # Initialize
        print("\n[1/4] Initializing...")
        if not await plugin.initialize():
            print("[X] Failed to initialize Finnhub plugin")
            return False
        print("[OK] Initialized successfully")

        # Get quote
        print("\n[2/4] Fetching quote for AAPL...")
        quote = await plugin.get_quote("AAPL")
        print("[OK] Quote received:")
        print(f"   Price: ${quote['last']:.2f}")
        print(f"   Change: {quote['change']:.2f} ({quote['change_percent']:.2f}%)")
        print(f"   High: ${quote['high']:.2f}")
        print(f"   Low: ${quote['low']:.2f}")

        # Get historical data
        print("\n[3/4] Fetching historical data (last 7 days)...")
        try:
            end = datetime.now(UTC)
            start = end - timedelta(days=7)
            bars = await plugin.get_historical("AAPL", start, end, "1d")
            print(f"[OK] Received {len(bars)} bars")
            if bars:
                latest = bars[-1]
                print(f"   Latest: {latest['timestamp']}")
                print(f"   Close: ${latest['close']:.2f}")
        except Exception as e:
            if "403" in str(e) or "Forbidden" in str(e):
                print("[SKIP] Historical data requires paid plan")
            else:
                raise

        # Get company info
        print("\n[4/4] Fetching company profile...")
        company = await plugin.get_company("AAPL")
        print("[OK] Company profile received:")
        print(f"   Name: {company['name']}")
        print(f"   Country: {company['country']}")
        print(f"   Market Cap: {company['market_cap']}M")

        return True

    except Exception as e:
        print(f"[X] Error: {e}")
        traceback.print_exc()
        return False

    finally:
        await plugin.shutdown()


async def test_massive():
    """Test Massive API."""
    print("\n" + "=" * 60)
    print("Testing Massive API")
    print("=" * 60)

    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        print("[X] MASSIVE_API_KEY not found in .env")
        return False

    config = PluginConfig(
        name="massive_test",
        api_key=api_key,
        enabled=True,
        rate_limit_per_minute=5,
        timeout_seconds=30,
    )

    plugin = MassiveDataPlugin(config)

    try:
        # Initialize
        print("\n[1/3] Initializing...")
        if not await plugin.initialize():
            print("[X] Failed to initialize Massive plugin")
            return False
        print("[OK] Initialized successfully")

        # Get previous close
        print("\n[2/3] Fetching previous close for AAPL...")
        prev_close = await plugin.get_previous_close("AAPL")
        print("[OK] Previous close received:")
        print(f"   Date: {prev_close['date']}")
        print(f"   Close: ${prev_close['close']:.2f}")
        print(f"   Volume: {prev_close['volume']:,}")

        # Get market status
        print("\n[3/3] Fetching market status...")
        status = await plugin.get_market_status()
        print("[OK] Market status received:")
        print(f"   Market: {status['market']}")

        return True

    except Exception as e:
        print(f"[X] Error: {e}")
        traceback.print_exc()
        return False

    finally:
        await plugin.shutdown()


async def test_twelvedata():
    """Test Twelve Data API."""
    print("\n" + "=" * 60)
    print("Testing Twelve Data API")
    print("=" * 60)

    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        print("[X] TWELVEDATA_API_KEY not found in .env")
        return False

    config = PluginConfig(
        name="twelvedata_test",
        api_key=api_key,
        enabled=True,
        rate_limit_per_minute=8,
        timeout_seconds=30,
    )

    plugin = TwelveDataPlugin(config)

    try:
        # Initialize
        print("\n[1/3] Initializing...")
        if not await plugin.initialize():
            print("[X] Failed to initialize Twelve Data plugin")
            return False
        print("[OK] Initialized successfully")

        # Get quote
        print("\n[2/3] Fetching quote for AAPL...")
        quote = await plugin.get_quote("AAPL")
        print("[OK] Quote received:")
        print(f"   Price: ${quote['last']:.2f}")
        print(f"   Change: {quote['change']:.2f} ({quote['change_percent']:.2f}%)")
        print(f"   Volume: {quote['volume']:,}")

        # Get company info
        print("\n[3/3] Fetching company profile...")
        company = await plugin.get_company("AAPL")
        print("[OK] Company profile received:")
        print(f"   Name: {company['name']}")
        print(f"   Exchange: {company['exchange']}")
        print(f"   Sector: {company['sector']}")

        return True

    except Exception as e:
        print(f"[X] Error: {e}")
        traceback.print_exc()
        return False

    finally:
        await plugin.shutdown()


async def main():
    """Run all API tests."""
    load_dotenv()

    print("\n" + "=" * 60)
    print("MARKET DATA API INTEGRATION TESTS")
    print("=" * 60)

    results = {}

    # Test Alpha Vantage
    results["Alpha Vantage"] = await test_alphavantage()
    await asyncio.sleep(1)

    # Test Finnhub
    results["Finnhub"] = await test_finnhub()
    await asyncio.sleep(1)

    # Test Massive
    results["Massive"] = await test_massive()
    await asyncio.sleep(1)

    # Test Twelve Data
    results["Twelve Data"] = await test_twelvedata()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for api, passed in results.items():
        status = "[OK] PASSED" if passed else "[X] FAILED"
        print(f"{api:20} {status}")

    passed_count = sum(results.values())
    total_count = len(results)
    print(f"\nTotal: {passed_count}/{total_count} APIs working")

    if passed_count == total_count:
        print("\n[PASS] All APIs are working correctly!")
    else:
        print("\n[WARN]  Some APIs failed. Check the logs above for details.")


if __name__ == "__main__":
    asyncio.run(main())
