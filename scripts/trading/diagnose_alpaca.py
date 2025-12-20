"""
Detailed Alpaca authentication diagnostic.

Tests different authentication methods and provides troubleshooting info.
"""

import os
import sys

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.trading.client import TradingClient

    print("[OK] alpaca-py library imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import alpaca-py: {e}")
    print("Install with: pip install alpaca-py")
    sys.exit(1)


def test_credentials():
    """Test Alpaca credentials with different methods."""
    print("\n" + "=" * 70)
    print("ALPACA AUTHENTICATION DIAGNOSTIC")
    print("=" * 70 + "\n")

    # Check environment
    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_SECRET_KEY")

    print("1. Environment Variables:")
    if api_key:
        print(f"   ALPACA_API_KEY: {api_key[:10]}... (length: {len(api_key)})")
    else:
        print("   ALPACA_API_KEY: [NOT SET]")

    if api_secret:
        print(f"   ALPACA_SECRET_KEY: {api_secret[:10]}... (length: {len(api_secret)})")
    else:
        print("   ALPACA_SECRET_KEY: [NOT SET]")

    if not api_key or not api_secret:
        print("\n[ERROR] Credentials not found in environment")
        return

    # Test paper trading connection
    print("\n2. Testing Paper Trading Connection:")
    try:
        client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=True,
        )
        account = client.get_account()
        print("   [OK] Paper trading connected successfully!")
        print(f"   Account ID: {account.id}")
        print(f"   Equity: ${float(account.equity):,.2f}")
        print(f"   Status: {account.status}")
        print(f"   Buying Power: ${float(account.buying_power):,.2f}")
    except Exception as e:
        print(f"   [ERROR] Paper trading failed: {e}")
        print(f"   Error type: {type(e).__name__}")

        # Check if it's an auth error
        error_str = str(e).lower()
        if "unauthorized" in error_str or "401" in error_str:
            print("\n   [INFO] This appears to be an authentication error.")
            print("   Possible causes:")
            print("   * API keys are for LIVE trading, not paper trading")
            print("   * API secret key is incorrect or expired")
            print("   * Keys were regenerated in Alpaca dashboard")
            print("\n   To fix:")
            print("   1. Log into https://app.alpaca.markets/paper/dashboard/overview")
            print("   2. Go to 'Your API Keys' section")
            print("   3. Regenerate paper trading keys")
            print("   4. Update your environment variables")

    # Test live trading connection (should fail with paper keys)
    print("\n3. Testing Live Trading Connection (should fail with paper keys):")
    try:
        client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=False,
        )
        account = client.get_account()
        print("   [WARNING] Live trading connected - you may be using LIVE keys!")
        print(f"   Account ID: {account.id}")
        print(f"   Equity: ${float(account.equity):,.2f}")
    except Exception as e:
        print(f"   [OK] Live trading failed (expected with paper keys): {type(e).__name__}")

    # Test data client
    print("\n4. Testing Market Data Access:")
    try:
        from alpaca.data.requests import StockLatestQuoteRequest

        data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret,
        )
        request = StockLatestQuoteRequest(symbol_or_symbols="SPY")
        quotes = data_client.get_stock_latest_quote(request)
        if "SPY" in quotes:
            quote = quotes["SPY"]
            print("   [OK] Market data access successful!")
            print(f"   SPY: Bid ${float(quote.bid_price):.2f} / Ask ${float(quote.ask_price):.2f}")
    except Exception as e:
        print(f"   [ERROR] Market data failed: {e}")

    print("\n" + "=" * 70)
    print("Diagnostic complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_credentials()
