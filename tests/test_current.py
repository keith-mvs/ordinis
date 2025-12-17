"""Test with current credentials."""

import os

from alpaca.trading.client import TradingClient

# Check what Python sees
print("Environment check:")
print(f"  ALPACA_API_KEY from os.getenv: {os.getenv('ALPACA_API_KEY', 'NOT SET')}")
print(
    f"  ALPACA_SECRET_KEY from os.getenv: {os.getenv('ALPACA_SECRET_KEY', 'NOT SET')[:20] if os.getenv('ALPACA_SECRET_KEY') else 'NOT SET'}..."
)

print("\nTesting connection...")
try:
    client = TradingClient(
        api_key=os.getenv("ALPACA_API_KEY"),
        secret_key=os.getenv("ALPACA_SECRET_KEY"),
        paper=True,
    )
    account = client.get_account()
    print("\n[SUCCESS] Connected!")
    print(f"Equity: ${float(account.equity):,.2f}")
    print(f"Status: {account.status}")
except Exception as e:
    print(f"\n[FAILED] {e}")
