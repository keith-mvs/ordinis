#!/usr/bin/env python
"""
Launch Ordinis v0.52 Enhanced with new account credentials.
Loads API keys from Windows User environment variables.
"""

import os
import sys
import subprocess

# Get keys from Windows User environment variables
def get_user_env_var(name):
    """Get environment variable from Windows User environment."""
    result = subprocess.run(
        ['powershell', '-NoProfile', '-Command',
         f'[System.Environment]::GetEnvironmentVariable("{name}", "User")'],
        capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None

print("=== Loading API Keys from User Environment Variables ===")

# Load the keys
api_key = get_user_env_var("APCA_API_KEY_ID")
api_secret = get_user_env_var("APCA_API_SECRET_KEY")
massive_key = get_user_env_var("MASSIVE_API_KEY")

if api_key:
    print(f"APCA_API_KEY_ID: {api_key[:10]}...")
    os.environ["APCA_API_KEY_ID"] = api_key
else:
    print("APCA_API_KEY_ID: NOT FOUND")

if api_secret:
    print("APCA_API_SECRET_KEY: SET")
    os.environ["APCA_API_SECRET_KEY"] = api_secret
else:
    print("APCA_API_SECRET_KEY: NOT FOUND")

if massive_key:
    print("MASSIVE_API_KEY: SET")
    os.environ["MASSIVE_API_KEY"] = massive_key
else:
    print("MASSIVE_API_KEY: NOT FOUND")

# Verify the account
print("\n=== Verifying Account ===")
try:
    from alpaca_trade_api import REST

    api = REST(api_key, api_secret, 'https://paper-api.alpaca.markets')
    account = api.get_account()

    print(f"Account Status: {account.status}")
    print(f"Equity: ${float(account.equity):,.2f}")
    print(f"Cash: ${float(account.cash):,.2f}")
    print(f"Buying Power: ${float(account.buying_power):,.2f}")

    positions = api.list_positions()
    print(f"Open Positions: {len(positions)}")

    if account.status != "ACTIVE":
        print(f"\nWARNING: Account status is {account.status}, not ACTIVE")

    print("\n" + "="*50)
    print("Launching Ordinis v0.52 Enhanced Demo")
    print("="*50)

    # Launch the enhanced demo
    exec(open('ordinis-v052-enhanced.py').read())

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)