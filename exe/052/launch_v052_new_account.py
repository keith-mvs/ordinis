#!/usr/bin/env python
"""
Launch Ordinis v0.52 Enhanced with new account credentials.
Properly loads both API key and SECRET from Windows User environment.
"""

import os
import sys
import subprocess

print("="*60)
print("ORDINIS v0.52 ENHANCED - NEW ACCOUNT LAUNCHER")
print("="*60)

# Force load from Windows User environment variables
def get_windows_env_var(name):
    """Get environment variable from Windows User environment."""
    cmd = ['powershell', '-NoProfile', '-Command',
           f'[System.Environment]::GetEnvironmentVariable("{name}", "User")']

    result = subprocess.run(cmd, capture_output=True, text=True)
    value = result.stdout.strip() if result.returncode == 0 else None

    if value:
        print(f"Loaded {name}: {value[:15]}..." if "SECRET" not in name else f"Loaded {name}: ***HIDDEN***")
    else:
        print(f"ERROR: Could not load {name}")

    return value

print("\n1. Loading credentials from Windows User Environment...")
print("-" * 50)

# Load all three keys
api_key = get_windows_env_var("APCA_API_KEY_ID")
api_secret = get_windows_env_var("APCA_API_SECRET_KEY")
massive_key = get_windows_env_var("MASSIVE_API_KEY")

if not api_key or not api_secret:
    print("\nERROR: Missing Alpaca API credentials!")
    sys.exit(1)

# Force set in Python environment
os.environ["APCA_API_KEY_ID"] = api_key
os.environ["APCA_API_SECRET_KEY"] = api_secret
if massive_key:
    os.environ["MASSIVE_API_KEY"] = massive_key

print("\n2. Verifying NEW Alpaca Account...")
print("-" * 50)

try:
    from alpaca_trade_api import REST

    # Create API connection with the new keys
    api = REST(
        key_id=api_key,
        secret_key=api_secret,
        base_url='https://paper-api.alpaca.markets'
    )

    # Get account info
    account = api.get_account()

    print(f"[OK] Account Status: {account.status}")
    print(f"[OK] Account Equity: ${float(account.equity):,.2f}")
    print(f"[OK] Cash Balance: ${float(account.cash):,.2f}")
    print(f"[OK] Buying Power: ${float(account.buying_power):,.2f}")
    print(f"[OK] Day Trading BP: ${float(account.daytrading_buying_power):,.2f}")

    # Check positions
    positions = api.list_positions()
    print(f"[OK] Open Positions: {len(positions)}")

    if len(positions) > 0:
        print("\nExisting positions:")
        for pos in positions[:5]:  # Show first 5
            print(f"  - {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")

    # Verify it's the right account
    if float(account.equity) > 50000:
        print(f"\nWARNING: This appears to be the old account (${float(account.equity):,.2f})")
        print("Please check your Windows User environment variables")
    else:
        print(f"\n[CONFIRMED] NEW ACCOUNT: ${float(account.equity):,.2f}")

    if account.status != "ACTIVE":
        print(f"\nWARNING: Account status is {account.status}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

except Exception as e:
    print(f"\nERROR connecting to Alpaca: {e}")
    print("Please verify your API keys in Windows User environment variables")
    sys.exit(1)

print("\n3. Launching Ordinis v0.52 Enhanced Demo...")
print("="*60)

# Now launch the enhanced demo directly
try:
    # Import and run the enhanced demo
    import asyncio
    from ordinis_v052_enhanced import main

    asyncio.run(main())

except ImportError:
    # If import fails, try executing the file directly
    print("Running enhanced demo directly...")
    import asyncio

    # Load the module by executing the file
    with open('ordinis-v052-enhanced.py', 'r') as f:
        code = compile(f.read(), 'ordinis-v052-enhanced.py', 'exec')
        exec(code)

except Exception as e:
    print(f"Error launching demo: {e}")
    sys.exit(1)