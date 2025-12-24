#!/usr/bin/env python
"""
Launch Ordinis v0.53 Alpaca Live with new account credentials.
Properly loads both API key and SECRET from Windows User environment.
Enhanced with traceable logging and comprehensive metrics.
"""
import os
import sys
import subprocess
from pathlib import Path

def get_user_env_variable(var_name):
    """Get environment variable from Windows User scope."""
    try:
        result = subprocess.run(
            ['pwsh', '-NoProfile', '-Command',
             f'[System.Environment]::GetEnvironmentVariable("{var_name}", "User")'],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception as e:
        print(f"Error getting {var_name}: {e}")
        return None

def main():
    """Main launcher."""
    try:
        print("="*60)
        print("ORDINIS v0.53 ALPACA LIVE - NEW ACCOUNT LAUNCHER")
        print("="*60)

        # Load API keys from Windows User environment
        api_key = get_user_env_variable('APCA_API_KEY_ID')
        api_secret = get_user_env_variable('APCA_API_SECRET_KEY')
        massive_key = get_user_env_variable('MASSIVE_API_KEY')

        if not all([api_key, api_secret]):
            print("ERROR: Missing Alpaca API credentials")
            print("Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY in User environment")
            sys.exit(1)

        # Set in current process
        os.environ['APCA_API_KEY_ID'] = api_key
        os.environ['APCA_API_SECRET_KEY'] = api_secret
        if massive_key:
            os.environ['MASSIVE_API_KEY'] = massive_key

        print("\n=== API Keys Loaded from User Environment ===")
        print(f"APCA_API_KEY_ID: {api_key[:10]}...")
        print(f"APCA_API_SECRET_KEY: {'SET' if api_secret else 'NOT SET'}")
        print(f"MASSIVE_API_KEY: {'SET' if massive_key else 'NOT SET'}")

        # Verify Alpaca connection
        print("\n=== Verifying Alpaca Connection ===")
        from alpaca.trading.client import TradingClient

        api = TradingClient(api_key, api_secret, paper=True)

        # Get account info
        account = api.get_account()

        print(f"[OK] Account Status: {account.status}")
        print(f"[OK] Account Equity: ${float(account.equity):,.2f}")
        print(f"[OK] Cash Balance: ${float(account.cash):,.2f}")
        print(f"[OK] Buying Power: ${float(account.buying_power):,.2f}")
        print(f"[OK] Day Trading BP: ${float(account.daytrading_buying_power):,.2f}")

        # Check positions
        positions = api.get_all_positions()
        print(f"[OK] Open Positions: {len(positions)}")

        if len(positions) > 0:
            print("\nExisting positions:")
            for pos in positions[:5]:  # Show first 5 positions
                print(f"  - {pos.symbol}: {pos.qty} shares @ ${float(pos.avg_entry_price):.2f}")

        # Account type check
        if float(account.equity) > 50000:
            print(f"\nWARNING: This appears to be the old account (${float(account.equity):,.2f})")
            print("Please check your Windows User environment variables")
        else:
            print(f"\n[CONFIRMED] NEW ACCOUNT: ${float(account.equity):,.2f}")

        if account.status != "ACTIVE":
            print(f"\nWARNING: Account status is {account.status}")
            print("Account must be ACTIVE to trade")

        print("\n" + "="*50)
        print("     LAUNCHING ORDINIS v0.53 ALPACA LIVE")
        print("     Enhanced with Traceable Logging")
        print("="*50)

        # Launch the v053 script as subprocess
        result = subprocess.run([sys.executable, 'ordinis-v053-alpaca-live.py'])
        sys.exit(result.returncode)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()