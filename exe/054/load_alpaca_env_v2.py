#!/usr/bin/env python3
"""
Alpaca Environment Loader v2 - Handles multiple naming conventions.
Checks for both ALPACA_* and APCA_* variables.
"""
import os
import sys
import subprocess

def load_alpaca_credentials():
    """Load Alpaca credentials from environment variables."""

    # First check what's available
    alpaca_key = os.environ.get('ALPACA_API_KEY')
    alpaca_secret = os.environ.get('ALPACA_SECRET') or os.environ.get('ALPACA_SECRET_KEY')
    apca_key = os.environ.get('APCA_API_KEY_ID')
    apca_secret = os.environ.get('APCA_API_SECRET_KEY')

    # If ALPACA variables not found in process, try Windows user environment
    if not alpaca_key and sys.platform == 'win32':
        try:
            ps_cmd = '[System.Environment]::GetEnvironmentVariable("ALPACA_API_KEY", "User")'
            result = subprocess.run(
                ["powershell", "-Command", ps_cmd],
                capture_output=True,
                text=True
            )
            if result.stdout and result.stdout.strip():
                alpaca_key = result.stdout.strip()
                print(f"[INFO] Loaded ALPACA_API_KEY from Windows user environment")

            ps_cmd = '[System.Environment]::GetEnvironmentVariable("ALPACA_SECRET", "User")'
            result = subprocess.run(
                ["powershell", "-Command", ps_cmd],
                capture_output=True,
                text=True
            )
            if result.stdout and result.stdout.strip():
                alpaca_secret = result.stdout.strip()
                print(f"[INFO] Loaded ALPACA_SECRET from Windows user environment")
        except Exception as e:
            print(f"[WARNING] Could not check Windows user environment: {e}")

    # Determine which credentials to use
    if alpaca_key and alpaca_secret:
        # Prefer new ALPACA_* variables
        api_key = alpaca_key
        api_secret = alpaca_secret
        source = "ALPACA"
    elif apca_key and apca_secret:
        # Fall back to APCA_* variables
        api_key = apca_key
        api_secret = apca_secret
        source = "APCA (legacy)"
    else:
        print("[ERROR] No Alpaca credentials found")
        print("  Checked: ALPACA_API_KEY, ALPACA_SECRET, ALPACA_SECRET_KEY")
        print("  Also checked: APCA_API_KEY_ID, APCA_API_SECRET_KEY")
        return None, None

    # Set all naming conventions for compatibility
    os.environ['ALPACA_API_KEY'] = api_key
    os.environ['ALPACA_SECRET_KEY'] = api_secret
    os.environ['ALPACA_SECRET'] = api_secret
    os.environ['APCA_API_KEY_ID'] = api_key
    os.environ['APCA_API_SECRET_KEY'] = api_secret

    # Set base URL if not already set
    if not os.environ.get('ALPACA_BASE_URL'):
        os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'

    print(f"[OK] Alpaca credentials loaded from {source}")
    print(f"  API Key: {api_key[:8]}...")
    print(f"  Base URL: {os.environ['ALPACA_BASE_URL']}")

    return api_key, api_secret

if __name__ == "__main__":
    api_key, api_secret = load_alpaca_credentials()
    if api_key:
        print("[OK] Environment configured successfully")

        # Also write a PowerShell script to set these in the current session
        ps_script = f"""
# Set Alpaca environment variables for current session
$env:ALPACA_API_KEY = "{api_key}"
$env:ALPACA_SECRET_KEY = "{api_secret}"
$env:ALPACA_SECRET = "{api_secret}"
$env:ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
Write-Host "[OK] Alpaca environment variables set in current PowerShell session"
"""
        with open("set_alpaca_env.ps1", "w") as f:
            f.write(ps_script)
        print("\nTo set in current PowerShell session, run:")
        print("  . .\\set_alpaca_env.ps1")

        sys.exit(0)
    else:
        sys.exit(1)