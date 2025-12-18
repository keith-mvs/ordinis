"""
Environment variable utilities.

CRITICAL: Alpaca API keys are stored in Windows User environment as:
- APCA_API_KEY_ID (NOT ALPACA_API_KEY)
- APCA_API_SECRET_KEY (NOT ALPACA_API_SECRET)

Windows User env is SOURCE OF TRUTH - process env may have stale values.
"""

import os
import subprocess


def _get_user_env(name: str) -> str:
    """
    Get environment variable from Windows User scope.

    NOTE: No caching - credentials may be rotated during session.
    """
    try:
        result = subprocess.run(
            [  # noqa: S607
                "powershell",
                "-NoProfile",
                "-Command",
                f'[Environment]::GetEnvironmentVariable("{name}", "User")',
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_alpaca_credentials() -> tuple[str, str]:
    """
    Get Alpaca API credentials.

    IMPORTANT: Windows User env is the SOURCE OF TRUTH.
    Process env may have stale values from old conda activations.

    Returns:
        Tuple of (api_key, api_secret)
    """
    # Windows User environment is authoritative (keys may be rotated)
    api_key = _get_user_env("APCA_API_KEY_ID")
    api_secret = _get_user_env("APCA_API_SECRET_KEY")

    # Only fall back to process env if User env is empty
    if not api_key:
        api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY", "")
    if not api_secret:
        api_secret = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get(
            "ALPACA_API_SECRET", ""
        )

    return api_key or "", api_secret or ""
