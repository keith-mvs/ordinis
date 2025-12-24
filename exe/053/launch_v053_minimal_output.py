#!/usr/bin/env python
"""
Launch Ordinis v0.53 with minimal output to preserve Claude Code context.
Redirects all output to log files instead of console.
"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def launch_v053_silent():
    """Launch v053 with output redirected to files."""

    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_log = log_dir / f"v053_stdout_{timestamp}.log"
    stderr_log = log_dir / f"v053_stderr_{timestamp}.log"

    # Load environment variables
    os.environ['ALPACA_API_KEY_PAPER_ENGINE'] = os.environ.get('ALPACA_API_KEY', '')
    os.environ['ALPACA_SECRET_KEY_PAPER_ENGINE'] = os.environ.get('ALPACA_SECRET', '')
    os.environ['MASSIVE_API_KEY'] = os.environ.get('MASSIVE_API_KEY', '')

    print("="*60)
    print("ORDINIS v0.53 - SILENT LAUNCH MODE")
    print("="*60)
    print(f"✓ Output redirected to: {stdout_log}")
    print(f"✓ Errors redirected to: {stderr_log}")
    print("✓ Running in background - context preserved")
    print("-"*60)
    print("Monitor with: tail -f", stderr_log)
    print("Check status: ps aux | grep ordinis")
    print("Stop with: pkill -f ordinis-v053")
    print("="*60)

    # Launch with output redirection
    with open(stdout_log, 'w') as out, open(stderr_log, 'w') as err:
        process = subprocess.Popen(
            [sys.executable, 'ordinis-v053-alpaca-live.py'],
            stdout=out,
            stderr=err,
            env=os.environ,
            cwd=Path(__file__).parent
        )

        print(f"✓ Process started with PID: {process.pid}")
        print(f"✓ System running in background")

        # Don't wait - let it run in background
        return process.pid

if __name__ == "__main__":
    pid = launch_v053_silent()
    print(f"\nTo stop: kill {pid}")
    print("Logs are being written to logs/ directory")