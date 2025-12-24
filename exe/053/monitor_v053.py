#!/usr/bin/env python
"""
Monitor v053 status without loading full output into context.
Shows only essential metrics and latest events.
"""
import os
import json
import time
from pathlib import Path
from datetime import datetime
import subprocess

def get_latest_log_dir():
    """Find the most recent session directory."""
    log_base = Path("logs")
    if not log_base.exists():
        log_base = Path("exe/053/logs")

    if not log_base.exists():
        return None

    session_dirs = [d for d in log_base.iterdir() if d.is_dir() and "session" in d.name]
    if not session_dirs:
        return None

    # Get most recent
    return max(session_dirs, key=lambda d: d.stat().st_mtime)

def check_process():
    """Check if v053 is running."""
    try:
        result = subprocess.run(
            ["pwsh", "-Command", "Get-Process python* | Where-Object {$_.CommandLine -like '*v053*'} | Select-Object Id, CPU, WorkingSet64"],
            capture_output=True,
            text=True
        )
        return len(result.stdout.strip()) > 0
    except:
        return False

def get_metrics_summary():
    """Get latest metrics without loading full log."""
    log_dir = get_latest_log_dir()
    if not log_dir:
        return None

    metrics_file = log_dir / "metrics_report.json"
    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except:
        return None

def get_last_n_lines(file_path, n=5):
    """Get last N lines from a file efficiently."""
    if not file_path.exists():
        return []

    try:
        # Use PowerShell for efficient tail
        result = subprocess.run(
            ["pwsh", "-Command", f"Get-Content '{file_path}' -Tail {n}"],
            capture_output=True,
            text=True
        )
        return result.stdout.strip().split('\n') if result.stdout else []
    except:
        return []

def monitor_status():
    """Display current status without loading full logs."""

    print("\n" + "="*70)
    print("ORDINIS v0.53 MONITOR - Minimal Context Mode")
    print("="*70)

    # Check if running
    is_running = check_process()
    print(f"Process Status: {'🟢 RUNNING' if is_running else '🔴 STOPPED'}")

    # Get latest session
    log_dir = get_latest_log_dir()
    if log_dir:
        print(f"Session: {log_dir.name}")

        # Get metrics
        metrics = get_metrics_summary()
        if metrics:
            account = metrics.get('account', {})
            system = metrics.get('system', {})

            print("\n📊 ACCOUNT METRICS:")
            print(f"  Equity: ${account.get('equity', 0):,.2f}")
            print(f"  Cash: ${account.get('cash', 0):,.2f}")
            print(f"  Positions: {account.get('positions_count', 0)}")
            print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")

            print("\n⚡ SYSTEM METRICS:")
            print(f"  Runtime: {system.get('runtime_minutes', 0):.1f} minutes")
            print(f"  Bars Processed: {system.get('bars_processed', 0):,}")
            print(f"  Signals Generated: {system.get('signals_generated', 0):,}")
            print(f"  Events Published: {system.get('events_published', 0):,}")

            # Signal breakdown
            signal_counts = metrics.get('signal_counts', {})
            if signal_counts:
                print(f"\n📈 SIGNALS:")
                for signal_type, count in signal_counts.items():
                    print(f"  {signal_type}: {count:,}")

        # Get last few orchestrator messages (lightweight)
        orch_log = log_dir / "orchestrator.jsonl"
        if orch_log.exists():
            print("\n📝 RECENT EVENTS (last 3):")
            lines = get_last_n_lines(orch_log, 3)
            for line in lines:
                if line.strip():
                    try:
                        # Parse JSONL and show just the message
                        data = json.loads(line)
                        msg = data.get('message', '')[:80]  # Truncate long messages
                        print(f"  • {msg}")
                    except:
                        # If not JSON, show raw (but truncated)
                        print(f"  • {line[:80]}")

        # Check for errors
        alpaca_log = log_dir / "alpaca.jsonl"
        if alpaca_log.exists():
            errors = get_last_n_lines(alpaca_log, 2)
            if any("error" in str(e).lower() for e in errors):
                print("\n⚠️ RECENT ERRORS:")
                for error in errors[:2]:
                    print(f"  • {error[:100]}")

    else:
        print("❌ No active session found")

    print("\n" + "-"*70)
    print("💡 Tips:")
    print("  • This monitor preserves context by showing only summaries")
    print("  • Full logs remain in files, not loaded into Claude Code")
    print("  • Use 'tail -f logs/session*/orchestrator.jsonl' for live view")
    print("="*70)

if __name__ == "__main__":
    monitor_status()