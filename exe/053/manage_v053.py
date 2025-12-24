#!/usr/bin/env python
"""
Manage v053 processes efficiently without loading outputs into context.
Provides start, stop, restart, and status commands.
"""
import os
import sys
import subprocess
import signal
import time
from pathlib import Path
from datetime import datetime

class V053Manager:
    """Context-efficient process manager for v053."""

    def __init__(self):
        self.pid_file = Path("logs/v053.pid")
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

    def get_pid(self):
        """Get stored PID if exists."""
        if self.pid_file.exists():
            try:
                return int(self.pid_file.read_text().strip())
            except:
                pass
        return None

    def save_pid(self, pid):
        """Save PID to file."""
        self.pid_file.write_text(str(pid))

    def is_running(self, pid=None):
        """Check if process is running."""
        if pid is None:
            pid = self.get_pid()

        if pid is None:
            return False

        try:
            # Check if process exists (Windows)
            result = subprocess.run(
                ["pwsh", "-Command", f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"],
                capture_output=True,
                text=True
            )
            return len(result.stdout.strip()) > 0
        except:
            return False

    def start(self, silent=True):
        """Start v053 with minimal output."""
        if self.is_running():
            print("⚠️ v053 is already running!")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if silent:
            # Silent mode - redirect all output to files
            stdout_log = self.log_dir / f"v053_stdout_{timestamp}.log"
            stderr_log = self.log_dir / f"v053_stderr_{timestamp}.log"

            print("🚀 Starting v053 in SILENT mode...")
            print(f"📁 Stdout: {stdout_log}")
            print(f"📁 Stderr: {stderr_log}")

            with open(stdout_log, 'w') as out, open(stderr_log, 'w') as err:
                process = subprocess.Popen(
                    [sys.executable, 'ordinis-v053-alpaca-live.py'],
                    stdout=out,
                    stderr=err,
                    env=os.environ,
                    cwd=Path(__file__).parent,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
                )

            self.save_pid(process.pid)
            print(f"✅ Started with PID: {process.pid}")

        else:
            # Background mode - still runs async but accessible via TaskOutput
            print("🚀 Starting v053 in BACKGROUND mode...")

            # Note: This would use Bash tool with run_in_background=True
            # For this script, we'll use the silent approach
            return self.start(silent=True)

        return True

    def stop(self):
        """Stop v053 gracefully."""
        pid = self.get_pid()

        if not pid or not self.is_running(pid):
            print("⚠️ v053 is not running")
            return False

        print(f"🛑 Stopping v053 (PID: {pid})...")

        try:
            if sys.platform == "win32":
                # Windows: Use taskkill
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
            else:
                # Unix: Send SIGTERM
                os.kill(pid, signal.SIGTERM)

            # Wait a moment for graceful shutdown
            time.sleep(2)

            if self.is_running(pid):
                print("⚠️ Process didn't stop gracefully, forcing...")
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
                else:
                    os.kill(pid, signal.SIGKILL)

            # Clean up PID file
            if self.pid_file.exists():
                self.pid_file.unlink()

            print("✅ v053 stopped successfully")
            return True

        except Exception as e:
            print(f"❌ Error stopping process: {e}")
            return False

    def restart(self):
        """Restart v053."""
        print("🔄 Restarting v053...")
        self.stop()
        time.sleep(2)
        return self.start()

    def status(self):
        """Show status without loading logs."""
        pid = self.get_pid()
        is_running = self.is_running(pid)

        print("\n" + "="*60)
        print("V053 STATUS (Context Efficient)")
        print("="*60)
        print(f"Status: {'🟢 RUNNING' if is_running else '🔴 STOPPED'}")

        if is_running:
            print(f"PID: {pid}")

            # Get process info without full output
            try:
                result = subprocess.run(
                    ["pwsh", "-Command",
                     f"Get-Process -Id {pid} | Select-Object CPU, WorkingSet64 | ConvertTo-Json"],
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    import json
                    info = json.loads(result.stdout)
                    cpu = info.get('CPU', 0)
                    mem_mb = info.get('WorkingSet64', 0) / 1024 / 1024
                    print(f"CPU Time: {cpu:.1f}s")
                    print(f"Memory: {mem_mb:.1f} MB")
            except:
                pass

        print("-"*60)
        print("Commands:")
        print("  python manage_v053.py start   - Start in silent mode")
        print("  python manage_v053.py stop    - Stop gracefully")
        print("  python manage_v053.py restart - Restart")
        print("  python manage_v053.py status  - Show this status")
        print("  python monitor_v053.py        - View metrics only")
        print("="*60)

def main():
    """Main entry point."""
    manager = V053Manager()

    if len(sys.argv) < 2:
        manager.status()
        return

    command = sys.argv[1].lower()

    if command == "start":
        manager.start()
    elif command == "stop":
        manager.stop()
    elif command == "restart":
        manager.restart()
    elif command == "status":
        manager.status()
    else:
        print(f"Unknown command: {command}")
        print("Available: start, stop, restart, status")

if __name__ == "__main__":
    main()