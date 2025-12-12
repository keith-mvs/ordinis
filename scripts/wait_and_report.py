"""Monitor backtest suite and auto-generate report on completion.

Usage:
    python scripts/wait_and_report.py --input results/comprehensive_suite_full_20251210
"""

import argparse
from pathlib import Path
import subprocess
import time


def check_suite_completion(log_file: Path) -> tuple[bool, int, int]:
    """Check if backtest suite has completed.

    Args:
        log_file: Path to suite execution log

    Returns:
        Tuple of (is_complete, completed_tests, total_tests)
    """
    if not log_file.exists():
        return False, 0, 0

    try:
        # Read log file
        with open(log_file, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Check for completion marker
        if "[COMPLETE]" in content or "All backtests completed" in content:
            return True, 0, 0

        # Count completed tests
        completed = content.count("[OK]") + content.count("[FAIL]")

        # Extract total from header
        total = 0
        if "Total Tests:" in content:
            for line in content.split("\n"):
                if "Total Tests:" in line:
                    try:
                        total = int(line.split("Total Tests:")[1].strip().split()[0])
                    except:
                        pass
                    break

        return False, completed, total

    except Exception as e:
        print(f"Error reading log: {e}")
        return False, 0, 0


def monitor_and_report(results_dir: Path, log_file: Path, check_interval: int = 60):
    """Monitor suite execution and generate report on completion.

    Args:
        results_dir: Directory containing results
        log_file: Path to execution log
        check_interval: Seconds between checks
    """
    print(f"Monitoring backtest suite: {results_dir}")
    print(f"Log file: {log_file}")
    print(f"Check interval: {check_interval} seconds\n")

    start_time = time.time()
    last_completed = 0

    while True:
        is_complete, completed, total = check_suite_completion(log_file)

        # Calculate progress
        if total > 0:
            progress_pct = (completed / total) * 100
        else:
            progress_pct = 0

        # Calculate rate
        elapsed = time.time() - start_time
        if elapsed > 0 and completed > last_completed:
            rate = (completed - last_completed) / (
                elapsed if last_completed == 0 else check_interval
            )
            remaining = (total - completed) / rate if rate > 0 else 0
            eta_min = remaining / 60
        else:
            rate = 0
            eta_min = 0

        # Status update
        print(
            f"\r[{time.strftime('%H:%M:%S')}] Progress: {progress_pct:.1f}% ({completed}/{total}) | "
            f"Rate: {rate:.1f} tests/s | ETA: {eta_min:.0f} min",
            end="",
            flush=True,
        )

        if is_complete:
            print("\n\n[COMPLETE] Suite finished!")
            break

        last_completed = completed
        time.sleep(check_interval)

    # Generate report
    print("\nGenerating consolidated report...")

    report_script = Path("scripts/generate_consolidated_report.py")
    output_report = Path("reports/CONSOLIDATED_BACKTEST_REPORT.md")

    if report_script.exists():
        try:
            subprocess.run(
                [
                    "python",
                    str(report_script),
                    "--input",
                    str(results_dir),
                    "--output",
                    str(output_report),
                ],
                check=True,
            )

            print(f"\n✅ Report generated: {output_report}")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error generating report: {e}")

    else:
        print(f"\n❌ Report script not found: {report_script}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor backtest suite and auto-generate report")
    parser.add_argument("--input", type=str, required=True, help="Results directory")
    parser.add_argument(
        "--log", type=str, default="results/suite_full_execution.log", help="Log file path"
    )
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")

    args = parser.parse_args()

    results_dir = Path(args.input)
    log_file = Path(args.log)

    monitor_and_report(results_dir, log_file, args.interval)


if __name__ == "__main__":
    main()
