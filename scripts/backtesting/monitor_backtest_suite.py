"""Real-time Monitoring Dashboard for Backtest Suite Execution.

Monitors running comprehensive_backtest_suite.py and displays:
- Real-time progress tracking
- Success/failure rates
- Estimated time remaining
- Performance preview from completed tests

Usage:
    python scripts/monitor_backtest_suite.py --log results/suite_execution.log
    python scripts/monitor_backtest_suite.py --results-dir results/comprehensive_suite_20251210
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import re
import time

import pandas as pd


class SuiteMonitor:
    """Monitor comprehensive backtest suite execution."""

    def __init__(self, log_file: Path = None, results_dir: Path = None):
        """Initialize monitor.

        Args:
            log_file: Path to execution log
            results_dir: Path to results directory
        """
        self.log_file = log_file
        self.results_dir = results_dir
        self.start_time = None
        self.total_tests = None
        self.completed_tests = 0
        self.failed_tests = 0
        self.last_update = None

    def parse_log(self):
        """Parse execution log for progress."""
        if not self.log_file or not self.log_file.exists():
            return None

        with open(self.log_file, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Extract total tests
        total_match = re.search(r"Total Tests:\s+(\d+)", content)
        if total_match:
            self.total_tests = int(total_match.group(1))

        # Count completed tests
        self.completed_tests = content.count("[OK]")
        self.failed_tests = content.count("[FAIL]")

        # Extract start time
        if not self.start_time and "RUNNING BACKTESTS" in content:
            self.start_time = datetime.now()

        return {
            "total": self.total_tests,
            "completed": self.completed_tests,
            "failed": self.failed_tests,
            "success_rate": (
                (self.completed_tests / (self.completed_tests + self.failed_tests)) * 100
                if (self.completed_tests + self.failed_tests) > 0
                else 0
            ),
        }

    def load_partial_results(self):
        """Load partial results from CSV if available."""
        if not self.results_dir:
            return None

        csv_files = list(self.results_dir.glob("raw_results_*.csv"))
        if not csv_files:
            return None

        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)

        try:
            df = pd.read_csv(latest_csv)
            return {
                "records": len(df),
                "mean_sharpe": df["sharpe_ratio"].mean(),
                "mean_return": df["total_return"].mean(),
                "best_strategy": df.groupby("strategy")["sharpe_ratio"].mean().idxmax(),
                "best_sharpe": df.groupby("strategy")["sharpe_ratio"].mean().max(),
            }
        except Exception:
            return None

    def estimate_remaining_time(self):
        """Estimate remaining execution time."""
        if not self.start_time or not self.total_tests or self.completed_tests == 0:
            return None

        elapsed = datetime.now() - self.start_time
        tests_processed = self.completed_tests + self.failed_tests
        avg_time_per_test = elapsed.total_seconds() / tests_processed

        remaining_tests = self.total_tests - tests_processed
        remaining_seconds = remaining_tests * avg_time_per_test

        return {
            "elapsed": elapsed,
            "remaining": timedelta(seconds=int(remaining_seconds)),
            "eta": datetime.now() + timedelta(seconds=remaining_seconds),
            "avg_per_test": avg_time_per_test,
        }

    def display_dashboard(self):
        """Display monitoring dashboard."""
        # Clear screen (works on both Windows and Unix)
        print("\033[2J\033[H", end="")

        print("=" * 100)
        print(" " * 30 + "BACKTEST SUITE MONITOR")
        print("=" * 100)
        print(f"Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Parse log
        stats = self.parse_log()

        if not stats:
            print("[WAITING] No execution data found yet...")
            print(f"  Log file: {self.log_file}")
            return

        # Progress bar
        progress = (
            ((stats["completed"] + stats["failed"]) / stats["total"]) * 100 if stats["total"] else 0
        )
        bar_width = 60
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        print("-" * 100)
        print("EXECUTION PROGRESS")
        print("-" * 100)
        print(f"[{bar}] {progress:.1f}%\n")

        print(f"Total Tests:       {stats['total']:>6}")
        print(f"Completed:         {stats['completed']:>6} [OK]")
        print(f"Failed:            {stats['failed']:>6} [FAIL]")
        print(f"Remaining:         {stats['total'] - stats['completed'] - stats['failed']:>6}")
        print(f"Success Rate:      {stats['success_rate']:>6.1f}%\n")

        # Time estimates
        time_est = self.estimate_remaining_time()
        if time_est:
            print("-" * 100)
            print("TIME ESTIMATES")
            print("-" * 100)
            print(f"Elapsed:           {str(time_est['elapsed']).split('.')[0]}")
            print(f"Remaining:         {str(time_est['remaining']).split('.')[0]}")
            print(f"ETA:               {time_est['eta'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Avg per test:      {time_est['avg_per_test']:.2f}s\n")

        # Partial results
        results = self.load_partial_results()
        if results:
            print("-" * 100)
            print("PRELIMINARY RESULTS")
            print("-" * 100)
            print(f"Tests analyzed:    {results['records']:>6}")
            print(f"Mean Sharpe:       {results['mean_sharpe']:>6.2f}")
            print(f"Mean Return:       {results['mean_return']:>6.2f}%")
            print(f"Best Strategy:     {results['best_strategy']}")
            print(f"  Sharpe Ratio:    {results['best_sharpe']:>6.2f}\n")

        print("=" * 100)
        print("Press Ctrl+C to exit monitoring")
        print("=" * 100)

    def monitor_loop(self, interval: int = 10):
        """Run monitoring loop.

        Args:
            interval: Update interval in seconds
        """
        print("\n[MONITOR] Starting real-time dashboard...")
        print(f"[MONITOR] Update interval: {interval}s")
        print("[MONITOR] Press Ctrl+C to stop\n")

        try:
            while True:
                self.display_dashboard()

                # Check if complete
                stats = self.parse_log()
                if stats and (stats["completed"] + stats["failed"]) >= stats["total"]:
                    print("\n[COMPLETE] Backtest suite finished!")
                    print(f"  Results: {self.results_dir}/")
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n[STOPPED] Monitoring stopped by user")


def main():
    """Main monitoring entry point."""
    parser = argparse.ArgumentParser(description="Monitor Backtest Suite")
    parser.add_argument(
        "--log",
        default="results/suite_execution.log",
        help="Path to execution log",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Results directory (for partial results)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Update interval in seconds",
    )

    args = parser.parse_args()

    log_file = Path(args.log)
    results_dir = Path(args.results_dir) if args.results_dir else None

    monitor = SuiteMonitor(log_file, results_dir)
    monitor.monitor_loop(args.interval)


if __name__ == "__main__":
    main()
