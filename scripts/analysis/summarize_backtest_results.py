"""
Summarize backtest results from all reports.
"""

import json
from pathlib import Path


def summarize_real_market():
    """Summarize real market backtest."""
    report_path = Path("reports/phase1_real_market_backtest.json")
    if not report_path.exists():
        print("Real market backtest report not found")
        return

    with open(report_path) as f:
        report = json.load(f)

    print("=" * 80)
    print("REAL MARKET BACKTEST RESULTS (2019-2024)")
    print("=" * 80)
    print()

    print(f"Period: {report['backtest_period']}")
    print(f"Symbols Tested: {report['symbols_tested']}")
    print(f"Data Source: {report['data_source']}")
    print(f"Total Trades Generated: {report['baseline_performance']['total_trades']:,}")
    print()

    print("--- BASELINE PERFORMANCE (All Trades) ---")
    baseline = report["baseline_performance"]
    print(f"  Win Rate: {baseline['win_rate']*100:.2f}%")
    print(f"  Total Return: {baseline['total_return_pct']:.2f}%")
    print(f"  Avg Return/Trade: {baseline['avg_return_pct']:.3f}%")
    print(f"  Sharpe Ratio: {baseline['sharpe_ratio']:.2f}")
    print(f"  Profit Factor: {baseline['profit_factor']:.2f}")
    print(f"  Avg Confidence: {baseline['avg_confidence']:.3f}")
    print()

    print("--- FILTERED PERFORMANCE (High Confidence via Calibration) ---")
    filtered = report["filtered_performance"]
    print(
        f"  Trades Executed: {filtered['total_trades']} ({filtered['total_trades']/baseline['total_trades']*100:.1f}% of baseline)"
    )
    print(f"  Win Rate: {filtered['win_rate']*100:.2f}%")
    print(f"  Total Return: {filtered['total_return_pct']:.2f}%")
    print(f"  Avg Return/Trade: {filtered['avg_return_pct']:.3f}%")
    print(f"  Sharpe Ratio: {filtered['sharpe_ratio']:.2f}")
    print(f"  Profit Factor: {filtered['profit_factor']:.2f}")
    print(f"  Avg Confidence: {filtered['avg_confidence']:.3f}")
    print()

    print("--- IMPROVEMENTS FROM FILTERING ---")
    improvement = report["improvement"]
    print(f"  Win Rate Δ: {improvement['win_rate_change_pct']:+.2f}%")
    print(f"  Sharpe Ratio Δ: {improvement['sharpe_ratio_change']:+.2f}")
    print(f"  Profit Factor Δ: {improvement['profit_factor_change']:+.2f}")
    print(
        f"  Trade Selectivity: {(1 - filtered['total_trades']/baseline['total_trades'])*100:.1f}% reduction"
    )
    print()

    if "calibration_metrics" in report:
        print("--- CALIBRATION QUALITY ---")
        cal = report["calibration_metrics"]
        print(f"  Brier Score: {cal['brier_score']:.4f} (lower is better)")
        print(f"  Log Loss: {cal['log_loss']:.4f} (lower is better)")
        print(f"  Accuracy: {cal['accuracy']*100:.1f}%")
        print()

        if cal.get("feature_importance"):
            print("  Feature Importance:")
            for feat, imp in sorted(
                cal["feature_importance"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"    - {feat}: {imp:.3f}")
        print()

    print(
        f"Learning Engine Integration: {'✓ Enabled' if report.get('use_learning_engine') else '✗ Disabled'}"
    )
    if report.get("learning_events_recorded"):
        print(f"  Events Recorded: {report['learning_events_recorded']:,}")
        print(f"  Data Dir: {report.get('learning_data_dir', 'N/A')}")
    print()


def summarize_confidence_backtest():
    """Summarize synthetic confidence backtest."""
    report_path = Path("reports/phase1_confidence_backtest_report.json")
    if not report_path.exists():
        print("Confidence backtest report not found")
        return

    with open(report_path) as f:
        report = json.load(f)

    print("=" * 80)
    print("SYNTHETIC CONFIDENCE BACKTEST")
    print("=" * 80)
    print()

    baseline = report["baseline_results"]
    filtered = report["filtered_results"]
    improvement = report["improvement"]

    print(f"Trades Generated: {baseline['total_trades']:,}")
    print(f"Baseline Win Rate: {baseline['win_rate']*100:.1f}%")
    print(f"Filtered Win Rate: {filtered['win_rate']*100:.1f}% (80%+ confidence)")
    print(f"Improvement: {improvement['win_rate_improvement_pct']:+.1f}%")
    print()

    print("Confidence Distribution:")
    for level, data in report["confidence_distribution"].items():
        print(
            f"  {level.title():12s} ({data['confidence_range']}): {data['count']:4d} trades, {data['win_rate']*100:.1f}% win rate"
        )
    print()


def calculate_annualized_returns():
    """Calculate annualized returns from real market backtest."""
    report_path = Path("reports/phase1_real_market_backtest.json")
    if not report_path.exists():
        return

    with open(report_path) as f:
        report = json.load(f)

    # Period is 2019-01-01 to 2024-12-01 = ~6 years
    years = 6.0

    baseline_total = report["baseline_performance"]["total_return_pct"]
    filtered_total = report["filtered_performance"]["total_return_pct"]

    baseline_annualized = ((1 + baseline_total / 100) ** (1 / years) - 1) * 100
    filtered_annualized = ((1 + filtered_total / 100) ** (1 / years) - 1) * 100

    print("=" * 80)
    print("ANNUALIZED RETURNS (2019-2024, ~6 years)")
    print("=" * 80)
    print()
    print("Baseline (All Trades):")
    print(f"  Total Return: {baseline_total:.2f}%")
    print(f"  Annualized Return: {baseline_annualized:.2f}%/year")
    print()
    print("Filtered (High Confidence):")
    print(f"  Total Return: {filtered_total:.2f}%")
    print(f"  Annualized Return: {filtered_annualized:.2f}%/year")
    print()

    # Compare to SPY benchmark (~10-12%/year historically)
    print("Benchmark Comparison:")
    print("  S&P 500 (SPY) historical avg: ~10-12%/year")
    print(f"  Our baseline: {baseline_annualized:.2f}%/year")
    print(f"  Our filtered: {filtered_annualized:.2f}%/year")
    print()


if __name__ == "__main__":
    summarize_real_market()
    print("\n")
    summarize_confidence_backtest()
    print("\n")
    calculate_annualized_returns()
