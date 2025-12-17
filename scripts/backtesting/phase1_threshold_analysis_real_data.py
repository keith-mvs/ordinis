"""
Phase 1 Real Market Threshold Analysis

Tests multiple confidence thresholds on real historical data to find
the optimal filtering level that actually improves performance.

Tests thresholds: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
Reports actual win rates and Sharpe ratios at each level.
"""

import json
from pathlib import Path

# Import helper functions from the main backtest script
import sys

from ordinis.optimizations.confidence_filter import ConfidenceFilter

sys.path.insert(0, str(Path(__file__).parent))

from phase1_real_market_backtest import (
    UNIVERSE,
    analyze_baseline_performance,
    analyze_filtered_performance,
    download_market_data,
    generate_trades_from_historical_data,
)


def test_threshold(trades: list, threshold: float):
    """Test a specific confidence threshold."""
    filter = ConfidenceFilter(min_confidence=threshold)

    passed = []
    for trade in trades:
        signal = {
            "confidence_score": trade["confidence_score"],
            "num_agreeing_models": trade["num_agreeing_models"],
            "market_volatility": trade["market_volatility"],
        }

        if filter.should_execute(signal):
            trade_copy = trade.copy()
            multiplier = filter.get_position_size_multiplier(trade["confidence_score"])
            trade_copy["position_multiplier"] = multiplier
            passed.append(trade_copy)

    return passed


def run_threshold_analysis():
    """Test multiple thresholds on real market data."""
    print("=" * 80)
    print("PHASE 1 THRESHOLD ANALYSIS - REAL MARKET DATA")
    print("=" * 80)
    print()

    # Download real data
    market_data = download_market_data(list(UNIVERSE.keys()), "2019-01-01", "2024-12-01")

    print()

    # Generate trades
    trades = generate_trades_from_historical_data(market_data)
    print(f"Generated {len(trades)} trades from real market data")
    print()

    # Baseline
    baseline = analyze_baseline_performance(trades)
    print(
        f"BASELINE: {baseline['win_rate']*100:.2f}% win rate, "
        f"{baseline['sharpe_ratio']:.2f} Sharpe, "
        f"{baseline['total_trades']} trades"
    )
    print()

    # Test thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    results = []

    print("=" * 80)
    print("THRESHOLD TESTING")
    print("=" * 80)
    print()

    for threshold in thresholds:
        passed = test_threshold(trades, threshold)
        metrics = analyze_filtered_performance(passed)

        results.append(
            {
                "threshold": threshold,
                "trades": metrics["total_trades"],
                "win_rate": metrics["win_rate"],
                "sharpe": metrics["sharpe_ratio"],
                "profit_factor": metrics["profit_factor"],
            }
        )

        print(f"Threshold {threshold:.2f}:")
        print(
            f"  Trades: {metrics['total_trades']:4} "
            f"({metrics['total_trades']/len(trades)*100:5.1f}%)"
        )
        print(f"  Win Rate: {metrics['win_rate']*100:6.2f}%")
        print(f"  Sharpe: {metrics['sharpe_ratio']:7.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']:5.2f}")
        print()

    # Find best threshold
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()

    # Best by win rate with minimum 100 trades
    viable = [r for r in results if r["trades"] >= 100]
    if viable:
        best_wr = max(viable, key=lambda x: x["win_rate"])
        print(
            f"Best Win Rate (100+ trades): {best_wr['threshold']:.2f} "
            f"({best_wr['win_rate']*100:.1f}% on {best_wr['trades']} trades)"
        )
    else:
        print("No threshold achieved 100+ trades")

    # Best by Sharpe
    best_sharpe = max(results, key=lambda x: x["sharpe"])
    print(
        f"Best Sharpe Ratio: {best_sharpe['threshold']:.2f} "
        f"({best_sharpe['sharpe']:.2f} on {best_sharpe['trades']} trades)"
    )

    print()

    # Save results
    report = {
        "baseline": {
            "win_rate": baseline["win_rate"],
            "sharpe": baseline["sharpe_ratio"],
            "trades": baseline["total_trades"],
        },
        "threshold_results": results,
    }

    report_path = Path("reports") / "phase1_threshold_analysis_real_data.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {report_path}")

    return results


if __name__ == "__main__":
    run_threshold_analysis()
