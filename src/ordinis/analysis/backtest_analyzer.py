"""
Advanced analysis and optimization based on backtest results.

Generates detailed trade analysis, sector comparisons, model rankings,
and recommendations for platform optimization.
"""

import json
from pathlib import Path

import numpy as np


class BacktestAnalyzer:
    """Comprehensive analysis of backtest results."""

    def __init__(self, report_path: str = "comprehensive_backtest_report.json"):
        """Initialize analyzer with backtest report."""
        with open(report_path) as f:
            self.report = json.load(f)

    def generate_detailed_report(self) -> str:
        """Generate comprehensive detailed report."""

        report = []
        report.append("=" * 100)
        report.append("COMPREHENSIVE BACKTEST ANALYSIS & OPTIMIZATION REPORT")
        report.append("=" * 100)

        # Executive Summary
        report.append("\n" + "=" * 100)
        report.append("EXECUTIVE SUMMARY")
        report.append("=" * 100)

        report.extend(self._executive_summary())

        # Performance Analysis by Period
        report.append("\n" + "=" * 100)
        report.append("DETAILED PERFORMANCE BY PERIOD")
        report.append("=" * 100)

        report.extend(self._period_analysis())

        # Model Performance Analysis
        report.append("\n" + "=" * 100)
        report.append("MODEL PERFORMANCE RANKINGS")
        report.append("=" * 100)

        report.extend(self._model_analysis())

        # Sector Analysis
        report.append("\n" + "=" * 100)
        report.append("SECTOR PERFORMANCE ANALYSIS")
        report.append("=" * 100)

        report.extend(self._sector_analysis())

        # Risk Analysis
        report.append("\n" + "=" * 100)
        report.append("RISK ANALYSIS & RECOMMENDATIONS")
        report.append("=" * 100)

        report.extend(self._risk_analysis())

        # Optimization Recommendations
        report.append("\n" + "=" * 100)
        report.append("OPTIMIZATION RECOMMENDATIONS")
        report.append("=" * 100)

        report.extend(self._optimization_recommendations())

        # Implementation Guide
        report.append("\n" + "=" * 100)
        report.append("IMPLEMENTATION GUIDE")
        report.append("=" * 100)

        report.extend(self._implementation_guide())

        return "\n".join(report)

    def _executive_summary(self) -> list[str]:
        """Executive summary statistics."""

        lines = []

        # Overall statistics
        results = self.report.get("results_by_period", {})

        if not results:
            lines.append("No results available")
            return lines

        # Calculate overall metrics
        returns = [p["metrics"]["total_return"] for p in results.values() if "metrics" in p]
        sharpes = [p["metrics"]["sharpe_ratio"] for p in results.values() if "metrics" in p]
        drawdowns = [abs(p["metrics"]["max_drawdown"]) for p in results.values() if "metrics" in p]

        lines.append(f"\nTest Universe: {self.report['metadata']['total_equities']} equities")
        lines.append(f"Time Periods Tested: {len(self.report['metadata']['test_periods'])}")
        lines.append("Sectors Covered: 10+")
        lines.append("")
        lines.append("Overall Statistics:")
        lines.append(f"  Average Return: {np.mean(returns):.2f}%")
        lines.append(f"  Average Sharpe: {np.mean(sharpes):.2f}")
        lines.append(f"  Average Max Drawdown: {np.mean(drawdowns):.2f}%")
        lines.append(f"  Best Sharpe: {max(sharpes):.2f}")
        lines.append(f"  Worst Sharpe: {min(sharpes):.2f}")
        lines.append(
            f"  Total Trades: {sum([p['metrics']['num_trades'] for p in results.values() if 'metrics' in p]):.0f}"
        )

        return lines

    def _period_analysis(self) -> list[str]:
        """Detailed analysis by time period."""

        lines = []
        results = self.report.get("results_by_period", {})

        lines.append(
            f"\n{'Period':<30} {'Return':<12} {'Sharpe':<12} {'Drawdown':<12} {'Trades':<10}"
        )
        lines.append("-" * 80)

        for period_name, period_data in sorted(results.items()):
            if "error" in period_data:
                lines.append(f"{period_name:<30} ERROR")
                continue

            m = period_data["metrics"]
            lines.append(
                f"{period_name:<30} "
                f"{m['total_return']:>10.2f}% "
                f"{m['sharpe_ratio']:>10.2f} "
                f"{m['max_drawdown']:>10.2f}% "
                f"{m['num_trades']:>8.0f}"
            )

        # Insights by period
        lines.append("\n\nKey Insights by Period:")

        # Find best and worst periods
        valid_results = {k: v for k, v in results.items() if "metrics" in v}

        if valid_results:
            best_period = max(valid_results.items(), key=lambda x: x[1]["metrics"]["sharpe_ratio"])
            worst_period = min(valid_results.items(), key=lambda x: x[1]["metrics"]["sharpe_ratio"])

            lines.append(
                f"  ✓ Best Period: {best_period[0]} (Sharpe: {best_period[1]['metrics']['sharpe_ratio']:.2f})"
            )
            lines.append(
                f"  ✗ Worst Period: {worst_period[0]} (Sharpe: {worst_period[1]['metrics']['sharpe_ratio']:.2f})"
            )

            # Market regime analysis
            if "20-Year Full" in valid_results:
                full_period = valid_results["20-Year Full"]
                lines.append("\n20-Year Performance:")
                lines.append(f"  Return: {full_period['metrics']['total_return']:.2f}%")
                lines.append(f"  Annualized: {full_period['metrics']['annualized_return']:.2f}%")
                lines.append(f"  Sharpe: {full_period['metrics']['sharpe_ratio']:.2f}")

        return lines

    def _model_analysis(self) -> list[str]:
        """Analyze model performance."""

        lines = []
        recommendations = self.report.get("recommendations", {})
        weights = recommendations.get("ensemble_weights", {})

        if weights:
            lines.append("\nRecommended IC-Weighted Ensemble Weights:")
            lines.append(f"{'Model':<35} {'Weight':<10}")
            lines.append("-" * 50)

            for model_name in sorted(weights.keys(), key=lambda x: weights[x], reverse=True):
                weight = weights[model_name]
                lines.append(f"{model_name:<35} {weight:>8.1%}")

        # Model statistics
        rankings = self.report.get("model_rankings", {})

        if rankings:
            lines.append("\n\nModel Performance Metrics:")
            lines.append(f"{'Model':<35} {'Avg IC':<12} {'Hit Rate':<12}")
            lines.append("-" * 60)

            for model_name in sorted(
                rankings.keys(), key=lambda x: rankings[x].get("avg_ic", 0), reverse=True
            ):
                metrics = rankings[model_name]
                lines.append(
                    f"{model_name:<35} "
                    f"{metrics.get('avg_ic', 0):>10.3f} "
                    f"{metrics.get('avg_hit_rate', 0):>10.1f}%"
                )

        return lines

    def _sector_analysis(self) -> list[str]:
        """Analyze performance by sector."""

        lines = []

        # Extract sector information from universe
        universe = self.report["metadata"]["equities"]
        sector_stats = {}

        for symbol, sector in universe.items():
            if sector not in sector_stats:
                sector_stats[sector] = {"symbols": [], "count": 0}
            sector_stats[sector]["symbols"].append(symbol)
            sector_stats[sector]["count"] += 1

        lines.append("\nSectors in Test Universe:")
        lines.append(f"{'Sector':<30} {'Symbols':<10}")
        lines.append("-" * 45)

        for sector in sorted(sector_stats.keys()):
            symbols = sector_stats[sector]["count"]
            lines.append(f"{sector:<30} {symbols:>8}")

        lines.append(f"\nTotal Symbols: {len(universe)}")
        lines.append(f"Total Sectors: {len(sector_stats)}")

        return lines

    def _risk_analysis(self) -> list[str]:
        """Analyze and recommend risk management."""

        lines = []
        results = self.report.get("results_by_period", {})

        # Extract risk metrics
        drawdowns = []
        win_rates = []
        sharpes = []

        for period_data in results.values():
            if "metrics" in period_data:
                m = period_data["metrics"]
                drawdowns.append(abs(m["max_drawdown"]))
                win_rates.append(m["win_rate"])
                sharpes.append(m["sharpe_ratio"])

        if drawdowns:
            lines.append("\nDrawdown Analysis:")
            lines.append(f"  Average Max Drawdown: {np.mean(drawdowns):.2f}%")
            lines.append(f"  Maximum Drawdown: {max(drawdowns):.2f}%")
            lines.append(f"  95th Percentile: {np.percentile(drawdowns, 95):.2f}%")

            lines.append("\nRecommended Risk Parameters:")
            max_dd = np.percentile(drawdowns, 95)
            lines.append("  Position Size: 5% (keep diversified)")
            lines.append("  Max Portfolio Exposure: 80-100%")
            lines.append(f"  Stop Loss: {min(10, max_dd/2):.1f}% (half max expected drawdown)")
            lines.append("  Daily Loss Limit: 2%")
            lines.append(f"  Max Drawdown Halt: {max_dd * 1.5:.1f}%")

        if win_rates:
            lines.append("\nWin Rate Analysis:")
            lines.append(f"  Average: {np.mean(win_rates):.2f}%")
            lines.append(f"  Minimum: {min(win_rates):.2f}%")
            lines.append(f"  Maximum: {max(win_rates):.2f}%")

        return lines

    def _optimization_recommendations(self) -> list[str]:
        """Optimization recommendations."""

        lines = []

        lines.append("\n1. ENSEMBLE OPTIMIZATION")
        lines.append("   ✓ Use Information Coefficient (IC) weighted ensemble")
        lines.append("   ✓ Models with IC > 0.05 have edge")
        lines.append("   ✓ Disable models with negative IC")
        lines.append("   ✓ Rebalance weights monthly based on recent IC")

        lines.append("\n2. SIGNAL GENERATION")
        lines.append("   ✓ Combine signals from multiple models (ensemble voting)")
        lines.append("   ✓ Require minimum 2-3 models to agree for high confidence")
        lines.append("   ✓ Use score weighting instead of binary voting")
        lines.append("   ✓ Consider signal decay (older signals less valuable)")

        lines.append("\n3. POSITION SIZING")
        lines.append("   ✓ Size positions by signal confidence × available capital")
        lines.append("   ✓ Keep max position 5-10% per symbol")
        lines.append("   ✓ Maintain 5-20% cash buffer")
        lines.append("   ✓ Adjust for portfolio concentration risk")

        lines.append("\n4. RISK MANAGEMENT")
        lines.append("   ✓ Use 8-10% stop losses")
        lines.append("   ✓ Monitor daily P&L (halt if -2% loss)")
        lines.append("   ✓ Set maximum drawdown circuit breaker at -20%")
        lines.append("   ✓ Rebalance portfolio daily")

        lines.append("\n5. MODEL TUNING")
        lines.append("   ✓ Test different market regimes separately")
        lines.append("   ✓ Ichimoku works best in trending markets")
        lines.append("   ✓ Chart patterns best in structured moves")
        lines.append("   ✓ Volume profile for mean reversion opportunities")

        lines.append("\n6. DATA QUALITY")
        lines.append("   ✓ Require data quality score > 95%")
        lines.append("   ✓ Monitor for gaps, outliers, staleness")
        lines.append("   ✓ Have automatic failover to backup provider")
        lines.append("   ✓ Alert on data quality degradation")

        return lines

    def _implementation_guide(self) -> list[str]:
        """Implementation guide for improvements."""

        lines = []

        lines.append("\nPHASE 1: IMMEDIATE (This Week)")
        lines.append("  1. Extract IC scores from comprehensive backtest")
        lines.append("  2. Update ensemble weights based on IC rankings")
        lines.append("  3. Disable low-IC models (< 0.02)")
        lines.append("  4. Test improved ensemble in paper trading")

        lines.append("\nPHASE 2: SHORT-TERM (Next 2 Weeks)")
        lines.append("  1. Implement signal decay mechanism")
        lines.append("  2. Add multi-model confirmation filter")
        lines.append("  3. Optimize position sizing formula")
        lines.append("  4. Back-test improved rules on historical data")

        lines.append("\nPHASE 3: MEDIUM-TERM (Next Month)")
        lines.append("  1. Market regime detection (trending vs ranging)")
        lines.append("  2. Dynamic model selection per regime")
        lines.append("  3. Sector-specific position sizing")
        lines.append("  4. Monthly model retraining pipeline")

        lines.append("\nPHASE 4: LONG-TERM (Next Quarter)")
        lines.append("  1. Machine learning model optimization")
        lines.append("  2. Advanced ensemble methods (stacking, etc.)")
        lines.append("  3. Multi-timeframe signal fusion")
        lines.append("  4. Adaptive risk management per market regime")

        lines.append("\n\nCRITICAL SUCCESS FACTORS")
        lines.append("  ✓ Monitor live IC scores vs backtest")
        lines.append("  ✓ Alert if IC drops by > 50%")
        lines.append("  ✓ Retrain if Sharpe drops by > 0.5")
        lines.append("  ✓ Maintain 80%+ test data quality")
        lines.append("  ✓ Document all changes for auditability")

        return lines


def generate_pdf_report(analyzer: BacktestAnalyzer):
    """Generate comprehensive PDF report."""

    report_text = analyzer.generate_detailed_report()

    # Save as text file (PDF generation would require additional library)
    output_path = Path("COMPREHENSIVE_BACKTEST_ANALYSIS.txt")
    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"✓ Report saved to {output_path}")
    print(f"✓ Report size: {len(report_text):,} characters")

    return report_text


async def main():
    """Generate comprehensive analysis."""

    print("Waiting for backtest to complete...")
    print("This will generate detailed analysis once backtest_report.json is available")

    # Try to load existing report
    report_path = Path("comprehensive_backtest_report.json")

    if report_path.exists():
        analyzer = BacktestAnalyzer(str(report_path))
        report = generate_pdf_report(analyzer)
        print("\n" + report[:5000] + "\n... (see full report in file)")
    else:
        print("Report not ready yet. When available, run: python -m ordinis.analysis.analyzer")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
