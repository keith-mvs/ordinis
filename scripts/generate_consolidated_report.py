"""Generate consolidated comprehensive backtest report.

This script analyzes backtest suite results and generates a comprehensive
markdown report with strategy rankings, comparative tables, regime analysis,
and model-specific insights.

Usage:
    python scripts/generate_consolidated_report.py --input results/comprehensive_suite_full_20251210
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════


def load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all backtest results from directory.

    Args:
        results_dir: Directory containing CSV results

    Returns:
        Dictionary of DataFrames for each results file
    """
    results = {}

    # Expected files
    files = {
        "raw": "backtest_results_raw.csv",
        "by_strategy": "backtest_results_by_strategy.csv",
        "by_sector": "backtest_results_by_sector.csv",
        "by_regime": "backtest_results_by_regime.csv",
        "by_cap": "backtest_results_by_market_cap.csv",
        "top20": "backtest_results_top20.csv",
        "robustness": "backtest_results_robustness.csv",
    }

    for key, filename in files.items():
        filepath = results_dir / filename
        if filepath.exists():
            results[key] = pd.read_csv(filepath)
            print(f"Loaded {key}: {len(results[key])} rows")
        else:
            print(f"Warning: {filename} not found")
            results[key] = pd.DataFrame()

    return results


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════


def calculate_strategy_rankings(df_strategy: pd.DataFrame) -> pd.DataFrame:
    """Calculate strategy rankings across multiple metrics.

    Args:
        df_strategy: Strategy-level results

    Returns:
        DataFrame with rankings and composite scores
    """
    if df_strategy.empty:
        return pd.DataFrame()

    # Metrics for ranking (higher is better)
    metrics = {
        "sharpe_ratio": 1.0,  # weight
        "total_return": 0.8,
        "win_rate": 0.6,
        "profit_factor": 0.5,
    }

    rankings = df_strategy.copy()
    rankings["composite_score"] = 0.0

    # Calculate percentile ranks for each metric
    for metric, weight in metrics.items():
        if metric in rankings.columns:
            rankings[f"{metric}_rank"] = rankings[metric].rank(pct=True)
            rankings["composite_score"] += rankings[f"{metric}_rank"] * weight

    # Normalize composite score to 0-100
    total_weight = sum(metrics.values())
    rankings["composite_score"] = (rankings["composite_score"] / total_weight) * 100

    # Overall rank
    rankings["overall_rank"] = (
        rankings["composite_score"].rank(ascending=False, method="min").astype(int)
    )

    # Sort by composite score
    rankings = rankings.sort_values("composite_score", ascending=False)

    return rankings


def analyze_regime_performance(df_regime: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Analyze strategy performance across different market regimes.

    Args:
        df_regime: Regime-level results

    Returns:
        Dictionary of regime-specific analysis
    """
    if df_regime.empty:
        return {}

    analysis = {}

    # Pivot table: Strategy x Regime
    for metric in ["sharpe_ratio", "total_return", "win_rate", "max_drawdown"]:
        if (
            metric in df_regime.columns
            and "strategy" in df_regime.columns
            and "regime" in df_regime.columns
        ):
            pivot = df_regime.pivot_table(
                index="strategy", columns="regime", values=metric, aggfunc="mean"
            )
            analysis[metric] = pivot

    return analysis


def identify_best_performers(df_top20: pd.DataFrame) -> dict[str, list[dict]]:
    """Identify best performing strategy-symbol combinations.

    Args:
        df_top20: Top 20 results

    Returns:
        Dictionary of categorized top performers
    """
    if df_top20.empty:
        return {}

    performers = {
        "highest_sharpe": [],
        "highest_return": [],
        "lowest_drawdown": [],
        "best_win_rate": [],
        "most_trades": [],
    }

    # Top 5 by each metric
    if "sharpe_ratio" in df_top20.columns:
        top_sharpe = df_top20.nlargest(5, "sharpe_ratio")
        performers["highest_sharpe"] = top_sharpe.to_dict("records")

    if "total_return" in df_top20.columns:
        top_return = df_top20.nlargest(5, "total_return")
        performers["highest_return"] = top_return.to_dict("records")

    if "max_drawdown" in df_top20.columns:
        top_dd = df_top20.nsmallest(5, "max_drawdown")  # Lower is better
        performers["lowest_drawdown"] = top_dd.to_dict("records")

    if "win_rate" in df_top20.columns:
        top_wr = df_top20.nlargest(5, "win_rate")
        performers["best_win_rate"] = top_wr.to_dict("records")

    if "total_trades" in df_top20.columns:
        top_trades = df_top20.nlargest(5, "total_trades")
        performers["most_trades"] = top_trades.to_dict("records")

    return performers


def analyze_robustness(df_robustness: pd.DataFrame) -> pd.DataFrame:
    """Analyze strategy robustness across conditions.

    Args:
        df_robustness: Robustness metrics

    Returns:
        DataFrame with robustness scores
    """
    if df_robustness.empty:
        return pd.DataFrame()

    # Calculate consistency score
    if "std_sharpe" in df_robustness.columns and "mean_sharpe" in df_robustness.columns:
        df_robustness["consistency_score"] = df_robustness["mean_sharpe"] / (
            df_robustness["std_sharpe"] + 0.01
        )

    # Calculate reliability (% of positive Sharpe tests)
    if "positive_sharpe_pct" in df_robustness.columns:
        df_robustness["reliability"] = df_robustness["positive_sharpe_pct"]

    return df_robustness


def calculate_sector_insights(df_sector: pd.DataFrame) -> dict[str, any]:
    """Calculate sector-specific insights.

    Args:
        df_sector: Sector-level results

    Returns:
        Dictionary of sector insights
    """
    if df_sector.empty:
        return {}

    insights = {}

    # Best performing sector by strategy
    if "strategy" in df_sector.columns and "sector" in df_sector.columns:
        pivot = df_sector.pivot_table(
            index="strategy", columns="sector", values="sharpe_ratio", aggfunc="mean"
        )

        best_sectors = {}
        for strategy in pivot.index:
            best_sector = pivot.loc[strategy].idxmax()
            best_sharpe = pivot.loc[strategy].max()
            best_sectors[strategy] = {"sector": best_sector, "sharpe": best_sharpe}

        insights["best_sectors_by_strategy"] = best_sectors

    # Overall sector rankings
    if "sector" in df_sector.columns and "sharpe_ratio" in df_sector.columns:
        sector_avg = df_sector.groupby("sector")["sharpe_ratio"].mean().sort_values(ascending=False)
        insights["sector_rankings"] = sector_avg.to_dict()

    return insights


# ═══════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════


def generate_executive_summary(results: dict[str, pd.DataFrame]) -> str:
    """Generate executive summary section.

    Args:
        results: Dictionary of all results DataFrames

    Returns:
        Markdown formatted executive summary
    """
    df_raw = results.get("raw", pd.DataFrame())
    df_strategy = results.get("by_strategy", pd.DataFrame())

    md = ["# Comprehensive Backtest Report - Ordinis Trading System", ""]
    md.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("**Project**: Ordinis - Algorithmic Trading Platform")
    md.append("**Version**: dev-build-0.3.0")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## Executive Summary")
    md.append("")

    if not df_raw.empty:
        total_tests = len(df_raw)
        strategies = df_raw["strategy"].nunique() if "strategy" in df_raw.columns else 0
        symbols = df_raw["symbol"].nunique() if "symbol" in df_raw.columns else 0
        regimes = df_raw["regime"].nunique() if "regime" in df_raw.columns else 0

        md.append(f"**Total Backtests**: {total_tests:,}")
        md.append(f"**Strategies Tested**: {strategies}")
        md.append(f"**Symbols Analyzed**: {symbols}")
        md.append(f"**Market Regimes**: {regimes}")
        md.append("")

    # Top level metrics
    if not df_strategy.empty and "sharpe_ratio" in df_strategy.columns:
        best_strategy = df_strategy.loc[df_strategy["sharpe_ratio"].idxmax(), "strategy"]
        best_sharpe = df_strategy["sharpe_ratio"].max()
        avg_sharpe = df_strategy["sharpe_ratio"].mean()

        md.append(f"**Best Strategy**: {best_strategy} (Sharpe: {best_sharpe:.2f})")
        md.append(f"**Average Sharpe Ratio**: {avg_sharpe:.2f}")
        md.append("")

    md.append("---")
    md.append("")

    return "\n".join(md)


def generate_strategy_rankings_table(rankings: pd.DataFrame) -> str:
    """Generate strategy rankings markdown table.

    Args:
        rankings: Strategy rankings DataFrame

    Returns:
        Markdown formatted table
    """
    if rankings.empty:
        return "No strategy rankings available.\n"

    md = ["## Strategy Rankings", ""]
    md.append("### Overall Performance")
    md.append("")

    # Table header
    md.append(
        "| Rank | Strategy | Composite Score | Sharpe | Return | Win Rate | Profit Factor | Max DD |"
    )
    md.append(
        "|------|----------|----------------|--------|--------|----------|---------------|--------|"
    )

    # Table rows
    for _, row in rankings.iterrows():
        rank = int(row.get("overall_rank", 0))
        strategy = row.get("strategy", "Unknown")
        score = row.get("composite_score", 0.0)
        sharpe = row.get("sharpe_ratio", 0.0)
        ret = row.get("total_return", 0.0) * 100
        wr = row.get("win_rate", 0.0)
        pf = row.get("profit_factor", 0.0)
        dd = row.get("max_drawdown", 0.0) * 100

        md.append(
            f"| {rank} | {strategy} | {score:.1f} | {sharpe:.2f} | {ret:.2f}% | {wr:.1f}% | {pf:.2f} | {dd:.2f}% |"
        )

    md.append("")
    md.append(
        "**Composite Score**: Weighted combination of Sharpe (1.0x), Return (0.8x), Win Rate (0.6x), Profit Factor (0.5x)"
    )
    md.append("")
    md.append("---")
    md.append("")

    return "\n".join(md)


def generate_regime_analysis_table(regime_analysis: dict[str, pd.DataFrame]) -> str:
    """Generate regime analysis markdown tables.

    Args:
        regime_analysis: Dictionary of regime analysis DataFrames

    Returns:
        Markdown formatted tables
    """
    if not regime_analysis:
        return "No regime analysis available.\n"

    md = ["## Regime Performance Analysis", ""]

    # Sharpe ratio by regime
    if "sharpe_ratio" in regime_analysis:
        md.append("### Sharpe Ratio by Regime")
        md.append("")
        md.append(regime_analysis["sharpe_ratio"].to_markdown())
        md.append("")

    # Total return by regime
    if "total_return" in regime_analysis:
        md.append("### Total Return by Regime")
        md.append("")
        df_return = regime_analysis["total_return"] * 100  # Convert to percentage
        md.append(df_return.to_markdown(floatfmt=".2f"))
        md.append("")

    # Win rate by regime
    if "win_rate" in regime_analysis:
        md.append("### Win Rate by Regime")
        md.append("")
        md.append(regime_analysis["win_rate"].to_markdown(floatfmt=".1f"))
        md.append("")

    md.append("---")
    md.append("")

    return "\n".join(md)


def generate_top_performers_section(performers: dict[str, list[dict]]) -> str:
    """Generate top performers section.

    Args:
        performers: Dictionary of top performers by category

    Returns:
        Markdown formatted section
    """
    if not performers:
        return "No top performers data available.\n"

    md = ["## Top Performing Combinations", ""]

    # Highest Sharpe
    if performers.get("highest_sharpe"):
        md.append("### Highest Sharpe Ratio (Top 5)")
        md.append("")
        md.append("| Rank | Strategy | Symbol | Sharpe | Return | Trades |")
        md.append("|------|----------|--------|--------|--------|--------|")

        for i, row in enumerate(performers["highest_sharpe"], 1):
            strategy = row.get("strategy", "Unknown")
            symbol = row.get("symbol", "Unknown")
            sharpe = row.get("sharpe_ratio", 0.0)
            ret = row.get("total_return", 0.0) * 100
            trades = row.get("total_trades", 0)

            md.append(f"| {i} | {strategy} | {symbol} | {sharpe:.2f} | {ret:.2f}% | {trades} |")

        md.append("")

    # Highest Return
    if performers.get("highest_return"):
        md.append("### Highest Total Return (Top 5)")
        md.append("")
        md.append("| Rank | Strategy | Symbol | Return | Sharpe | Trades |")
        md.append("|------|----------|--------|--------|--------|--------|")

        for i, row in enumerate(performers["highest_return"], 1):
            strategy = row.get("strategy", "Unknown")
            symbol = row.get("symbol", "Unknown")
            ret = row.get("total_return", 0.0) * 100
            sharpe = row.get("sharpe_ratio", 0.0)
            trades = row.get("total_trades", 0)

            md.append(f"| {i} | {strategy} | {symbol} | {ret:.2f}% | {sharpe:.2f} | {trades} |")

        md.append("")

    md.append("---")
    md.append("")

    return "\n".join(md)


def generate_model_insights(df_strategy: pd.DataFrame) -> str:
    """Generate model-specific insights.

    Args:
        df_strategy: Strategy-level results

    Returns:
        Markdown formatted insights
    """
    if df_strategy.empty:
        return "No model insights available.\n"

    md = ["## Model-Specific Insights", ""]

    # Group strategies by model type
    model_groups = {
        "New Technical Indicators": ["ADX_TrendFilter", "Fibonacci_Retracement", "ParabolicSAR"],
        "Classic Technical": ["RSI_MeanReversion", "MACD_Crossover", "BollingerBands"],
    }

    for group_name, strategies in model_groups.items():
        md.append(f"### {group_name}")
        md.append("")

        group_data = df_strategy[df_strategy["strategy"].isin(strategies)]

        if not group_data.empty:
            # Summary statistics
            avg_sharpe = (
                group_data["sharpe_ratio"].mean() if "sharpe_ratio" in group_data.columns else 0
            )
            avg_return = (
                group_data["total_return"].mean() * 100
                if "total_return" in group_data.columns
                else 0
            )
            avg_wr = group_data["win_rate"].mean() if "win_rate" in group_data.columns else 0

            md.append(f"**Average Sharpe**: {avg_sharpe:.2f}")
            md.append(f"**Average Return**: {avg_return:.2f}%")
            md.append(f"**Average Win Rate**: {avg_wr:.1f}%")
            md.append("")

            # Individual strategy performance
            for strategy in strategies:
                strat_data = group_data[group_data["strategy"] == strategy]
                if not strat_data.empty:
                    row = strat_data.iloc[0]
                    sharpe = row.get("sharpe_ratio", 0.0)
                    ret = row.get("total_return", 0.0) * 100
                    trades = row.get("total_trades", 0)

                    md.append(
                        f"- **{strategy}**: Sharpe {sharpe:.2f}, Return {ret:.2f}%, Trades {trades}"
                    )

        md.append("")

    md.append("---")
    md.append("")

    return "\n".join(md)


def generate_recommendations(rankings: pd.DataFrame, robustness: pd.DataFrame) -> str:
    """Generate trading recommendations.

    Args:
        rankings: Strategy rankings
        robustness: Robustness analysis

    Returns:
        Markdown formatted recommendations
    """
    md = ["## Recommendations", ""]

    if not rankings.empty:
        # Top 3 strategies overall
        top3 = rankings.head(3)

        md.append("### Recommended Strategies for Live Trading")
        md.append("")

        for i, (_, row) in enumerate(top3.iterrows(), 1):
            strategy = row.get("strategy", "Unknown")
            sharpe = row.get("sharpe_ratio", 0.0)
            ret = row.get("total_return", 0.0) * 100
            score = row.get("composite_score", 0.0)

            md.append(f"{i}. **{strategy}**")
            md.append(f"   - Composite Score: {score:.1f}/100")
            md.append(f"   - Sharpe Ratio: {sharpe:.2f}")
            md.append(f"   - Expected Return: {ret:.2f}%")

            # Add robustness info if available
            if not robustness.empty and "strategy" in robustness.columns:
                rob_row = robustness[robustness["strategy"] == strategy]
                if not rob_row.empty:
                    consistency = rob_row.iloc[0].get("consistency_score", 0.0)
                    reliability = rob_row.iloc[0].get("reliability", 0.0)
                    md.append(f"   - Consistency: {consistency:.2f}")
                    md.append(f"   - Reliability: {reliability:.1f}%")

            md.append("")

    md.append("### Risk Management Notes")
    md.append("")
    md.append("- Use 10% maximum position sizing")
    md.append("- Maintain 90% cash buffer")
    md.append("- Monitor drawdowns closely")
    md.append("- Diversify across top 3 strategies")
    md.append("")

    md.append("---")
    md.append("")

    return "\n".join(md)


def generate_full_report(results_dir: Path, output_path: Path) -> None:
    """Generate complete consolidated backtest report.

    Args:
        results_dir: Directory containing backtest results
        output_path: Path to output markdown file
    """
    print("\nGenerating Consolidated Backtest Report")
    print(f"Input: {results_dir}")
    print(f"Output: {output_path}\n")

    # Load results
    results = load_results(results_dir)

    if all(df.empty for df in results.values()):
        print("Error: No results data found")
        return

    # Perform analyses
    print("Calculating strategy rankings...")
    rankings = calculate_strategy_rankings(results.get("by_strategy", pd.DataFrame()))

    print("Analyzing regime performance...")
    regime_analysis = analyze_regime_performance(results.get("by_regime", pd.DataFrame()))

    print("Identifying top performers...")
    performers = identify_best_performers(results.get("top20", pd.DataFrame()))

    print("Analyzing robustness...")
    robustness = analyze_robustness(results.get("robustness", pd.DataFrame()))

    print("Calculating sector insights...")
    sector_insights = calculate_sector_insights(results.get("by_sector", pd.DataFrame()))

    # Generate report sections
    print("Generating report sections...")

    report_sections = []

    # Executive summary
    report_sections.append(generate_executive_summary(results))

    # Strategy rankings
    report_sections.append(generate_strategy_rankings_table(rankings))

    # Regime analysis
    report_sections.append(generate_regime_analysis_table(regime_analysis))

    # Top performers
    report_sections.append(generate_top_performers_section(performers))

    # Model insights
    report_sections.append(generate_model_insights(results.get("by_strategy", pd.DataFrame())))

    # Recommendations
    report_sections.append(generate_recommendations(rankings, robustness))

    # Write report
    full_report = "\n".join(report_sections)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\nReport generated successfully: {output_path}")
    print(f"Report size: {len(full_report):,} characters\n")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate consolidated comprehensive backtest report"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing backtest results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for markdown report (default: reports/CONSOLIDATED_BACKTEST_REPORT.md)",
    )

    args = parser.parse_args()

    # Paths
    results_dir = Path(args.input)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("reports") / "CONSOLIDATED_BACKTEST_REPORT.md"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate report
    generate_full_report(results_dir, output_path)


if __name__ == "__main__":
    main()
