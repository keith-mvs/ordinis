"""Backtest Results Analysis and Visualization Tool.

Analyzes CSV outputs from comprehensive_backtest_suite.py and generates:
- Statistical summaries and rankings
- Performance visualizations
- Risk-adjusted return analysis
- Regime/sector/cap dependency analysis
- Strategy comparison matrices

Usage:
    python scripts/analyze_backtest_results.py --input results/comprehensive_suite_20251210
    python scripts/analyze_backtest_results.py --input results/comprehensive_suite_20251210 --format html
"""

import argparse
from datetime import datetime
from pathlib import Path
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # Non-interactive backend
warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


class BacktestAnalyzer:
    """Analyze and visualize backtest results."""

    def __init__(self, results_dir: Path):
        """Initialize analyzer with results directory.

        Args:
            results_dir: Directory containing CSV outputs
        """
        self.results_dir = Path(results_dir)
        self.df = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_data(self):
        """Load raw results CSV."""
        # Find most recent raw_results file
        raw_files = list(self.results_dir.glob("raw_results_*.csv"))
        if not raw_files:
            raise FileNotFoundError(f"No raw_results_*.csv found in {self.results_dir}")

        latest_file = max(raw_files, key=lambda p: p.stat().st_mtime)
        print(f"\nLoading: {latest_file.name}")

        self.df = pd.read_csv(latest_file)
        print(f"Loaded {len(self.df)} backtest results")
        print(f"Columns: {', '.join(self.df.columns)}")

        return self.df

    def generate_visualizations(self, output_dir: Path):
        """Generate all visualization charts.

        Args:
            output_dir: Directory to save charts
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nGenerating visualizations in {output_dir}/")

        # 1. Strategy Performance Comparison
        self._plot_strategy_comparison(output_dir)

        # 2. Regime Performance Heatmap
        self._plot_regime_heatmap(output_dir)

        # 3. Sector Performance
        self._plot_sector_performance(output_dir)

        # 4. Risk-Return Scatter
        self._plot_risk_return_scatter(output_dir)

        # 5. Market Cap Analysis
        self._plot_market_cap_analysis(output_dir)

        # 6. Timeframe Comparison
        self._plot_timeframe_comparison(output_dir)

        # 7. Distribution Analysis
        self._plot_return_distributions(output_dir)

        # 8. Win Rate vs Profit Factor
        self._plot_win_rate_profit_factor(output_dir)

        # 9. Drawdown Analysis
        self._plot_drawdown_analysis(output_dir)

        # 10. Strategy Correlation Matrix
        self._plot_strategy_correlation(output_dir)

        print(f"\n[COMPLETE] All visualizations saved to {output_dir}/")

    def _plot_strategy_comparison(self, output_dir: Path):
        """Plot strategy performance comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Sharpe ratio
        strategy_sharpe = self.df.groupby("strategy")["sharpe_ratio"].agg(["mean", "median", "std"])
        strategy_sharpe.sort_values("mean", ascending=False).plot(
            kind="bar", ax=axes[0, 0], title="Sharpe Ratio by Strategy"
        )
        axes[0, 0].set_ylabel("Sharpe Ratio")
        axes[0, 0].legend(["Mean", "Median", "Std"])
        axes[0, 0].grid(True, alpha=0.3)

        # Total return
        strategy_returns = self.df.groupby("strategy")["total_return"].agg(["mean", "median"])
        strategy_returns.sort_values("mean", ascending=False).plot(
            kind="bar", ax=axes[0, 1], title="Total Return by Strategy"
        )
        axes[0, 1].set_ylabel("Total Return (%)")
        axes[0, 1].legend(["Mean", "Median"])
        axes[0, 1].grid(True, alpha=0.3)

        # Win rate
        strategy_winrate = (
            self.df.groupby("strategy")["win_rate"].mean().sort_values(ascending=False)
        )
        strategy_winrate.plot(
            kind="bar", ax=axes[1, 0], title="Win Rate by Strategy", color="green"
        )
        axes[1, 0].set_ylabel("Win Rate (%)")
        axes[1, 0].grid(True, alpha=0.3)

        # Max drawdown
        strategy_dd = self.df.groupby("strategy")["max_drawdown"].mean().sort_values()
        strategy_dd.plot(
            kind="barh", ax=axes[1, 1], title="Avg Max Drawdown by Strategy", color="red"
        )
        axes[1, 1].set_xlabel("Max Drawdown (%)")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "01_strategy_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [OK] 01_strategy_comparison.png")

    def _plot_regime_heatmap(self, output_dir: Path):
        """Plot regime performance heatmap."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Sharpe ratio heatmap
        pivot_sharpe = self.df.pivot_table(
            values="sharpe_ratio", index="strategy", columns="regime", aggfunc="mean"
        )
        sns.heatmap(pivot_sharpe, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=axes[0])
        axes[0].set_title("Sharpe Ratio by Strategy × Regime")

        # Total return heatmap
        pivot_returns = self.df.pivot_table(
            values="total_return", index="strategy", columns="regime", aggfunc="mean"
        )
        sns.heatmap(pivot_returns, annot=True, fmt=".1f", cmap="RdYlGn", center=0, ax=axes[1])
        axes[1].set_title("Avg Total Return (%) by Strategy × Regime")

        plt.tight_layout()
        plt.savefig(output_dir / "02_regime_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [OK] 02_regime_heatmap.png")

    def _plot_sector_performance(self, output_dir: Path):
        """Plot sector performance analysis."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Sharpe by sector
        sector_perf = (
            self.df.groupby("sector")
            .agg({"sharpe_ratio": "mean", "total_return": "mean", "win_rate": "mean"})
            .sort_values("sharpe_ratio", ascending=False)
        )

        sector_perf["sharpe_ratio"].plot(
            kind="barh", ax=axes[0], title="Average Sharpe Ratio by Sector", color="steelblue"
        )
        axes[0].set_xlabel("Sharpe Ratio")
        axes[0].grid(True, alpha=0.3)

        # Return by sector
        sector_perf["total_return"].plot(
            kind="barh", ax=axes[1], title="Average Total Return by Sector", color="coral"
        )
        axes[1].set_xlabel("Total Return (%)")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "03_sector_performance.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [OK] 03_sector_performance.png")

    def _plot_risk_return_scatter(self, output_dir: Path):
        """Plot risk-return scatter by strategy."""
        fig, ax = plt.subplots(figsize=(12, 8))

        strategies = self.df["strategy"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

        for strategy, color in zip(strategies, colors, strict=False):
            data = self.df[self.df["strategy"] == strategy]
            ax.scatter(
                data["volatility"],
                data["total_return"],
                alpha=0.6,
                s=50,
                color=color,
                label=strategy,
            )

        ax.set_xlabel("Volatility (%)")
        ax.set_ylabel("Total Return (%)")
        ax.set_title("Risk-Return Profile by Strategy")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "04_risk_return_scatter.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [OK] 04_risk_return_scatter.png")

    def _plot_market_cap_analysis(self, output_dir: Path):
        """Plot market cap bucket analysis."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        cap_order = ["SMALL", "MID", "LARGE"]

        # Sharpe by cap
        cap_sharpe = self.df.groupby("market_cap")["sharpe_ratio"].mean().reindex(cap_order)
        cap_sharpe.plot(kind="bar", ax=axes[0], title="Sharpe Ratio by Market Cap", color="purple")
        axes[0].set_ylabel("Sharpe Ratio")
        axes[0].set_xlabel("Market Cap")
        axes[0].grid(True, alpha=0.3)

        # Return by cap
        cap_return = self.df.groupby("market_cap")["total_return"].mean().reindex(cap_order)
        cap_return.plot(kind="bar", ax=axes[1], title="Total Return by Market Cap", color="orange")
        axes[1].set_ylabel("Total Return (%)")
        axes[1].set_xlabel("Market Cap")
        axes[1].grid(True, alpha=0.3)

        # Win rate by cap
        cap_winrate = self.df.groupby("market_cap")["win_rate"].mean().reindex(cap_order)
        cap_winrate.plot(kind="bar", ax=axes[2], title="Win Rate by Market Cap", color="green")
        axes[2].set_ylabel("Win Rate (%)")
        axes[2].set_xlabel("Market Cap")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "05_market_cap_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [OK] 05_market_cap_analysis.png")

    def _plot_timeframe_comparison(self, output_dir: Path):
        """Plot daily vs weekly timeframe comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for idx, strategy in enumerate(self.df["strategy"].unique()[:4]):
            ax = axes[idx // 2, idx % 2]
            data = self.df[self.df["strategy"] == strategy]

            timeframe_perf = data.groupby("timeframe")[["sharpe_ratio", "total_return"]].mean()

            x = np.arange(len(timeframe_perf))
            width = 0.35

            ax.bar(x - width / 2, timeframe_perf["sharpe_ratio"], width, label="Sharpe", alpha=0.8)
            ax.bar(
                x + width / 2,
                timeframe_perf["total_return"] / 10,
                width,
                label="Return/10",
                alpha=0.8,
            )

            ax.set_xlabel("Timeframe")
            ax.set_ylabel("Value")
            ax.set_title(f"{strategy}: Daily vs Weekly")
            ax.set_xticks(x)
            ax.set_xticklabels(timeframe_perf.index)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "06_timeframe_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [OK] 06_timeframe_comparison.png")

    def _plot_return_distributions(self, output_dir: Path):
        """Plot return distributions by strategy."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        strategies = self.df["strategy"].unique()

        for idx, strategy in enumerate(strategies):
            if idx >= 6:
                break
            data = self.df[self.df["strategy"] == strategy]["total_return"]

            axes[idx].hist(data, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
            axes[idx].axvline(data.mean(), color="red", linestyle="--", linewidth=2, label="Mean")
            axes[idx].axvline(
                data.median(), color="green", linestyle="--", linewidth=2, label="Median"
            )
            axes[idx].set_title(f"{strategy} Return Distribution")
            axes[idx].set_xlabel("Total Return (%)")
            axes[idx].set_ylabel("Frequency")
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "07_return_distributions.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [OK] 07_return_distributions.png")

    def _plot_win_rate_profit_factor(self, output_dir: Path):
        """Plot win rate vs profit factor."""
        fig, ax = plt.subplots(figsize=(12, 8))

        strategies = self.df["strategy"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))

        for strategy, color in zip(strategies, colors, strict=False):
            data = self.df[self.df["strategy"] == strategy]
            ax.scatter(
                data["win_rate"],
                data["profit_factor"],
                alpha=0.6,
                s=50,
                color=color,
                label=strategy,
            )

        ax.set_xlabel("Win Rate (%)")
        ax.set_ylabel("Profit Factor")
        ax.set_title("Win Rate vs Profit Factor by Strategy")
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Break-even")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "08_win_rate_profit_factor.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [OK] 08_win_rate_profit_factor.png")

    def _plot_drawdown_analysis(self, output_dir: Path):
        """Plot drawdown analysis."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Drawdown by strategy
        dd_stats = self.df.groupby("strategy")["max_drawdown"].agg(["mean", "min", "max"])
        dd_stats.sort_values("mean").plot(kind="barh", ax=axes[0], title="Max Drawdown by Strategy")
        axes[0].set_xlabel("Max Drawdown (%)")
        axes[0].legend(["Mean", "Best", "Worst"])
        axes[0].grid(True, alpha=0.3)

        # Drawdown distribution
        axes[1].hist(self.df["max_drawdown"], bins=50, alpha=0.7, color="red", edgecolor="black")
        axes[1].set_xlabel("Max Drawdown (%)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Max Drawdown Distribution (All Tests)")
        axes[1].axvline(
            self.df["max_drawdown"].mean(),
            color="black",
            linestyle="--",
            linewidth=2,
            label="Mean",
        )
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "09_drawdown_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [OK] 09_drawdown_analysis.png")

    def _plot_strategy_correlation(self, output_dir: Path):
        """Plot strategy correlation matrix."""
        # Pivot returns by strategy and test
        self.df["test_id"] = self.df.groupby(["symbol", "regime", "timeframe"]).ngroup()
        pivot_returns = self.df.pivot_table(
            values="total_return", index="test_id", columns="strategy", aggfunc="first"
        )

        # Calculate correlation
        corr_matrix = pivot_returns.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, square=True
        )
        ax.set_title("Strategy Return Correlation Matrix")

        plt.tight_layout()
        plt.savefig(output_dir / "10_strategy_correlation.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [OK] 10_strategy_correlation.png")

    def generate_summary_report(self, output_file: Path):
        """Generate text summary report.

        Args:
            output_file: Path to save report
        """
        with open(output_file, "w") as f:
            f.write("=" * 100 + "\n")
            f.write(" " * 30 + "BACKTEST RESULTS SUMMARY\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tests: {len(self.df)}\n\n")

            # Overall statistics
            f.write("-" * 100 + "\n")
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 100 + "\n")
            f.write(f"Mean Total Return:     {self.df['total_return'].mean():>10.2f}%\n")
            f.write(f"Median Total Return:   {self.df['total_return'].median():>10.2f}%\n")
            f.write(f"Mean Sharpe Ratio:     {self.df['sharpe_ratio'].mean():>10.2f}\n")
            f.write(f"Mean Win Rate:         {self.df['win_rate'].mean():>10.2f}%\n")
            f.write(f"Mean Max Drawdown:     {self.df['max_drawdown'].mean():>10.2f}%\n\n")

            # Top strategies
            f.write("-" * 100 + "\n")
            f.write("TOP 5 STRATEGIES BY SHARPE RATIO\n")
            f.write("-" * 100 + "\n")
            top_sharpe = (
                self.df.groupby("strategy")["sharpe_ratio"]
                .mean()
                .sort_values(ascending=False)
                .head(5)
            )
            for strategy, sharpe in top_sharpe.items():
                f.write(f"{strategy:30s} {sharpe:>10.2f}\n")

            f.write("\n")

            # Best regimes
            f.write("-" * 100 + "\n")
            f.write("BEST PERFORMING REGIMES\n")
            f.write("-" * 100 + "\n")
            regime_perf = (
                self.df.groupby("regime")["total_return"].mean().sort_values(ascending=False)
            )
            for regime, ret in regime_perf.items():
                f.write(f"{regime:15s} {ret:>10.2f}%\n")

            f.write("\n")

            # Best sectors
            f.write("-" * 100 + "\n")
            f.write("BEST PERFORMING SECTORS\n")
            f.write("-" * 100 + "\n")
            sector_perf = (
                self.df.groupby("sector")["sharpe_ratio"]
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            for sector, sharpe in sector_perf.items():
                f.write(f"{sector:20s} {sharpe:>10.2f}\n")

            f.write("\n")

            # Strategy × Regime matrix
            f.write("-" * 100 + "\n")
            f.write("STRATEGY × REGIME SHARPE RATIOS\n")
            f.write("-" * 100 + "\n")
            pivot = self.df.pivot_table(
                values="sharpe_ratio", index="strategy", columns="regime", aggfunc="mean"
            )
            f.write(pivot.to_string())
            f.write("\n\n")

            f.write("=" * 100 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 100 + "\n")

        print(f"\n[SAVED] Summary report: {output_file}")

    def generate_html_report(self, output_file: Path, charts_dir: Path):
        """Generate interactive HTML report.

        Args:
            output_file: Path to save HTML
            charts_dir: Directory with chart images
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Results Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            margin-top: 10px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Comprehensive Backtest Results Analysis</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Tests</div>
                <div class="stat-value">{len(self.df):,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Return</div>
                <div class="stat-value">{self.df['total_return'].mean():.2f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Sharpe</div>
                <div class="stat-value">{self.df['sharpe_ratio'].mean():.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Win Rate</div>
                <div class="stat-value">{self.df['win_rate'].mean():.1f}%</div>
            </div>
        </div>

        <h2>Strategy Performance Comparison</h2>
        <img src="{charts_dir.name}/01_strategy_comparison.png" alt="Strategy Comparison">

        <h2>Regime Analysis</h2>
        <img src="{charts_dir.name}/02_regime_heatmap.png" alt="Regime Heatmap">

        <h2>Sector Performance</h2>
        <img src="{charts_dir.name}/03_sector_performance.png" alt="Sector Performance">

        <h2>Risk-Return Profile</h2>
        <img src="{charts_dir.name}/04_risk_return_scatter.png" alt="Risk Return Scatter">

        <h2>Market Cap Analysis</h2>
        <img src="{charts_dir.name}/05_market_cap_analysis.png" alt="Market Cap Analysis">

        <h2>Timeframe Comparison</h2>
        <img src="{charts_dir.name}/06_timeframe_comparison.png" alt="Timeframe Comparison">

        <h2>Return Distributions</h2>
        <img src="{charts_dir.name}/07_return_distributions.png" alt="Return Distributions">

        <h2>Win Rate vs Profit Factor</h2>
        <img src="{charts_dir.name}/08_win_rate_profit_factor.png" alt="Win Rate Profit Factor">

        <h2>Drawdown Analysis</h2>
        <img src="{charts_dir.name}/09_drawdown_analysis.png" alt="Drawdown Analysis">

        <h2>Strategy Correlation</h2>
        <img src="{charts_dir.name}/10_strategy_correlation.png" alt="Strategy Correlation">

        <h2>Top 20 Performers</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Strategy</th>
                <th>Symbol</th>
                <th>Regime</th>
                <th>Timeframe</th>
                <th>Sharpe</th>
                <th>Return (%)</th>
                <th>Win Rate (%)</th>
            </tr>
"""

        # Add top 20 performers
        top_20 = self.df.nlargest(20, "sharpe_ratio")[
            [
                "strategy",
                "symbol",
                "regime",
                "timeframe",
                "sharpe_ratio",
                "total_return",
                "win_rate",
            ]
        ]

        for idx, row in enumerate(top_20.itertuples(), 1):
            html += f"""
            <tr>
                <td>{idx}</td>
                <td>{row.strategy}</td>
                <td>{row.symbol}</td>
                <td>{row.regime}</td>
                <td>{row.timeframe}</td>
                <td>{row.sharpe_ratio:.2f}</td>
                <td>{row.total_return:.2f}</td>
                <td>{row.win_rate:.1f}</td>
            </tr>
"""

        html += """
        </table>
    </div>
</body>
</html>
"""

        with open(output_file, "w") as f:
            f.write(html)

        print(f"[SAVED] HTML report: {output_file}")


def main():
    """Main analysis entry point."""
    parser = argparse.ArgumentParser(description="Analyze Backtest Results")
    parser.add_argument(
        "--input",
        required=True,
        help="Results directory (e.g., results/comprehensive_suite_20251210)",
    )
    parser.add_argument(
        "--format",
        default="both",
        choices=["text", "html", "both"],
        help="Report format",
    )

    args = parser.parse_args()

    results_dir = Path(args.input)
    if not results_dir.exists():
        print(f"[ERROR] Directory not found: {results_dir}")
        return

    print("\n" + "=" * 100)
    print(" " * 35 + "BACKTEST RESULTS ANALYZER")
    print("=" * 100)

    # Initialize analyzer
    analyzer = BacktestAnalyzer(results_dir)

    # Load data
    analyzer.load_data()

    # Generate visualizations
    charts_dir = results_dir / "charts"
    analyzer.generate_visualizations(charts_dir)

    # Generate reports
    if args.format in ["text", "both"]:
        report_file = results_dir / "ANALYSIS_SUMMARY.txt"
        analyzer.generate_summary_report(report_file)

    if args.format in ["html", "both"]:
        html_file = results_dir / "ANALYSIS_REPORT.html"
        analyzer.generate_html_report(html_file, charts_dir)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print(f"\nOutputs saved to: {results_dir}/")
    print(f"  - Charts: {charts_dir}/")
    if args.format in ["text", "both"]:
        print("  - Text Report: ANALYSIS_SUMMARY.txt")
    if args.format in ["html", "both"]:
        print("  - HTML Report: ANALYSIS_REPORT.html")
        print(f"\nOpen in browser: file:///{html_file.absolute()}")


if __name__ == "__main__":
    main()
