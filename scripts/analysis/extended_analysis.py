"""Extended Analysis Dimensions for Backtest Results.

Adds sophisticated analysis beyond basic metrics:
- Drawdown recovery analysis
- Trade duration patterns
- Turnover efficiency
- Capacity estimation
- Portfolio construction recommendations
- Risk factor decomposition
- Strategy diversification benefits

Usage:
    python scripts/extended_analysis.py --input results/comprehensive_suite_20251210
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

sns.set_style("whitegrid")


class ExtendedAnalyzer:
    """Extended analysis beyond basic metrics."""

    def __init__(self, results_dir: Path):
        """Initialize extended analyzer.

        Args:
            results_dir: Directory with backtest results
        """
        self.results_dir = Path(results_dir)
        self.df = None

    def load_data(self):
        """Load results data."""
        raw_files = list(self.results_dir.glob("raw_results_*.csv"))
        if not raw_files:
            raise FileNotFoundError("No raw_results_*.csv found")

        latest = max(raw_files, key=lambda p: p.stat().st_mtime)
        self.df = pd.read_csv(latest)
        print(f"Loaded {len(self.df)} results from {latest.name}")

    def analyze_drawdown_recovery(self):
        """Analyze drawdown recovery patterns."""
        print("\n" + "=" * 100)
        print("DRAWDOWN RECOVERY ANALYSIS")
        print("=" * 100)

        # Calculate recovery ratios (return / abs(drawdown))
        self.df["recovery_ratio"] = self.df["total_return"] / (-self.df["max_drawdown"] + 0.01)

        # Best recovery by strategy
        recovery_by_strategy = (
            self.df.groupby("strategy")
            .agg(
                {
                    "recovery_ratio": ["mean", "median"],
                    "max_drawdown": "mean",
                    "total_return": "mean",
                }
            )
            .round(2)
        )

        print("\nRecovery Ratios by Strategy:")
        print(recovery_by_strategy.to_string())

        # Strategies with best drawdown control
        low_dd = self.df[self.df["max_drawdown"] > -20].groupby("strategy").size()
        print("\nTests with Drawdown < 20% by Strategy:")
        print(low_dd.sort_values(ascending=False).to_string())

        return recovery_by_strategy

    def analyze_trade_duration_patterns(self):
        """Analyze trade duration and turnover patterns."""
        print("\n" + "=" * 100)
        print("TRADE DURATION & TURNOVER ANALYSIS")
        print("=" * 100)

        # Trade activity by strategy
        trade_stats = (
            self.df.groupby("strategy")
            .agg(
                {
                    "num_trades": ["mean", "median", "max"],
                    "turnover": ["mean", "median"],
                    "avg_trade_duration": ["mean", "median"],
                }
            )
            .round(2)
        )

        print("\nTrade Activity Statistics:")
        print(trade_stats.to_string())

        # High activity strategies (potential over-trading)
        high_turnover = self.df[self.df["turnover"] > 10].groupby("strategy")["turnover"].mean()
        print("\nHigh Turnover Strategies (>10 annual):")
        print(high_turnover.sort_values(ascending=False).to_string())

        # Low activity strategies (potential under-trading)
        low_trades = self.df[self.df["num_trades"] < 5].groupby("strategy").size()
        print("\nLow Activity Strategies (<5 trades):")
        print(low_trades.sort_values(ascending=False).to_string())

        return trade_stats

    def estimate_capacity(self):
        """Estimate strategy capacity and scalability."""
        print("\n" + "=" * 100)
        print("CAPACITY & SCALABILITY ANALYSIS")
        print("=" * 100)

        # Capacity proxies
        # Lower turnover + higher avg position size = better capacity
        self.df["capacity_score"] = (1 / (self.df["turnover"] + 1)) * (
            self.df["avg_position_size"] / 10000
        )

        capacity_by_strategy = (
            self.df.groupby("strategy")
            .agg(
                {
                    "capacity_score": "mean",
                    "turnover": "mean",
                    "avg_position_size": "mean",
                    "num_trades": "mean",
                }
            )
            .sort_values("capacity_score", ascending=False)
            .round(2)
        )

        print("\nCapacity Scores (Higher = Better Scalability):")
        print(capacity_by_strategy.to_string())

        # Strategies suitable for large capital
        print("\nHigh Capacity Strategies (low turnover, reasonable trade count):")
        high_cap = self.df[
            (self.df["turnover"] < 5) & (self.df["num_trades"] > 10) & (self.df["num_trades"] < 100)
        ]
        if len(high_cap) > 0:
            high_cap_stats = (
                high_cap.groupby("strategy")["sharpe_ratio"].mean().sort_values(ascending=False)
            )
            print(high_cap_stats.to_string())
        else:
            print("  No strategies meet criteria")

        return capacity_by_strategy

    def build_optimal_portfolio(self):
        """Build optimal strategy portfolio using correlation."""
        print("\n" + "=" * 100)
        print("PORTFOLIO CONSTRUCTION RECOMMENDATIONS")
        print("=" * 100)

        # Create strategy return matrix
        self.df["test_id"] = self.df.groupby(["symbol", "regime", "timeframe"]).ngroup()
        pivot = self.df.pivot_table(values="total_return", index="test_id", columns="strategy")

        # Calculate correlation
        corr = pivot.corr()
        print("\nStrategy Return Correlations:")
        print(corr.round(2).to_string())

        # Find diversifying pairs (low correlation, high individual Sharpe)
        strategy_sharpe = self.df.groupby("strategy")["sharpe_ratio"].mean()

        print("\nDiversifying Strategy Pairs (correlation < 0.3, both Sharpe > 0.5):")
        found_pairs = False
        for i, strat1 in enumerate(corr.columns):
            for strat2 in corr.columns[i + 1 :]:
                if (
                    corr.loc[strat1, strat2] < 0.3
                    and strategy_sharpe[strat1] > 0.5
                    and strategy_sharpe[strat2] > 0.5
                ):
                    print(
                        f"  {strat1:30s} <-> {strat2:30s} | Corr: {corr.loc[strat1, strat2]:>5.2f} | "
                        f"Sharpe: {strategy_sharpe[strat1]:.2f}, {strategy_sharpe[strat2]:.2f}"
                    )
                    found_pairs = True

        if not found_pairs:
            print("  No pairs meet criteria")

        # Recommend 3-strategy portfolio
        print("\nRecommended 3-Strategy Portfolio (low correlation, high Sharpe):")
        top_sharpe_strategies = strategy_sharpe.nlargest(6).index.tolist()

        best_combo = None
        best_score = -np.inf

        for i in range(len(top_sharpe_strategies)):
            for j in range(i + 1, len(top_sharpe_strategies)):
                for k in range(j + 1, len(top_sharpe_strategies)):
                    s1, s2, s3 = (
                        top_sharpe_strategies[i],
                        top_sharpe_strategies[j],
                        top_sharpe_strategies[k],
                    )
                    avg_corr = (corr.loc[s1, s2] + corr.loc[s1, s3] + corr.loc[s2, s3]) / 3
                    avg_sharpe = (
                        strategy_sharpe[s1] + strategy_sharpe[s2] + strategy_sharpe[s3]
                    ) / 3

                    # Score: favor low correlation and high Sharpe
                    score = avg_sharpe * (1 - avg_corr)

                    if score > best_score:
                        best_score = score
                        best_combo = (s1, s2, s3, avg_corr, avg_sharpe)

        if best_combo:
            s1, s2, s3, avg_corr, avg_sharpe = best_combo
            print(f"  1. {s1} (Sharpe: {strategy_sharpe[s1]:.2f})")
            print(f"  2. {s2} (Sharpe: {strategy_sharpe[s2]:.2f})")
            print(f"  3. {s3} (Sharpe: {strategy_sharpe[s3]:.2f})")
            print(f"  Avg Correlation: {avg_corr:.2f}")
            print(f"  Avg Sharpe:      {avg_sharpe:.2f}")
            print(f"  Portfolio Score: {best_score:.2f}")

        return corr

    def analyze_risk_factors(self):
        """Decompose risk factors (regime, sector, cap)."""
        print("\n" + "=" * 100)
        print("RISK FACTOR DECOMPOSITION")
        print("=" * 100)

        # Regime sensitivity
        print("\nRegime Sensitivity (Sharpe Ratio Range):")
        regime_sensitivity = {}
        for strategy in self.df["strategy"].unique():
            strat_data = self.df[self.df["strategy"] == strategy]
            regime_sharpes = strat_data.groupby("regime")["sharpe_ratio"].mean()
            sensitivity = regime_sharpes.max() - regime_sharpes.min()
            regime_sensitivity[strategy] = sensitivity

        regime_df = pd.Series(regime_sensitivity).sort_values(ascending=False)
        print(regime_df.to_string())
        print("\n  Low sensitivity (< 0.5) = regime-robust")
        print("  High sensitivity (> 1.5) = regime-dependent")

        # Sector bias
        print("\nSector Bias (Performance Spread):")
        sector_bias = {}
        for strategy in self.df["strategy"].unique():
            strat_data = self.df[self.df["strategy"] == strategy]
            sector_returns = strat_data.groupby("sector")["total_return"].mean()
            bias = sector_returns.max() - sector_returns.min()
            sector_bias[strategy] = bias

        sector_df = pd.Series(sector_bias).sort_values(ascending=False)
        print(sector_df.round(2).to_string())

        # Cap sensitivity
        print("\nMarket Cap Sensitivity:")
        cap_sensitivity = {}
        for strategy in self.df["strategy"].unique():
            strat_data = self.df[self.df["strategy"] == strategy]
            cap_sharpes = strat_data.groupby("market_cap")["sharpe_ratio"].mean()
            if len(cap_sharpes) >= 2:
                sensitivity = cap_sharpes.max() - cap_sharpes.min()
                cap_sensitivity[strategy] = sensitivity

        cap_df = pd.Series(cap_sensitivity).sort_values(ascending=False)
        print(cap_df.to_string())

        return {
            "regime": regime_df,
            "sector": sector_df,
            "cap": cap_df,
        }

    def calculate_diversification_benefit(self):
        """Calculate diversification benefit of combining strategies."""
        print("\n" + "=" * 100)
        print("DIVERSIFICATION BENEFIT ANALYSIS")
        print("=" * 100)

        # Calculate equal-weight portfolio returns
        self.df["test_id"] = self.df.groupby(["symbol", "regime", "timeframe"]).ngroup()
        pivot = self.df.pivot_table(values="total_return", index="test_id", columns="strategy")

        # Individual strategy stats
        individual_mean = pivot.mean()
        individual_std = pivot.std()
        individual_sharpe = individual_mean / individual_std

        # Equal-weight portfolio
        portfolio_returns = pivot.mean(axis=1)
        portfolio_mean = portfolio_returns.mean()
        portfolio_std = portfolio_returns.std()
        portfolio_sharpe = portfolio_mean / portfolio_std

        print("\nIndividual Strategy Average:")
        print(f"  Mean Return: {individual_mean.mean():.2f}%")
        print(f"  Std Dev:     {individual_std.mean():.2f}%")
        print(f"  Sharpe:      {individual_sharpe.mean():.2f}")

        print("\nEqual-Weight Portfolio (All Strategies):")
        print(f"  Mean Return: {portfolio_mean:.2f}%")
        print(f"  Std Dev:     {portfolio_std:.2f}%")
        print(f"  Sharpe:      {portfolio_sharpe:.2f}")

        diversification_benefit = portfolio_sharpe / individual_sharpe.mean()
        print(f"\nDiversification Benefit: {diversification_benefit:.2f}x")
        print(
            f"  (Portfolio Sharpe is {(diversification_benefit-1)*100:.0f}% higher than avg individual)"
        )

    def generate_extended_report(self):
        """Generate complete extended analysis report."""
        self.load_data()

        output_file = self.results_dir / "EXTENDED_ANALYSIS.txt"

        with open(output_file, "w") as f:
            from io import StringIO
            import sys

            # Capture print output
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()

            # Run all analyses
            self.analyze_drawdown_recovery()
            self.analyze_trade_duration_patterns()
            self.estimate_capacity()
            self.build_optimal_portfolio()
            self.analyze_risk_factors()
            self.calculate_diversification_benefit()

            # Restore stdout and write to file
            sys.stdout = old_stdout
            report_content = mystdout.getvalue()
            f.write(report_content)

            # Also print to console
            print(report_content)

        print(f"\n[SAVED] Extended analysis: {output_file}")


def main():
    """Main extended analysis entry point."""
    parser = argparse.ArgumentParser(description="Extended Backtest Analysis")
    parser.add_argument(
        "--input",
        required=True,
        help="Results directory",
    )

    args = parser.parse_args()
    results_dir = Path(args.input)

    if not results_dir.exists():
        print(f"[ERROR] Directory not found: {results_dir}")
        return

    analyzer = ExtendedAnalyzer(results_dir)
    analyzer.generate_extended_report()


if __name__ == "__main__":
    main()
