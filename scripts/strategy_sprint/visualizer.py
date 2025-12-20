"""
Strategy Sprint Visualization Module.

Generates equity curves, performance charts, and comparison dashboards
for strategy backtesting results.
"""

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Try to import plotting libraries
try:
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available, visualization disabled")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class StrategyVisualizer:
    """Generates visualizations for strategy performance."""

    def __init__(self, output_dir: str = "artifacts/reports/strategy_sprint/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Color palette for strategies
        self.colors = {
            "garch": "#1f77b4",
            "kalman": "#ff7f0e",
            "hmm": "#2ca02c",
            "ou_pairs": "#d62728",
            "evt": "#9467bd",
            "mtf": "#8c564b",
            "mi": "#e377c2",
            "network": "#7f7f7f",
        }

    def generate_equity_curve(
        self,
        trades: list[dict[str, Any]],
        strategy_name: str,
        symbol: str,
        initial_capital: float = 10000,
    ) -> Path | None:
        """Generate equity curve from trade list."""
        if not MATPLOTLIB_AVAILABLE or not trades:
            return None

        # Build equity curve from trade PnLs
        pnls = [t["pnl"] for t in trades]
        equity = [initial_capital]

        for pnl in pnls:
            # Apply percentage return
            new_equity = equity[-1] * (1 + pnl / 100)
            equity.append(new_equity)

        # Calculate drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

        # Equity curve
        ax1.plot(equity, color=self.colors.get(strategy_name, "#1f77b4"), linewidth=1.5)
        ax1.axhline(y=initial_capital, color="gray", linestyle="--", alpha=0.5)
        ax1.set_title(
            f"{strategy_name.upper()} - {symbol} Equity Curve", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("Equity ($)")
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, len(equity))

        # Add metrics annotation
        total_return = (equity[-1] - initial_capital) / initial_capital * 100
        max_dd = np.max(drawdown)
        sharpe = (
            np.mean(pnls)
            / (np.std(pnls) + 1e-10)
            * np.sqrt(252 / max(np.mean([t.get("bars", 5) for t in trades]), 1))
        )

        metrics_text = f"Return: {total_return:.1f}%\nMax DD: {max_dd:.1f}%\nSharpe: {sharpe:.2f}\nTrades: {len(trades)}"
        ax1.text(
            0.02,
            0.98,
            metrics_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Drawdown
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color="red", alpha=0.3)
        ax2.plot(drawdown, color="red", linewidth=1)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Trade #")
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()

        plt.tight_layout()

        # Save
        filename = self.output_dir / f"{strategy_name}_{symbol}_equity.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved equity curve: {filename}")
        return filename

    def generate_comparison_chart(
        self,
        results: dict[str, dict[str, Any]],
        metric: str = "sharpe",
        title: str = "Strategy Comparison",
    ) -> Path | None:
        """Generate bar chart comparing strategies."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        strategies = []
        values = []
        colors = []

        for name, result in results.items():
            if metric == "sharpe":
                val = result.get("avg_sharpe", result.get("test_avg_sharpe", 0))
            elif metric == "return":
                val = result.get("avg_annual_return", result.get("test_avg_annual_return", 0))
            elif metric == "trades":
                val = result.get("total_trades", result.get("test_total_trades", 0))
            else:
                val = result.get(metric, 0)

            strategies.append(name.upper())
            values.append(float(val))
            colors.append(self.colors.get(name, "#1f77b4"))

        # Sort by value
        sorted_idx = np.argsort(values)[::-1]
        strategies = [strategies[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]
        colors = [colors[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(12, 6))

        bars = ax.barh(strategies, values, color=colors, edgecolor="black", linewidth=0.5)

        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(
                width + 0.01 * max(values),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}",
                ha="left",
                va="center",
                fontsize=10,
            )

        # Styling
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        filename = (
            self.output_dir / f"comparison_{metric}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved comparison chart: {filename}")
        return filename

    def generate_walk_forward_chart(
        self,
        results: dict[str, dict[str, Any]],
    ) -> Path | None:
        """Generate train vs test comparison chart for walk-forward results."""
        if not MATPLOTLIB_AVAILABLE:
            return None

        strategies = []
        train_sharpes = []
        test_sharpes = []

        for name, result in results.items():
            if "train_avg_sharpe" in result and "test_avg_sharpe" in result:
                strategies.append(name.upper())
                train_sharpes.append(float(result["train_avg_sharpe"]))
                test_sharpes.append(float(result["test_avg_sharpe"]))

        if not strategies:
            return None

        x = np.arange(len(strategies))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 7))

        bars1 = ax.bar(
            x - width / 2,
            train_sharpes,
            width,
            label="Train (In-Sample)",
            color="#2ecc71",
            edgecolor="black",
            linewidth=0.5,
        )
        bars2 = ax.bar(
            x + width / 2,
            test_sharpes,
            width,
            label="Test (Out-of-Sample)",
            color="#e74c3c",
            edgecolor="black",
            linewidth=0.5,
        )

        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for bar in bars2:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Add overfit indicators
        for i, (train, test) in enumerate(zip(train_sharpes, test_sharpes)):
            if train > 0 and test < 0:
                ax.annotate(
                    "⚠️ OVERFIT", (i, max(train, test) + 0.1), ha="center", fontsize=10, color="red"
                )
            elif train > 0.5 and test > 0:
                ax.annotate(
                    "✓", (i, max(train, test) + 0.1), ha="center", fontsize=12, color="green"
                )

        ax.set_ylabel("Sharpe Ratio")
        ax.set_title(
            "Walk-Forward Validation: Train vs Test Performance", fontsize=14, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.legend(loc="upper right")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        filename = self.output_dir / f"walk_forward_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved walk-forward chart: {filename}")
        return filename

    def generate_dashboard(
        self,
        sprint_results: dict[str, Any],
        trades_by_strategy: dict[str, list[dict[str, Any]]] = None,
    ) -> Path | None:
        """Generate comprehensive dashboard with all visualizations."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping dashboard generation")
            return None

        strategy_results = sprint_results.get("detailed_results", {})
        is_walk_forward = any("train_avg_sharpe" in r for r in strategy_results.values())

        # Create multi-panel figure
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.25)

        # Panel 1: Strategy comparison (Sharpe)
        ax1 = fig.add_subplot(gs[0, 0])
        strategies = list(strategy_results.keys())
        if is_walk_forward:
            train_vals = [float(strategy_results[s].get("train_avg_sharpe", 0)) for s in strategies]
            test_vals = [float(strategy_results[s].get("test_avg_sharpe", 0)) for s in strategies]
            x = np.arange(len(strategies))
            width = 0.35
            ax1.bar(x - width / 2, train_vals, width, label="Train", color="#2ecc71")
            ax1.bar(x + width / 2, test_vals, width, label="Test", color="#e74c3c")
            ax1.set_xticks(x)
            ax1.set_xticklabels([s.upper() for s in strategies], rotation=45, ha="right")
            ax1.legend()
        else:
            sharpes = [float(strategy_results[s].get("avg_sharpe", 0)) for s in strategies]
            colors = [self.colors.get(s, "#1f77b4") for s in strategies]
            ax1.bar([s.upper() for s in strategies], sharpes, color=colors)
            ax1.set_xticklabels([s.upper() for s in strategies], rotation=45, ha="right")
        ax1.set_ylabel("Sharpe Ratio")
        ax1.set_title("Sharpe Ratio Comparison", fontweight="bold")
        ax1.axhline(y=0, color="black", linewidth=0.5)
        ax1.grid(axis="y", alpha=0.3)

        # Panel 2: Annual returns
        ax2 = fig.add_subplot(gs[0, 1])
        if is_walk_forward:
            train_rets = [
                float(strategy_results[s].get("train_avg_annual_return", 0)) for s in strategies
            ]
            test_rets = [
                float(strategy_results[s].get("test_avg_annual_return", 0)) for s in strategies
            ]
            x = np.arange(len(strategies))
            ax2.bar(x - width / 2, train_rets, width, label="Train", color="#2ecc71")
            ax2.bar(x + width / 2, test_rets, width, label="Test", color="#e74c3c")
            ax2.set_xticks(x)
            ax2.set_xticklabels([s.upper() for s in strategies], rotation=45, ha="right")
            ax2.legend()
        else:
            returns = [float(strategy_results[s].get("avg_annual_return", 0)) for s in strategies]
            colors = [self.colors.get(s, "#1f77b4") for s in strategies]
            ax2.bar([s.upper() for s in strategies], returns, color=colors)
            ax2.set_xticklabels([s.upper() for s in strategies], rotation=45, ha="right")
        ax2.set_ylabel("Annual Return (%)")
        ax2.set_title("Annual Return Comparison", fontweight="bold")
        ax2.axhline(y=0, color="black", linewidth=0.5)
        ax2.grid(axis="y", alpha=0.3)

        # Panel 3: Trade counts
        ax3 = fig.add_subplot(gs[1, 0])
        if is_walk_forward:
            train_trades = [
                int(strategy_results[s].get("train_total_trades", 0)) for s in strategies
            ]
            test_trades = [int(strategy_results[s].get("test_total_trades", 0)) for s in strategies]
            x = np.arange(len(strategies))
            ax3.bar(x - width / 2, train_trades, width, label="Train", color="#2ecc71")
            ax3.bar(x + width / 2, test_trades, width, label="Test", color="#e74c3c")
            ax3.set_xticks(x)
            ax3.set_xticklabels([s.upper() for s in strategies], rotation=45, ha="right")
            ax3.legend()
        else:
            trades = [int(strategy_results[s].get("total_trades", 0)) for s in strategies]
            colors = [self.colors.get(s, "#1f77b4") for s in strategies]
            ax3.bar([s.upper() for s in strategies], trades, color=colors)
            ax3.set_xticklabels([s.upper() for s in strategies], rotation=45, ha="right")
        ax3.set_ylabel("Total Trades")
        ax3.set_title("Trade Activity", fontweight="bold")
        ax3.grid(axis="y", alpha=0.3)

        # Panel 4: Overfit ratio (walk-forward only)
        ax4 = fig.add_subplot(gs[1, 1])
        if is_walk_forward:
            overfit_ratios = [
                float(strategy_results[s].get("overfit_ratio", 1)) for s in strategies
            ]
            colors = ["red" if r > 2 else "green" if r < 1.5 else "orange" for r in overfit_ratios]
            ax4.bar([s.upper() for s in strategies], overfit_ratios, color=colors)
            ax4.axhline(y=1, color="green", linestyle="--", label="Ideal (1.0)")
            ax4.axhline(y=2, color="red", linestyle="--", label="Overfit threshold")
            ax4.set_xticklabels([s.upper() for s in strategies], rotation=45, ha="right")
            ax4.set_ylabel("Train/Test Sharpe Ratio")
            ax4.set_title("Overfit Detection", fontweight="bold")
            ax4.legend(loc="upper right", fontsize=8)
        else:
            # Win rates
            win_rates = [float(strategy_results[s].get("avg_win_rate", 50)) for s in strategies]
            colors = [self.colors.get(s, "#1f77b4") for s in strategies]
            ax4.bar([s.upper() for s in strategies], win_rates, color=colors)
            ax4.axhline(y=50, color="gray", linestyle="--")
            ax4.set_xticklabels([s.upper() for s in strategies], rotation=45, ha="right")
            ax4.set_ylabel("Win Rate (%)")
            ax4.set_title("Win Rates", fontweight="bold")
        ax4.grid(axis="y", alpha=0.3)

        # Panel 5-6: Summary table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")

        # Create summary table
        if is_walk_forward:
            columns = [
                "Strategy",
                "Train Sharpe",
                "Test Sharpe",
                "Train Return",
                "Test Return",
                "Overfit",
            ]
            cell_data = []
            for s in strategies:
                r = strategy_results[s]
                overfit = float(r.get("overfit_ratio", 1))
                overfit_str = f"{overfit:.2f} {'⚠️' if overfit > 2 else '✓'}"
                cell_data.append(
                    [
                        s.upper(),
                        f"{float(r.get('train_avg_sharpe', 0)):.3f}",
                        f"{float(r.get('test_avg_sharpe', 0)):.3f}",
                        f"{float(r.get('train_avg_annual_return', 0)):.1f}%",
                        f"{float(r.get('test_avg_annual_return', 0)):.1f}%",
                        overfit_str,
                    ]
                )
        else:
            columns = ["Strategy", "Sharpe", "Annual Return", "Win Rate", "Trades"]
            cell_data = []
            for s in strategies:
                r = strategy_results[s]
                cell_data.append(
                    [
                        s.upper(),
                        f"{float(r.get('avg_sharpe', 0)):.3f}",
                        f"{float(r.get('avg_annual_return', 0)):.1f}%",
                        f"{float(r.get('avg_win_rate', 0)):.1f}%",
                        f"{int(r.get('total_trades', 0))}",
                    ]
                )

        table = ax5.table(
            cellText=cell_data,
            colLabels=columns,
            loc="center",
            cellLoc="center",
            colColours=["#f0f0f0"] * len(columns),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Title
        fig.suptitle(
            f"Strategy Sprint Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        filename = self.output_dir / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        logger.info(f"Saved dashboard: {filename}")
        return filename

    def generate_html_report(
        self,
        sprint_results: dict[str, Any],
    ) -> Path | None:
        """Generate interactive HTML report."""
        strategy_results = sprint_results.get("detailed_results", {})
        is_walk_forward = any("train_avg_sharpe" in r for r in strategy_results.values())

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Strategy Sprint Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #2ecc71; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #2ecc71; color: white; }}
        tr:hover {{ background: #f5f5f5; }}
        .positive {{ color: #27ae60; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
        .warning {{ color: #f39c12; }}
        .metric-card {{ display: inline-block; padding: 20px; margin: 10px; background: #ecf0f1; border-radius: 8px; text-align: center; min-width: 150px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; }}
        .timestamp {{ color: #999; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Strategy Sprint Report</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div style="text-align: center; margin: 30px 0;">
            <div class="metric-card">
                <div class="metric-value">{len(strategy_results)}</div>
                <div class="metric-label">Strategies Tested</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sprint_results.get('sprint_time', 0):.0f}s</div>
                <div class="metric-label">Sprint Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{'Walk-Forward' if is_walk_forward else 'Full Sample'}</div>
                <div class="metric-label">Validation Mode</div>
            </div>
        </div>

        <h2>📊 Strategy Comparison</h2>
        <table>
            <tr>
                <th>Strategy</th>
                {'<th>Train Sharpe</th><th>Test Sharpe</th><th>Train Return</th><th>Test Return</th><th>Status</th>' if is_walk_forward else '<th>Sharpe</th><th>Annual Return</th><th>Win Rate</th><th>Trades</th>'}
            </tr>
"""

        for name, result in strategy_results.items():
            if is_walk_forward:
                train_sharpe = float(result.get("train_avg_sharpe", 0))
                test_sharpe = float(result.get("test_avg_sharpe", 0))
                train_ret = float(result.get("train_avg_annual_return", 0))
                test_ret = float(result.get("test_avg_annual_return", 0))
                overfit = float(result.get("overfit_ratio", 1))

                status = (
                    "✅ Valid"
                    if overfit < 1.5 and test_sharpe > 0
                    else ("⚠️ Marginal" if overfit < 2 else "❌ Overfit")
                )
                status_class = (
                    "positive"
                    if "Valid" in status
                    else ("warning" if "Marginal" in status else "negative")
                )

                html_content += f"""            <tr>
                <td><strong>{name.upper()}</strong></td>
                <td class="{'positive' if train_sharpe > 0 else 'negative'}">{train_sharpe:.3f}</td>
                <td class="{'positive' if test_sharpe > 0 else 'negative'}">{test_sharpe:.3f}</td>
                <td class="{'positive' if train_ret > 0 else 'negative'}">{train_ret:.1f}%</td>
                <td class="{'positive' if test_ret > 0 else 'negative'}">{test_ret:.1f}%</td>
                <td class="{status_class}">{status}</td>
            </tr>
"""
            else:
                sharpe = float(result.get("avg_sharpe", 0))
                ret = float(result.get("avg_annual_return", 0))
                win = float(result.get("avg_win_rate", 0))
                trades = int(result.get("total_trades", 0))

                html_content += f"""            <tr>
                <td><strong>{name.upper()}</strong></td>
                <td class="{'positive' if sharpe > 0 else 'negative'}">{sharpe:.3f}</td>
                <td class="{'positive' if ret > 0 else 'negative'}">{ret:.1f}%</td>
                <td>{win:.1f}%</td>
                <td>{trades}</td>
            </tr>
"""

        html_content += """        </table>

        <h2>📈 Key Insights</h2>
        <ul>
"""

        # Add insights
        if is_walk_forward:
            best_test = max(
                strategy_results.items(), key=lambda x: float(x[1].get("test_avg_sharpe", 0))
            )
            html_content += f"            <li><strong>Best Out-of-Sample:</strong> {best_test[0].upper()} (Test Sharpe: {float(best_test[1].get('test_avg_sharpe', 0)):.3f})</li>\n"

            overfit_strategies = [
                n for n, r in strategy_results.items() if float(r.get("overfit_ratio", 1)) > 2
            ]
            if overfit_strategies:
                html_content += f"            <li class='warning'><strong>Overfit Warning:</strong> {', '.join([s.upper() for s in overfit_strategies])}</li>\n"
        else:
            best = max(strategy_results.items(), key=lambda x: float(x[1].get("avg_sharpe", 0)))
            html_content += f"            <li><strong>Best Strategy:</strong> {best[0].upper()} (Sharpe: {float(best[1].get('avg_sharpe', 0)):.3f})</li>\n"

        html_content += """        </ul>
    </div>
</body>
</html>
"""

        filename = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, "w") as f:
            f.write(html_content)

        logger.info(f"Saved HTML report: {filename}")
        return filename


async def generate_all_visualizations(sprint_results: dict[str, Any]) -> dict[str, Path]:
    """Generate all visualizations for sprint results."""
    visualizer = StrategyVisualizer()

    generated = {}

    # Dashboard
    dashboard = visualizer.generate_dashboard(sprint_results)
    if dashboard:
        generated["dashboard"] = dashboard

    # Walk-forward chart
    if any("train_avg_sharpe" in r for r in sprint_results.get("detailed_results", {}).values()):
        wf_chart = visualizer.generate_walk_forward_chart(
            sprint_results.get("detailed_results", {})
        )
        if wf_chart:
            generated["walk_forward"] = wf_chart

    # Comparison charts
    for metric in ["sharpe", "return", "trades"]:
        chart = visualizer.generate_comparison_chart(
            sprint_results.get("detailed_results", {}),
            metric=metric,
            title=f"Strategy {metric.title()} Comparison",
        )
        if chart:
            generated[f"comparison_{metric}"] = chart

    # HTML report
    html = visualizer.generate_html_report(sprint_results)
    if html:
        generated["html_report"] = html

    return generated
