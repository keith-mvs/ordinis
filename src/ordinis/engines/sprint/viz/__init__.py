"""Sprint Visualizer - generates charts, dashboards, and reports."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StrategyVisualizer:
    """Generates visualizations for strategy backtests."""

    def __init__(self, output_dir: str = "artifacts/sprint/viz"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check matplotlib availability
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self.plt = plt
            self._has_matplotlib = True
        except ImportError:
            self._has_matplotlib = False
            logger.warning("matplotlib not available - visualizations disabled")

    def generate_equity_curve(
        self,
        equity: pd.Series | np.ndarray,
        name: str,
        benchmark: pd.Series | np.ndarray | None = None,
    ) -> Path | None:
        """Generate equity curve chart."""
        if not self._has_matplotlib:
            return None

        # Validate equity data
        if equity is None or len(equity) < 2:
            logger.warning(f"Insufficient equity data for {name}")
            return None

        fig, ax = self.plt.subplots(figsize=(12, 6))

        # Strategy equity
        if isinstance(equity, pd.Series):
            x = range(len(equity))  # Use integer index for consistent alignment
            y = equity.values
        else:
            x = range(len(equity))
            y = equity

        # Check for valid starting value
        if y[0] == 0:
            y = y + 1  # Shift to avoid division by zero

        ax.plot(x, y, label=name, linewidth=2, color="steelblue")

        # Benchmark if provided
        if benchmark is not None and len(benchmark) > 0:
            if isinstance(benchmark, pd.Series):
                by = benchmark.values
            else:
                by = benchmark

            # Ensure benchmark has valid starting value
            if by[0] != 0:
                # Normalize benchmark to same starting point as strategy
                by_norm = by / by[0] * y[0]
                # Align lengths - use minimum length
                min_len = min(len(x), len(by_norm))
                ax.plot(
                    range(min_len),
                    by_norm[:min_len],
                    label="Benchmark",
                    linewidth=1.5,
                    linestyle="--",
                    color="gray",
                    alpha=0.7,
                )
            else:
                logger.warning("Benchmark has zero starting value, skipping")

        ax.set_title(f"{name} - Equity Curve", fontsize=14, fontweight="bold")
        ax.set_xlabel("Trading Days")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Add drawdown shading
        if len(y) > 0:
            peak = np.maximum.accumulate(y)
            drawdown = (peak - y) / peak

            ax2 = ax.twinx()
            ax2.fill_between(x, 0, -drawdown * 100, alpha=0.2, color="red", label="Drawdown")
            ax2.set_ylabel("Drawdown (%)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.set_ylim(-50, 5)

        self.plt.tight_layout()

        filepath = self.output_dir / f"{name.lower().replace(' ', '_')}_equity.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        self.plt.close(fig)

        logger.info(f"Saved equity curve: {filepath}")
        return filepath

    def generate_comparison_chart(
        self,
        results: dict[str, Any],
        metric: str = "sharpe_ratio",
    ) -> Path | None:
        """Generate strategy comparison bar chart."""
        if not self._has_matplotlib:
            return None

        # Extract metrics
        names = []
        values = []

        for name, result in results.items():
            if hasattr(result, metric):
                names.append(result.name if hasattr(result, "name") else name)
                values.append(getattr(result, metric))
            elif isinstance(result, dict) and metric in result:
                names.append(name)
                values.append(result[metric])

        if not names:
            logger.warning(f"No results found for metric: {metric}")
            return None

        fig, ax = self.plt.subplots(figsize=(10, 6))

        # Color based on value
        colors = ["green" if v > 0 else "red" for v in values]

        bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor="black")

        # Add value labels
        for bar, val in zip(bars, values, strict=False):
            height = bar.get_height()
            ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        ax.set_title(
            f"Strategy Comparison - {metric.replace('_', ' ').title()}",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        self.plt.xticks(rotation=45, ha="right")
        self.plt.tight_layout()

        filepath = self.output_dir / f"comparison_{metric}.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        self.plt.close(fig)

        logger.info(f"Saved comparison chart: {filepath}")
        return filepath

    def generate_walk_forward_chart(
        self,
        results: dict[str, Any],
    ) -> Path | None:
        """Generate walk-forward validation chart."""
        if not self._has_matplotlib:
            return None

        names = []
        train_sharpes = []
        test_sharpes = []

        for name, result in results.items():
            train = (
                getattr(result, "train_sharpe", None) if hasattr(result, "train_sharpe") else None
            )
            test = getattr(result, "test_sharpe", None) if hasattr(result, "test_sharpe") else None

            if train is not None and test is not None:
                names.append(result.name if hasattr(result, "name") else name)
                train_sharpes.append(train)
                test_sharpes.append(test)

        if not names:
            logger.warning("No walk-forward results found")
            return None

        fig, ax = self.plt.subplots(figsize=(10, 6))

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, train_sharpes, width, label="Train Sharpe", color="steelblue", alpha=0.8
        )
        bars2 = ax.bar(
            x + width / 2, test_sharpes, width, label="Test Sharpe", color="coral", alpha=0.8
        )

        ax.set_title("Walk-Forward Validation Results", fontsize=14, fontweight="bold")
        ax.set_ylabel("Sharpe Ratio")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        # Add overfit indicator
        for i, (train, test) in enumerate(zip(train_sharpes, test_sharpes, strict=False)):
            if test > 0:
                overfit = train / test
                color = "green" if overfit < 1.5 else "orange" if overfit < 2.0 else "red"
                ax.annotate(
                    f"OF: {overfit:.1f}x",
                    xy=(i, max(train, test) + 0.1),
                    ha="center",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                )

        self.plt.tight_layout()

        filepath = self.output_dir / "walk_forward_validation.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        self.plt.close(fig)

        logger.info(f"Saved walk-forward chart: {filepath}")
        return filepath

    def generate_dashboard(
        self,
        results: dict[str, Any],
    ) -> Path | None:
        """Generate combined dashboard with multiple charts."""
        if not self._has_matplotlib:
            return None

        fig = self.plt.figure(figsize=(16, 12))

        # Create grid
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Equity curves (top left, large)
        ax1 = fig.add_subplot(gs[0, :])

        # Collect valid equity curves and find max length for proper alignment
        valid_curves = []
        for name, result in results.items():
            equity = getattr(result, "equity_curve", None)
            if equity is not None and len(equity) > 1:
                label = result.name if hasattr(result, "name") else name
                # Skip flat curves (no actual trading)
                if isinstance(equity, pd.Series):
                    values = equity.values
                else:
                    values = equity
                # Check for actual price movement (not just flat line)
                if np.std(values) > 1e-6:
                    valid_curves.append((label, equity))

        # Plot each curve with proper x-axis alignment
        for label, equity in valid_curves:
            if isinstance(equity, pd.Series):
                # Normalize to 100
                normalized = equity / equity.iloc[0] * 100
                # Use index for x-axis if available
                ax1.plot(range(len(normalized)), normalized.values, label=label, linewidth=1.5)
            else:
                normalized = equity / equity[0] * 100 if equity[0] != 0 else equity
                ax1.plot(range(len(normalized)), normalized, label=label, linewidth=1.5)

        ax1.set_title("Normalized Equity Curves (Base=100)", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper left", fontsize=9, ncol=2)  # Two columns for many strategies
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel("Value")
        ax1.set_xlabel("Trading Days")

        # 2. Sharpe comparison (middle left)
        ax2 = fig.add_subplot(gs[1, 0])

        names = [r.name if hasattr(r, "name") else k for k, r in results.items()]
        sharpes = [getattr(r, "sharpe_ratio", 0) for r in results.values()]
        colors = ["green" if s > 0 else "red" for s in sharpes]

        ax2.barh(names, sharpes, color=colors, alpha=0.8)
        ax2.set_title("Sharpe Ratio", fontsize=12, fontweight="bold")
        ax2.axvline(x=0, color="black", linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis="x")

        # 3. Returns comparison (middle right)
        ax3 = fig.add_subplot(gs[1, 1])

        returns = [getattr(r, "annualized_return", 0) * 100 for r in results.values()]
        colors = ["green" if r > 0 else "red" for r in returns]

        ax3.barh(names, returns, color=colors, alpha=0.8)
        ax3.set_title("Annualized Return (%)", fontsize=12, fontweight="bold")
        ax3.axvline(x=0, color="black", linewidth=0.5)
        ax3.grid(True, alpha=0.3, axis="x")

        # 4. Drawdown comparison (bottom left)
        ax4 = fig.add_subplot(gs[2, 0])

        drawdowns = [getattr(r, "max_drawdown", 0) * 100 for r in results.values()]

        ax4.barh(names, drawdowns, color="salmon", alpha=0.8)
        ax4.set_title("Max Drawdown (%)", fontsize=12, fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="x")

        # 5. Summary table (bottom right)
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")

        table_data = []
        headers = ["Strategy", "Return", "Sharpe", "MaxDD", "Trades"]

        for name, result in results.items():
            row = [
                result.name if hasattr(result, "name") else name,
                f"{getattr(result, 'annualized_return', 0) * 100:.1f}%",
                f"{getattr(result, 'sharpe_ratio', 0):.2f}",
                f"{getattr(result, 'max_drawdown', 0) * 100:.1f}%",
                f"{getattr(result, 'total_trades', 0)}",
            ]
            table_data.append(row)

        table = ax5.table(
            cellText=table_data,
            colLabels=headers,
            loc="center",
            cellLoc="center",
            colWidths=[0.3, 0.15, 0.15, 0.15, 0.15],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(color="white", fontweight="bold")

        fig.suptitle("Strategy Sprint Dashboard", fontsize=16, fontweight="bold", y=0.98)

        filepath = self.output_dir / "dashboard.png"
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        self.plt.close(fig)

        logger.info(f"Saved dashboard: {filepath}")
        return filepath

    def generate_html_report(
        self,
        results: dict[str, Any],
        title: str = "Strategy Sprint Report",
    ) -> Path:
        """Generate HTML report with embedded charts."""

        # Generate all charts
        chart_paths = {}
        chart_paths["dashboard"] = self.generate_dashboard(results)
        chart_paths["comparison_sharpe"] = self.generate_comparison_chart(results, "sharpe_ratio")
        chart_paths["comparison_return"] = self.generate_comparison_chart(
            results, "annualized_return"
        )
        chart_paths["walk_forward"] = self.generate_walk_forward_chart(results)

        for name, result in results.items():
            equity = getattr(result, "equity_curve", None)
            if equity is not None:
                label = result.name if hasattr(result, "name") else name
                chart_paths[f"equity_{name}"] = self.generate_equity_curve(equity, label)

        # Build HTML
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            f"<title>{title}</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }",
            ".container { max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            "h1 { color: #333; border-bottom: 2px solid #4472C4; padding-bottom: 10px; }",
            "h2 { color: #4472C4; margin-top: 30px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 12px; text-align: center; }",
            "th { background: #4472C4; color: white; }",
            "tr:nth-child(even) { background: #f9f9f9; }",
            ".positive { color: green; font-weight: bold; }",
            ".negative { color: red; font-weight: bold; }",
            "img { max-width: 100%; height: auto; margin: 20px 0; border-radius: 4px; }",
            ".metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }",
            ".metric-card { background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; }",
            ".metric-value { font-size: 24px; font-weight: bold; color: #333; }",
            ".metric-label { font-size: 12px; color: #666; margin-top: 5px; }",
            "</style>",
            "</head><body>",
            '<div class="container">',
            f"<h1>{title}</h1>",
            f"<p>Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]

        # Summary metrics
        if results:
            best_sharpe = max(results.values(), key=lambda r: getattr(r, "sharpe_ratio", 0))
            best_return = max(results.values(), key=lambda r: getattr(r, "annualized_return", 0))
            avg_sharpe = np.mean([getattr(r, "sharpe_ratio", 0) for r in results.values()])

            html_parts.extend(
                [
                    '<div class="metric-grid">',
                    f'<div class="metric-card"><div class="metric-value">{len(results)}</div><div class="metric-label">Strategies Tested</div></div>',
                    f'<div class="metric-card"><div class="metric-value">{avg_sharpe:.2f}</div><div class="metric-label">Avg Sharpe Ratio</div></div>',
                    f'<div class="metric-card"><div class="metric-value">{getattr(best_sharpe, "name", "N/A")}</div><div class="metric-label">Best Sharpe</div></div>',
                    f'<div class="metric-card"><div class="metric-value">{getattr(best_return, "annualized_return", 0)*100:.1f}%</div><div class="metric-label">Best Return</div></div>',
                    "</div>",
                ]
            )

        # Dashboard
        if chart_paths.get("dashboard"):
            html_parts.extend(
                [
                    "<h2>Dashboard</h2>",
                    f'<img src="{chart_paths["dashboard"].name}" alt="Dashboard">',
                ]
            )

        # Results table
        html_parts.extend(
            [
                "<h2>Strategy Results</h2>",
                "<table>",
                "<tr><th>Strategy</th><th>Return</th><th>Sharpe</th><th>Max DD</th><th>Win Rate</th><th>Trades</th><th>Train/Test</th></tr>",
            ]
        )

        for name, result in results.items():
            ret = getattr(result, "annualized_return", 0)
            sharpe = getattr(result, "sharpe_ratio", 0)
            ret_class = "positive" if ret > 0 else "negative"
            sharpe_class = "positive" if sharpe > 0 else "negative"

            train = getattr(result, "train_sharpe", None)
            test = getattr(result, "test_sharpe", None)
            wf = f"{train:.2f}/{test:.2f}" if train is not None else "N/A"

            html_parts.append(
                f'<tr>'
                f'<td>{getattr(result, "name", name)}</td>'
                f'<td class="{ret_class}">{ret*100:.2f}%</td>'
                f'<td class="{sharpe_class}">{sharpe:.2f}</td>'
                f'<td>{getattr(result, "max_drawdown", 0)*100:.2f}%</td>'
                f'<td>{getattr(result, "win_rate", 0)*100:.1f}%</td>'
                f'<td>{getattr(result, "total_trades", 0)}</td>'
                f'<td>{wf}</td>'
                f'</tr>'
            )

        html_parts.append("</table>")

        # Comparison charts
        for key in ["comparison_sharpe", "comparison_return", "walk_forward"]:
            if chart_paths.get(key):
                title_map = {
                    "comparison_sharpe": "Sharpe Ratio Comparison",
                    "comparison_return": "Return Comparison",
                    "walk_forward": "Walk-Forward Validation",
                }
                html_parts.extend(
                    [
                        f"<h2>{title_map.get(key, key)}</h2>",
                        f'<img src="{chart_paths[key].name}" alt="{key}">',
                    ]
                )

        # Individual equity curves
        html_parts.append("<h2>Individual Equity Curves</h2>")
        for key, path in chart_paths.items():
            if key.startswith("equity_") and path:
                html_parts.append(f'<img src="{path.name}" alt="{key}">')

        html_parts.extend(["</div></body></html>"])

        # Write HTML
        filepath = self.output_dir / "report.html"
        filepath.write_text("\n".join(html_parts))

        logger.info(f"Saved HTML report: {filepath}")
        return filepath


def generate_all_visualizations(
    results: dict[str, Any],
    output_dir: str = "artifacts/sprint/viz",
) -> dict[str, Path]:
    """Convenience function to generate all visualizations."""
    viz = StrategyVisualizer(output_dir)

    paths = {}
    paths["dashboard"] = viz.generate_dashboard(results)
    paths["comparison_sharpe"] = viz.generate_comparison_chart(results, "sharpe_ratio")
    paths["comparison_return"] = viz.generate_comparison_chart(results, "annualized_return")
    paths["walk_forward"] = viz.generate_walk_forward_chart(results)
    paths["report"] = viz.generate_html_report(results)

    return paths
