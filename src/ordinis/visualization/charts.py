"""
Chart utilities for visualization module.

Provides theming, export, and comparison utilities for trading charts.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


class ChartUtils:
    """Utility functions for chart generation, theming, and export."""

    # Color schemes
    DARK_THEME = {
        "template": "plotly_dark",
        "paper_bgcolor": "#1e1e1e",
        "plot_bgcolor": "#1e1e1e",
        "font_color": "#e0e0e0",
        "grid_color": "#333333",
    }

    LIGHT_THEME = {
        "template": "plotly_white",
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#ffffff",
        "font_color": "#2a2a2a",
        "grid_color": "#e5e5e5",
    }

    @staticmethod
    def apply_theme(fig: go.Figure, theme: str = "dark") -> go.Figure:
        """
        Apply consistent theme to charts.

        Args:
            fig: Plotly Figure object
            theme: Theme name ("dark" or "light")

        Returns:
            Figure with theme applied
        """
        theme_config = ChartUtils.DARK_THEME if theme == "dark" else ChartUtils.LIGHT_THEME

        fig.update_layout(
            template=theme_config["template"],
            paper_bgcolor=theme_config["paper_bgcolor"],
            plot_bgcolor=theme_config["plot_bgcolor"],
            font={"color": theme_config["font_color"]},
        )

        fig.update_xaxes(gridcolor=theme_config["grid_color"])
        fig.update_yaxes(gridcolor=theme_config["grid_color"])

        return fig

    @staticmethod
    def export_chart(
        fig: go.Figure,
        filename: str | Path,
        file_format: str = "html",
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """
        Export chart to file.

        Args:
            fig: Plotly Figure object
            filename: Output file path
            file_format: Export format ("html", "png", "jpg", "svg", "json")
            width: Image width in pixels (for image exports)
            height: Image height in pixels (for image exports)
        """
        filename = Path(filename)

        if file_format == "html":
            fig.write_html(str(filename), include_plotlyjs="cdn")
        elif file_format == "json":
            fig.write_json(str(filename))
        elif file_format in ("png", "jpg", "svg"):
            # Requires kaleido
            fig.write_image(str(filename), format=file_format, width=width, height=height)
        else:
            raise ValueError(f"Unsupported export format: {file_format}")

    @staticmethod
    def create_comparison_chart(
        strategies: dict[str, pd.DataFrame],
        metric: str = "cumulative_return",
        title: str = "Strategy Comparison",
    ) -> go.Figure:
        """
        Create multi-strategy comparison chart.

        Args:
            strategies: Dict mapping strategy name to DataFrame with performance metrics
            metric: Column name to plot (e.g., "cumulative_return", "equity_curve")
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Color palette
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
        ]

        for idx, (name, data) in enumerate(strategies.items()):
            color = colors[idx % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[metric],
                    name=name,
                    line={"color": color, "width": 2},
                    mode="lines",
                )
            )

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title=metric.replace("_", " ").title(),
            height=600,
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )

        return fig

    @staticmethod
    def create_equity_curve(
        equity_series: pd.Series,
        trades: pd.DataFrame | None = None,
        title: str = "Equity Curve",
    ) -> go.Figure:
        """
        Create equity curve with optional trade markers.

        Args:
            equity_series: Series with equity values over time
            trades: DataFrame with trade information (optional)
            title: Chart title

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=equity_series.index,
                y=equity_series.values,
                name="Equity",
                line={"color": "blue", "width": 2},
                fill="tozeroy",
                fillcolor="rgba(0, 100, 255, 0.1)",
            )
        )

        # Add trade markers if provided
        if trades is not None and len(trades) > 0:
            # Entry points
            entries = trades[trades["type"] == "entry"]
            if len(entries) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=entries["timestamp"],
                        y=entries["equity"],
                        mode="markers",
                        name="Entry",
                        marker={"color": "green", "size": 10, "symbol": "triangle-up"},
                    )
                )

            # Exit points
            exits = trades[trades["type"] == "exit"]
            if len(exits) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=exits["timestamp"],
                        y=exits["equity"],
                        mode="markers",
                        name="Exit",
                        marker={"color": "red", "size": 10, "symbol": "triangle-down"},
                    )
                )

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            height=500,
            hovermode="x unified",
        )

        return fig

    @staticmethod
    def create_drawdown_chart(
        equity_series: pd.Series,
        title: str = "Drawdown",
    ) -> go.Figure:
        """
        Create drawdown chart showing peak-to-trough declines.

        Args:
            equity_series: Series with equity values over time
            title: Chart title

        Returns:
            Plotly Figure object
        """
        # Calculate drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100  # Percentage drawdown

        fig = go.Figure()

        # Drawdown area
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                name="Drawdown",
                line={"color": "red", "width": 0},
                fill="tozeroy",
                fillcolor="rgba(255, 0, 0, 0.3)",
            )
        )

        # Zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400,
            hovermode="x unified",
        )

        return fig

    @staticmethod
    def create_returns_distribution(
        returns: pd.Series,
        title: str = "Returns Distribution",
        bins: int = 50,
    ) -> go.Figure:
        """
        Create histogram of returns distribution.

        Args:
            returns: Series of returns
            title: Chart title
            bins: Number of histogram bins

        Returns:
            Plotly Figure object
        """
        fig = go.Figure()

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns.values,
                name="Returns",
                nbinsx=bins,
                marker_color="blue",
                opacity=0.7,
            )
        )

        # Mean line
        mean_return = returns.mean()
        fig.add_vline(x=mean_return, line_dash="dash", line_color="green", annotation_text="Mean")

        # Layout
        fig.update_layout(
            title=title,
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            height=400,
            showlegend=False,
        )

        return fig
