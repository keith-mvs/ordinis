"""
Performance Dashboard.

Provides interactive dashboard for trading performance monitoring,
KPI visualization, and system health status.
"""

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ordinis.adapters.telemetry.kpi import (
    AlertSeverity,
    KPIStatus,
    KPITracker,
    KPIValue,
)


class PerformanceDashboard:
    """
    Interactive performance dashboard for trading system monitoring.

    Displays KPIs, alerts, equity curves, and system health status
    using Plotly for interactive visualizations.
    """

    # Status colors
    STATUS_COLORS = {
        KPIStatus.HEALTHY: "#00cc00",
        KPIStatus.WARNING: "#ffcc00",
        KPIStatus.CRITICAL: "#ff0000",
        KPIStatus.UNKNOWN: "#888888",
    }

    # Severity colors
    SEVERITY_COLORS = {
        AlertSeverity.INFO: "#3498db",
        AlertSeverity.WARNING: "#f39c12",
        AlertSeverity.CRITICAL: "#e74c3c",
    }

    def __init__(self, kpi_tracker: KPITracker | None = None):
        """
        Initialize dashboard.

        Args:
            kpi_tracker: KPITracker instance (optional, creates new if not provided)
        """
        self.kpi_tracker = kpi_tracker or KPITracker()

    def create_kpi_gauge(
        self,
        kpi: KPIValue,
        min_val: float = 0.0,
        max_val: float = 1.0,
        title: str | None = None,
    ) -> go.Figure:
        """
        Create gauge chart for single KPI.

        Args:
            kpi: KPIValue object
            min_val: Minimum gauge value
            max_val: Maximum gauge value
            title: Custom title (uses KPI name if not provided)

        Returns:
            Plotly Figure object
        """
        title = title or kpi.name.replace("_", " ").title()
        color = self.STATUS_COLORS.get(kpi.status, "#888888")

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=kpi.value,
                title={"text": title, "font": {"size": 16}},
                number={"font": {"size": 24}},
                gauge={
                    "axis": {"range": [min_val, max_val], "tickwidth": 1},
                    "bar": {"color": color},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {
                            "range": [min_val, (max_val - min_val) * 0.3 + min_val],
                            "color": "#ffcccc",
                        },
                        {
                            "range": [
                                (max_val - min_val) * 0.3 + min_val,
                                (max_val - min_val) * 0.7 + min_val,
                            ],
                            "color": "#ffffcc",
                        },
                        {
                            "range": [(max_val - min_val) * 0.7 + min_val, max_val],
                            "color": "#ccffcc",
                        },
                    ],
                },
            )
        )

        fig.update_layout(
            height=200,
            margin={"l": 30, "r": 30, "t": 50, "b": 20},
        )

        return fig

    def create_kpi_summary_table(self, kpis: dict[str, KPIValue]) -> go.Figure:
        """
        Create table showing all KPIs with status indicators.

        Args:
            kpis: Dictionary of KPI name to KPIValue

        Returns:
            Plotly Figure object with table
        """
        # Prepare data
        names = []
        values = []
        statuses = []
        colors = []

        for name, kpi in kpis.items():
            names.append(name.replace("_", " ").title())
            values.append(f"{kpi.value:.4f}" if isinstance(kpi.value, float) else str(kpi.value))
            statuses.append(kpi.status.value.upper())
            colors.append(self.STATUS_COLORS.get(kpi.status, "#888888"))

        fig = go.Figure(
            data=[
                go.Table(
                    header={
                        "values": ["KPI", "Value", "Status"],
                        "fill_color": "#2c3e50",
                        "font": {"color": "white", "size": 14},
                        "align": "left",
                        "height": 40,
                    },
                    cells={
                        "values": [names, values, statuses],
                        "fill_color": [["white"] * len(names), ["white"] * len(names), colors],
                        "font": {"color": ["black", "black", "white"], "size": 12},
                        "align": "left",
                        "height": 30,
                    },
                )
            ]
        )

        fig.update_layout(
            title="KPI Summary",
            height=max(400, len(names) * 35 + 100),
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
        )

        return fig

    def create_health_status_card(self) -> go.Figure:
        """
        Create health status summary card.

        Returns:
            Plotly Figure object
        """
        status = self.kpi_tracker.get_health_status()
        overall = status["overall_status"]

        # Color based on overall status
        if overall == "healthy":
            color = "#00cc00"
            icon = "checkmark"
        elif overall == "warning":
            color = "#ffcc00"
            icon = "warning"
        else:
            color = "#ff0000"
            icon = "alert"

        fig = go.Figure()

        # Status indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=status["healthy_count"],
                title={
                    "text": f"System Status: {overall.upper()}",
                    "font": {"size": 20, "color": color},
                },
                number={"suffix": " healthy KPIs", "font": {"size": 16}},
                delta={
                    "reference": status["healthy_count"]
                    + status["warning_count"]
                    + status["critical_count"],
                    "position": "bottom",
                },
            )
        )

        # Add status counts as annotation
        fig.add_annotation(
            x=0.5,
            y=-0.1,
            text=f"<span style='color:#00cc00'>Healthy: {status['healthy_count']}</span> | "
            f"<span style='color:#ffcc00'>Warning: {status['warning_count']}</span> | "
            f"<span style='color:#ff0000'>Critical: {status['critical_count']}</span>",
            showarrow=False,
            font={"size": 14},
            xanchor="center",
        )

        fig.update_layout(
            height=200,
            margin={"l": 20, "r": 20, "t": 50, "b": 50},
        )

        return fig

    def create_alerts_timeline(
        self,
        hours: int = 24,
        severity_filter: AlertSeverity | None = None,
    ) -> go.Figure:
        """
        Create timeline of recent alerts.

        Args:
            hours: Number of hours to look back
            severity_filter: Filter by severity (optional)

        Returns:
            Plotly Figure object
        """
        # Get alert history
        alerts = self.kpi_tracker.get_alert_history(
            since=datetime.now() - timedelta(hours=hours),
            severity=severity_filter,
        )

        if not alerts:
            # Empty state
            fig = go.Figure()
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text="No alerts in the selected time period",
                showarrow=False,
                font={"size": 16, "color": "gray"},
                xanchor="center",
            )
            fig.update_layout(
                title=f"Alerts (Last {hours} hours)",
                height=300,
                xaxis={"visible": False},
                yaxis={"visible": False},
            )
            return fig

        # Prepare data
        timestamps = [a.timestamp for a in alerts]
        severities = [a.severity.value for a in alerts]
        colors = [self.SEVERITY_COLORS.get(a.severity, "#888888") for a in alerts]
        messages = [f"{a.kpi_name}: {a.message}" for a in alerts]

        fig = go.Figure()

        # Scatter plot for alerts
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=severities,
                mode="markers",
                marker={"color": colors, "size": 15, "symbol": "diamond"},
                text=messages,
                hovertemplate="<b>%{y}</b><br>%{text}<br>%{x}<extra></extra>",
                name="Alerts",
            )
        )

        fig.update_layout(
            title=f"Alerts Timeline (Last {hours} hours)",
            xaxis_title="Time",
            yaxis_title="Severity",
            height=300,
            margin={"l": 60, "r": 20, "t": 50, "b": 50},
            hovermode="closest",
        )

        return fig

    def create_kpi_trend_chart(
        self,
        kpi_name: str,
        periods: int = 50,
    ) -> go.Figure:
        """
        Create trend chart for a specific KPI over time.

        Args:
            kpi_name: Name of the KPI to chart
            periods: Number of periods to display

        Returns:
            Plotly Figure object
        """
        # Get KPI history
        history = self.kpi_tracker._kpi_history[-periods:] if self.kpi_tracker._kpi_history else []

        if not history:
            fig = go.Figure()
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text="No historical data available",
                showarrow=False,
                font={"size": 16, "color": "gray"},
                xanchor="center",
            )
            fig.update_layout(
                title=f"{kpi_name.replace('_', ' ').title()} Trend",
                height=300,
                xaxis={"visible": False},
                yaxis={"visible": False},
            )
            return fig

        # Extract KPI values
        timestamps = []
        values = []

        for kpis in history:
            if hasattr(kpis, kpi_name):
                timestamps.append(kpis.timestamp if hasattr(kpis, "timestamp") else datetime.now())
                values.append(getattr(kpis, kpi_name))

        if not values:
            fig = go.Figure()
            fig.add_annotation(
                x=0.5,
                y=0.5,
                text=f"No data for {kpi_name}",
                showarrow=False,
                font={"size": 16, "color": "gray"},
                xanchor="center",
            )
            fig.update_layout(
                title=f"{kpi_name.replace('_', ' ').title()} Trend",
                height=300,
                xaxis={"visible": False},
                yaxis={"visible": False},
            )
            return fig

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=values,
                mode="lines+markers",
                line={"color": "#3498db", "width": 2},
                marker={"size": 6},
                name=kpi_name,
            )
        )

        fig.update_layout(
            title=f"{kpi_name.replace('_', ' ').title()} Trend",
            xaxis_title="Time",
            yaxis_title=kpi_name.replace("_", " ").title(),
            height=300,
            margin={"l": 60, "r": 20, "t": 50, "b": 50},
            hovermode="x unified",
        )

        return fig

    def create_performance_overview(
        self,
        equity_data: pd.Series | None = None,
        trades_data: pd.DataFrame | None = None,
    ) -> go.Figure:
        """
        Create comprehensive performance overview with multiple panels.

        Args:
            equity_data: Equity curve series (optional)
            trades_data: Trade history DataFrame (optional)

        Returns:
            Plotly Figure object with subplots
        """
        kpis = self.kpi_tracker.get_current_kpis()

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}],
            ],
            subplot_titles=(
                "Total Return",
                "Win Rate",
                "Sharpe Ratio",
                "Max Drawdown",
            ),
        )

        # Total Return
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=kpis.total_return * 100,
                number={"suffix": "%", "font": {"size": 28}},
                delta={"reference": 0, "relative": False, "position": "bottom"},
            ),
            row=1,
            col=1,
        )

        # Win Rate
        fig.add_trace(
            go.Indicator(
                mode="number+gauge",
                value=kpis.win_rate * 100,
                number={"suffix": "%", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#00cc00" if kpis.win_rate >= 0.5 else "#ff9900"},
                    "steps": [
                        {"range": [0, 40], "color": "#ffcccc"},
                        {"range": [40, 60], "color": "#ffffcc"},
                        {"range": [60, 100], "color": "#ccffcc"},
                    ],
                },
            ),
            row=1,
            col=2,
        )

        # Sharpe Ratio
        sharpe_color = (
            "#00cc00"
            if kpis.sharpe_ratio >= 1.0
            else "#ff9900"
            if kpis.sharpe_ratio >= 0
            else "#ff0000"
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=kpis.sharpe_ratio,
                number={"font": {"size": 28, "color": sharpe_color}},
            ),
            row=2,
            col=1,
        )

        # Max Drawdown
        dd_color = (
            "#00cc00"
            if kpis.max_drawdown >= -0.10
            else "#ff9900"
            if kpis.max_drawdown >= -0.20
            else "#ff0000"
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=kpis.max_drawdown * 100,
                number={"suffix": "%", "font": {"size": 28, "color": dd_color}},
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title="Performance Overview",
            height=500,
            margin={"l": 30, "r": 30, "t": 80, "b": 30},
        )

        return fig

    def create_trading_metrics_panel(self) -> go.Figure:
        """
        Create panel showing detailed trading metrics.

        Returns:
            Plotly Figure object
        """
        kpis = self.kpi_tracker.get_current_kpis()

        # Prepare metrics
        metrics = [
            ("Total Trades", kpis.total_trades, ""),
            ("Winning Trades", kpis.winning_trades, ""),
            ("Losing Trades", kpis.losing_trades, ""),
            ("Win Rate", kpis.win_rate * 100, "%"),
            ("Avg Win", kpis.avg_win, "$"),
            ("Avg Loss", kpis.avg_loss, "$"),
            ("Profit Factor", kpis.profit_factor, ""),
            ("Expectancy", kpis.expectancy, "$"),
            ("Win/Loss Ratio", kpis.win_loss_ratio, ""),
        ]

        names = [m[0] for m in metrics]
        values = [
            f"{m[1]:.2f}{m[2]}" if isinstance(m[1], float) else f"{m[1]}{m[2]}" for m in metrics
        ]

        fig = go.Figure(
            data=[
                go.Table(
                    header={
                        "values": ["Metric", "Value"],
                        "fill_color": "#34495e",
                        "font": {"color": "white", "size": 14},
                        "align": "left",
                        "height": 40,
                    },
                    cells={
                        "values": [names, values],
                        "fill_color": [["#ecf0f1"] * len(names), ["white"] * len(names)],
                        "font": {"color": "black", "size": 12},
                        "align": "left",
                        "height": 30,
                    },
                )
            ]
        )

        fig.update_layout(
            title="Trading Metrics",
            height=400,
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
        )

        return fig

    def create_risk_metrics_panel(self) -> go.Figure:
        """
        Create panel showing risk metrics.

        Returns:
            Plotly Figure object
        """
        kpis = self.kpi_tracker.get_current_kpis()

        # Prepare risk metrics (using available TradingKPIs attributes)
        metrics = [
            ("Sharpe Ratio", kpis.sharpe_ratio, ""),
            ("Sortino Ratio", kpis.sortino_ratio, ""),
            ("Max Drawdown", kpis.max_drawdown * 100, "%"),
            ("Current Drawdown", kpis.current_drawdown * 100, "%"),
            ("Volatility", kpis.volatility * 100, "%"),
            ("Downside Dev", kpis.downside_deviation * 100, "%"),
        ]

        names = [m[0] for m in metrics]
        values = [
            f"{m[1]:.2f}{m[2]}" if isinstance(m[1], float) else f"{m[1]}{m[2]}" for m in metrics
        ]

        # Color based on risk level
        colors = []
        for name, val, unit in metrics:
            if "drawdown" in name.lower():
                # val is already in %, so -10 means -10%
                color = "#ccffcc" if val >= -10 else "#ffffcc" if val >= -20 else "#ffcccc"
            elif "ratio" in name.lower():
                color = "#ccffcc" if val >= 1 else "#ffffcc" if val >= 0 else "#ffcccc"
            elif "volatility" in name.lower() or "deviation" in name.lower():
                # Lower volatility is better
                color = "#ccffcc" if val <= 15 else "#ffffcc" if val <= 25 else "#ffcccc"
            else:
                color = "white"
            colors.append(color)

        fig = go.Figure(
            data=[
                go.Table(
                    header={
                        "values": ["Metric", "Value"],
                        "fill_color": "#8e44ad",
                        "font": {"color": "white", "size": 14},
                        "align": "left",
                        "height": 40,
                    },
                    cells={
                        "values": [names, values],
                        "fill_color": [["#ecf0f1"] * len(names), colors],
                        "font": {"color": "black", "size": 12},
                        "align": "left",
                        "height": 30,
                    },
                )
            ]
        )

        fig.update_layout(
            title="Risk Metrics",
            height=400,
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
        )

        return fig

    def create_system_metrics_panel(self) -> go.Figure:
        """
        Create panel showing system/operational metrics.

        Returns:
            Plotly Figure object
        """
        kpis = self.kpi_tracker.get_current_kpis()

        # System metrics (using available TradingKPIs attributes)
        metrics = [
            ("Uptime", kpis.uptime_percent, "%"),
            ("API Success Rate", kpis.api_success_rate, "%"),
            ("Data Freshness", kpis.data_freshness_seconds, "s"),
            ("Signals Generated", kpis.signals_generated, ""),
            ("Signals Executed", kpis.signals_executed, ""),
            ("Signal Accuracy", kpis.signal_accuracy * 100, "%"),
        ]

        names = [m[0] for m in metrics]
        values = [
            f"{m[1]:.2f}{m[2]}" if isinstance(m[1], float) else f"{m[1]}{m[2]}" for m in metrics
        ]

        # Status colors
        colors = []
        for name, val, _ in metrics:
            if "uptime" in name.lower() or "success" in name.lower():
                color = "#ccffcc" if val >= 99 else "#ffffcc" if val >= 95 else "#ffcccc"
            elif "freshness" in name.lower():
                color = "#ccffcc" if val <= 60 else "#ffffcc" if val <= 300 else "#ffcccc"
            elif "accuracy" in name.lower():
                # val is already in %, higher is better
                color = "#ccffcc" if val >= 80 else "#ffffcc" if val >= 60 else "#ffcccc"
            else:
                color = "white"
            colors.append(color)

        fig = go.Figure(
            data=[
                go.Table(
                    header={
                        "values": ["Metric", "Value"],
                        "fill_color": "#16a085",
                        "font": {"color": "white", "size": 14},
                        "align": "left",
                        "height": 40,
                    },
                    cells={
                        "values": [names, values],
                        "fill_color": [["#ecf0f1"] * len(names), colors],
                        "font": {"color": "black", "size": 12},
                        "align": "left",
                        "height": 30,
                    },
                )
            ]
        )

        fig.update_layout(
            title="System Metrics",
            height=350,
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
        )

        return fig

    def create_full_dashboard(
        self,
        equity_data: pd.Series | None = None,
        show_alerts: bool = True,
    ) -> go.Figure:
        """
        Create complete dashboard with all panels.

        Args:
            equity_data: Equity curve series (optional)
            show_alerts: Whether to show alerts timeline

        Returns:
            Combined Plotly Figure object
        """
        kpis = self.kpi_tracker.get_current_kpis()
        status = self.kpi_tracker.get_health_status()

        # Create complex subplot layout
        fig = make_subplots(
            rows=4,
            cols=3,
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"type": "table", "colspan": 2}, None, {"type": "table"}],
                [{"type": "scatter", "colspan": 3}, None, None],
            ],
            subplot_titles=(
                "Total Return",
                "Win Rate",
                "Sharpe Ratio",
                "Max Drawdown",
                "Profit Factor",
                "System Health",
                "Key Metrics",
                "",
                "Risk Metrics",
                "Alerts Timeline" if show_alerts else "Equity Curve",
            ),
            row_heights=[0.2, 0.2, 0.3, 0.3],
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
        )

        # Row 1: Key Performance Indicators
        # Total Return
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=kpis.total_return * 100,
                number={
                    "suffix": "%",
                    "font": {
                        "size": 24,
                        "color": "#00cc00" if kpis.total_return >= 0 else "#ff0000",
                    },
                },
            ),
            row=1,
            col=1,
        )

        # Win Rate
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=kpis.win_rate * 100,
                number={
                    "suffix": "%",
                    "font": {"size": 24, "color": "#00cc00" if kpis.win_rate >= 0.5 else "#ff9900"},
                },
            ),
            row=1,
            col=2,
        )

        # Sharpe Ratio
        sharpe_color = (
            "#00cc00"
            if kpis.sharpe_ratio >= 1.0
            else "#ff9900"
            if kpis.sharpe_ratio >= 0
            else "#ff0000"
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=kpis.sharpe_ratio,
                number={"font": {"size": 24, "color": sharpe_color}},
            ),
            row=1,
            col=3,
        )

        # Row 2: More KPIs
        # Max Drawdown
        dd_color = (
            "#00cc00"
            if kpis.max_drawdown >= -0.10
            else "#ff9900"
            if kpis.max_drawdown >= -0.20
            else "#ff0000"
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=kpis.max_drawdown * 100,
                number={"suffix": "%", "font": {"size": 24, "color": dd_color}},
            ),
            row=2,
            col=1,
        )

        # Profit Factor
        pf_color = (
            "#00cc00"
            if kpis.profit_factor >= 1.5
            else "#ff9900"
            if kpis.profit_factor >= 1.0
            else "#ff0000"
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=kpis.profit_factor,
                number={"font": {"size": 24, "color": pf_color}},
            ),
            row=2,
            col=2,
        )

        # System Health
        health_color = (
            "#00cc00"
            if status["overall_status"] == "healthy"
            else "#ff9900"
            if status["overall_status"] == "warning"
            else "#ff0000"
        )
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=status["healthy_count"],
                number={
                    "suffix": f"/{status['healthy_count'] + status['warning_count'] + status['critical_count']}",
                    "font": {"size": 24, "color": health_color},
                },
            ),
            row=2,
            col=3,
        )

        # Row 3: Tables
        # Key Metrics Table
        key_metrics = [
            ("Total Trades", str(kpis.total_trades)),
            ("Winning", str(kpis.winning_trades)),
            ("Losing", str(kpis.losing_trades)),
            ("Avg Win", f"${kpis.avg_win:.2f}"),
            ("Avg Loss", f"${kpis.avg_loss:.2f}"),
            ("Expectancy", f"${kpis.expectancy:.2f}"),
        ]

        fig.add_trace(
            go.Table(
                header={
                    "values": ["Metric", "Value"],
                    "fill_color": "#34495e",
                    "font": {"color": "white", "size": 11},
                    "height": 25,
                },
                cells={
                    "values": [[m[0] for m in key_metrics], [m[1] for m in key_metrics]],
                    "fill_color": "white",
                    "font": {"size": 10},
                    "height": 22,
                },
            ),
            row=3,
            col=1,
        )

        # Risk Metrics Table (using available TradingKPIs attributes)
        risk_metrics = [
            ("Sortino", f"{kpis.sortino_ratio:.2f}"),
            ("Volatility", f"{kpis.volatility * 100:.1f}%"),
            ("Current DD", f"{kpis.current_drawdown * 100:.1f}%"),
            ("Downside Dev", f"{kpis.downside_deviation * 100:.1f}%"),
        ]

        fig.add_trace(
            go.Table(
                header={
                    "values": ["Metric", "Value"],
                    "fill_color": "#8e44ad",
                    "font": {"color": "white", "size": 11},
                    "height": 25,
                },
                cells={
                    "values": [[m[0] for m in risk_metrics], [m[1] for m in risk_metrics]],
                    "fill_color": "white",
                    "font": {"size": 10},
                    "height": 22,
                },
            ),
            row=3,
            col=3,
        )

        # Row 4: Alerts or Equity
        if show_alerts:
            alerts = self.kpi_tracker.get_alert_history(since=datetime.now() - timedelta(hours=24))
            if alerts:
                timestamps = [a.timestamp for a in alerts]
                severities = [a.severity.value for a in alerts]
                colors = [self.SEVERITY_COLORS.get(a.severity, "#888888") for a in alerts]
                messages = [f"{a.kpi_name}: {a.message}" for a in alerts]

                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=severities,
                        mode="markers",
                        marker={"color": colors, "size": 10, "symbol": "diamond"},
                        text=messages,
                        hovertemplate="%{text}<extra></extra>",
                        name="Alerts",
                    ),
                    row=4,
                    col=1,
                )
        elif equity_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=equity_data.index,
                    y=equity_data.values,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor="rgba(0, 100, 255, 0.1)",
                    line={"color": "blue", "width": 2},
                    name="Equity",
                ),
                row=4,
                col=1,
            )

        # Layout
        fig.update_layout(
            title={
                "text": "Trading Performance Dashboard",
                "font": {"size": 24},
                "x": 0.5,
                "xanchor": "center",
            },
            height=900,
            showlegend=False,
            margin={"l": 40, "r": 40, "t": 80, "b": 40},
        )

        return fig

    def export_dashboard(
        self,
        filename: str,
        file_format: str = "html",
        width: int = 1200,
        height: int = 900,
    ) -> None:
        """
        Export full dashboard to file.

        Args:
            filename: Output filename
            file_format: Export format (html, png, jpg, svg)
            width: Image width (for image formats)
            height: Image height (for image formats)
        """
        fig = self.create_full_dashboard()

        if file_format == "html":
            fig.write_html(filename, include_plotlyjs="cdn")
        elif file_format in ("png", "jpg", "svg"):
            fig.write_image(filename, format=file_format, width=width, height=height)
        else:
            raise ValueError(f"Unsupported format: {file_format}")
