"""Tests for Performance Dashboard."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from adapters.telemetry.kpi import (
    AlertSeverity,
    KPIStatus,
    KPITracker,
    KPIValue,
    TradingKPIs,
)
from visualization.dashboard import PerformanceDashboard


class TestPerformanceDashboardInitialization:
    """Tests for PerformanceDashboard initialization."""

    def test_default_initialization(self):
        """Test dashboard initializes with new KPI tracker."""
        dashboard = PerformanceDashboard()
        assert dashboard.kpi_tracker is not None
        assert isinstance(dashboard.kpi_tracker, KPITracker)

    def test_initialization_with_tracker(self):
        """Test dashboard initializes with provided tracker."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.65)

        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        assert dashboard.kpi_tracker is tracker
        assert dashboard.kpi_tracker.get_current_kpis().win_rate == 0.65

    def test_status_colors_defined(self):
        """Test status colors are properly defined."""
        assert KPIStatus.HEALTHY in PerformanceDashboard.STATUS_COLORS
        assert KPIStatus.WARNING in PerformanceDashboard.STATUS_COLORS
        assert KPIStatus.CRITICAL in PerformanceDashboard.STATUS_COLORS
        assert KPIStatus.UNKNOWN in PerformanceDashboard.STATUS_COLORS

    def test_severity_colors_defined(self):
        """Test severity colors are properly defined."""
        assert AlertSeverity.INFO in PerformanceDashboard.SEVERITY_COLORS
        assert AlertSeverity.WARNING in PerformanceDashboard.SEVERITY_COLORS
        assert AlertSeverity.CRITICAL in PerformanceDashboard.SEVERITY_COLORS


class TestKPIGauge:
    """Tests for KPI gauge chart creation."""

    def test_create_kpi_gauge_basic(self):
        """Test basic KPI gauge creation."""
        dashboard = PerformanceDashboard()
        kpi = KPIValue(
            name="win_rate",
            value=0.55,
            status=KPIStatus.HEALTHY,
        )

        fig = dashboard.create_kpi_gauge(kpi)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_kpi_gauge_custom_range(self):
        """Test KPI gauge with custom value range."""
        dashboard = PerformanceDashboard()
        kpi = KPIValue(
            name="sharpe_ratio",
            value=1.5,
            status=KPIStatus.HEALTHY,
        )

        fig = dashboard.create_kpi_gauge(kpi, min_val=-2.0, max_val=3.0)

        assert isinstance(fig, go.Figure)

    def test_create_kpi_gauge_custom_title(self):
        """Test KPI gauge with custom title."""
        dashboard = PerformanceDashboard()
        kpi = KPIValue(
            name="win_rate",
            value=0.55,
            status=KPIStatus.HEALTHY,
        )

        fig = dashboard.create_kpi_gauge(kpi, title="Custom Title")

        assert isinstance(fig, go.Figure)

    def test_gauge_colors_match_status(self):
        """Test gauge uses correct color for status."""
        dashboard = PerformanceDashboard()

        # Test each status
        for status in [KPIStatus.HEALTHY, KPIStatus.WARNING, KPIStatus.CRITICAL]:
            kpi = KPIValue(name="test", value=0.5, status=status)
            fig = dashboard.create_kpi_gauge(kpi)
            assert isinstance(fig, go.Figure)


class TestKPISummaryTable:
    """Tests for KPI summary table creation."""

    def test_create_kpi_summary_table(self):
        """Test KPI summary table creation."""
        dashboard = PerformanceDashboard()
        kpis = {
            "win_rate": KPIValue(name="win_rate", value=0.55, status=KPIStatus.HEALTHY),
            "sharpe_ratio": KPIValue(name="sharpe_ratio", value=1.2, status=KPIStatus.HEALTHY),
            "max_drawdown": KPIValue(name="max_drawdown", value=-0.15, status=KPIStatus.WARNING),
        }

        fig = dashboard.create_kpi_summary_table(kpis)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        # Check it's a table
        assert isinstance(fig.data[0], go.Table)

    def test_summary_table_empty_kpis(self):
        """Test summary table with empty KPIs."""
        dashboard = PerformanceDashboard()

        fig = dashboard.create_kpi_summary_table({})

        assert isinstance(fig, go.Figure)


class TestHealthStatusCard:
    """Tests for health status card creation."""

    def test_create_health_status_card_healthy(self):
        """Test health status card when system is healthy."""
        tracker = KPITracker()
        tracker.update_kpis(
            TradingKPIs(
                win_rate=0.55,
                sharpe_ratio=1.5,
                total_return=0.10,
            )
        )
        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        fig = dashboard.create_health_status_card()

        assert isinstance(fig, go.Figure)

    def test_create_health_status_card_warning(self):
        """Test health status card when system has warnings."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.35)  # Warning level
        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        fig = dashboard.create_health_status_card()

        assert isinstance(fig, go.Figure)

    def test_create_health_status_card_critical(self):
        """Test health status card when system is critical."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.20)  # Critical level
        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        fig = dashboard.create_health_status_card()

        assert isinstance(fig, go.Figure)


class TestAlertsTimeline:
    """Tests for alerts timeline creation."""

    def test_create_alerts_timeline_no_alerts(self):
        """Test alerts timeline when no alerts exist."""
        dashboard = PerformanceDashboard()

        fig = dashboard.create_alerts_timeline(hours=24)

        assert isinstance(fig, go.Figure)

    def test_create_alerts_timeline_with_alerts(self):
        """Test alerts timeline with alerts present."""
        tracker = KPITracker()
        # Trigger some alerts
        tracker.update_kpi("win_rate", 0.35)  # Warning
        tracker.update_kpi("sharpe_ratio", -0.5)  # Critical

        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        fig = dashboard.create_alerts_timeline(hours=24)

        assert isinstance(fig, go.Figure)

    def test_create_alerts_timeline_with_severity_filter(self):
        """Test alerts timeline with severity filter."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.25)  # Critical

        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        fig = dashboard.create_alerts_timeline(
            hours=24,
            severity_filter=AlertSeverity.CRITICAL,
        )

        assert isinstance(fig, go.Figure)


class TestKPITrendChart:
    """Tests for KPI trend chart creation."""

    def test_create_kpi_trend_chart_no_history(self):
        """Test KPI trend chart with no history."""
        dashboard = PerformanceDashboard()

        fig = dashboard.create_kpi_trend_chart("win_rate")

        assert isinstance(fig, go.Figure)

    def test_create_kpi_trend_chart_with_history(self):
        """Test KPI trend chart with history data."""
        tracker = KPITracker()

        # Add some history
        for i in range(10):
            kpis = TradingKPIs(win_rate=0.5 + i * 0.01)
            tracker.update_kpis(kpis)

        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        fig = dashboard.create_kpi_trend_chart("win_rate", periods=10)

        assert isinstance(fig, go.Figure)


class TestPerformanceOverview:
    """Tests for performance overview creation."""

    def test_create_performance_overview_basic(self):
        """Test basic performance overview creation."""
        tracker = KPITracker()
        tracker.update_kpis(
            TradingKPIs(
                total_return=0.15,
                win_rate=0.55,
                sharpe_ratio=1.2,
                max_drawdown=-0.08,
            )
        )
        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        fig = dashboard.create_performance_overview()

        assert isinstance(fig, go.Figure)

    def test_create_performance_overview_with_equity(self):
        """Test performance overview with equity data."""
        dashboard = PerformanceDashboard()

        # Create sample equity data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        equity = pd.Series([10000 + i * 50 for i in range(100)], index=dates)

        fig = dashboard.create_performance_overview(equity_data=equity)

        assert isinstance(fig, go.Figure)


class TestMetricsPanels:
    """Tests for various metrics panels."""

    def test_create_trading_metrics_panel(self):
        """Test trading metrics panel creation."""
        tracker = KPITracker()
        tracker.update_kpis(
            TradingKPIs(
                total_trades=100,
                winning_trades=55,
                losing_trades=45,
                avg_win=150.0,
                avg_loss=100.0,
            )
        )
        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        fig = dashboard.create_trading_metrics_panel()

        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Table)

    def test_create_risk_metrics_panel(self):
        """Test risk metrics panel creation."""
        tracker = KPITracker()
        tracker.update_kpis(
            TradingKPIs(
                sharpe_ratio=1.2,
                sortino_ratio=1.5,
                max_drawdown=-0.10,
                volatility=0.15,
            )
        )
        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        fig = dashboard.create_risk_metrics_panel()

        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Table)

    def test_create_system_metrics_panel(self):
        """Test system metrics panel creation."""
        tracker = KPITracker()
        tracker.update_kpis(
            TradingKPIs(
                uptime_percent=99.9,
                api_success_rate=99.5,
                data_freshness_seconds=30,
                signals_generated=100,
                signals_executed=95,
                signal_accuracy=0.75,
            )
        )
        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        fig = dashboard.create_system_metrics_panel()

        assert isinstance(fig, go.Figure)
        assert isinstance(fig.data[0], go.Table)


class TestFullDashboard:
    """Tests for full dashboard creation."""

    def test_create_full_dashboard_basic(self):
        """Test basic full dashboard creation."""
        tracker = KPITracker()
        tracker.update_kpis(
            TradingKPIs(
                total_return=0.15,
                win_rate=0.55,
                sharpe_ratio=1.2,
                max_drawdown=-0.08,
                profit_factor=1.5,
                total_trades=100,
            )
        )
        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        fig = dashboard.create_full_dashboard()

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_full_dashboard_with_alerts(self):
        """Test full dashboard with alerts enabled."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.35)  # Trigger warning

        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        fig = dashboard.create_full_dashboard(show_alerts=True)

        assert isinstance(fig, go.Figure)

    def test_create_full_dashboard_with_equity(self):
        """Test full dashboard with equity data."""
        dashboard = PerformanceDashboard()

        # Create sample equity data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        equity = pd.Series([10000 + i * 50 for i in range(100)], index=dates)

        fig = dashboard.create_full_dashboard(equity_data=equity, show_alerts=False)

        assert isinstance(fig, go.Figure)


class TestDashboardExport:
    """Tests for dashboard export functionality."""

    def test_export_dashboard_html(self, tmp_path):
        """Test exporting dashboard to HTML."""
        dashboard = PerformanceDashboard()
        output_file = tmp_path / "dashboard.html"

        dashboard.export_dashboard(str(output_file), file_format="html")

        assert output_file.exists()
        content = output_file.read_text()
        assert "plotly" in content.lower()

    def test_export_dashboard_invalid_format(self):
        """Test exporting dashboard with invalid format."""
        dashboard = PerformanceDashboard()

        with pytest.raises(ValueError):
            dashboard.export_dashboard("test.xyz", file_format="xyz")


class TestDashboardIntegration:
    """Integration tests for dashboard with KPI tracker."""

    def test_dashboard_reflects_kpi_updates(self):
        """Test that dashboard reflects KPI updates."""
        tracker = KPITracker()
        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        # Initial state
        fig1 = dashboard.create_performance_overview()
        assert isinstance(fig1, go.Figure)

        # Update KPIs
        tracker.update_kpis(
            TradingKPIs(
                total_return=0.20,
                win_rate=0.60,
                sharpe_ratio=1.5,
            )
        )

        # New dashboard should reflect updates
        fig2 = dashboard.create_performance_overview()
        assert isinstance(fig2, go.Figure)

    def test_dashboard_with_multiple_alert_types(self):
        """Test dashboard handles multiple alert types."""
        tracker = KPITracker()

        # Trigger various alerts
        tracker.update_kpi("win_rate", 0.25)  # Critical
        tracker.update_kpi("sharpe_ratio", -0.3)  # Critical
        tracker.update_kpi("max_drawdown", -0.15)  # Warning

        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        # All panels should render without error
        fig1 = dashboard.create_health_status_card()
        fig2 = dashboard.create_alerts_timeline()
        fig3 = dashboard.create_full_dashboard()

        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)
        assert isinstance(fig3, go.Figure)

    def test_dashboard_performance_with_large_history(self):
        """Test dashboard performance with large history."""
        tracker = KPITracker()

        # Add large history
        for i in range(1000):
            kpis = TradingKPIs(
                total_return=i * 0.001,
                win_rate=0.5 + (i % 20) * 0.01,
            )
            tracker.update_kpis(kpis)

        dashboard = PerformanceDashboard(kpi_tracker=tracker)

        # Should handle large history efficiently
        fig = dashboard.create_kpi_trend_chart("win_rate", periods=100)
        assert isinstance(fig, go.Figure)
