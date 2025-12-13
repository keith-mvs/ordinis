"""Tests for KPI Tracking System."""

from datetime import timedelta

from adapters.telemetry.kpi import (
    Alert,
    AlertSeverity,
    KPIStatus,
    KPIThreshold,
    KPITracker,
    KPIValue,
    TradingKPIs,
    get_kpi_tracker,
    reset_kpi_tracker,
)


class TestKPIThreshold:
    """Tests for KPIThreshold class."""

    def test_evaluate_healthy(self):
        """Test healthy evaluation when within thresholds."""
        threshold = KPIThreshold(warning_min=0.3, warning_max=0.7)
        assert threshold.evaluate(0.5) == KPIStatus.HEALTHY

    def test_evaluate_warning_below_min(self):
        """Test warning when below minimum."""
        threshold = KPIThreshold(warning_min=0.3, critical_min=0.1)
        assert threshold.evaluate(0.25) == KPIStatus.WARNING

    def test_evaluate_warning_above_max(self):
        """Test warning when above maximum."""
        threshold = KPIThreshold(warning_max=0.7, critical_max=0.9)
        assert threshold.evaluate(0.8) == KPIStatus.WARNING

    def test_evaluate_critical_below_min(self):
        """Test critical when below critical minimum."""
        threshold = KPIThreshold(warning_min=0.3, critical_min=0.1)
        assert threshold.evaluate(0.05) == KPIStatus.CRITICAL

    def test_evaluate_critical_above_max(self):
        """Test critical when above critical maximum."""
        threshold = KPIThreshold(warning_max=0.7, critical_max=0.9)
        assert threshold.evaluate(0.95) == KPIStatus.CRITICAL

    def test_evaluate_no_thresholds(self):
        """Test evaluation with no thresholds returns healthy."""
        threshold = KPIThreshold()
        assert threshold.evaluate(100.0) == KPIStatus.HEALTHY
        assert threshold.evaluate(-100.0) == KPIStatus.HEALTHY

    def test_evaluate_only_min_threshold(self):
        """Test evaluation with only minimum threshold."""
        threshold = KPIThreshold(warning_min=0.0)
        assert threshold.evaluate(0.5) == KPIStatus.HEALTHY
        assert threshold.evaluate(-0.5) == KPIStatus.WARNING


class TestKPIValue:
    """Tests for KPIValue class."""

    def test_kpi_value_creation(self):
        """Test KPIValue creation."""
        kpi = KPIValue(
            name="win_rate",
            value=0.55,
            status=KPIStatus.HEALTHY,
            unit="%",
            description="Percentage of winning trades",
        )
        assert kpi.name == "win_rate"
        assert kpi.value == 0.55
        assert kpi.status == KPIStatus.HEALTHY
        assert kpi.unit == "%"

    def test_kpi_value_to_dict(self):
        """Test KPIValue to_dict method."""
        kpi = KPIValue(name="test", value=1.23, status=KPIStatus.WARNING)
        result = kpi.to_dict()
        assert result["name"] == "test"
        assert result["value"] == 1.23
        assert result["status"] == "warning"
        assert "timestamp" in result


class TestTradingKPIs:
    """Tests for TradingKPIs class."""

    def test_trading_kpis_defaults(self):
        """Test TradingKPIs default values."""
        kpis = TradingKPIs()
        assert kpis.total_return == 0.0
        assert kpis.win_rate == 0.0
        assert kpis.total_trades == 0
        assert kpis.sharpe_ratio == 0.0

    def test_trading_kpis_custom_values(self):
        """Test TradingKPIs with custom values."""
        kpis = TradingKPIs(
            total_return=0.15,
            win_rate=0.55,
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
        )
        assert kpis.total_return == 0.15
        assert kpis.win_rate == 0.55
        assert kpis.total_trades == 100

    def test_win_loss_ratio(self):
        """Test win/loss ratio calculation."""
        kpis = TradingKPIs(avg_win=100.0, avg_loss=50.0)
        assert kpis.win_loss_ratio == 2.0

    def test_win_loss_ratio_zero_loss(self):
        """Test win/loss ratio with zero loss."""
        kpis = TradingKPIs(avg_win=100.0, avg_loss=0.0)
        assert kpis.win_loss_ratio == float("inf")

    def test_expectancy(self):
        """Test expectancy calculation."""
        kpis = TradingKPIs(win_rate=0.6, avg_win=100.0, avg_loss=50.0)
        # Expectancy = (0.6 * 100) - (0.4 * 50) = 60 - 20 = 40
        assert kpis.expectancy == 40.0

    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        kpis = TradingKPIs(total_trades=100, winning_trades=60, losing_trades=40)
        kpis.calculate_win_rate()
        assert kpis.win_rate == 0.6

    def test_calculate_win_rate_no_trades(self):
        """Test win rate with no trades."""
        kpis = TradingKPIs()
        kpis.calculate_win_rate()
        assert kpis.win_rate == 0.0

    def test_to_dict(self):
        """Test TradingKPIs to_dict method."""
        kpis = TradingKPIs(total_return=0.15, win_rate=0.55)
        result = kpis.to_dict()
        assert result["total_return"] == 0.15
        assert result["win_rate"] == 0.55
        assert "timestamp" in result
        assert "expectancy" in result
        assert "win_loss_ratio" in result


class TestAlert:
    """Tests for Alert class."""

    def test_alert_creation(self):
        """Test Alert creation."""
        alert = Alert(
            kpi_name="win_rate",
            severity=AlertSeverity.WARNING,
            message="Win rate below threshold",
            value=0.35,
            threshold=0.4,
        )
        assert alert.kpi_name == "win_rate"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.value == 0.35
        assert alert.threshold == 0.4

    def test_alert_to_dict(self):
        """Test Alert to_dict method."""
        alert = Alert(
            kpi_name="test",
            severity=AlertSeverity.CRITICAL,
            message="Test alert",
            value=0.0,
        )
        result = alert.to_dict()
        assert result["kpi_name"] == "test"
        assert result["severity"] == "critical"
        assert "timestamp" in result


class TestKPITracker:
    """Tests for KPITracker class."""

    def test_tracker_initialization(self):
        """Test KPITracker initialization."""
        tracker = KPITracker()
        kpis = tracker.get_current_kpis()
        assert isinstance(kpis, TradingKPIs)

    def test_tracker_custom_thresholds(self):
        """Test KPITracker with custom thresholds."""
        custom = {"custom_kpi": KPIThreshold(warning_min=0.5)}
        tracker = KPITracker(thresholds=custom)
        tracker._thresholds["custom_kpi"]
        assert tracker._thresholds["custom_kpi"].warning_min == 0.5

    def test_update_kpis(self):
        """Test updating KPIs."""
        tracker = KPITracker()
        new_kpis = TradingKPIs(
            total_return=0.15,
            win_rate=0.55,
            sharpe_ratio=1.5,
        )
        tracker.update_kpis(new_kpis)
        current = tracker.get_current_kpis()
        assert current.total_return == 0.15
        assert current.win_rate == 0.55

    def test_update_single_kpi(self):
        """Test updating a single KPI."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.65)
        current = tracker.get_current_kpis()
        assert current.win_rate == 0.65

    def test_get_kpi_value(self):
        """Test getting a single KPI value."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.55)
        kpi = tracker.get_kpi_value("win_rate")
        assert kpi.name == "win_rate"
        assert kpi.value == 0.55
        assert kpi.status == KPIStatus.HEALTHY

    def test_get_kpi_value_unknown(self):
        """Test getting unknown KPI returns unknown status."""
        tracker = KPITracker()
        kpi = tracker.get_kpi_value("nonexistent")
        assert kpi.status == KPIStatus.UNKNOWN

    def test_get_kpi_summary(self):
        """Test getting KPI summary."""
        tracker = KPITracker()
        tracker.update_kpis(TradingKPIs(total_return=0.15, win_rate=0.55))
        summary = tracker.get_kpi_summary()
        assert "total_return" in summary
        assert "win_rate" in summary
        assert summary["total_return"].value == 0.15

    def test_alert_triggered_on_threshold_breach(self):
        """Test alert is triggered when threshold is breached."""
        alerts_received = []

        def handler(alert: Alert):
            alerts_received.append(alert)

        tracker = KPITracker(alert_handlers=[handler])
        # Win rate below warning threshold (0.40)
        tracker.update_kpi("win_rate", 0.35)

        assert len(alerts_received) == 1
        assert alerts_received[0].kpi_name == "win_rate"
        assert alerts_received[0].severity == AlertSeverity.WARNING

    def test_critical_alert_on_severe_breach(self):
        """Test critical alert on severe threshold breach."""
        alerts_received = []

        def handler(alert: Alert):
            alerts_received.append(alert)

        tracker = KPITracker(alert_handlers=[handler])
        # Win rate below critical threshold (0.30)
        tracker.update_kpi("win_rate", 0.25)

        assert len(alerts_received) == 1
        assert alerts_received[0].severity == AlertSeverity.CRITICAL

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.25)  # Critical
        tracker.update_kpi("sharpe_ratio", -0.5)  # Critical

        alerts = tracker.get_active_alerts()
        assert len(alerts) >= 2

    def test_clear_alert(self):
        """Test clearing an alert."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.25)  # Trigger alert
        assert "win_rate" in tracker._active_alerts

        tracker.clear_alert("win_rate")
        assert "win_rate" not in tracker._active_alerts

    def test_alert_history(self):
        """Test getting alert history."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.35)  # Warning
        tracker.update_kpi("win_rate", 0.25)  # Critical

        history = tracker.get_alert_history()
        assert len(history) >= 1

    def test_alert_history_filter_by_severity(self):
        """Test filtering alert history by severity."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.35)  # Warning
        tracker.update_kpi("sharpe_ratio", -0.5)  # Critical

        critical_alerts = tracker.get_alert_history(severity=AlertSeverity.CRITICAL)
        assert all(a.severity == AlertSeverity.CRITICAL for a in critical_alerts)

    def test_get_health_status(self):
        """Test getting overall health status."""
        tracker = KPITracker()
        tracker.update_kpis(
            TradingKPIs(
                win_rate=0.55,
                sharpe_ratio=1.5,
                total_return=0.10,
            )
        )
        status = tracker.get_health_status()
        assert "overall_status" in status
        assert "critical_count" in status
        assert "warning_count" in status
        assert "healthy_count" in status

    def test_health_status_critical(self):
        """Test health status is critical when any KPI is critical."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.20)  # Critical
        status = tracker.get_health_status()
        assert status["overall_status"] == "critical"

    def test_add_alert_handler(self):
        """Test adding alert handler."""
        tracker = KPITracker()
        alerts = []
        tracker.add_alert_handler(lambda a: alerts.append(a))
        tracker.update_kpi("win_rate", 0.35)
        assert len(alerts) == 1

    def test_set_threshold(self):
        """Test setting custom threshold."""
        tracker = KPITracker()
        tracker.set_threshold("custom", KPIThreshold(warning_min=0.5))
        assert "custom" in tracker._thresholds


class TestGlobalKPITracker:
    """Tests for global KPI tracker functions."""

    def test_get_kpi_tracker(self):
        """Test getting global tracker."""
        reset_kpi_tracker()
        tracker = get_kpi_tracker()
        assert isinstance(tracker, KPITracker)

    def test_get_kpi_tracker_singleton(self):
        """Test global tracker is singleton."""
        reset_kpi_tracker()
        tracker1 = get_kpi_tracker()
        tracker2 = get_kpi_tracker()
        assert tracker1 is tracker2

    def test_reset_kpi_tracker(self):
        """Test resetting global tracker."""
        tracker1 = get_kpi_tracker()
        tracker1.update_kpi("win_rate", 0.50)
        reset_kpi_tracker()
        tracker2 = get_kpi_tracker()
        assert tracker1 is not tracker2


class TestKPIIntegration:
    """Integration tests for KPI system."""

    def test_full_kpi_workflow(self):
        """Test complete KPI tracking workflow."""
        alerts = []
        tracker = KPITracker(alert_handlers=[lambda a: alerts.append(a)])

        # Initial state - healthy (all values must be within thresholds)
        kpis = TradingKPIs(
            total_return=0.10,
            annualized_return=0.15,
            win_rate=0.55,
            sharpe_ratio=1.2,
            sortino_ratio=1.5,
            max_drawdown=-0.05,
            current_drawdown=-0.02,
            volatility=0.15,
            profit_factor=1.5,
            total_trades=100,
            winning_trades=55,
            losing_trades=45,
            avg_win=150.0,
            avg_loss=100.0,
            uptime_percent=99.9,
            api_success_rate=99.5,
            data_freshness_seconds=30,
        )
        tracker.update_kpis(kpis)

        status = tracker.get_health_status()
        # Should have no critical alerts with healthy initial values
        assert (
            status["critical_count"] == 0
        ), f"Unexpected critical alerts: {tracker.get_active_alerts()}"

        # Deteriorating performance
        tracker.update_kpi("win_rate", 0.35)  # Warning
        assert len(alerts) == 1

        tracker.update_kpi("win_rate", 0.25)  # Critical
        assert len(alerts) == 2

        # Check active alerts
        active = tracker.get_active_alerts()
        assert any(a.kpi_name == "win_rate" for a in active)

        # Recovery
        tracker.update_kpi("win_rate", 0.55)
        assert "win_rate" not in tracker._active_alerts

    def test_multiple_kpi_alerts(self):
        """Test multiple KPI alerts simultaneously."""
        tracker = KPITracker()

        # Multiple KPIs breaching thresholds
        tracker.update_kpis(
            TradingKPIs(
                win_rate=0.25,  # Critical
                sharpe_ratio=-0.5,  # Critical
                max_drawdown=-0.25,  # Critical
            )
        )

        status = tracker.get_health_status()
        assert status["overall_status"] == "critical"
        assert status["critical_count"] >= 2

    def test_kpi_history_retention(self):
        """Test KPI history retention."""
        tracker = KPITracker(history_retention=timedelta(hours=1))

        # Add some history
        for i in range(5):
            kpis = TradingKPIs(total_return=i * 0.01)
            tracker.update_kpis(kpis)

        assert len(tracker._kpi_history) == 5
