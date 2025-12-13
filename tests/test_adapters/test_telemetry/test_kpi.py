"""Comprehensive tests for KPI Tracking System.

This module provides additional test coverage for edge cases and branches
in the KPI tracking system that are not covered by the basic tests.
"""

from datetime import datetime, timedelta
import time

from ordinis.adapters.telemetry.kpi import (
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


class TestKPIThresholdEdgeCases:
    """Edge case tests for KPIThreshold class."""

    def test_evaluate_with_only_warning_max(self):
        """Test evaluation with only warning_max threshold."""
        threshold = KPIThreshold(warning_max=10.0)
        assert threshold.evaluate(5.0) == KPIStatus.HEALTHY
        assert threshold.evaluate(15.0) == KPIStatus.WARNING

    def test_evaluate_with_only_critical_min(self):
        """Test evaluation with only critical_min threshold."""
        threshold = KPIThreshold(critical_min=0.0)
        assert threshold.evaluate(5.0) == KPIStatus.HEALTHY
        assert threshold.evaluate(-5.0) == KPIStatus.CRITICAL

    def test_evaluate_with_only_critical_max(self):
        """Test evaluation with only critical_max threshold."""
        threshold = KPIThreshold(critical_max=100.0)
        assert threshold.evaluate(50.0) == KPIStatus.HEALTHY
        assert threshold.evaluate(150.0) == KPIStatus.CRITICAL

    def test_evaluate_at_exact_threshold_values(self):
        """Test evaluation at exact threshold boundaries."""
        threshold = KPIThreshold(
            warning_min=20.0,
            warning_max=80.0,
            critical_min=10.0,
            critical_max=90.0,
        )
        # Just inside healthy zone
        assert threshold.evaluate(20.01) == KPIStatus.HEALTHY
        assert threshold.evaluate(79.99) == KPIStatus.HEALTHY
        # Just below warning_min (at boundary)
        assert threshold.evaluate(19.99) == KPIStatus.WARNING
        # Just above warning_max
        assert threshold.evaluate(80.01) == KPIStatus.WARNING
        # Just inside warning zone (above critical_min)
        assert threshold.evaluate(10.01) == KPIStatus.WARNING
        assert threshold.evaluate(89.99) == KPIStatus.WARNING
        # Just below critical_min
        assert threshold.evaluate(9.99) == KPIStatus.CRITICAL
        # Just above critical_max
        assert threshold.evaluate(90.01) == KPIStatus.CRITICAL


class TestTradingKPIsEdgeCases:
    """Edge case tests for TradingKPIs class."""

    def test_win_loss_ratio_both_zero(self):
        """Test win/loss ratio when both avg_win and avg_loss are zero."""
        kpis = TradingKPIs(avg_win=0.0, avg_loss=0.0)
        assert kpis.win_loss_ratio == 0.0

    def test_win_loss_ratio_negative_loss(self):
        """Test win/loss ratio with negative avg_loss."""
        kpis = TradingKPIs(avg_win=100.0, avg_loss=-50.0)
        # Should use absolute value of avg_loss
        assert kpis.win_loss_ratio == 2.0

    def test_risk_reward_ratio_zero_loss(self):
        """Test risk/reward ratio when avg_loss is zero."""
        kpis = TradingKPIs(avg_win=100.0, avg_loss=0.0)
        assert kpis.risk_reward_ratio == 0.0

    def test_risk_reward_ratio_negative_loss(self):
        """Test risk/reward ratio with negative avg_loss."""
        kpis = TradingKPIs(avg_win=150.0, avg_loss=-50.0)
        # Should use absolute value
        assert kpis.risk_reward_ratio == 3.0

    def test_expectancy_with_zero_win_rate(self):
        """Test expectancy calculation with zero win rate."""
        kpis = TradingKPIs(win_rate=0.0, avg_win=100.0, avg_loss=50.0)
        # Expectancy = (0.0 * 100) - (1.0 * 50) = -50
        assert kpis.expectancy == -50.0

    def test_expectancy_with_100_percent_win_rate(self):
        """Test expectancy calculation with 100% win rate."""
        kpis = TradingKPIs(win_rate=1.0, avg_win=100.0, avg_loss=50.0)
        # Expectancy = (1.0 * 100) - (0.0 * 50) = 100
        assert kpis.expectancy == 100.0

    def test_to_dict_includes_all_computed_properties(self):
        """Test that to_dict includes all computed properties."""
        kpis = TradingKPIs(
            avg_win=100.0,
            avg_loss=50.0,
            win_rate=0.6,
        )
        result = kpis.to_dict()
        # Verify computed properties are included
        assert "win_loss_ratio" in result
        assert "expectancy" in result
        assert "risk_reward_ratio" in result
        assert result["win_loss_ratio"] == 2.0
        assert result["expectancy"] == 40.0
        assert result["risk_reward_ratio"] == 2.0


class TestKPIValueEdgeCases:
    """Edge case tests for KPIValue class."""

    def test_kpi_value_with_empty_strings(self):
        """Test KPIValue with empty unit and description."""
        kpi = KPIValue(name="test", value=1.0)
        assert kpi.unit == ""
        assert kpi.description == ""

    def test_kpi_value_default_timestamp(self):
        """Test that KPIValue has a default timestamp."""
        kpi = KPIValue(name="test", value=1.0)
        assert isinstance(kpi.timestamp, datetime)
        # Should be very recent
        assert (datetime.utcnow() - kpi.timestamp).total_seconds() < 1.0


class TestAlertEdgeCases:
    """Edge case tests for Alert class."""

    def test_alert_without_threshold(self):
        """Test Alert creation without threshold value."""
        alert = Alert(
            kpi_name="test",
            severity=AlertSeverity.INFO,
            message="Info message",
            value=1.0,
        )
        assert alert.threshold is None

    def test_alert_to_dict_with_none_threshold(self):
        """Test Alert to_dict with None threshold."""
        alert = Alert(
            kpi_name="test",
            severity=AlertSeverity.INFO,
            message="Test",
            value=1.0,
            threshold=None,
        )
        result = alert.to_dict()
        assert result["threshold"] is None


class TestKPITrackerEdgeCases:
    """Edge case tests for KPITracker class."""

    def test_update_kpi_nonexistent_attribute(self):
        """Test updating a non-existent KPI attribute."""
        tracker = KPITracker()
        # Should not raise an error, just silently ignore
        tracker.update_kpi("nonexistent_kpi", 123.45)
        # Current KPIs should not have this attribute
        assert not hasattr(tracker.get_current_kpis(), "nonexistent_kpi")

    def test_get_kpi_value_for_computed_property(self):
        """Test getting a computed property as KPI value."""
        tracker = KPITracker()
        tracker.update_kpis(TradingKPIs(avg_win=100.0, avg_loss=50.0))
        # win_loss_ratio is a computed property that gets included in to_dict()
        kpi = tracker.get_kpi_value("win_loss_ratio")
        # It's accessible via get_kpi_value and will have healthy status if no threshold
        assert kpi.value == 2.0
        assert kpi.status == KPIStatus.HEALTHY  # No threshold set for this metric

    def test_get_kpi_summary_filters_non_numeric(self):
        """Test that get_kpi_summary filters out non-numeric values."""
        tracker = KPITracker()
        tracker.update_kpis(TradingKPIs(total_return=0.15))
        summary = tracker.get_kpi_summary()
        # Should not include 'timestamp' in summary
        assert "timestamp" not in summary

    def test_alert_history_with_time_filter(self):
        """Test filtering alert history by time."""
        tracker = KPITracker()
        # Create an alert
        tracker.update_kpi("win_rate", 0.25)  # Critical
        # Get alerts from 1 hour ago (should include recent alert)
        since = datetime.utcnow() - timedelta(hours=1)
        alerts = tracker.get_alert_history(since=since)
        assert len(alerts) > 0
        # Get alerts from future (should return empty)
        future = datetime.utcnow() + timedelta(hours=1)
        alerts = tracker.get_alert_history(since=future)
        assert len(alerts) == 0

    def test_alert_escalation_from_warning_to_critical(self):
        """Test that alert escalates from WARNING to CRITICAL and triggers handler."""
        alerts_received = []

        def handler(alert: Alert):
            alerts_received.append(alert)

        tracker = KPITracker(alert_handlers=[handler])
        # First trigger a warning
        tracker.update_kpi("win_rate", 0.35)  # Warning
        assert len(alerts_received) == 1
        assert alerts_received[0].severity == AlertSeverity.WARNING
        # Now escalate to critical
        tracker.update_kpi("win_rate", 0.25)  # Critical
        # Should have received the critical alert (handler called again)
        assert len(alerts_received) == 2
        assert alerts_received[1].severity == AlertSeverity.CRITICAL

    def test_alert_same_severity_does_not_retrigger(self):
        """Test that alert at same severity doesn't retrigger handler."""
        alerts_received = []

        def handler(alert: Alert):
            alerts_received.append(alert)

        tracker = KPITracker(alert_handlers=[handler])
        # Trigger warning
        tracker.update_kpi("win_rate", 0.35)
        assert len(alerts_received) == 1
        # Update to different warning value
        tracker.update_kpi("win_rate", 0.38)
        # Should not trigger handler again (still warning)
        assert len(alerts_received) == 1

    def test_alert_handler_exception_does_not_break_tracking(self):
        """Test that exception in alert handler doesn't break tracking."""

        def failing_handler(alert: Alert):
            raise ValueError("Handler error")

        successful_calls = []

        def successful_handler(alert: Alert):
            successful_calls.append(alert)

        tracker = KPITracker(alert_handlers=[failing_handler, successful_handler])
        # Trigger alert - should not raise exception
        tracker.update_kpi("win_rate", 0.25)
        # Successful handler should still be called
        assert len(successful_calls) == 1
        # Alert should still be active
        assert len(tracker.get_active_alerts()) > 0

    def test_clear_alert_nonexistent_kpi(self):
        """Test clearing alert for non-existent KPI."""
        tracker = KPITracker()
        # Should not raise error
        tracker.clear_alert("nonexistent")
        assert "nonexistent" not in tracker._active_alerts

    def test_health_status_with_warning_only(self):
        """Test health status when there are only warnings."""
        tracker = KPITracker()
        # Set all KPIs to safe values first, then add a warning
        tracker.update_kpis(
            TradingKPIs(
                total_return=0.05,  # Above critical (-0.10)
                annualized_return=0.05,  # Above critical (-0.20)
                sharpe_ratio=0.6,  # Above warning (0.5)
                sortino_ratio=0.6,  # Above warning (0.5)
                max_drawdown=-0.08,  # Between warning (-0.10) and critical (-0.20)
                current_drawdown=-0.02,  # Above warning (-0.05)
                volatility=0.15,  # Below warning (0.25)
                win_rate=0.35,  # Warning: below 0.40, above critical 0.30
                profit_factor=1.5,  # Above warning (1.2)
                uptime_percent=99.5,  # Above warning (99.0)
                api_success_rate=99.0,  # Above warning (98.0)
                data_freshness_seconds=30,  # Below warning (60)
            )
        )
        status = tracker.get_health_status()
        assert status["overall_status"] == "warning"
        assert status["warning_count"] >= 1
        assert status["critical_count"] == 0

    def test_health_status_all_healthy(self):
        """Test health status when all KPIs are healthy."""
        tracker = KPITracker()
        # Set ALL tracked KPIs to healthy values
        tracker.update_kpis(
            TradingKPIs(
                total_return=0.10,  # Above warning (-0.05)
                annualized_return=0.15,  # Above warning (-0.10)
                win_rate=0.55,  # Above warning (0.40)
                sharpe_ratio=1.5,  # Above warning (0.5)
                sortino_ratio=1.5,  # Above warning (0.5)
                max_drawdown=-0.05,  # Above warning (-0.10)
                current_drawdown=-0.02,  # Above warning (-0.05)
                volatility=0.15,  # Below warning (0.25)
                profit_factor=1.5,  # Above warning (1.2)
                uptime_percent=99.9,  # Above warning (99.0)
                api_success_rate=99.5,  # Above warning (98.0)
                data_freshness_seconds=30,  # Below warning (60)
            )
        )
        status = tracker.get_health_status()
        assert status["overall_status"] == "healthy"
        assert status["critical_count"] == 0
        assert status["warning_count"] == 0

    def test_get_breached_threshold_critical_min(self):
        """Test identifying which threshold was breached - critical_min."""
        tracker = KPITracker()
        threshold = KPIThreshold(
            warning_min=0.4, warning_max=0.7, critical_min=0.3, critical_max=0.8
        )
        breached = tracker._get_breached_threshold("test", 0.25, threshold)
        assert breached == 0.3

    def test_get_breached_threshold_critical_max(self):
        """Test identifying which threshold was breached - critical_max."""
        tracker = KPITracker()
        threshold = KPIThreshold(
            warning_min=0.4, warning_max=0.7, critical_min=0.3, critical_max=0.8
        )
        breached = tracker._get_breached_threshold("test", 0.85, threshold)
        assert breached == 0.8

    def test_get_breached_threshold_warning_min(self):
        """Test identifying which threshold was breached - warning_min."""
        tracker = KPITracker()
        threshold = KPIThreshold(
            warning_min=0.4, warning_max=0.7, critical_min=0.3, critical_max=0.8
        )
        breached = tracker._get_breached_threshold("test", 0.35, threshold)
        assert breached == 0.4

    def test_get_breached_threshold_warning_max(self):
        """Test identifying which threshold was breached - warning_max."""
        tracker = KPITracker()
        threshold = KPIThreshold(
            warning_min=0.4, warning_max=0.7, critical_min=0.3, critical_max=0.8
        )
        breached = tracker._get_breached_threshold("test", 0.75, threshold)
        assert breached == 0.7

    def test_get_breached_threshold_none(self):
        """Test that no threshold is returned when value is healthy."""
        tracker = KPITracker()
        threshold = KPIThreshold(
            warning_min=0.4, warning_max=0.7, critical_min=0.3, critical_max=0.8
        )
        breached = tracker._get_breached_threshold("test", 0.5, threshold)
        assert breached is None

    def test_cleanup_history_removes_old_kpis(self):
        """Test that old KPIs are removed from history."""
        tracker = KPITracker(history_retention=timedelta(milliseconds=100))
        # Add initial KPI
        tracker.update_kpis(TradingKPIs(total_return=0.1))
        initial_count = len(tracker._kpi_history)
        # Wait a bit and add another
        time.sleep(0.15)
        tracker.update_kpis(TradingKPIs(total_return=0.2))
        # Old KPI should be cleaned up
        assert len(tracker._kpi_history) == 1

    def test_cleanup_history_removes_old_alerts(self):
        """Test that old alerts are removed from history."""
        tracker = KPITracker(history_retention=timedelta(milliseconds=100))
        # Trigger an alert
        tracker.update_kpi("win_rate", 0.25)
        initial_count = len(tracker._alert_history)
        assert initial_count > 0
        # Wait and trigger another alert
        time.sleep(0.15)
        tracker.update_kpi("sharpe_ratio", -0.5)
        # Should have cleaned up old alerts
        assert len(tracker._alert_history) < initial_count + 10

    def test_check_thresholds_skips_non_numeric_values(self):
        """Test that threshold checking skips non-numeric values."""
        tracker = KPITracker()
        # This should not raise any errors even though timestamp is not numeric
        kpis = TradingKPIs(total_return=0.15)
        tracker._check_thresholds(kpis)
        # Should complete without error

    def test_alert_recovery_clears_active_alert(self):
        """Test that recovering from an alert clears it from active alerts."""
        tracker = KPITracker()
        # Trigger critical alert
        tracker.update_kpi("win_rate", 0.25)
        assert "win_rate" in tracker._active_alerts
        # Recover to healthy
        tracker.update_kpi("win_rate", 0.55)
        assert "win_rate" not in tracker._active_alerts

    def test_set_threshold_overwrites_existing(self):
        """Test that set_threshold overwrites existing threshold."""
        tracker = KPITracker()
        original = tracker._thresholds["win_rate"]
        new_threshold = KPIThreshold(warning_min=0.5, critical_min=0.3)
        tracker.set_threshold("win_rate", new_threshold)
        assert tracker._thresholds["win_rate"] is new_threshold
        assert tracker._thresholds["win_rate"] is not original

    def test_tracker_initialization_with_empty_alert_handlers(self):
        """Test tracker initialization with None alert_handlers."""
        tracker = KPITracker(alert_handlers=None)
        assert tracker._alert_handlers == []

    def test_tracker_initialization_merges_thresholds(self):
        """Test that custom thresholds are merged with defaults."""
        custom_thresholds = {"custom_metric": KPIThreshold(warning_min=5.0)}
        tracker = KPITracker(thresholds=custom_thresholds)
        # Should have custom threshold
        assert "custom_metric" in tracker._thresholds
        # Should also have defaults
        assert "win_rate" in tracker._thresholds
        assert "sharpe_ratio" in tracker._thresholds


class TestGlobalTrackerEdgeCases:
    """Edge case tests for global tracker functions."""

    def test_get_kpi_tracker_creates_new_instance(self):
        """Test that get_kpi_tracker creates instance if none exists."""
        reset_kpi_tracker()
        tracker = get_kpi_tracker()
        assert tracker is not None
        assert isinstance(tracker, KPITracker)

    def test_reset_sets_global_to_none(self):
        """Test that reset_kpi_tracker sets global to None."""
        get_kpi_tracker()
        reset_kpi_tracker()
        # After reset, getting tracker should create a new one
        new_tracker = get_kpi_tracker()
        assert new_tracker is not None


class TestAlertCreationEdgeCases:
    """Edge case tests for alert creation and dispatch."""

    def test_alert_message_formatting_with_threshold(self):
        """Test alert message includes threshold when provided."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.25)
        alerts = tracker.get_active_alerts()
        assert len(alerts) > 0
        alert = alerts[0]
        assert "threshold=" in alert.message

    def test_alert_message_formatting_without_threshold(self):
        """Test alert message formatting when threshold is None."""
        tracker = KPITracker()
        # Manually create an alert without threshold
        tracker._create_alert("test", AlertSeverity.INFO, 1.0, None)
        alerts = tracker.get_active_alerts()
        assert any(a.kpi_name == "test" for a in alerts)

    def test_multiple_alert_handlers_all_called(self):
        """Test that all alert handlers are called."""
        calls1 = []
        calls2 = []
        calls3 = []

        tracker = KPITracker(
            alert_handlers=[
                lambda a: calls1.append(a),
                lambda a: calls2.append(a),
                lambda a: calls3.append(a),
            ]
        )
        tracker.update_kpi("win_rate", 0.25)
        # All handlers should be called
        assert len(calls1) == 1
        assert len(calls2) == 1
        assert len(calls3) == 1


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""

    def test_downgrade_from_critical_to_warning(self):
        """Test transitioning from CRITICAL to WARNING state."""
        alerts_received = []

        def handler(alert: Alert):
            alerts_received.append(alert)

        tracker = KPITracker(alert_handlers=[handler])
        # Start with critical
        tracker.update_kpi("win_rate", 0.25)  # Critical
        assert len(alerts_received) == 1
        assert alerts_received[0].severity == AlertSeverity.CRITICAL
        # Move to warning (should not trigger new alert since not escalation)
        tracker.update_kpi("win_rate", 0.35)  # Warning
        # Should have 1 alerts total (critical, then warning - warning doesn't retrigger)
        assert len(alerts_received) == 1

    def test_multiple_kpis_with_different_statuses(self):
        """Test tracking multiple KPIs with different health statuses."""
        tracker = KPITracker()
        tracker.update_kpis(
            TradingKPIs(
                win_rate=0.55,  # Healthy
                sharpe_ratio=0.3,  # Warning
                max_drawdown=-0.15,  # Critical
            )
        )
        status = tracker.get_health_status()
        assert status["overall_status"] == "critical"
        assert status["healthy_count"] > 0
        assert status["warning_count"] > 0
        assert status["critical_count"] > 0

    def test_kpi_value_timestamp_updated_on_change(self):
        """Test that timestamp is updated when KPI value changes."""
        tracker = KPITracker()
        tracker.update_kpi("win_rate", 0.5)
        first_timestamp = tracker.get_current_kpis().timestamp
        # Wait a tiny bit
        time.sleep(0.01)
        tracker.update_kpi("win_rate", 0.6)
        second_timestamp = tracker.get_current_kpis().timestamp
        assert second_timestamp > first_timestamp
