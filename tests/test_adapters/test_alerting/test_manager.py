"""Tests for adapters.alerting.manager module."""

import asyncio
from datetime import datetime, timedelta

import pytest

from ordinis.adapters.alerting.manager import (
    Alert,
    AlertChannel,
    AlertManager,
    AlertSeverity,
    AlertType,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_alert_severity_values(self):
        """Test AlertSeverity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.EMERGENCY.value == "emergency"

    def test_alert_severity_members(self):
        """Test AlertSeverity has all expected members."""
        severities = list(AlertSeverity)
        assert len(severities) == 4
        assert AlertSeverity.INFO in severities
        assert AlertSeverity.WARNING in severities
        assert AlertSeverity.CRITICAL in severities
        assert AlertSeverity.EMERGENCY in severities


class TestAlertType:
    """Tests for AlertType enum."""

    def test_alert_type_values(self):
        """Test AlertType enum values."""
        assert AlertType.KILL_SWITCH.value == "kill_switch"
        assert AlertType.RISK_BREACH.value == "risk_breach"
        assert AlertType.ORDER_REJECTED.value == "order_rejected"
        assert AlertType.POSITION_RECONCILIATION.value == "position_reconciliation"
        assert AlertType.API_CONNECTIVITY.value == "api_connectivity"
        assert AlertType.SYSTEM_HEALTH.value == "system_health"
        assert AlertType.TRADE_EXECUTED.value == "trade_executed"
        assert AlertType.DAILY_SUMMARY.value == "daily_summary"
        assert AlertType.CUSTOM.value == "custom"

    def test_alert_type_members(self):
        """Test AlertType has all expected members."""
        types = list(AlertType)
        assert len(types) == 9


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_initialization(self):
        """Test Alert initialization with required fields."""
        alert = Alert(
            alert_id="test_001",
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test Alert",
            message="This is a test",
        )

        assert alert.alert_id == "test_001"
        assert alert.alert_type == AlertType.CUSTOM
        assert alert.severity == AlertSeverity.INFO
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test"
        assert isinstance(alert.timestamp, datetime)
        assert isinstance(alert.metadata, dict)
        assert alert.acknowledged is False
        assert isinstance(alert.channels_sent, list)
        assert len(alert.channels_sent) == 0

    def test_alert_with_metadata(self):
        """Test Alert initialization with metadata."""
        metadata = {"user_id": 123, "source": "test"}
        alert = Alert(
            alert_id="test_002",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.WARNING,
            title="System Warning",
            message="CPU usage high",
            metadata=metadata,
        )

        assert alert.metadata == metadata
        assert alert.metadata["user_id"] == 123
        assert alert.metadata["source"] == "test"

    def test_alert_with_custom_timestamp(self):
        """Test Alert with custom timestamp."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        alert = Alert(
            alert_id="test_003",
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
            timestamp=custom_time,
        )

        assert alert.timestamp == custom_time

    def test_alert_to_dict(self):
        """Test Alert to_dict conversion."""
        metadata = {"key": "value"}
        alert = Alert(
            alert_id="test_004",
            alert_type=AlertType.RISK_BREACH,
            severity=AlertSeverity.CRITICAL,
            title="Risk Breach",
            message="Position limit exceeded",
            metadata=metadata,
            acknowledged=True,
            channels_sent=["email", "slack"],
        )

        result = alert.to_dict()

        assert isinstance(result, dict)
        assert result["alert_id"] == "test_004"
        assert result["alert_type"] == "risk_breach"
        assert result["severity"] == "critical"
        assert result["title"] == "Risk Breach"
        assert result["message"] == "Position limit exceeded"
        assert result["metadata"] == metadata
        assert result["acknowledged"] is True
        assert result["channels_sent"] == ["email", "slack"]
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)

    def test_alert_to_dict_timestamp_format(self):
        """Test Alert to_dict timestamp is ISO format."""
        alert = Alert(
            alert_id="test_005",
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
        )

        result = alert.to_dict()
        timestamp_str = result["timestamp"]

        # Verify it can be parsed back
        parsed_time = datetime.fromisoformat(timestamp_str)
        assert isinstance(parsed_time, datetime)


class TestAlertChannel:
    """Tests for AlertChannel dataclass."""

    def test_alert_channel_initialization(self):
        """Test AlertChannel initialization."""
        channel = AlertChannel(name="test_channel")

        assert channel.name == "test_channel"
        assert channel.enabled is True
        assert channel.min_severity == AlertSeverity.WARNING
        assert channel.send_func is None
        assert channel.async_send_func is None

    def test_alert_channel_with_sync_func(self):
        """Test AlertChannel with sync send function."""

        def send_func(alert: Alert) -> bool:
            return True

        channel = AlertChannel(
            name="sync_channel",
            send_func=send_func,
            min_severity=AlertSeverity.CRITICAL,
            enabled=False,
        )

        assert channel.send_func == send_func
        assert channel.min_severity == AlertSeverity.CRITICAL
        assert channel.enabled is False

    def test_alert_channel_with_async_func(self):
        """Test AlertChannel with async send function."""

        async def async_send_func(alert: Alert):
            return True

        channel = AlertChannel(
            name="async_channel",
            async_send_func=async_send_func,
        )

        assert channel.async_send_func == async_send_func


class TestAlertManager:
    """Tests for AlertManager class."""

    def test_alert_manager_initialization(self):
        """Test AlertManager initialization with defaults."""
        manager = AlertManager()

        assert manager._rate_limit_seconds == 60.0
        assert manager._dedup_window_seconds == 300.0
        assert manager._max_history == 1000
        assert len(manager._channels) == 0
        assert len(manager._history) == 0
        assert manager._alert_count == 0
        assert manager._suppressed_count == 0

    def test_alert_manager_custom_parameters(self):
        """Test AlertManager initialization with custom parameters."""
        manager = AlertManager(
            rate_limit_seconds=30.0,
            dedup_window_seconds=600.0,
            max_history=500,
        )

        assert manager._rate_limit_seconds == 30.0
        assert manager._dedup_window_seconds == 600.0
        assert manager._max_history == 500

    def test_register_channel_sync(self):
        """Test registering a sync alert channel."""
        manager = AlertManager()

        def send_func(alert: Alert) -> bool:
            return True

        manager.register_channel(
            name="test_channel",
            send_func=send_func,
            min_severity=AlertSeverity.CRITICAL,
            enabled=True,
        )

        assert "test_channel" in manager._channels
        channel = manager._channels["test_channel"]
        assert channel.name == "test_channel"
        assert channel.send_func == send_func
        assert channel.min_severity == AlertSeverity.CRITICAL
        assert channel.enabled is True

    def test_register_channel_async(self):
        """Test registering an async alert channel."""
        manager = AlertManager()

        async def async_send_func(alert: Alert):
            return True

        manager.register_channel(
            name="async_channel",
            async_send_func=async_send_func,
            min_severity=AlertSeverity.INFO,
        )

        assert "async_channel" in manager._channels
        channel = manager._channels["async_channel"]
        assert channel.async_send_func == async_send_func
        assert channel.min_severity == AlertSeverity.INFO

    def test_disable_channel(self):
        """Test disabling an alert channel."""
        manager = AlertManager()
        manager.register_channel("test_channel")

        assert manager._channels["test_channel"].enabled is True

        manager.disable_channel("test_channel")

        assert manager._channels["test_channel"].enabled is False

    def test_disable_nonexistent_channel(self):
        """Test disabling a non-existent channel does not raise."""
        manager = AlertManager()
        manager.disable_channel("nonexistent")  # Should not raise

    def test_enable_channel(self):
        """Test enabling an alert channel."""
        manager = AlertManager()
        manager.register_channel("test_channel", enabled=False)

        assert manager._channels["test_channel"].enabled is False

        manager.enable_channel("test_channel")

        assert manager._channels["test_channel"].enabled is True

    def test_enable_nonexistent_channel(self):
        """Test enabling a non-existent channel does not raise."""
        manager = AlertManager()
        manager.enable_channel("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_send_basic_alert(self):
        """Test sending a basic alert."""
        manager = AlertManager()

        result = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test Alert",
            message="Test message",
        )

        assert result is not None
        assert result.alert_type == AlertType.CUSTOM
        assert result.severity == AlertSeverity.INFO
        assert result.title == "Test Alert"
        assert result.message == "Test message"
        assert manager._alert_count == 1

    @pytest.mark.asyncio
    async def test_send_alert_with_metadata(self):
        """Test sending alert with metadata."""
        manager = AlertManager()
        metadata = {"user_id": 123, "source": "test"}

        result = await manager.send(
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.WARNING,
            title="System Warning",
            message="CPU high",
            metadata=metadata,
        )

        assert result is not None
        assert result.metadata == metadata

    @pytest.mark.asyncio
    async def test_send_alert_rate_limiting(self):
        """Test alert rate limiting by type."""
        manager = AlertManager(rate_limit_seconds=10.0)

        # First alert should succeed
        result1 = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="First",
            message="First",
        )
        assert result1 is not None

        # Second alert of same type should be suppressed
        result2 = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Second",
            message="Second",
        )
        assert result2 is None
        assert manager._suppressed_count == 1

    @pytest.mark.asyncio
    async def test_send_alert_force_bypasses_rate_limit(self):
        """Test force flag bypasses rate limiting."""
        manager = AlertManager(rate_limit_seconds=10.0)

        # First alert
        result1 = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="First",
            message="First",
        )
        assert result1 is not None

        # Second alert with force=True should succeed
        result2 = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Second",
            message="Second",
            force=True,
        )
        assert result2 is not None
        assert manager._suppressed_count == 0

    @pytest.mark.asyncio
    async def test_send_alert_deduplication(self):
        """Test alert deduplication within window."""
        manager = AlertManager(
            rate_limit_seconds=0.1,
            dedup_window_seconds=10.0,
        )

        # First alert
        result1 = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Duplicate Alert",
            message="Same message",
        )
        assert result1 is not None

        # Wait for rate limit to expire
        await asyncio.sleep(0.2)

        # Same alert should be deduplicated
        result2 = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Duplicate Alert",
            message="Same message",
        )
        assert result2 is None
        assert manager._suppressed_count == 1

    @pytest.mark.asyncio
    async def test_send_alert_different_content_not_deduplicated(self):
        """Test different alerts are not deduplicated."""
        manager = AlertManager(rate_limit_seconds=0.1)

        result1 = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="First Alert",
            message="First message",
        )
        assert result1 is not None

        await asyncio.sleep(0.2)

        result2 = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Second Alert",
            message="Different message",
        )
        assert result2 is not None

    @pytest.mark.asyncio
    async def test_send_to_sync_channel(self):
        """Test sending alert to sync channel."""
        manager = AlertManager()
        sent_alerts = []

        def send_func(alert: Alert) -> bool:
            sent_alerts.append(alert)
            return True

        manager.register_channel(
            name="test_channel",
            send_func=send_func,
            min_severity=AlertSeverity.INFO,
        )

        result = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
        )

        assert result is not None
        assert len(sent_alerts) == 1
        assert sent_alerts[0].title == "Test"
        assert "test_channel" in result.channels_sent

    @pytest.mark.asyncio
    async def test_send_to_async_channel(self):
        """Test sending alert to async channel."""
        manager = AlertManager()
        sent_alerts = []

        async def async_send_func(alert: Alert):
            sent_alerts.append(alert)

        manager.register_channel(
            name="async_channel",
            async_send_func=async_send_func,
            min_severity=AlertSeverity.INFO,
        )

        result = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
        )

        assert result is not None
        assert len(sent_alerts) == 1
        assert "async_channel" in result.channels_sent

    @pytest.mark.asyncio
    async def test_send_respects_channel_severity(self):
        """Test channel minimum severity filtering."""
        manager = AlertManager()
        sent_alerts = []

        def send_func(alert: Alert) -> bool:
            sent_alerts.append(alert)
            return True

        # Channel only accepts CRITICAL and above
        manager.register_channel(
            name="critical_channel",
            send_func=send_func,
            min_severity=AlertSeverity.CRITICAL,
        )

        # Send INFO alert - should not reach channel
        result1 = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Info",
            message="Info",
            force=True,
        )
        assert len(sent_alerts) == 0
        assert "critical_channel" not in result1.channels_sent

        # Send CRITICAL alert - should reach channel
        result2 = await manager.send(
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.CRITICAL,
            title="Critical",
            message="Critical",
            force=True,
        )
        assert len(sent_alerts) == 1
        assert "critical_channel" in result2.channels_sent

    @pytest.mark.asyncio
    async def test_send_skips_disabled_channel(self):
        """Test disabled channels are skipped."""
        manager = AlertManager()
        sent_alerts = []

        def send_func(alert: Alert) -> bool:
            sent_alerts.append(alert)
            return True

        manager.register_channel(
            name="disabled_channel",
            send_func=send_func,
            enabled=False,
        )

        result = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
        )

        assert len(sent_alerts) == 0
        assert "disabled_channel" not in result.channels_sent

    @pytest.mark.asyncio
    async def test_send_handles_channel_failure(self):
        """Test alert manager handles channel send failures gracefully."""
        manager = AlertManager()

        def failing_send_func(alert: Alert) -> bool:
            raise RuntimeError("Send failed")

        manager.register_channel(
            name="failing_channel",
            send_func=failing_send_func,
        )

        # Should not raise exception
        result = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test",
        )

        assert result is not None
        assert "failing_channel" not in result.channels_sent

    @pytest.mark.asyncio
    async def test_send_handles_async_channel_failure(self):
        """Test alert manager handles async channel failures gracefully."""
        manager = AlertManager()

        async def failing_async_send_func(alert: Alert):
            raise RuntimeError("Async send failed")

        manager.register_channel(
            name="failing_async_channel",
            async_send_func=failing_async_send_func,
        )

        result = await manager.send(
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.WARNING,
            title="Test",
            message="Test",
        )

        assert result is not None
        assert "failing_async_channel" not in result.channels_sent

    @pytest.mark.asyncio
    async def test_send_emergency(self):
        """Test send_emergency convenience method."""
        manager = AlertManager()

        result = await manager.send_emergency(
            title="Emergency",
            message="System failure",
            metadata={"severity_level": "critical"},
        )

        assert result is not None
        assert result.alert_type == AlertType.KILL_SWITCH
        assert result.severity == AlertSeverity.EMERGENCY
        assert result.title == "Emergency"
        assert result.message == "System failure"

    @pytest.mark.asyncio
    async def test_send_emergency_bypasses_rate_limit(self):
        """Test emergency alerts bypass rate limiting."""
        manager = AlertManager(rate_limit_seconds=10.0)

        # Send multiple emergency alerts
        result1 = await manager.send_emergency(
            title="Emergency 1",
            message="First",
        )
        result2 = await manager.send_emergency(
            title="Emergency 2",
            message="Second",
        )

        assert result1 is not None
        assert result2 is not None
        assert manager._suppressed_count == 0

    @pytest.mark.asyncio
    async def test_send_risk_breach(self):
        """Test send_risk_breach convenience method."""
        manager = AlertManager()

        result = await manager.send_risk_breach(
            title="Risk Limit Exceeded",
            message="Position size over limit",
            metadata={"position_size": 1000},
        )

        assert result is not None
        assert result.alert_type == AlertType.RISK_BREACH
        assert result.severity == AlertSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_send_warning(self):
        """Test send_warning convenience method."""
        manager = AlertManager()

        result = await manager.send_warning(
            title="Warning",
            message="CPU usage elevated",
        )

        assert result is not None
        assert result.severity == AlertSeverity.WARNING
        assert result.alert_type == AlertType.CUSTOM

    @pytest.mark.asyncio
    async def test_send_warning_custom_type(self):
        """Test send_warning with custom alert type."""
        manager = AlertManager()

        result = await manager.send_warning(
            title="System Warning",
            message="CPU high",
            alert_type=AlertType.SYSTEM_HEALTH,
        )

        assert result is not None
        assert result.alert_type == AlertType.SYSTEM_HEALTH

    @pytest.mark.asyncio
    async def test_send_info(self):
        """Test send_info convenience method."""
        manager = AlertManager()

        result = await manager.send_info(
            title="Info",
            message="Trade executed",
        )

        assert result is not None
        assert result.severity == AlertSeverity.INFO
        assert result.alert_type == AlertType.CUSTOM

    @pytest.mark.asyncio
    async def test_send_info_custom_type(self):
        """Test send_info with custom alert type."""
        manager = AlertManager()

        result = await manager.send_info(
            title="Daily Summary",
            message="Trades: 10",
            alert_type=AlertType.DAILY_SUMMARY,
        )

        assert result is not None
        assert result.alert_type == AlertType.DAILY_SUMMARY

    @pytest.mark.asyncio
    async def test_alert_history(self):
        """Test alerts are added to history."""
        manager = AlertManager(rate_limit_seconds=0.0)

        await manager.send_info(title="Alert 1", message="First")
        await manager.send_info(title="Alert 2", message="Second")

        history = manager.get_history()

        assert len(history) == 2
        assert history[0].title == "Alert 1"
        assert history[1].title == "Alert 2"

    @pytest.mark.asyncio
    async def test_history_limit(self):
        """Test history respects limit parameter."""
        manager = AlertManager(rate_limit_seconds=0.0)

        for i in range(10):
            await manager.send_info(title=f"Alert {i}", message=f"Message {i}")

        history = manager.get_history(limit=5)

        assert len(history) == 5
        # Should get last 5
        assert history[-1].title == "Alert 9"

    @pytest.mark.asyncio
    async def test_history_filter_by_severity(self):
        """Test filtering history by severity."""
        manager = AlertManager(rate_limit_seconds=0.0)

        await manager.send_info(title="Info 1", message="Info")
        await manager.send_warning(title="Warning 1", message="Warning")
        await manager.send_info(title="Info 2", message="Info")

        history = manager.get_history(severity=AlertSeverity.INFO)

        assert len(history) == 2
        assert all(a.severity == AlertSeverity.INFO for a in history)

    @pytest.mark.asyncio
    async def test_history_filter_by_type(self):
        """Test filtering history by alert type."""
        manager = AlertManager(rate_limit_seconds=0.0)

        await manager.send_info(
            title="Custom 1",
            message="Custom",
            alert_type=AlertType.CUSTOM,
        )
        await manager.send_info(
            title="Daily",
            message="Summary",
            alert_type=AlertType.DAILY_SUMMARY,
        )
        await manager.send_info(
            title="Custom 2",
            message="Custom",
            alert_type=AlertType.CUSTOM,
        )

        history = manager.get_history(alert_type=AlertType.CUSTOM)

        assert len(history) == 2
        assert all(a.alert_type == AlertType.CUSTOM for a in history)

    @pytest.mark.asyncio
    async def test_history_filter_combined(self):
        """Test filtering history by both severity and type."""
        manager = AlertManager()

        await manager.send_info(
            title="Info Custom",
            message="Info",
            alert_type=AlertType.CUSTOM,
        )
        await manager.send_warning(
            title="Warning Custom",
            message="Warning",
            alert_type=AlertType.CUSTOM,
        )
        await manager.send_info(
            title="Info System",
            message="Info",
            alert_type=AlertType.SYSTEM_HEALTH,
        )

        history = manager.get_history(
            severity=AlertSeverity.INFO,
            alert_type=AlertType.CUSTOM,
        )

        assert len(history) == 1
        assert history[0].title == "Info Custom"

    @pytest.mark.asyncio
    async def test_max_history_limit(self):
        """Test history respects max_history limit."""
        manager = AlertManager(max_history=5, rate_limit_seconds=0.0)

        # Send 10 alerts
        for i in range(10):
            await manager.send_info(title=f"Alert {i}", message=f"Message {i}")

        # Should only keep last 5
        assert len(manager._history) == 5
        assert manager._history[0].title == "Alert 5"
        assert manager._history[-1].title == "Alert 9"

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting alert statistics."""
        manager = AlertManager(rate_limit_seconds=0.0)

        manager.register_channel(
            name="channel1",
            min_severity=AlertSeverity.INFO,
            enabled=True,
        )
        manager.register_channel(
            name="channel2",
            min_severity=AlertSeverity.CRITICAL,
            enabled=False,
        )

        await manager.send_info(title="Test 1", message="Test 1")
        await manager.send_info(title="Test 2", message="Test 2")

        stats = manager.get_stats()

        assert stats["total_alerts"] == 2
        assert stats["suppressed_alerts"] == 0
        assert len(stats["channels"]) == 2
        assert stats["channels"]["channel1"]["enabled"] is True
        assert stats["channels"]["channel1"]["min_severity"] == "info"
        assert stats["channels"]["channel2"]["enabled"] is False
        assert stats["channels"]["channel2"]["min_severity"] == "critical"
        assert stats["history_count"] == 2

    @pytest.mark.asyncio
    async def test_get_stats_with_suppression(self):
        """Test statistics include suppressed count."""
        manager = AlertManager(rate_limit_seconds=10.0)

        await manager.send_info(title="First", message="First")
        await manager.send_info(title="Second", message="Second")  # Suppressed

        stats = manager.get_stats()

        assert stats["total_alerts"] == 1
        assert stats["suppressed_alerts"] == 1

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        manager = AlertManager()

        result = await manager.send_info(title="Test", message="Test")
        alert_id = result.alert_id

        assert result.acknowledged is False

        success = manager.acknowledge(alert_id)

        assert success is True
        assert result.acknowledged is True

    @pytest.mark.asyncio
    async def test_acknowledge_nonexistent_alert(self):
        """Test acknowledging non-existent alert returns False."""
        manager = AlertManager()

        success = manager.acknowledge("nonexistent_id")

        assert success is False

    @pytest.mark.asyncio
    async def test_get_unacknowledged(self):
        """Test getting unacknowledged alerts."""
        manager = AlertManager(rate_limit_seconds=0.0)

        result1 = await manager.send_info(title="Alert 1", message="First")
        result2 = await manager.send_info(title="Alert 2", message="Second")
        result3 = await manager.send_info(title="Alert 3", message="Third")

        manager.acknowledge(result2.alert_id)

        unacknowledged = manager.get_unacknowledged()

        assert len(unacknowledged) == 2
        assert result1.alert_id in [a.alert_id for a in unacknowledged]
        assert result3.alert_id in [a.alert_id for a in unacknowledged]
        assert result2.alert_id not in [a.alert_id for a in unacknowledged]

    @pytest.mark.asyncio
    async def test_get_unacknowledged_filter_by_severity(self):
        """Test filtering unacknowledged alerts by severity."""
        manager = AlertManager(rate_limit_seconds=0.0)

        await manager.send_info(title="Info 1", message="Info")
        await manager.send_warning(title="Warning 1", message="Warning")
        await manager.send_info(title="Info 2", message="Info")

        unacknowledged = manager.get_unacknowledged(severity=AlertSeverity.INFO)

        assert len(unacknowledged) == 2
        assert all(a.severity == AlertSeverity.INFO for a in unacknowledged)

    def test_compute_hash(self):
        """Test alert hash computation for deduplication."""
        manager = AlertManager()

        alert1 = Alert(
            alert_id="id1",
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test message",
        )

        alert2 = Alert(
            alert_id="id2",  # Different ID
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test message",
        )

        hash1 = manager._compute_hash(alert1)
        hash2 = manager._compute_hash(alert2)

        # Same content should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest

    def test_compute_hash_different_content(self):
        """Test different alerts produce different hashes."""
        manager = AlertManager()

        alert1 = Alert(
            alert_id="id1",
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test 1",
            message="Message 1",
        )

        alert2 = Alert(
            alert_id="id2",
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test 2",
            message="Message 2",
        )

        hash1 = manager._compute_hash(alert1)
        hash2 = manager._compute_hash(alert2)

        assert hash1 != hash2

    def test_is_rate_limited(self):
        """Test rate limiting check."""
        manager = AlertManager(rate_limit_seconds=60.0)

        # No previous send
        assert manager._is_rate_limited(AlertType.CUSTOM) is False

        # Record a send
        manager._last_sent[AlertType.CUSTOM.value] = datetime.utcnow()

        # Should be rate limited now
        assert manager._is_rate_limited(AlertType.CUSTOM) is True

        # Simulate time passing
        manager._last_sent[AlertType.CUSTOM.value] = datetime.utcnow() - timedelta(seconds=61)

        # Should not be rate limited anymore
        assert manager._is_rate_limited(AlertType.CUSTOM) is False

    def test_is_duplicate(self):
        """Test duplicate detection."""
        manager = AlertManager(dedup_window_seconds=300.0)

        alert = Alert(
            alert_id="id1",
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
        )

        # Not a duplicate initially
        assert manager._is_duplicate(alert) is False

        # Record the alert
        manager._record_dedup(alert)

        # Should be duplicate now
        assert manager._is_duplicate(alert) is True

    @pytest.mark.asyncio
    async def test_severity_ordering_in_channels(self):
        """Test severity ordering for channel filtering."""
        manager = AlertManager(rate_limit_seconds=0.0)
        received_severities = []

        def send_func(alert: Alert) -> bool:
            received_severities.append(alert.severity)
            return True

        # Channel accepts WARNING and above
        manager.register_channel(
            name="warning_channel",
            send_func=send_func,
            min_severity=AlertSeverity.WARNING,
        )

        # Send all severity levels
        await manager.send_info(title="Info", message="Info")
        await manager.send_warning(title="Warning", message="Warning")
        await manager.send(
            AlertType.SYSTEM_HEALTH,
            AlertSeverity.CRITICAL,
            "Critical",
            "Critical",
        )
        await manager.send_emergency(title="Emergency", message="Emergency")

        # Should receive WARNING, CRITICAL, EMERGENCY (not INFO)
        assert len(received_severities) == 3
        assert AlertSeverity.INFO not in received_severities
        assert AlertSeverity.WARNING in received_severities
        assert AlertSeverity.CRITICAL in received_severities
        assert AlertSeverity.EMERGENCY in received_severities

    @pytest.mark.asyncio
    async def test_dedup_hash_cleanup(self):
        """Test dedup hash cleanup removes old entries."""
        manager = AlertManager(dedup_window_seconds=1.0)

        alert = Alert(
            alert_id="id1",
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test",
            message="Test",
        )

        # Record dedup
        manager._record_dedup(alert)
        assert len(manager._dedup_hashes) == 1

        # Simulate old timestamp
        hash_key = manager._compute_hash(alert)
        manager._dedup_hashes[hash_key] = datetime.utcnow() - timedelta(seconds=10)

        # Record another alert to trigger cleanup
        alert2 = Alert(
            alert_id="id2",
            alert_type=AlertType.CUSTOM,
            severity=AlertSeverity.INFO,
            title="Test2",
            message="Test2",
        )
        manager._record_dedup(alert2)

        # Old hash should be cleaned up
        assert hash_key not in manager._dedup_hashes

    @pytest.mark.asyncio
    async def test_multiple_channels_receive_alert(self):
        """Test alert sent to multiple channels."""
        manager = AlertManager()
        channel1_alerts = []
        channel2_alerts = []

        def send_func1(alert: Alert) -> bool:
            channel1_alerts.append(alert)
            return True

        def send_func2(alert: Alert) -> bool:
            channel2_alerts.append(alert)
            return True

        manager.register_channel("channel1", send_func=send_func1)
        manager.register_channel("channel2", send_func=send_func2)

        result = await manager.send_warning(title="Test", message="Test")

        assert len(channel1_alerts) == 1
        assert len(channel2_alerts) == 1
        assert len(result.channels_sent) == 2
        assert "channel1" in result.channels_sent
        assert "channel2" in result.channels_sent

    @pytest.mark.asyncio
    async def test_sync_channel_return_false_not_recorded(self):
        """Test sync channel returning False is not recorded in channels_sent."""
        manager = AlertManager()

        def send_func(alert: Alert) -> bool:
            return False

        manager.register_channel("failing_channel", send_func=send_func)

        result = await manager.send_warning(title="Test", message="Test")

        assert "failing_channel" not in result.channels_sent

    @pytest.mark.asyncio
    async def test_alert_id_generation(self):
        """Test alert ID generation is unique."""
        manager = AlertManager(rate_limit_seconds=0.0)

        result1 = await manager.send_info(title="Test 1", message="Test 1")
        result2 = await manager.send_info(title="Test 2", message="Test 2")

        assert result1.alert_id != result2.alert_id
        assert AlertType.CUSTOM.value in result1.alert_id
        assert AlertType.CUSTOM.value in result2.alert_id

    @pytest.mark.asyncio
    async def test_concurrent_alert_sending(self):
        """Test concurrent alert sending is handled safely."""
        manager = AlertManager(rate_limit_seconds=0.0)

        async def send_alert(i: int):
            return await manager.send_info(
                title=f"Alert {i}",
                message=f"Message {i}",
            )

        # Send multiple alerts concurrently
        results = await asyncio.gather(*[send_alert(i) for i in range(10)])

        # All should succeed
        assert all(r is not None for r in results)
        assert manager._alert_count == 10
        assert len(manager._history) == 10

        # All alert IDs should be unique
        alert_ids = [r.alert_id for r in results]
        assert len(alert_ids) == len(set(alert_ids))


class TestAlertManagerIntegration:
    """Integration tests for AlertManager."""

    @pytest.mark.asyncio
    async def test_complete_alert_workflow(self):
        """Test complete alert workflow with multiple channels."""
        manager = AlertManager(
            rate_limit_seconds=0.0,
            dedup_window_seconds=5.0,
            max_history=100,
        )

        # Setup channels
        email_alerts = []
        slack_alerts = []

        def send_email(alert: Alert) -> bool:
            email_alerts.append(alert)
            return True

        async def send_slack(alert: Alert):
            slack_alerts.append(alert)

        manager.register_channel(
            "email",
            send_func=send_email,
            min_severity=AlertSeverity.CRITICAL,
        )
        manager.register_channel(
            "slack",
            async_send_func=send_slack,
            min_severity=AlertSeverity.WARNING,
        )

        # Send various alerts
        info = await manager.send_info(title="Info", message="Info message")
        warning = await manager.send_warning(title="Warning", message="Warning message")
        critical = await manager.send_risk_breach(
            title="Risk Breach",
            message="Limit exceeded",
        )
        emergency = await manager.send_emergency(
            title="Emergency",
            message="System failure",
        )

        # Verify routing
        assert len(email_alerts) == 2  # CRITICAL and EMERGENCY
        assert len(slack_alerts) == 3  # WARNING, CRITICAL, EMERGENCY

        # Verify history
        history = manager.get_history()
        assert len(history) == 4

        # Verify stats
        stats = manager.get_stats()
        assert stats["total_alerts"] == 4
        assert stats["suppressed_alerts"] == 0

        # Acknowledge critical alert
        manager.acknowledge(critical.alert_id)
        unack = manager.get_unacknowledged()
        assert len(unack) == 3
        assert critical.alert_id not in [a.alert_id for a in unack]

    @pytest.mark.asyncio
    async def test_rate_limiting_and_deduplication_interaction(self):
        """Test interaction between rate limiting and deduplication."""
        manager = AlertManager(
            rate_limit_seconds=0.5,
            dedup_window_seconds=2.0,
        )

        # First alert
        result1 = await manager.send_info(title="Test", message="Test")
        assert result1 is not None

        # Immediate duplicate - rate limited
        result2 = await manager.send_info(title="Test", message="Test")
        assert result2 is None

        # Wait for rate limit to expire
        await asyncio.sleep(0.6)

        # Same content - deduplicated
        result3 = await manager.send_info(title="Test", message="Test")
        assert result3 is None

        # Different content - should succeed
        result4 = await manager.send_info(title="Different", message="Different")
        assert result4 is not None

        assert manager._alert_count == 2
        assert manager._suppressed_count == 2

    @pytest.mark.asyncio
    async def test_channel_enable_disable_workflow(self):
        """Test enabling and disabling channels dynamically."""
        manager = AlertManager(rate_limit_seconds=0.0, dedup_window_seconds=0.0)
        received = []

        def send_func(alert: Alert) -> bool:
            received.append(alert)
            return True

        manager.register_channel(
            "dynamic_channel", send_func=send_func, min_severity=AlertSeverity.INFO
        )

        # Send with channel enabled
        await manager.send(AlertType.TRADE_EXECUTED, AlertSeverity.INFO, "Test 1", "Message 1")
        assert len(received) == 1

        # Disable channel
        manager.disable_channel("dynamic_channel")
        await manager.send(AlertType.DAILY_SUMMARY, AlertSeverity.INFO, "Test 2", "Message 2")
        assert len(received) == 1  # Still 1

        # Re-enable channel
        manager.enable_channel("dynamic_channel")
        await manager.send(AlertType.SYSTEM_HEALTH, AlertSeverity.INFO, "Test 3", "Message 3")
        assert len(received) == 2
