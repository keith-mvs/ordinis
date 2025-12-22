"""Tests for AlertManager module.

Tests cover:
- AlertSeverity enum
- AlertType enum
- Alert dataclass
- AlertChannel dataclass
- AlertManager class
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

import pytest

from ordinis.adapters.alerting.manager import (
    AlertSeverity,
    AlertType,
    Alert,
    AlertChannel,
    AlertManager,
)


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    @pytest.mark.unit
    def test_severity_levels(self):
        """Test all severity levels exist."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"
        assert AlertSeverity.EMERGENCY.value == "emergency"

    @pytest.mark.unit
    def test_severity_count(self):
        """Test correct number of severity levels."""
        assert len(AlertSeverity) == 4


class TestAlertType:
    """Tests for AlertType enum."""

    @pytest.mark.unit
    def test_alert_types(self):
        """Test key alert types exist."""
        assert AlertType.KILL_SWITCH.value == "kill_switch"
        assert AlertType.RISK_BREACH.value == "risk_breach"
        assert AlertType.ORDER_REJECTED.value == "order_rejected"
        assert AlertType.SYSTEM_HEALTH.value == "system_health"
        assert AlertType.CUSTOM.value == "custom"

    @pytest.mark.unit
    def test_alert_type_count(self):
        """Test correct number of alert types."""
        assert len(AlertType) >= 8


class TestAlert:
    """Tests for Alert dataclass."""

    @pytest.mark.unit
    def test_create_alert_basic(self):
        """Test creating a basic alert."""
        alert = Alert(
            alert_id="test_123",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.INFO,
            title="Test Alert",
            message="This is a test alert",
        )

        assert alert.alert_id == "test_123"
        assert alert.alert_type == AlertType.SYSTEM_HEALTH
        assert alert.severity == AlertSeverity.INFO
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"

    @pytest.mark.unit
    def test_alert_default_timestamp(self):
        """Test alert has default timestamp."""
        alert = Alert(
            alert_id="test_123",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.INFO,
            title="Test Alert",
            message="Test message",
        )

        assert isinstance(alert.timestamp, datetime)
        # Timestamp should be recent (within last minute)
        assert datetime.utcnow() - alert.timestamp < timedelta(minutes=1)

    @pytest.mark.unit
    def test_alert_default_metadata(self):
        """Test alert has empty default metadata."""
        alert = Alert(
            alert_id="test_123",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.INFO,
            title="Test Alert",
            message="Test message",
        )

        assert alert.metadata == {}
        assert alert.acknowledged is False
        assert alert.channels_sent == []

    @pytest.mark.unit
    def test_alert_with_metadata(self):
        """Test alert with custom metadata."""
        metadata = {"symbol": "AAPL", "value": 150.0}
        alert = Alert(
            alert_id="test_123",
            alert_type=AlertType.TRADE_EXECUTED,
            severity=AlertSeverity.INFO,
            title="Trade Alert",
            message="Trade executed",
            metadata=metadata,
        )

        assert alert.metadata == metadata
        assert alert.metadata["symbol"] == "AAPL"

    @pytest.mark.unit
    def test_alert_to_dict(self):
        """Test alert to_dict method."""
        alert = Alert(
            alert_id="test_123",
            alert_type=AlertType.SYSTEM_HEALTH,
            severity=AlertSeverity.INFO,
            title="Test Alert",
            message="Test message",
        )

        result = alert.to_dict()

        assert result["alert_id"] == "test_123"
        assert result["alert_type"] == "system_health"
        assert result["severity"] == "info"
        assert result["title"] == "Test Alert"
        assert result["message"] == "Test message"
        assert "timestamp" in result
        assert result["acknowledged"] is False


class TestAlertChannel:
    """Tests for AlertChannel dataclass."""

    @pytest.mark.unit
    def test_channel_basic(self):
        """Test creating a basic channel."""
        channel = AlertChannel(name="test_channel")

        assert channel.name == "test_channel"
        assert channel.enabled is True
        assert channel.min_severity == AlertSeverity.WARNING
        assert channel.send_func is None
        assert channel.async_send_func is None

    @pytest.mark.unit
    def test_channel_with_func(self):
        """Test channel with send function."""
        def mock_send(alert: Alert) -> bool:
            return True

        channel = AlertChannel(
            name="email",
            send_func=mock_send,
            min_severity=AlertSeverity.CRITICAL,
        )

        assert channel.name == "email"
        assert channel.send_func is mock_send
        assert channel.min_severity == AlertSeverity.CRITICAL

    @pytest.mark.unit
    def test_channel_disabled(self):
        """Test channel can be disabled."""
        channel = AlertChannel(name="disabled", enabled=False)

        assert not channel.enabled


class TestAlertManager:
    """Tests for AlertManager class."""

    @pytest.mark.unit
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        manager = AlertManager()

        assert manager._channels == {}
        assert manager._history == []
        assert manager._rate_limit_seconds == 60.0
        assert manager._dedup_window_seconds == 300.0
        assert manager._max_history == 1000

    @pytest.mark.unit
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        manager = AlertManager(
            rate_limit_seconds=30.0,
            dedup_window_seconds=120.0,
            max_history=500,
        )

        assert manager._rate_limit_seconds == 30.0
        assert manager._dedup_window_seconds == 120.0
        assert manager._max_history == 500

    @pytest.mark.unit
    def test_register_channel(self):
        """Test registering a channel."""
        manager = AlertManager()

        manager.register_channel(
            name="email",
            send_func=lambda a: True,
        )

        assert "email" in manager._channels
        assert manager._channels["email"].name == "email"

    @pytest.mark.unit
    def test_register_multiple_channels(self):
        """Test registering multiple channels."""
        manager = AlertManager()

        manager.register_channel(name="email", send_func=lambda a: True)
        manager.register_channel(name="slack", send_func=lambda a: True)
        manager.register_channel(name="sms", send_func=lambda a: True)

        assert len(manager._channels) == 3
        assert "email" in manager._channels
        assert "slack" in manager._channels
        assert "sms" in manager._channels

    @pytest.mark.unit
    def test_disable_channel(self):
        """Test disabling a channel."""
        manager = AlertManager()
        manager.register_channel(name="email", send_func=lambda a: True)

        assert manager._channels["email"].enabled is True

        manager.disable_channel("email")

        assert manager._channels["email"].enabled is False

    @pytest.mark.unit
    def test_enable_channel(self):
        """Test enabling a channel."""
        manager = AlertManager()
        manager.register_channel(name="email", send_func=lambda a: True, enabled=False)

        assert manager._channels["email"].enabled is False

        manager.enable_channel("email")

        assert manager._channels["email"].enabled is True

    @pytest.mark.unit
    def test_disable_nonexistent_channel(self):
        """Test disabling nonexistent channel doesn't raise."""
        manager = AlertManager()
        # Should not raise
        manager.disable_channel("nonexistent")

    @pytest.mark.unit
    def test_enable_nonexistent_channel(self):
        """Test enabling nonexistent channel doesn't raise."""
        manager = AlertManager()
        # Should not raise
        manager.enable_channel("nonexistent")
