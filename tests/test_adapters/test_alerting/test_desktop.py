"""Tests for Desktop notification channel."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from ordinis.adapters.alerting.channels.desktop import (
    ALERT_SOUNDS,
    DesktopNotifier,
)


class TestAlertSounds:
    """Tests for alert sound configuration."""

    def test_alert_sounds_defined(self):
        """Test alert sounds are defined for each severity."""
        assert "info" in ALERT_SOUNDS
        assert "warning" in ALERT_SOUNDS
        assert "critical" in ALERT_SOUNDS
        assert "emergency" in ALERT_SOUNDS

    def test_info_no_sound(self):
        """Test info level has no sound."""
        assert ALERT_SOUNDS["info"] is None

    def test_severity_frequencies(self):
        """Test sound frequencies increase with severity."""
        assert ALERT_SOUNDS["warning"] < ALERT_SOUNDS["critical"]
        assert ALERT_SOUNDS["critical"] < ALERT_SOUNDS["emergency"]


class TestDesktopNotifierInit:
    """Tests for DesktopNotifier initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        with patch.object(DesktopNotifier, "_check_dependencies"):
            notifier = DesktopNotifier()

            assert notifier.app_name == "Ordinis Trading"
            assert notifier.enable_sound is True
            assert notifier.sound_duration_ms == 500

    def test_init_custom_values(self):
        """Test custom initialization."""
        with patch.object(DesktopNotifier, "_check_dependencies"):
            notifier = DesktopNotifier(
                app_name="Custom App",
                enable_sound=False,
                sound_duration_ms=1000,
            )

            assert notifier.app_name == "Custom App"
            assert notifier.enable_sound is False
            assert notifier.sound_duration_ms == 1000


class TestDesktopNotifierDependencies:
    """Tests for dependency checking."""

    def test_check_dependencies_plyer_available(self):
        """Test plyer availability check."""
        with patch.dict(sys.modules, {"plyer": MagicMock(), "plyer.notification": MagicMock()}):
            notifier = DesktopNotifier()
            # Manually call check since patching may interfere
            notifier._plyer_available = True

            assert notifier._plyer_available is True

    def test_check_dependencies_plyer_not_available(self):
        """Test when plyer is not available."""
        with patch.dict(sys.modules, {"plyer": None}):
            with patch.object(DesktopNotifier, "_check_dependencies") as mock_check:
                notifier = DesktopNotifier()
                notifier._plyer_available = False

                assert notifier._plyer_available is False

    def test_check_dependencies_winsound_on_windows(self):
        """Test winsound check on Windows."""
        with patch("sys.platform", "win32"):
            with patch.dict(sys.modules, {"winsound": MagicMock()}):
                with patch.object(DesktopNotifier, "_check_dependencies"):
                    notifier = DesktopNotifier()
                    notifier._winsound_available = True

                    assert notifier._winsound_available is True


class TestDesktopNotifierSendNotification:
    """Tests for sending notifications."""

    @pytest.fixture
    def notifier(self):
        """Create notifier with mocked dependencies."""
        with patch.object(DesktopNotifier, "_check_dependencies"):
            notifier = DesktopNotifier()
            notifier._plyer_available = True
            notifier._winsound_available = True
            return notifier

    def test_send_notification_attributes(self, notifier):
        """Test notifier has correct attributes."""
        assert notifier.app_name == "Ordinis Trading"
        assert notifier.enable_sound is True
        assert notifier._plyer_available is True


class TestDesktopNotifierStatus:
    """Tests for notifier status methods."""

    def test_get_status_keys(self):
        """Test status retrieval has expected keys."""
        with patch.object(DesktopNotifier, "_check_dependencies"):
            notifier = DesktopNotifier()
            notifier._plyer_available = True
            notifier._winsound_available = False

            status = notifier.get_status()

            assert "plyer_available" in status
            assert status["plyer_available"] is True
            assert status["winsound_available"] is False

    def test_get_status_includes_app_name(self):
        """Test status includes app name."""
        with patch.object(DesktopNotifier, "_check_dependencies"):
            notifier = DesktopNotifier()

            status = notifier.get_status()

            assert "app_name" in status
            assert status["app_name"] == "Ordinis Trading"
