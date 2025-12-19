"""
Desktop notification channel for Windows.

Provides:
- Windows toast notifications
- Audio alerts for critical/emergency
- System tray integration (future)

Uses plyer for cross-platform notifications with winsound for audio.
"""

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ordinis.adapters.alerting.manager import Alert

logger = logging.getLogger(__name__)

# Audio alert frequencies by severity
ALERT_SOUNDS = {
    "info": None,  # No sound
    "warning": 800,  # Hz
    "critical": 1200,  # Hz
    "emergency": 1600,  # Hz
}


class DesktopNotifier:
    """
    Desktop notification sender for Windows.

    Uses plyer for toast notifications and winsound for audio alerts.
    Falls back gracefully if dependencies not available.
    """

    def __init__(
        self,
        app_name: str = "Ordinis Trading",
        enable_sound: bool = True,
        sound_duration_ms: int = 500,
    ):
        """
        Initialize desktop notifier.

        Args:
            app_name: Application name shown in notifications
            enable_sound: Whether to play audio alerts
            sound_duration_ms: Duration of audio alert in ms
        """
        self.app_name = app_name
        self.enable_sound = enable_sound
        self.sound_duration_ms = sound_duration_ms

        self._plyer_available = False
        self._winsound_available = False

        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """Check for available notification libraries."""
        # Check plyer
        try:
            from plyer import notification  # noqa: F401

            self._plyer_available = True
            logger.debug("plyer notification available")
        except ImportError:
            logger.warning("plyer not available - install with: pip install plyer")

        # Check winsound (Windows only)
        if sys.platform == "win32":
            try:
                import winsound  # noqa: F401

                self._winsound_available = True
                logger.debug("winsound available")
            except ImportError:
                pass

    def send(self, alert: "Alert") -> bool:
        """
        Send desktop notification.

        Args:
            alert: Alert to send

        Returns:
            True if notification sent successfully
        """
        try:
            # Send toast notification
            if self._plyer_available:
                self._send_toast(alert)

            # Play audio alert if enabled and severe enough
            if self.enable_sound and alert.severity.value in ["critical", "emergency"]:
                self._play_sound(alert.severity.value)

            return True
        except Exception as e:
            logger.exception(f"Failed to send desktop notification: {e}")
            return False

    def _send_toast(self, alert: "Alert") -> None:
        """Send toast notification using plyer."""
        from plyer import notification

        # Set timeout based on severity
        timeout_map = {
            "info": 5,
            "warning": 10,
            "critical": 15,
            "emergency": 30,
        }
        timeout = timeout_map.get(alert.severity.value, 10)

        # Format message
        title = f"[{alert.severity.value.upper()}] {alert.title}"
        message = alert.message[:250]  # Truncate for toast

        notification.notify(
            title=title,
            message=message,
            app_name=self.app_name,
            timeout=timeout,
        )

        logger.debug(f"Toast notification sent: {title}")

    def _play_sound(self, severity: str) -> None:
        """Play audio alert using winsound."""
        if not self._winsound_available:
            return

        frequency = ALERT_SOUNDS.get(severity)
        if frequency is None:
            return

        try:
            import winsound

            winsound.Beep(frequency, self.sound_duration_ms)

            # Emergency gets double beep
            if severity == "emergency":
                import time

                time.sleep(0.1)
                winsound.Beep(frequency + 200, self.sound_duration_ms)
        except Exception as e:
            logger.warning(f"Failed to play alert sound: {e}")

    def test(self) -> bool:
        """
        Test notification system.

        Returns:
            True if test successful
        """
        try:
            if self._plyer_available:
                from plyer import notification

                notification.notify(
                    title="Ordinis Test",
                    message="Desktop notifications are working!",
                    app_name=self.app_name,
                    timeout=5,
                )

            if self._winsound_available:
                import winsound

                winsound.Beep(800, 200)

            return True
        except Exception as e:
            logger.exception(f"Notification test failed: {e}")
            return False

    @property
    def is_available(self) -> bool:
        """Check if any notification method is available."""
        return self._plyer_available or self._winsound_available

    def get_status(self) -> dict:
        """Get notifier status."""
        return {
            "plyer_available": self._plyer_available,
            "winsound_available": self._winsound_available,
            "sound_enabled": self.enable_sound,
            "app_name": self.app_name,
        }
