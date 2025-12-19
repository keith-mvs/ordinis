"""
Alert manager for centralized alerting.

Features:
- Multiple alert channels
- Rate limiting per alert type
- Deduplication
- Severity-based routing
- Alert history
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Types of alerts."""

    KILL_SWITCH = "kill_switch"
    RISK_BREACH = "risk_breach"
    ORDER_REJECTED = "order_rejected"
    POSITION_RECONCILIATION = "position_reconciliation"
    API_CONNECTIVITY = "api_connectivity"
    SYSTEM_HEALTH = "system_health"
    TRADE_EXECUTED = "trade_executed"
    DAILY_SUMMARY = "daily_summary"
    CUSTOM = "custom"


@dataclass
class Alert:
    """Alert data structure."""

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    channels_sent: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "acknowledged": self.acknowledged,
            "channels_sent": self.channels_sent,
        }


@dataclass
class AlertChannel:
    """Base alert channel."""

    name: str
    enabled: bool = True
    min_severity: AlertSeverity = AlertSeverity.WARNING
    send_func: Callable[[Alert], bool] | None = None
    async_send_func: Callable[[Alert], Any] | None = None


class AlertManager:
    """
    Central alert manager.

    Coordinates alert delivery across multiple channels with:
    - Rate limiting per alert type
    - Deduplication within time window
    - Severity-based channel routing
    - Alert history
    """

    def __init__(
        self,
        rate_limit_seconds: float = 60.0,
        dedup_window_seconds: float = 300.0,
        max_history: int = 1000,
    ):
        """
        Initialize alert manager.

        Args:
            rate_limit_seconds: Minimum time between same alert type
            dedup_window_seconds: Window for deduplication
            max_history: Maximum alerts to keep in history
        """
        self._rate_limit_seconds = rate_limit_seconds
        self._dedup_window_seconds = dedup_window_seconds
        self._max_history = max_history

        self._channels: dict[str, AlertChannel] = {}
        self._history: list[Alert] = []
        self._last_sent: dict[str, datetime] = {}
        self._dedup_hashes: dict[str, datetime] = {}
        self._lock = asyncio.Lock()

        self._alert_count = 0
        self._suppressed_count = 0

    def register_channel(
        self,
        name: str,
        send_func: Callable[[Alert], bool] | None = None,
        async_send_func: Callable[[Alert], Any] | None = None,
        min_severity: AlertSeverity = AlertSeverity.WARNING,
        enabled: bool = True,
    ) -> None:
        """
        Register an alert channel.

        Args:
            name: Channel name
            send_func: Sync send function
            async_send_func: Async send function
            min_severity: Minimum severity to send
            enabled: Whether channel is enabled
        """
        self._channels[name] = AlertChannel(
            name=name,
            enabled=enabled,
            min_severity=min_severity,
            send_func=send_func,
            async_send_func=async_send_func,
        )
        logger.info(f"Registered alert channel: {name}")

    def disable_channel(self, name: str) -> None:
        """Disable an alert channel."""
        if name in self._channels:
            self._channels[name].enabled = False

    def enable_channel(self, name: str) -> None:
        """Enable an alert channel."""
        if name in self._channels:
            self._channels[name].enabled = True

    async def send(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        title: str,
        message: str,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> Alert | None:
        """
        Send alert through configured channels.

        Args:
            alert_type: Type of alert
            severity: Alert severity
            title: Alert title
            message: Alert message
            metadata: Additional data
            force: Skip rate limiting and dedup

        Returns:
            Alert if sent, None if suppressed
        """
        async with self._lock:
            # Generate alert ID
            alert_id = f"{alert_type.value}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{self._alert_count}"

            alert = Alert(
                alert_id=alert_id,
                alert_type=alert_type,
                severity=severity,
                title=title,
                message=message,
                metadata=metadata or {},
            )

            if not force:
                # Check rate limit
                if self._is_rate_limited(alert_type):
                    self._suppressed_count += 1
                    logger.debug(f"Alert rate limited: {alert_type.value}")
                    return None

                # Check deduplication
                if self._is_duplicate(alert):
                    self._suppressed_count += 1
                    logger.debug(f"Alert deduplicated: {title}")
                    return None

            # Send to channels
            await self._send_to_channels(alert)

            # Update tracking
            self._alert_count += 1
            self._last_sent[alert_type.value] = datetime.utcnow()
            self._record_dedup(alert)
            self._add_to_history(alert)

            return alert

    async def send_emergency(
        self,
        title: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> Alert | None:
        """
        Send emergency alert (bypasses rate limiting).

        Args:
            title: Alert title
            message: Alert message
            metadata: Additional data

        Returns:
            Alert if sent
        """
        return await self.send(
            alert_type=AlertType.KILL_SWITCH,
            severity=AlertSeverity.EMERGENCY,
            title=title,
            message=message,
            metadata=metadata,
            force=True,
        )

    async def send_risk_breach(
        self,
        title: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> Alert | None:
        """Send risk breach alert."""
        return await self.send(
            alert_type=AlertType.RISK_BREACH,
            severity=AlertSeverity.CRITICAL,
            title=title,
            message=message,
            metadata=metadata,
        )

    async def send_warning(
        self,
        title: str,
        message: str,
        alert_type: AlertType = AlertType.CUSTOM,
        metadata: dict[str, Any] | None = None,
    ) -> Alert | None:
        """Send warning alert."""
        return await self.send(
            alert_type=alert_type,
            severity=AlertSeverity.WARNING,
            title=title,
            message=message,
            metadata=metadata,
        )

    async def send_info(
        self,
        title: str,
        message: str,
        alert_type: AlertType = AlertType.CUSTOM,
        metadata: dict[str, Any] | None = None,
    ) -> Alert | None:
        """Send info alert."""
        return await self.send(
            alert_type=alert_type,
            severity=AlertSeverity.INFO,
            title=title,
            message=message,
            metadata=metadata,
        )

    def _is_rate_limited(self, alert_type: AlertType) -> bool:
        """Check if alert type is rate limited."""
        key = alert_type.value
        if key not in self._last_sent:
            return False

        elapsed = (datetime.utcnow() - self._last_sent[key]).total_seconds()
        return elapsed < self._rate_limit_seconds

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is duplicate within window."""
        hash_key = self._compute_hash(alert)
        if hash_key not in self._dedup_hashes:
            return False

        elapsed = (datetime.utcnow() - self._dedup_hashes[hash_key]).total_seconds()
        return elapsed < self._dedup_window_seconds

    def _compute_hash(self, alert: Alert) -> str:
        """Compute hash for deduplication."""
        content = f"{alert.alert_type.value}:{alert.severity.value}:{alert.title}:{alert.message}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _record_dedup(self, alert: Alert) -> None:
        """Record alert hash for deduplication."""
        hash_key = self._compute_hash(alert)
        self._dedup_hashes[hash_key] = datetime.utcnow()

        # Clean old hashes
        cutoff = datetime.utcnow() - timedelta(seconds=self._dedup_window_seconds * 2)
        self._dedup_hashes = {k: v for k, v in self._dedup_hashes.items() if v > cutoff}

    async def _send_to_channels(self, alert: Alert) -> None:
        """Send alert to all enabled channels."""
        severity_order = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.CRITICAL,
            AlertSeverity.EMERGENCY,
        ]
        alert_severity_index = severity_order.index(alert.severity)

        for name, channel in self._channels.items():
            if not channel.enabled:
                continue

            # Check minimum severity
            channel_severity_index = severity_order.index(channel.min_severity)
            if alert_severity_index < channel_severity_index:
                continue

            # Send through channel
            try:
                if channel.async_send_func:
                    await channel.async_send_func(alert)
                    alert.channels_sent.append(name)
                elif channel.send_func:
                    if channel.send_func(alert):
                        alert.channels_sent.append(name)
            except Exception as e:
                logger.exception(f"Failed to send alert through {name}: {e}")

    def _add_to_history(self, alert: Alert) -> None:
        """Add alert to history."""
        self._history.append(alert)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

    def get_history(
        self,
        limit: int = 100,
        severity: AlertSeverity | None = None,
        alert_type: AlertType | None = None,
    ) -> list[Alert]:
        """
        Get alert history.

        Args:
            limit: Maximum alerts to return
            severity: Filter by severity
            alert_type: Filter by type

        Returns:
            List of alerts
        """
        alerts = self._history.copy()

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]

        return alerts[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get alerting statistics."""
        return {
            "total_alerts": self._alert_count,
            "suppressed_alerts": self._suppressed_count,
            "channels": {
                name: {
                    "enabled": channel.enabled,
                    "min_severity": channel.min_severity.value,
                }
                for name, channel in self._channels.items()
            },
            "history_count": len(self._history),
        }

    def acknowledge(self, alert_id: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID

        Returns:
            True if acknowledged
        """
        for alert in self._history:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_unacknowledged(
        self,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """
        Get unacknowledged alerts.

        Args:
            severity: Filter by severity

        Returns:
            List of unacknowledged alerts
        """
        alerts = [a for a in self._history if not a.acknowledged]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts
