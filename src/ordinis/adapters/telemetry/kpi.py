"""
KPI (Key Performance Indicator) Tracking System.

Provides comprehensive KPI tracking, thresholds, and alerting
for trading system performance monitoring.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class KPIStatus(Enum):
    """KPI status indicators."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class KPIThreshold:
    """
    Threshold configuration for a KPI.

    Attributes:
        warning_min: Minimum value before warning (None = no min warning)
        warning_max: Maximum value before warning (None = no max warning)
        critical_min: Minimum value before critical (None = no min critical)
        critical_max: Maximum value before critical (None = no max critical)
    """

    warning_min: float | None = None
    warning_max: float | None = None
    critical_min: float | None = None
    critical_max: float | None = None

    def evaluate(self, value: float) -> KPIStatus:
        """
        Evaluate a value against thresholds.

        Args:
            value: The value to evaluate

        Returns:
            KPIStatus based on threshold evaluation
        """
        # Check critical thresholds first
        if self.critical_min is not None and value < self.critical_min:
            return KPIStatus.CRITICAL
        if self.critical_max is not None and value > self.critical_max:
            return KPIStatus.CRITICAL

        # Check warning thresholds
        if self.warning_min is not None and value < self.warning_min:
            return KPIStatus.WARNING
        if self.warning_max is not None and value > self.warning_max:
            return KPIStatus.WARNING

        return KPIStatus.HEALTHY


@dataclass
class KPIValue:
    """
    A single KPI measurement.

    Attributes:
        name: KPI name
        value: Current value
        timestamp: When the value was recorded
        status: KPI status based on thresholds
        unit: Unit of measurement
        description: Human-readable description
    """

    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: KPIStatus = KPIStatus.UNKNOWN
    unit: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "unit": self.unit,
            "description": self.description,
        }


@dataclass
class Alert:
    """
    KPI alert notification.

    Attributes:
        kpi_name: Name of the KPI that triggered the alert
        severity: Alert severity level
        message: Alert message
        value: Current value that triggered the alert
        threshold: Threshold that was breached
        timestamp: When the alert was created
    """

    kpi_name: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float | None = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "kpi_name": self.kpi_name,
            "severity": self.severity.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class TradingKPIs:
    """
    Trading-specific KPIs.

    Comprehensive set of key performance indicators for trading systems.
    """

    # Return metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    daily_return_avg: float = 0.0
    monthly_return_avg: float = 0.0

    # Risk metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    downside_deviation: float = 0.0

    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0

    # Signal metrics
    signals_generated: int = 0
    signals_executed: int = 0
    signal_accuracy: float = 0.0

    # Position metrics
    avg_position_size: float = 0.0
    max_position_size: float = 0.0
    avg_exposure: float = 0.0

    # System metrics
    uptime_percent: float = 100.0
    api_success_rate: float = 100.0
    data_freshness_seconds: float = 0.0

    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def win_loss_ratio(self) -> float:
        """Calculate win/loss ratio."""
        if self.avg_loss == 0:
            return float("inf") if self.avg_win > 0 else 0.0
        return abs(self.avg_win / self.avg_loss) if self.avg_loss != 0 else 0.0

    @property
    def expectancy(self) -> float:
        """Calculate trade expectancy."""
        return (self.win_rate * self.avg_win) - ((1 - self.win_rate) * abs(self.avg_loss))

    @property
    def risk_reward_ratio(self) -> float:
        """Calculate risk/reward ratio."""
        if self.avg_loss == 0:
            return 0.0
        return abs(self.avg_win / self.avg_loss)

    def calculate_win_rate(self):
        """Recalculate win rate from trade counts."""
        if self.total_trades == 0:
            self.win_rate = 0.0
        else:
            self.win_rate = self.winning_trades / self.total_trades

    def to_dict(self) -> dict[str, Any]:
        """Convert KPIs to dictionary."""
        return {
            # Return metrics
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "daily_return_avg": self.daily_return_avg,
            "monthly_return_avg": self.monthly_return_avg,
            # Risk metrics
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "current_drawdown": self.current_drawdown,
            "volatility": self.volatility,
            "downside_deviation": self.downside_deviation,
            # Trade metrics
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "largest_win": self.largest_win,
            "largest_loss": self.largest_loss,
            "profit_factor": self.profit_factor,
            "avg_trade_duration": self.avg_trade_duration,
            "win_loss_ratio": self.win_loss_ratio,
            "expectancy": self.expectancy,
            "risk_reward_ratio": self.risk_reward_ratio,
            # Signal metrics
            "signals_generated": self.signals_generated,
            "signals_executed": self.signals_executed,
            "signal_accuracy": self.signal_accuracy,
            # Position metrics
            "avg_position_size": self.avg_position_size,
            "max_position_size": self.max_position_size,
            "avg_exposure": self.avg_exposure,
            # System metrics
            "uptime_percent": self.uptime_percent,
            "api_success_rate": self.api_success_rate,
            "data_freshness_seconds": self.data_freshness_seconds,
            # Timestamp
            "timestamp": self.timestamp.isoformat(),
        }


# Default KPI thresholds for trading systems
DEFAULT_KPI_THRESHOLDS: dict[str, KPIThreshold] = {
    # Return thresholds
    "total_return": KPIThreshold(warning_min=-0.05, critical_min=-0.10),
    "annualized_return": KPIThreshold(warning_min=-0.10, critical_min=-0.20),
    # Risk thresholds
    "sharpe_ratio": KPIThreshold(warning_min=0.5, critical_min=0.0),
    "sortino_ratio": KPIThreshold(warning_min=0.5, critical_min=0.0),
    # Drawdowns are negative, so worse = more negative = use min thresholds
    "max_drawdown": KPIThreshold(warning_min=-0.10, critical_min=-0.20),
    "current_drawdown": KPIThreshold(warning_min=-0.05, critical_min=-0.10),
    "volatility": KPIThreshold(warning_max=0.25, critical_max=0.40),
    # Trade thresholds
    "win_rate": KPIThreshold(warning_min=0.40, critical_min=0.30),
    "profit_factor": KPIThreshold(warning_min=1.2, critical_min=1.0),
    # System thresholds
    "uptime_percent": KPIThreshold(warning_min=99.0, critical_min=95.0),
    "api_success_rate": KPIThreshold(warning_min=98.0, critical_min=95.0),
    "data_freshness_seconds": KPIThreshold(warning_max=60, critical_max=300),
}


class KPITracker:
    """
    Tracks KPIs over time with alerting capabilities.

    Provides:
    - Real-time KPI tracking
    - Threshold-based alerting
    - Historical KPI storage
    - Trend analysis
    """

    def __init__(
        self,
        thresholds: dict[str, KPIThreshold] | None = None,
        alert_handlers: list[Callable[[Alert], None]] | None = None,
        history_retention: timedelta = timedelta(days=30),
    ):
        """
        Initialize KPI tracker.

        Args:
            thresholds: Custom thresholds (merged with defaults)
            alert_handlers: Functions to call when alerts are triggered
            history_retention: How long to keep historical KPIs
        """
        self._thresholds = {**DEFAULT_KPI_THRESHOLDS}
        if thresholds:
            self._thresholds.update(thresholds)

        self._alert_handlers = alert_handlers or []
        self._history_retention = history_retention

        # Current KPIs
        self._current_kpis = TradingKPIs()

        # Historical data
        self._kpi_history: list[TradingKPIs] = []
        self._alert_history: list[Alert] = []

        # Active alerts (not yet resolved)
        self._active_alerts: dict[str, Alert] = {}

    def update_kpis(self, kpis: TradingKPIs):
        """
        Update current KPIs and check thresholds.

        Args:
            kpis: New KPI values
        """
        self._current_kpis = kpis
        self._kpi_history.append(kpis)
        self._cleanup_history()
        self._check_thresholds(kpis)

    def update_kpi(self, name: str, value: float):
        """
        Update a single KPI value.

        Args:
            name: KPI name
            value: New value
        """
        if hasattr(self._current_kpis, name):
            setattr(self._current_kpis, name, value)
            self._current_kpis.timestamp = datetime.utcnow()
            self._check_single_threshold(name, value)

    def get_current_kpis(self) -> TradingKPIs:
        """Get current KPIs."""
        return self._current_kpis

    def get_kpi_value(self, name: str) -> KPIValue:
        """
        Get a single KPI with status.

        Args:
            name: KPI name

        Returns:
            KPIValue with current value and status
        """
        if not hasattr(self._current_kpis, name):
            return KPIValue(
                name=name,
                value=0.0,
                status=KPIStatus.UNKNOWN,
                description=f"Unknown KPI: {name}",
            )

        value = getattr(self._current_kpis, name)
        status = KPIStatus.HEALTHY

        if name in self._thresholds:
            status = self._thresholds[name].evaluate(value)

        return KPIValue(
            name=name,
            value=value,
            timestamp=self._current_kpis.timestamp,
            status=status,
        )

    def get_kpi_summary(self) -> dict[str, KPIValue]:
        """
        Get summary of all KPIs with status.

        Returns:
            Dictionary of KPI name to KPIValue
        """
        kpi_dict = self._current_kpis.to_dict()
        summary = {}

        for name, value in kpi_dict.items():
            if name == "timestamp" or not isinstance(value, int | float):
                continue

            status = KPIStatus.HEALTHY
            if name in self._thresholds:
                status = self._thresholds[name].evaluate(value)

            summary[name] = KPIValue(
                name=name,
                value=value,
                timestamp=self._current_kpis.timestamp,
                status=status,
            )

        return summary

    def get_active_alerts(self) -> list[Alert]:
        """Get list of active alerts."""
        return list(self._active_alerts.values())

    def get_alert_history(
        self,
        since: datetime | None = None,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """
        Get historical alerts.

        Args:
            since: Only include alerts after this time
            severity: Filter by severity

        Returns:
            List of matching alerts
        """
        alerts = self._alert_history

        if since:
            alerts = [a for a in alerts if a.timestamp >= since]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts

    def set_threshold(self, name: str, threshold: KPIThreshold):
        """
        Set or update a KPI threshold.

        Args:
            name: KPI name
            threshold: New threshold configuration
        """
        self._thresholds[name] = threshold

    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """
        Add an alert handler.

        Args:
            handler: Function to call when an alert is triggered
        """
        self._alert_handlers.append(handler)

    def clear_alert(self, kpi_name: str):
        """
        Clear an active alert.

        Args:
            kpi_name: KPI name to clear alert for
        """
        if kpi_name in self._active_alerts:
            del self._active_alerts[kpi_name]

    def get_health_status(self) -> dict[str, Any]:
        """
        Get overall system health status.

        Returns:
            Health status summary
        """
        summary = self.get_kpi_summary()

        critical_count = sum(1 for v in summary.values() if v.status == KPIStatus.CRITICAL)
        warning_count = sum(1 for v in summary.values() if v.status == KPIStatus.WARNING)
        healthy_count = sum(1 for v in summary.values() if v.status == KPIStatus.HEALTHY)

        if critical_count > 0:
            overall_status = KPIStatus.CRITICAL
        elif warning_count > 0:
            overall_status = KPIStatus.WARNING
        else:
            overall_status = KPIStatus.HEALTHY

        return {
            "overall_status": overall_status.value,
            "critical_count": critical_count,
            "warning_count": warning_count,
            "healthy_count": healthy_count,
            "active_alerts": len(self._active_alerts),
            "kpis": {name: v.to_dict() for name, v in summary.items()},
        }

    def _check_thresholds(self, kpis: TradingKPIs):
        """Check all thresholds and trigger alerts."""
        kpi_dict = kpis.to_dict()

        for name, threshold in self._thresholds.items():
            if name in kpi_dict and isinstance(kpi_dict[name], int | float):
                self._check_single_threshold(name, kpi_dict[name])

    def _check_single_threshold(self, name: str, value: float):
        """Check a single threshold and handle alerts."""
        if name not in self._thresholds:
            return

        threshold = self._thresholds[name]
        status = threshold.evaluate(value)

        # Create or clear alerts based on status
        if status == KPIStatus.CRITICAL:
            self._create_alert(
                name,
                AlertSeverity.CRITICAL,
                value,
                self._get_breached_threshold(name, value, threshold),
            )
        elif status == KPIStatus.WARNING:
            self._create_alert(
                name,
                AlertSeverity.WARNING,
                value,
                self._get_breached_threshold(name, value, threshold),
            )
        else:
            # Clear any existing alert for this KPI
            self.clear_alert(name)

    def _create_alert(
        self,
        kpi_name: str,
        severity: AlertSeverity,
        value: float,
        threshold: float | None,
    ):
        """Create and dispatch an alert."""
        message = f"KPI '{kpi_name}' is {severity.value}: value={value:.4f}"
        if threshold is not None:
            message += f", threshold={threshold:.4f}"

        alert = Alert(
            kpi_name=kpi_name,
            severity=severity,
            message=message,
            value=value,
            threshold=threshold,
        )

        # Only trigger handlers for new or escalated alerts
        existing = self._active_alerts.get(kpi_name)
        should_notify = existing is None or (
            existing.severity != AlertSeverity.CRITICAL and severity == AlertSeverity.CRITICAL
        )

        self._active_alerts[kpi_name] = alert
        self._alert_history.append(alert)

        if should_notify:
            for handler in self._alert_handlers:
                try:
                    handler(alert)
                except Exception:
                    # Don't let handler failures break tracking
                    pass

    def _get_breached_threshold(
        self, name: str, value: float, threshold: KPIThreshold
    ) -> float | None:
        """Determine which threshold was breached."""
        if threshold.critical_min is not None and value < threshold.critical_min:
            return threshold.critical_min
        if threshold.critical_max is not None and value > threshold.critical_max:
            return threshold.critical_max
        if threshold.warning_min is not None and value < threshold.warning_min:
            return threshold.warning_min
        if threshold.warning_max is not None and value > threshold.warning_max:
            return threshold.warning_max
        return None

    def _cleanup_history(self):
        """Remove old historical data."""
        cutoff = datetime.utcnow() - self._history_retention
        self._kpi_history = [k for k in self._kpi_history if k.timestamp >= cutoff]
        self._alert_history = [a for a in self._alert_history if a.timestamp >= cutoff]


# Global KPI tracker instance
_global_tracker: KPITracker | None = None


def get_kpi_tracker() -> KPITracker:
    """
    Get global KPI tracker instance.

    Returns:
        Global KPI tracker
    """
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = KPITracker()
    return _global_tracker


def reset_kpi_tracker():
    """Reset global KPI tracker."""
    global _global_tracker
    _global_tracker = None
