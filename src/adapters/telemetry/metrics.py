"""
Metrics Collection and Tracking.

Collects and tracks performance metrics, trading statistics,
and system health indicators.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""

    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Execution metrics
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0

    # Timing metrics
    avg_execution_time: float = 0.0
    min_execution_time: float = float("inf")
    max_execution_time: float = 0.0

    # Trading metrics
    signals_generated: int = 0
    signals_executed: int = 0
    signals_rejected: int = 0

    # API metrics
    api_calls: int = 0
    api_errors: int = 0
    api_avg_response_time: float = 0.0

    # Custom metrics
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_operations == 0:
            return 0.0
        return self.successful_operations / self.total_operations

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_operations == 0:
            return 0.0
        return self.failed_operations / self.total_operations

    @property
    def signal_execution_rate(self) -> float:
        """Calculate signal execution rate."""
        if self.signals_generated == 0:
            return 0.0
        return self.signals_executed / self.signals_generated

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": self.success_rate,
            "error_rate": self.error_rate,
            "avg_execution_time": self.avg_execution_time,
            "min_execution_time": (
                self.min_execution_time if self.min_execution_time != float("inf") else 0
            ),
            "max_execution_time": self.max_execution_time,
            "signals_generated": self.signals_generated,
            "signals_executed": self.signals_executed,
            "signals_rejected": self.signals_rejected,
            "signal_execution_rate": self.signal_execution_rate,
            "api_calls": self.api_calls,
            "api_errors": self.api_errors,
            "api_avg_response_time": self.api_avg_response_time,
            "custom_metrics": self.custom_metrics,
        }


class MetricsCollector:
    """
    Collects and aggregates system metrics.

    Thread-safe metrics collection with periodic reporting.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = PerformanceMetrics()
        self._execution_times: list[float] = []
        self._api_response_times: list[float] = []
        self._counters: dict[str, int] = defaultdict(int)

    def record_operation(self, success: bool, execution_time: float | None = None):
        """
        Record an operation execution.

        Args:
            success: Whether operation succeeded
            execution_time: Execution time in seconds
        """
        self.metrics.total_operations += 1

        if success:
            self.metrics.successful_operations += 1
        else:
            self.metrics.failed_operations += 1

        if execution_time is not None:
            self._execution_times.append(execution_time)
            self.metrics.min_execution_time = min(self.metrics.min_execution_time, execution_time)
            self.metrics.max_execution_time = max(self.metrics.max_execution_time, execution_time)
            self.metrics.avg_execution_time = sum(self._execution_times) / len(
                self._execution_times
            )

    def record_signal(
        self, generated: bool = False, executed: bool = False, rejected: bool = False
    ):
        """
        Record signal generation and execution.

        Args:
            generated: Signal was generated
            executed: Signal was executed
            rejected: Signal was rejected
        """
        if generated:
            self.metrics.signals_generated += 1
        if executed:
            self.metrics.signals_executed += 1
        if rejected:
            self.metrics.signals_rejected += 1

    def record_api_call(self, success: bool, response_time: float | None = None):
        """
        Record API call.

        Args:
            success: Whether call succeeded
            response_time: Response time in seconds
        """
        self.metrics.api_calls += 1

        if not success:
            self.metrics.api_errors += 1

        if response_time is not None:
            self._api_response_times.append(response_time)
            self.metrics.api_avg_response_time = sum(self._api_response_times) / len(
                self._api_response_times
            )

    def increment_counter(self, name: str, value: int = 1):
        """
        Increment a custom counter.

        Args:
            name: Counter name
            value: Increment value
        """
        self._counters[name] += value
        self.metrics.custom_metrics[name] = self._counters[name]

    def set_gauge(self, name: str, value: float):
        """
        Set a gauge metric.

        Args:
            name: Gauge name
            value: Gauge value
        """
        self.metrics.custom_metrics[name] = value

    def get_metrics(self) -> PerformanceMetrics:
        """
        Get current metrics snapshot.

        Returns:
            Current performance metrics
        """
        return self.metrics

    def get_summary(self) -> dict[str, Any]:
        """
        Get metrics summary.

        Returns:
            Dictionary with metrics summary
        """
        return self.metrics.to_dict()

    def reset(self):
        """Reset all metrics."""
        self.metrics = PerformanceMetrics()
        self._execution_times.clear()
        self._api_response_times.clear()
        self._counters.clear()


# Global metrics collector instance
_global_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """
    Get global metrics collector instance.

    Returns:
        Global metrics collector
    """
    global _global_collector  # noqa: PLW0603
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_metrics():
    """Reset global metrics."""
    global _global_collector  # noqa: PLW0602
    if _global_collector is not None:
        _global_collector.reset()
