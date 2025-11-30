"""
Health Check System.

Monitors system health and component status for observability
and alerting.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class HealthStatus(Enum):
    """Health check status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: dict[str, Any] = field(default_factory=dict)
    response_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "response_time": self.response_time,
        }


class HealthCheck:
    """
    System health check coordinator.

    Manages and executes health checks for various components.
    """

    def __init__(self):
        """Initialize health check system."""
        self._checks: dict[str, Callable[[], HealthCheckResult]] = {}
        self._last_results: dict[str, HealthCheckResult] = {}

    def register_check(self, name: str, check_func: Callable[[], HealthCheckResult]):
        """
        Register a health check.

        Args:
            name: Check name
            check_func: Function that performs the check
        """
        self._checks[name] = check_func

    def run_check(self, name: str) -> HealthCheckResult:
        """
        Run a specific health check.

        Args:
            name: Check name

        Returns:
            Health check result
        """
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Check '{name}' not found",
            )

        from time import perf_counter

        start = perf_counter()

        try:
            result = self._checks[name]()
            result.response_time = perf_counter() - start
            self._last_results[name] = result
            return result
        except Exception as e:
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                response_time=perf_counter() - start,
            )
            self._last_results[name] = result
            return result

    def run_all_checks(self) -> dict[str, HealthCheckResult]:
        """
        Run all registered health checks.

        Returns:
            Dictionary of check results
        """
        results = {}
        for name in self._checks:
            results[name] = self.run_check(name)
        return results

    def get_overall_status(self) -> HealthStatus:
        """
        Get overall system health status.

        Returns:
            Overall health status (worst status among all checks)
        """
        if not self._last_results:
            return HealthStatus.UNKNOWN

        statuses = [result.status for result in self._last_results.values()]

        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        if HealthStatus.HEALTHY in statuses:
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def get_health_report(self) -> dict[str, Any]:
        """
        Get comprehensive health report.

        Returns:
            Health report with all check results
        """
        results = self.run_all_checks()

        return {
            "overall_status": self.get_overall_status().value,
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {name: result.to_dict() for name, result in results.items()},
        }

    def get_last_results(self) -> dict[str, HealthCheckResult]:
        """
        Get last health check results without re-running checks.

        Returns:
            Dictionary of last check results
        """
        return self._last_results.copy()


# Common health check implementations
def database_health_check() -> HealthCheckResult:
    """Check database connectivity and health."""
    try:
        # TODO: Implement actual database check
        return HealthCheckResult(
            name="database", status=HealthStatus.HEALTHY, message="Database is accessible"
        )
    except Exception as e:
        return HealthCheckResult(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database error: {e}",
        )


def api_health_check() -> HealthCheckResult:
    """Check external API connectivity."""
    try:
        # TODO: Implement actual API check
        return HealthCheckResult(
            name="api", status=HealthStatus.HEALTHY, message="APIs are accessible"
        )
    except Exception as e:
        return HealthCheckResult(
            name="api", status=HealthStatus.UNHEALTHY, message=f"API error: {e}"
        )


def disk_space_health_check() -> HealthCheckResult:
    """Check disk space availability."""
    import shutil

    try:
        total, used, free = shutil.disk_usage("/")
        free_percent = (free / total) * 100

        if free_percent < 10:
            status = HealthStatus.UNHEALTHY
            message = f"Disk space critical: {free_percent:.1f}% free"
        elif free_percent < 20:
            status = HealthStatus.DEGRADED
            message = f"Disk space low: {free_percent:.1f}% free"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk space OK: {free_percent:.1f}% free"

        return HealthCheckResult(
            name="disk_space",
            status=status,
            message=message,
            details={
                "total_gb": total / (1024**3),
                "used_gb": used / (1024**3),
                "free_gb": free / (1024**3),
                "free_percent": free_percent,
            },
        )
    except Exception as e:
        return HealthCheckResult(
            name="disk_space",
            status=HealthStatus.UNHEALTHY,
            message=f"Disk check failed: {e}",
        )


def memory_health_check() -> HealthCheckResult:
    """Check memory usage."""
    import psutil

    try:
        memory = psutil.virtual_memory()
        available_percent = memory.available / memory.total * 100

        if available_percent < 10:
            status = HealthStatus.UNHEALTHY
            message = f"Memory critical: {available_percent:.1f}% available"
        elif available_percent < 20:
            status = HealthStatus.DEGRADED
            message = f"Memory low: {available_percent:.1f}% available"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory OK: {available_percent:.1f}% available"

        return HealthCheckResult(
            name="memory",
            status=status,
            message=message,
            details={
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_percent": memory.percent,
                "available_percent": available_percent,
            },
        )
    except Exception as e:
        return HealthCheckResult(
            name="memory",
            status=HealthStatus.UNHEALTHY,
            message=f"Memory check failed: {e}",
        )


# Global health check instance
_global_health_check: HealthCheck | None = None


def get_health_check() -> HealthCheck:
    """
    Get global health check instance.

    Returns:
        Global health check coordinator
    """
    global _global_health_check  # noqa: PLW0603
    if _global_health_check is None:
        _global_health_check = HealthCheck()

        # Register default checks
        _global_health_check.register_check("disk_space", disk_space_health_check)
        _global_health_check.register_check("memory", memory_health_check)

    return _global_health_check
