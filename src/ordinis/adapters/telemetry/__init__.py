"""
Telemetry adapters for Ordinis trading system.

Provides comprehensive monitoring, logging, health checks, KPI tracking,
metrics collection, and distributed tracing for observability.
"""

from .health import HealthCheck, HealthStatus
from .kpi import (
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
from .logger import get_logger, setup_logging
from .metrics import MetricsCollector, PerformanceMetrics
from .tracing import TracingConfig, get_tracer, setup_tracing, shutdown_tracing

__all__ = [
    "Alert",
    "AlertSeverity",
    "HealthCheck",
    "HealthStatus",
    "KPIStatus",
    "KPIThreshold",
    "KPITracker",
    "KPIValue",
    "MetricsCollector",
    "PerformanceMetrics",
    "TracingConfig",
    "TradingKPIs",
    "get_kpi_tracker",
    "get_logger",
    "get_tracer",
    "reset_kpi_tracker",
    "setup_logging",
    "setup_tracing",
    "shutdown_tracing",
]
