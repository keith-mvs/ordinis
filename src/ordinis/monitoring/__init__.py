"""
Ordinis Monitoring Module.

Provides Prometheus metrics export for paper trading workflow monitoring.
Exposes throughput, latency, and workflow health metrics to Grafana.
"""

from __future__ import annotations

from typing import Any

from ordinis.monitoring.collectors import MetricsCollector

# `MetricsExporter` requires optional web dependencies (fastapi/uvicorn).
# Keep the core monitoring utilities importable even when those extras aren't.
try:
    from ordinis.monitoring.metrics_exporter import MetricsExporter, start_metrics_server
except ModuleNotFoundError as e:  # pragma: no cover
    MetricsExporter = None  # type: ignore[assignment]

    def start_metrics_server(*args: Any, **kwargs: Any):
        raise ImportError(
            "MetricsExporter requires optional dependencies. Install 'fastapi' and "
            "'uvicorn' (or the project's monitoring extras) to use start_metrics_server()."
        ) from e

__all__ = [
    "MetricsCollector",
    "MetricsExporter",
    "start_metrics_server",
]
