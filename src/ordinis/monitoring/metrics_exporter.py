"""
Prometheus Metrics Exporter for Ordinis Paper Trading.

Exposes metrics via HTTP endpoint for Prometheus scraping.
Default port: 3005 (configured in prometheus.yml)
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import signal
import sys
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
import uvicorn

from ordinis.monitoring.collectors import MetricsCollector

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)

# Global collector instance
_collector: MetricsCollector | None = None


def get_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _collector
    if _collector is None:
        _collector = MetricsCollector()
    return _collector


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan context manager."""
    collector = get_collector()
    collector.start()
    logger.info("Metrics exporter started")
    yield
    collector.stop()
    logger.info("Metrics exporter stopped")


class MetricsExporter:
    """
    HTTP server exposing Prometheus metrics.

    Usage:
        exporter = MetricsExporter(port=3005)
        await exporter.start()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",  # noqa: S104
        port: int = 3005,
    ) -> None:
        """
        Initialize the metrics exporter.

        Args:
            host: Host to bind to
            port: Port to listen on (default 3005)
        """
        self.host = host
        self.port = port
        self._app = self._create_app()
        self._server: uvicorn.Server | None = None
        self._collector = get_collector()

    def _create_app(self) -> FastAPI:
        """Create the FastAPI application."""
        app = FastAPI(
            title="Ordinis Metrics Exporter",
            description="Prometheus metrics for paper trading monitoring",
            version="1.0.0",
            lifespan=lifespan,
        )

        @app.get("/")
        async def root() -> dict[str, str]:
            """Root endpoint with service info."""
            return {
                "service": "ordinis-metrics-exporter",
                "version": "1.0.0",
                "metrics_endpoint": "/metrics",
            }

        @app.get("/health")
        async def health() -> dict[str, str]:
            """Health check endpoint."""
            return {"status": "healthy"}

        @app.get("/ready")
        async def ready() -> dict[str, Any]:
            """Readiness check endpoint."""
            collector = get_collector()
            return {
                "ready": True,
                "collector_running": collector._running,
            }

        @app.get("/metrics")
        async def metrics() -> Response:
            """Prometheus metrics endpoint."""
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST,
            )

        return app

    @property
    def collector(self) -> MetricsCollector:
        """Get the metrics collector."""
        return self._collector

    async def start(self) -> None:
        """Start the metrics server."""
        config = uvicorn.Config(
            self._app,
            host=self.host,
            port=self.port,
            log_level="info",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        logger.info(f"Starting metrics exporter on http://{self.host}:{self.port}")
        await self._server.serve()

    async def stop(self) -> None:
        """Stop the metrics server."""
        if self._server:
            self._server.should_exit = True
            logger.info("Stopping metrics exporter")


async def start_metrics_server(
    host: str = "0.0.0.0",  # noqa: S104
    port: int = 3005,
) -> MetricsExporter:
    """
    Start the metrics server.

    Args:
        host: Host to bind to
        port: Port to listen on

    Returns:
        MetricsExporter instance
    """
    exporter = MetricsExporter(host=host, port=port)
    await exporter.start()
    return exporter


def run_standalone() -> None:
    """Run the metrics exporter as a standalone service."""
    import argparse

    parser = argparse.ArgumentParser(description="Ordinis Metrics Exporter")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")  # noqa: S104
    parser.add_argument("--port", type=int, default=3005, help="Port to listen on")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Starting Ordinis Metrics Exporter on {args.host}:{args.port}")

    # Handle shutdown signals
    def handle_shutdown(signum: int, frame: Any) -> None:
        logger.info("Received shutdown signal")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Run the server
    asyncio.run(start_metrics_server(host=args.host, port=args.port))


if __name__ == "__main__":
    run_standalone()
