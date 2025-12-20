"""Health check and readiness probe HTTP server."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading
from typing import Callable


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for health and readiness checks."""

    health_check_fn: Callable = None

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self._handle_health()
        elif self.path == "/ready":
            self._handle_ready()
        elif self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"See port 3005 for Prometheus metrics\n")
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_health(self):
        """Health check - is the service running?"""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        health = {"status": "healthy", "service": "ordinis-trading"}
        self.wfile.write(json.dumps(health).encode())

    def _handle_ready(self):
        """Readiness check - is the service ready to trade?"""
        if self.health_check_fn:
            status = self.health_check_fn()
            if status["healthy"]:
                self.send_response(200)
            else:
                self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_response(503)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default logging."""


def start_health_server(port: int, health_check_fn: Callable):
    """Start health check HTTP server in background thread."""
    HealthCheckHandler.health_check_fn = health_check_fn
    server = HTTPServer(("", port), HealthCheckHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server
