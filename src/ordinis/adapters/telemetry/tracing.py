"""
OpenTelemetry Tracing Configuration for Ordinis.

This module sets up distributed tracing with OpenTelemetry for the Ordinis trading system.
It instruments the OpenAI SDK and exports traces to AI Toolkit's OTLP endpoint.
"""

import logging
import os
from typing import Any

from opentelemetry import _events, _logs, trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._events import EventLoggerProvider
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_logger = logging.getLogger(__name__)


class TracingConfig:
    """Configuration for OpenTelemetry tracing."""

    def __init__(
        self,
        service_name: str = "ordinis",
        otlp_endpoint: str = "http://localhost:4318",
        capture_message_content: bool = True,
        enabled: bool = True,
    ):
        """
        Initialize tracing configuration.

        Args:
            service_name: Name of the service for trace identification
            otlp_endpoint: OTLP collector endpoint (AI Toolkit default: http://localhost:4318)
            capture_message_content: Whether to capture LLM message content (prompts/completions)
            enabled: Whether tracing is enabled
        """
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.capture_message_content = capture_message_content
        self.enabled = enabled


def setup_tracing(config: TracingConfig | None = None) -> bool:
    """
    Set up OpenTelemetry tracing for the Ordinis application.

    This function:
    1. Configures OpenTelemetry with AI Toolkit's OTLP endpoint
    2. Instruments the OpenAI SDK for automatic trace generation
    3. Sets up logging and event providers for LLM message content capture

    Args:
        config: Tracing configuration. Uses defaults if None.

    Returns:
        True if tracing was successfully configured, False otherwise.
    """
    if config is None:
        config = TracingConfig()

    if not config.enabled:
        _logger.info("Tracing is disabled")
        return False

    try:
        # Enable message content capture for LLM operations
        if config.capture_message_content:
            os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"

        # Create resource with service name
        resource = Resource(attributes={"service.name": config.service_name})

        # Configure trace provider
        provider = TracerProvider(resource=resource)
        otlp_trace_exporter = OTLPSpanExporter(
            endpoint=f"{config.otlp_endpoint}/v1/traces",
        )
        trace_processor = BatchSpanProcessor(otlp_trace_exporter)
        provider.add_span_processor(trace_processor)
        trace.set_tracer_provider(provider)

        # Configure logging and events for LLM message content
        _logs.set_logger_provider(LoggerProvider(resource=resource))
        _logs.get_logger_provider().add_log_record_processor(
            BatchLogRecordProcessor(
                OTLPLogExporter(endpoint=f"{config.otlp_endpoint}/v1/logs")
            )
        )
        _events.set_event_logger_provider(EventLoggerProvider())

        # Instrument OpenAI SDK
        try:
            from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

            OpenAIInstrumentor().instrument()
            _logger.info("OpenAI SDK instrumentation enabled")
        except ImportError:
            _logger.warning(
                "opentelemetry-instrumentation-openai-v2 not installed. "
                "OpenAI SDK will not be automatically instrumented. "
                "Install with: pip install opentelemetry-instrumentation-openai-v2==2.1b0"
            )

        _logger.info(
            "OpenTelemetry tracing configured: service=%s, endpoint=%s",
            config.service_name,
            config.otlp_endpoint,
        )
        return True

    except Exception as e:
        _logger.error("Failed to setup tracing: %s", e)
        return False


def get_tracer(name: str) -> Any:
    """
    Get a tracer instance for manual instrumentation.

    Args:
        name: Name of the tracer (typically module name)

    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """
    Gracefully shutdown tracing.

    Flushes all pending spans and logs before shutdown.
    """
    try:
        provider = trace.get_tracer_provider()
        if hasattr(provider, "shutdown"):
            provider.shutdown()
        _logger.info("Tracing shutdown complete")
    except Exception as e:
        _logger.error("Error during tracing shutdown: %s", e)
