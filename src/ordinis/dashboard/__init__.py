"""
Dashboard backend and logging infrastructure.
"""

from .trace_logger import TraceEvent, TraceLogger, TraceType, get_trace_logger

__all__ = ["TraceLogger", "TraceEvent", "TraceType", "get_trace_logger"]
