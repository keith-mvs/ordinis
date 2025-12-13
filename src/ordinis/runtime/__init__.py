"""
Runtime layer for Ordinis trading system.

Contains:
- config: Configuration management with Pydantic BaseSettings
- container: Dependency injection container (re-exported from core)
- bootstrap: Application startup and wiring
- logging: Centralized logging configuration
"""

from ordinis.runtime.bootstrap import ApplicationContext, bootstrap, get_app_context, shutdown
from ordinis.runtime.config import Settings, get_settings, reset_settings
from ordinis.runtime.logging import configure_logging, get_logger

__all__ = [
    # Bootstrap
    "ApplicationContext",
    "bootstrap",
    "get_app_context",
    "shutdown",
    # Config
    "Settings",
    "get_settings",
    "reset_settings",
    # Logging
    "configure_logging",
    "get_logger",
]
