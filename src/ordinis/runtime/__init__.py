"""
Runtime layer for Ordinis trading system.

Contains:
- config: Configuration management with Pydantic BaseSettings
- container: Dependency injection container (re-exported from core)
- bootstrap: Application startup and wiring
- logging: Centralized logging configuration
"""

from ordinis.runtime.bootstrap import (
    ApplicationContext,
    bootstrap,
    get_app_context,
    initialize,
    load_config,
    shutdown,
)
from ordinis.runtime.config import Settings, get_settings, reset_settings
from ordinis.runtime.logging import configure_logging, get_logger

__all__ = [
    # Bootstrap
    "ApplicationContext",
    # Config
    "Settings",
    "bootstrap",
    # Logging
    "configure_logging",
    "get_app_context",
    "get_logger",
    "get_settings",
    "initialize",
    "load_config",
    "reset_settings",
    "shutdown",
]
