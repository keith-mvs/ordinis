"""
Application bootstrap for Ordinis trading system.

Handles application startup, configuration loading, and component wiring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from runtime.config import Settings, get_settings, reset_settings
from runtime.logging import configure_logging

if TYPE_CHECKING:
    from core.container import Container

logger = logging.getLogger(__name__)


class ApplicationContext:
    """
    Application context holding runtime configuration and components.

    Provides a single entry point for accessing configured services.
    """

    def __init__(
        self,
        settings: Settings,
        container: Container | None = None,
    ) -> None:
        """
        Initialize application context.

        Args:
            settings: Application settings
            container: DI container (optional, created if not provided)
        """
        self.settings = settings
        self._container = container
        self._initialized = False

    @property
    def container(self) -> Container:
        """Get or create the DI container."""
        if self._container is None:
            from core.container import Container, ContainerConfig

            config = ContainerConfig(
                broker_type=self.settings.broker.provider,
                paper_initial_cash=100000.0,
                enable_kill_switch=True,
                enable_persistence=True,
                enable_alerting=self.settings.alerting.enabled,
                db_path=self.settings.database.path,
            )
            self._container = Container(config)
        return self._container

    def ensure_directories(self) -> None:
        """Create required artifact directories."""
        dirs = [
            self.settings.artifacts.runs_dir,
            self.settings.artifacts.reports_dir,
            self.settings.artifacts.logs_dir,
            self.settings.artifacts.cache_dir,
            self.settings.database.backup_dir,
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """Initialize application components."""
        if self._initialized:
            return

        logger.info(
            "Initializing Ordinis %s (%s)",
            self.settings.system.version,
            self.settings.system.environment,
        )

        self.ensure_directories()
        self._initialized = True

        logger.info("Application initialized successfully")

    def shutdown(self) -> None:
        """Shutdown application components."""
        if self._container is not None:
            self._container.reset()

        logger.info("Application shutdown complete")


class _AppContextHolder:
    """Internal holder for application context instance."""

    instance: ApplicationContext | None = None


def bootstrap(
    environment: str | None = None,
    log_level: str | None = None,
) -> ApplicationContext:
    """
    Bootstrap the application.

    Loads configuration, configures logging, and initializes components.

    Args:
        environment: Environment name (dev, test, prod). Defaults to
                     ORDINIS_ENVIRONMENT env var or 'dev'.
        log_level: Override log level from config.

    Returns:
        Initialized ApplicationContext.
    """
    # Reset cached settings if re-bootstrapping
    reset_settings()

    # Load settings
    settings = get_settings(environment)

    # Configure logging
    configure_logging(settings.logging, level=log_level)

    # Create context
    _AppContextHolder.instance = ApplicationContext(settings)
    _AppContextHolder.instance.initialize()

    return _AppContextHolder.instance


def get_app_context() -> ApplicationContext:
    """
    Get the current application context.

    Raises:
        RuntimeError: If application has not been bootstrapped.
    """
    if _AppContextHolder.instance is None:
        raise RuntimeError("Application not bootstrapped. Call bootstrap() first.")
    return _AppContextHolder.instance


def shutdown() -> None:
    """Shutdown the application."""
    if _AppContextHolder.instance is not None:
        _AppContextHolder.instance.shutdown()
        _AppContextHolder.instance = None
