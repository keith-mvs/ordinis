"""
Application bootstrap for Ordinis trading system.

Handles application startup, configuration loading, and component wiring.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ordinis.runtime.config import Settings, _deep_merge, get_settings, reset_settings
from ordinis.runtime.logging import configure_logging

if TYPE_CHECKING:
    from ordinis.core.container import Container

logger = logging.getLogger(__name__)


def load_config(path: str) -> Settings:
    """
    Load configuration from a specific file path.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Settings object populated from the file.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    # Load default config as base
    default_path = Path("configs/default.yaml")
    base_config = {}
    if default_path.exists():
        with default_path.open() as f:
            base_config = yaml.safe_load(f) or {}

    # Load override config
    with config_path.open() as f:
        override_config = yaml.safe_load(f) or {}

    # Merge configs
    merged_config = _deep_merge(base_config, override_config)

    # Create Settings object
    return Settings.model_validate(merged_config)


def initialize(settings: Settings) -> Any:
    """
    Initialize the application container with the given settings.

    Args:
        settings: Application settings.

    Returns:
        Container instance with all engines wired up.
    """
    from ordinis.core.container import Container, ContainerConfig

    # Configure logging
    configure_logging(settings.logging)

    # Create container config
    # Map settings to ContainerConfig fields
    # Note: This mapping is best-effort based on available fields
    container_config = ContainerConfig(
        broker_type=settings.broker.provider if hasattr(settings, "broker") else "paper",
        enable_kill_switch=True,  # Default to true, can be overridden
        enable_persistence=True,
        enable_alerting=settings.alerting.enabled if hasattr(settings, "alerting") else False,
        db_path=settings.database.path if hasattr(settings, "database") else None,
    )

    # Handle dev overrides if present in settings (e.g. from dev.yaml)
    # Since Settings is strict, we might not see 'execution' field if it wasn't in Settings model
    # But we can check the raw config if needed, or rely on what mapped to 'broker'

    container = Container(container_config)

    # --- Wire up Engines (Monkey-patching for Dev Playground) ---

    # 1. Helix (LLM Provider)
    try:
        from ordinis.ai.helix.config import HelixConfig
        from ordinis.ai.helix.engine import Helix

        # Create config from settings if possible, else default
        helix_config = HelixConfig()
        # TODO: Populate helix_config from settings.helix if available

        container.helix = Helix(helix_config)
    except ImportError:
        logger.warning("Could not initialize Helix")

    # 2. Synapse (RAG)
    try:
        from ordinis.ai.synapse.config import SynapseConfig
        from ordinis.ai.synapse.engine import Synapse

        if hasattr(container, "helix"):
            synapse_config = SynapseConfig()
            container.synapse = Synapse(container.helix, synapse_config)
    except ImportError:
        logger.warning("Could not initialize Synapse")

    # 3. Cortex (Reasoning)
    try:
        from ordinis.engines.cortex.core.engine import CortexEngine

        if hasattr(container, "helix"):
            container.cortex = CortexEngine(container.helix)
    except ImportError:
        logger.warning("Could not initialize Cortex")

    # 4. SignalCore (Signals)
    try:
        from ordinis.engines.signalcore.core.config import SignalCoreEngineConfig
        from ordinis.engines.signalcore.core.engine import SignalCoreEngine

        container.signal_engine = SignalCoreEngine(SignalCoreEngineConfig())
    except ImportError:
        logger.warning("Could not initialize SignalCore")

    # 5. RiskGuard (Risk)
    try:
        from ordinis.engines.riskguard.core.engine import RiskGuardEngine

        container.risk_engine = RiskGuardEngine()
    except ImportError:
        logger.warning("Could not initialize RiskGuard")

    # 6. Portfolio (Positions)
    try:
        from ordinis.engines.portfolio.core.config import PortfolioEngineConfig
        from ordinis.engines.portfolio.core.engine import PortfolioEngine

        container.portfolio_engine = PortfolioEngine(PortfolioEngineConfig())
    except ImportError:
        logger.warning("Could not initialize PortfolioEngine")

    # 7. Analytics (ProofBench)
    try:
        from ordinis.engines.proofbench.core.config import ProofBenchEngineConfig
        from ordinis.engines.proofbench.core.engine import ProofBenchEngine

        container.analytics_engine = ProofBenchEngine(ProofBenchEngineConfig())
    except ImportError:
        logger.warning("Could not initialize AnalyticsEngine")

    # 8. Execution (FlowRoute)
    container.execution_engine = container.get_flowroute_engine()

    # 9. StreamingBus (Mock)
    # For dev, we use a mock/in-memory bus
    class InMemoryBus:
        def __init__(self):
            self.subscribers = {}

        async def start(self):
            logger.info("InMemoryBus started")

        async def publish(self, topic, event):
            logger.debug(f"Publishing to {topic}: {event}")
            if topic in self.subscribers:
                for handler in self.subscribers[topic]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Error in handler for {topic}: {e}")

        async def subscribe(self, topic, handler):
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(handler)
            logger.info(f"Subscribed to {topic}")

    container.bus = InMemoryBus()

    # 10. Orchestration
    try:
        from ordinis.engines.orchestration.core.engine import (
            OrchestrationEngine,
            OrchestrationEngineConfig,
        )

        orch_config = OrchestrationEngineConfig()
        container.orchestration = OrchestrationEngine(orch_config)

        # Register engines with orchestration
        if hasattr(container.orchestration, "register_engines"):
            # This method might not exist or might be named differently
            # Based on docstring: engine.register_engines(signal_engine, risk_engine, execution_engine)
            # But looking at code, it has _engines = PipelineEngines()
            # We can manually set them
            container.orchestration._engines.signal_engine = getattr(
                container, "signal_engine", None
            )
            container.orchestration._engines.risk_engine = getattr(container, "risk_engine", None)
            container.orchestration._engines.execution_engine = getattr(
                container, "execution_engine", None
            )
            container.orchestration._engines.analytics_engine = getattr(
                container, "analytics_engine", None
            )
            container.orchestration._engines.portfolio_engine = getattr(
                container, "portfolio_engine", None
            )
            # container.orchestration._engines.data_source = container.bus # Protocol mismatch likely

    except ImportError:
        logger.warning("Could not initialize OrchestrationEngine")

    return container


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
            from ordinis.core.container import Container, ContainerConfig

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
