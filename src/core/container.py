"""
Dependency injection container for Ordinis.

Provides factory functions and a composition root for wiring up
the application with proper dependency injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alerting import AlertManager
    from core.protocols import BrokerAdapter
    from engines.flowroute.core.engine import FlowRouteEngine
    from persistence.repositories.order import OrderRepository
    from safety.kill_switch import KillSwitch

logger = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    """Configuration for dependency container."""

    broker_type: str = "paper"  # "paper" or "alpaca"
    paper_slippage_bps: float = 5.0
    paper_commission_per_share: float = 0.005
    paper_fill_delay_ms: float = 100.0
    paper_initial_cash: float = 100000.0

    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    alpaca_paper_trading: bool = True

    enable_kill_switch: bool = True
    enable_persistence: bool = False
    enable_alerting: bool = False

    db_path: str | None = None
    alert_config: dict[str, Any] = field(default_factory=dict)


class Container:
    """
    Dependency injection container.

    Central composition root for creating and wiring up application
    components with their dependencies.
    """

    def __init__(self, config: ContainerConfig | None = None) -> None:
        """
        Initialize container.

        Args:
            config: Container configuration. Defaults to paper trading setup.
        """
        self.config = config or ContainerConfig()
        self._instances: dict[str, Any] = {}

    def get_broker_adapter(self) -> BrokerAdapter:
        """
        Get or create broker adapter.

        Returns:
            BrokerAdapter instance based on configuration.
        """
        if "broker_adapter" in self._instances:
            return self._instances["broker_adapter"]

        adapter: BrokerAdapter

        if self.config.broker_type == "paper":
            from engines.flowroute.adapters.paper import PaperBrokerAdapter

            adapter = PaperBrokerAdapter(
                slippage_bps=self.config.paper_slippage_bps,
                commission_per_share=self.config.paper_commission_per_share,
                fill_delay_ms=self.config.paper_fill_delay_ms,
            )
            logger.info("Created PaperBrokerAdapter")

        elif self.config.broker_type == "alpaca":
            from engines.flowroute.adapters.alpaca import AlpacaBrokerAdapter

            adapter = AlpacaBrokerAdapter(
                api_key=self.config.alpaca_api_key,
                api_secret=self.config.alpaca_api_secret,
                paper=self.config.alpaca_paper_trading,
            )
            logger.info(f"Created AlpacaBrokerAdapter (paper={self.config.alpaca_paper_trading})")

        else:
            raise ValueError(f"Unknown broker type: {self.config.broker_type}")

        self._instances["broker_adapter"] = adapter
        return adapter

    def get_kill_switch(self) -> KillSwitch | None:
        """
        Get or create kill switch.

        Returns:
            KillSwitch instance if enabled, None otherwise.
        """
        if not self.config.enable_kill_switch:
            return None

        if "kill_switch" in self._instances:
            return self._instances["kill_switch"]

        from safety.kill_switch import KillSwitch

        switch = KillSwitch()
        self._instances["kill_switch"] = switch
        logger.info("Created KillSwitch")
        return switch

    def get_order_repository(self) -> OrderRepository | None:
        """
        Get or create order repository.

        Returns:
            OrderRepository instance if persistence enabled, None otherwise.
        """
        if not self.config.enable_persistence:
            return None

        if "order_repository" in self._instances:
            return self._instances["order_repository"]

        if not self.config.db_path:
            logger.warning("Persistence enabled but no db_path configured")
            return None

        from persistence.repositories.order import OrderRepository

        repo = OrderRepository(db_path=self.config.db_path)
        self._instances["order_repository"] = repo
        logger.info(f"Created OrderRepository (db={self.config.db_path})")
        return repo

    def get_alert_manager(self) -> AlertManager | None:
        """
        Get or create alert manager.

        Returns:
            AlertManager instance if alerting enabled, None otherwise.
        """
        if not self.config.enable_alerting:
            return None

        if "alert_manager" in self._instances:
            return self._instances["alert_manager"]

        from alerting import AlertManager

        manager = AlertManager(**self.config.alert_config)
        self._instances["alert_manager"] = manager
        logger.info("Created AlertManager")
        return manager

    def get_flowroute_engine(self) -> FlowRouteEngine:
        """
        Get or create FlowRoute execution engine.

        Returns:
            FlowRouteEngine with all dependencies wired.
        """
        if "flowroute_engine" in self._instances:
            return self._instances["flowroute_engine"]

        from engines.flowroute.core.engine import FlowRouteEngine

        engine = FlowRouteEngine(
            broker_adapter=self.get_broker_adapter(),
            kill_switch=self.get_kill_switch(),
            order_repository=self.get_order_repository(),
            alert_manager=self.get_alert_manager(),
        )

        self._instances["flowroute_engine"] = engine
        logger.info("Created FlowRouteEngine")
        return engine

    def reset(self) -> None:
        """Clear all cached instances."""
        self._instances.clear()
        logger.info("Container reset - all instances cleared")


# Convenience factory functions for simple use cases


def create_paper_trading_engine(
    initial_cash: float = 100000.0,
    slippage_bps: float = 5.0,
    enable_kill_switch: bool = True,
) -> FlowRouteEngine:
    """
    Create FlowRouteEngine configured for paper trading.

    Args:
        initial_cash: Starting capital for paper trading
        slippage_bps: Simulated slippage in basis points
        enable_kill_switch: Whether to enable kill switch

    Returns:
        Configured FlowRouteEngine
    """
    config = ContainerConfig(
        broker_type="paper",
        paper_slippage_bps=slippage_bps,
        paper_initial_cash=initial_cash,
        enable_kill_switch=enable_kill_switch,
        enable_persistence=False,
        enable_alerting=False,
    )
    container = Container(config)
    return container.get_flowroute_engine()


def create_alpaca_engine(
    api_key: str | None = None,
    api_secret: str | None = None,
    paper: bool = True,
    enable_kill_switch: bool = True,
    db_path: str | None = None,
) -> FlowRouteEngine:
    """
    Create FlowRouteEngine configured for Alpaca trading.

    Args:
        api_key: Alpaca API key (defaults to env var)
        api_secret: Alpaca API secret (defaults to env var)
        paper: Use Alpaca paper trading
        enable_kill_switch: Whether to enable kill switch
        db_path: Database path for order persistence

    Returns:
        Configured FlowRouteEngine
    """
    config = ContainerConfig(
        broker_type="alpaca",
        alpaca_api_key=api_key,
        alpaca_api_secret=api_secret,
        alpaca_paper_trading=paper,
        enable_kill_switch=enable_kill_switch,
        enable_persistence=db_path is not None,
        db_path=db_path,
        enable_alerting=False,
    )
    container = Container(config)
    return container.get_flowroute_engine()


class _DefaultContainerHolder:
    """Internal holder for default container instance."""

    instance: Container | None = None


def get_default_container() -> Container:
    """
    Get default container instance.

    Creates with default paper trading config if not exists.
    """
    if _DefaultContainerHolder.instance is None:
        _DefaultContainerHolder.instance = Container()
    return _DefaultContainerHolder.instance


def set_default_container(container: Container) -> None:
    """Set the default container instance."""
    _DefaultContainerHolder.instance = container


def reset_default_container() -> None:
    """Reset the default container."""
    if _DefaultContainerHolder.instance is not None:
        _DefaultContainerHolder.instance.reset()
    _DefaultContainerHolder.instance = None
