"""Tests for dependency injection container."""

import pytest

from core.container import (
    Container,
    ContainerConfig,
    create_paper_trading_engine,
    get_default_container,
    reset_default_container,
    set_default_container,
)
from engines.flowroute.core.engine import FlowRouteEngine


class TestContainerConfig:
    """Tests for ContainerConfig."""

    def test_default_config(self):
        """Default config uses paper trading."""
        config = ContainerConfig()
        assert config.broker_type == "paper"
        assert config.paper_slippage_bps == 5.0
        assert config.enable_kill_switch is True
        assert config.enable_persistence is False

    def test_custom_config(self):
        """Custom config values are set correctly."""
        config = ContainerConfig(
            broker_type="alpaca",
            paper_slippage_bps=10.0,
            enable_kill_switch=False,
        )
        assert config.broker_type == "alpaca"
        assert config.paper_slippage_bps == 10.0
        assert config.enable_kill_switch is False


class TestContainer:
    """Tests for Container."""

    def test_create_paper_broker(self):
        """Container creates paper broker adapter."""
        config = ContainerConfig(broker_type="paper")
        container = Container(config)

        adapter = container.get_broker_adapter()

        assert adapter.__class__.__name__ == "PaperBrokerAdapter"

    def test_broker_singleton(self):
        """Container returns same broker instance."""
        container = Container()

        adapter1 = container.get_broker_adapter()
        adapter2 = container.get_broker_adapter()

        assert adapter1 is adapter2

    def test_kill_switch_disabled(self):
        """Kill switch returns None when disabled."""
        config = ContainerConfig(enable_kill_switch=False)
        container = Container(config)

        switch = container.get_kill_switch()

        assert switch is None

    def test_kill_switch_enabled(self):
        """Kill switch is created when enabled."""
        config = ContainerConfig(enable_kill_switch=True)
        container = Container(config)

        switch = container.get_kill_switch()

        assert switch is not None

    def test_flowroute_engine_creation(self):
        """Container creates FlowRouteEngine with dependencies."""
        config = ContainerConfig(
            broker_type="paper",
            enable_kill_switch=True,
        )
        container = Container(config)

        engine = container.get_flowroute_engine()

        assert isinstance(engine, FlowRouteEngine)
        assert engine._broker is not None

    def test_flowroute_engine_singleton(self):
        """Container returns same engine instance."""
        container = Container()

        engine1 = container.get_flowroute_engine()
        engine2 = container.get_flowroute_engine()

        assert engine1 is engine2

    def test_reset_clears_instances(self):
        """Reset clears cached instances."""
        container = Container()
        engine1 = container.get_flowroute_engine()

        container.reset()

        engine2 = container.get_flowroute_engine()
        assert engine1 is not engine2

    def test_unknown_broker_type_raises(self):
        """Unknown broker type raises ValueError."""
        config = ContainerConfig(broker_type="unknown")
        container = Container(config)

        with pytest.raises(ValueError, match="Unknown broker type"):
            container.get_broker_adapter()


class TestFactoryFunctions:
    """Tests for factory convenience functions."""

    def test_create_paper_trading_engine(self):
        """Factory creates paper trading engine."""
        engine = create_paper_trading_engine(
            initial_cash=50000.0,
            slippage_bps=10.0,
        )

        assert isinstance(engine, FlowRouteEngine)
        assert engine._broker.__class__.__name__ == "PaperBrokerAdapter"


class TestDefaultContainer:
    """Tests for default container management."""

    def test_get_default_container(self):
        """Gets or creates default container."""
        reset_default_container()

        container = get_default_container()

        assert container is not None
        assert isinstance(container, Container)

    def test_set_default_container(self):
        """Sets custom default container."""
        reset_default_container()
        custom = Container(ContainerConfig(paper_slippage_bps=99.0))

        set_default_container(custom)
        retrieved = get_default_container()

        assert retrieved is custom
        assert retrieved.config.paper_slippage_bps == 99.0

    def test_reset_default_container(self):
        """Resets default container."""
        _ = get_default_container()

        reset_default_container()

        new = get_default_container()
        assert new is not None
