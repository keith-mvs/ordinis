"""Tests for dependency injection container."""

from unittest.mock import Mock, patch

import pytest

from ordinis.core.container import (
    Container,
    ContainerConfig,
    create_paper_trading_engine,
    get_default_container,
    reset_default_container,
    set_default_container,
)
from ordinis.engines.flowroute.core.engine import FlowRouteEngine


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

    @pytest.mark.skip(reason="Alpaca adapter requires optional dependencies")
    def test_create_alpaca_broker(self):
        """Container creates Alpaca broker adapter."""
        config = ContainerConfig(
            broker_type="alpaca",
            alpaca_api_key="test_key",
            alpaca_api_secret="test_secret",  # noqa: S106
            alpaca_paper_trading=True,
        )
        container = Container(config)

        # Would test: adapter = container.get_broker_adapter()
        # Skipped due to missing alpaca-py dependency

    @pytest.mark.skip(reason="Alpaca adapter requires optional dependencies")
    def test_alpaca_broker_singleton(self):
        """Container returns same Alpaca broker instance."""
        config = ContainerConfig(
            broker_type="alpaca",
            alpaca_api_key="test_key",
            alpaca_api_secret="test_secret",  # noqa: S106
        )
        container = Container(config)

        # Would test singleton behavior
        # Skipped due to missing alpaca-py dependency

    @pytest.mark.skip(reason="Alpaca adapter requires optional dependencies")
    def test_alpaca_broker_paper_false(self):
        """Container creates Alpaca broker with paper=False."""
        config = ContainerConfig(
            broker_type="alpaca",
            alpaca_api_key="test_key",
            alpaca_api_secret="test_secret",  # noqa: S106
            alpaca_paper_trading=False,
        )
        container = Container(config)

        # Would test paper=False configuration
        # Skipped due to missing alpaca-py dependency

    def test_kill_switch_singleton(self):
        """Container returns same kill switch instance."""
        config = ContainerConfig(enable_kill_switch=True)
        container = Container(config)

        switch1 = container.get_kill_switch()
        switch2 = container.get_kill_switch()

        assert switch1 is switch2

    def test_order_repository_disabled(self):
        """Order repository returns None when persistence disabled."""
        config = ContainerConfig(enable_persistence=False)
        container = Container(config)

        repo = container.get_order_repository()

        assert repo is None

    def test_order_repository_no_db_path(self):
        """Order repository returns None when db_path not configured."""
        config = ContainerConfig(enable_persistence=True, db_path=None)
        container = Container(config)

        repo = container.get_order_repository()

        assert repo is None

    @patch("ordinis.adapters.storage.repositories.order.OrderRepository")
    def test_order_repository_enabled(self, mock_repo_cls):
        """Order repository is created when persistence enabled."""
        mock_repo = Mock()
        mock_repo_cls.return_value = mock_repo

        config = ContainerConfig(
            enable_persistence=True,
            db_path=":memory:",
        )
        container = Container(config)

        repo = container.get_order_repository()

        assert repo is mock_repo
        mock_repo_cls.assert_called_once()

    @patch("ordinis.adapters.storage.repositories.order.OrderRepository")
    def test_order_repository_singleton(self, mock_repo_cls):
        """Container returns same order repository instance."""
        mock_repo = Mock()
        mock_repo_cls.return_value = mock_repo

        config = ContainerConfig(
            enable_persistence=True,
            db_path=":memory:",
        )
        container = Container(config)

        repo1 = container.get_order_repository()
        repo2 = container.get_order_repository()

        assert repo1 is repo2
        mock_repo_cls.assert_called_once()

    def test_alert_manager_disabled(self):
        """Alert manager returns None when alerting disabled."""
        config = ContainerConfig(enable_alerting=False)
        container = Container(config)

        manager = container.get_alert_manager()

        assert manager is None

    def test_alert_manager_enabled(self):
        """Alert manager is created when alerting enabled."""
        config = ContainerConfig(
            enable_alerting=True,
            alert_config={},
        )
        container = Container(config)

        manager = container.get_alert_manager()

        assert manager is not None
        assert manager.__class__.__name__ == "AlertManager"

    def test_alert_manager_singleton(self):
        """Container returns same alert manager instance."""
        config = ContainerConfig(
            enable_alerting=True,
            alert_config={},
        )
        container = Container(config)

        manager1 = container.get_alert_manager()
        manager2 = container.get_alert_manager()

        assert manager1 is manager2

    def test_alert_manager_with_config(self):
        """Alert manager is created with custom config."""
        alert_config = {"rate_limit_seconds": 30.0, "max_history": 500}
        config = ContainerConfig(
            enable_alerting=True,
            alert_config=alert_config,
        )
        container = Container(config)

        manager = container.get_alert_manager()

        assert manager is not None

    @patch("ordinis.adapters.storage.repositories.order.OrderRepository")
    def test_flowroute_engine_with_all_dependencies(self, mock_repo_cls):
        """FlowRouteEngine created with all optional dependencies."""
        mock_repo = Mock()
        mock_repo_cls.return_value = mock_repo

        config = ContainerConfig(
            broker_type="paper",
            enable_kill_switch=True,
            enable_persistence=True,
            db_path=":memory:",
            enable_alerting=True,
            alert_config={},
        )
        container = Container(config)

        engine = container.get_flowroute_engine()

        assert isinstance(engine, FlowRouteEngine)
        assert engine._broker is not None
        assert engine._kill_switch is not None
        assert engine._order_repo is not None
        assert engine._alert_manager is not None

    def test_flowroute_engine_minimal_dependencies(self):
        """FlowRouteEngine created with minimal dependencies."""
        config = ContainerConfig(
            broker_type="paper",
            enable_kill_switch=False,
            enable_persistence=False,
            enable_alerting=False,
        )
        container = Container(config)

        engine = container.get_flowroute_engine()

        assert isinstance(engine, FlowRouteEngine)
        assert engine._broker is not None
        assert engine._kill_switch is None
        assert engine._order_repo is None
        assert engine._alert_manager is None

    def test_paper_broker_custom_params(self):
        """Paper broker is created with custom parameters."""
        config = ContainerConfig(
            broker_type="paper",
            paper_slippage_bps=10.0,
            paper_commission_per_share=0.01,
            paper_fill_delay_ms=200.0,
            paper_initial_cash=50000.0,
        )
        container = Container(config)

        adapter = container.get_broker_adapter()

        assert adapter.__class__.__name__ == "PaperBrokerAdapter"
        # Verify params were passed (adapter stores them without underscore)
        assert adapter.slippage_bps == 10.0
        assert adapter.commission_per_share == 0.01

    def test_container_default_config(self):
        """Container initializes with default config when None provided."""
        container = Container(config=None)

        assert container.config is not None
        assert container.config.broker_type == "paper"

    @patch("ordinis.adapters.storage.repositories.order.OrderRepository")
    def test_reset_with_multiple_instances(self, mock_repo_cls):
        """Reset clears all cached instances."""
        # Create new mock for each call
        mock_repo_cls.side_effect = [Mock(), Mock()]

        config = ContainerConfig(
            enable_kill_switch=True,
            enable_persistence=True,
            db_path=":memory:",
            enable_alerting=True,
            alert_config={},
        )
        container = Container(config)

        # Create all instances
        broker = container.get_broker_adapter()
        kill_switch = container.get_kill_switch()
        repo = container.get_order_repository()
        alert_mgr = container.get_alert_manager()
        engine = container.get_flowroute_engine()

        # Reset
        container.reset()

        # Verify new instances are created
        assert container.get_broker_adapter() is not broker
        assert container.get_kill_switch() is not kill_switch
        assert container.get_order_repository() is not repo
        assert container.get_alert_manager() is not alert_mgr
        assert container.get_flowroute_engine() is not engine


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

    def test_create_paper_trading_engine_defaults(self):
        """Factory creates paper trading engine with defaults."""
        engine = create_paper_trading_engine()

        assert isinstance(engine, FlowRouteEngine)
        assert engine._broker is not None
        assert engine._kill_switch is not None
        assert engine._order_repo is None
        assert engine._alert_manager is None

    def test_create_paper_trading_engine_custom_params(self):
        """Factory creates paper trading engine with custom parameters."""
        engine = create_paper_trading_engine(
            initial_cash=25000.0,
            slippage_bps=2.5,
            enable_kill_switch=False,
        )

        assert isinstance(engine, FlowRouteEngine)
        assert engine._broker.__class__.__name__ == "PaperBrokerAdapter"
        assert engine._kill_switch is None

    @pytest.mark.skip(reason="Alpaca adapter requires optional dependencies")
    def test_create_alpaca_engine_defaults(self):
        """Factory creates Alpaca engine with defaults."""
        # Would test: create_alpaca_engine(api_key="test_key", api_secret="test_secret")
        # Skipped due to missing alpaca-py dependency

    @pytest.mark.skip(reason="Alpaca adapter requires optional dependencies")
    def test_create_alpaca_engine_with_persistence(self):
        """Factory creates Alpaca engine with persistence enabled."""
        # Would test: create_alpaca_engine with db_path
        # Skipped due to missing alpaca-py dependency

    @pytest.mark.skip(reason="Alpaca adapter requires optional dependencies")
    def test_create_alpaca_engine_paper_false(self):
        """Factory creates Alpaca engine with live trading."""
        # Would test: create_alpaca_engine with paper=False
        # Skipped due to missing alpaca-py dependency

    @pytest.mark.skip(reason="Alpaca adapter requires optional dependencies")
    def test_create_alpaca_engine_no_kill_switch(self):
        """Factory creates Alpaca engine without kill switch."""
        # Would test: create_alpaca_engine with enable_kill_switch=False
        # Skipped due to missing alpaca-py dependency

    @pytest.mark.skip(reason="Alpaca adapter requires optional dependencies")
    def test_create_alpaca_engine_none_credentials(self):
        """Factory creates Alpaca engine with None credentials."""
        # Would test: create_alpaca_engine with api_key=None, api_secret=None
        # Skipped due to missing alpaca-py dependency


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

    def test_get_default_container_singleton(self):
        """Default container returns same instance."""
        reset_default_container()

        container1 = get_default_container()
        container2 = get_default_container()

        assert container1 is container2

    def test_reset_default_container_calls_reset(self):
        """Reset default container calls reset on instance."""
        reset_default_container()
        container = get_default_container()
        engine = container.get_flowroute_engine()

        reset_default_container()

        # Getting default container again should create new instance
        new_container = get_default_container()
        assert new_container is not container

    def test_reset_default_container_when_none(self):
        """Reset default container when None does not raise."""
        reset_default_container()
        reset_default_container()  # Should not raise

        container = get_default_container()
        assert container is not None
