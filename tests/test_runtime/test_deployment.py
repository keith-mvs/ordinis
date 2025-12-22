"""Tests for Production Deployment and System Integration."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ordinis.runtime.deployment import (
    ComponentHealth,
    ComponentStatus,
    DeploymentConfig,
    DeploymentEnvironment,
    GracefulShutdownManager,
    HealthMonitor,
    IntegrationTestRunner,
    SystemHealth,
    SystemIntegrator,
    create_production_system,
)


class TestComponentStatus:
    """Tests for ComponentStatus enum."""

    def test_status_values_exist(self):
        """Test all status values exist."""
        assert ComponentStatus.UNKNOWN is not None
        assert ComponentStatus.STARTING is not None
        assert ComponentStatus.HEALTHY is not None
        assert ComponentStatus.DEGRADED is not None
        assert ComponentStatus.UNHEALTHY is not None
        assert ComponentStatus.STOPPED is not None


class TestDeploymentEnvironment:
    """Tests for DeploymentEnvironment enum."""

    def test_environment_values_exist(self):
        """Test all environment values exist."""
        assert DeploymentEnvironment.DEVELOPMENT is not None
        assert DeploymentEnvironment.STAGING is not None
        assert DeploymentEnvironment.PAPER is not None
        assert DeploymentEnvironment.PRODUCTION is not None


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_create_health(self):
        """Test creating component health."""
        health = ComponentHealth(
            component_name="MarketData",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.utcnow(),
            latency_ms=50.0,
        )

        assert health.component_name == "MarketData"
        assert health.status == ComponentStatus.HEALTHY
        assert health.latency_ms == 50.0

    def test_is_healthy_when_healthy(self):
        """Test is_healthy when status is HEALTHY."""
        health = ComponentHealth(
            component_name="Test",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.utcnow(),
        )

        assert health.is_healthy is True

    def test_is_healthy_when_starting(self):
        """Test is_healthy when status is STARTING."""
        health = ComponentHealth(
            component_name="Test",
            status=ComponentStatus.STARTING,
            last_check=datetime.utcnow(),
        )

        assert health.is_healthy is True

    def test_is_healthy_when_degraded(self):
        """Test is_healthy when status is DEGRADED."""
        health = ComponentHealth(
            component_name="Test",
            status=ComponentStatus.DEGRADED,
            last_check=datetime.utcnow(),
        )

        assert health.is_healthy is False

    def test_is_operational_when_healthy(self):
        """Test is_operational when HEALTHY."""
        health = ComponentHealth(
            component_name="Test",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.utcnow(),
        )

        assert health.is_operational is True

    def test_is_operational_when_degraded(self):
        """Test is_operational when DEGRADED."""
        health = ComponentHealth(
            component_name="Test",
            status=ComponentStatus.DEGRADED,
            last_check=datetime.utcnow(),
        )

        assert health.is_operational is True

    def test_is_operational_when_unhealthy(self):
        """Test is_operational when UNHEALTHY."""
        health = ComponentHealth(
            component_name="Test",
            status=ComponentStatus.UNHEALTHY,
            last_check=datetime.utcnow(),
        )

        assert health.is_operational is False

    def test_with_error_message(self):
        """Test health with error message."""
        health = ComponentHealth(
            component_name="Test",
            status=ComponentStatus.UNHEALTHY,
            last_check=datetime.utcnow(),
            error_message="Connection failed",
        )

        assert health.error_message == "Connection failed"

    def test_with_metrics(self):
        """Test health with metrics."""
        health = ComponentHealth(
            component_name="Test",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.utcnow(),
            metrics={"requests_per_second": 100.0, "error_rate": 0.01},
        )

        assert health.metrics["requests_per_second"] == 100.0
        assert health.metrics["error_rate"] == 0.01

    def test_with_dependencies(self):
        """Test health with dependencies."""
        health = ComponentHealth(
            component_name="OrderEngine",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.utcnow(),
            dependencies=["MarketData", "BrokerAdapter"],
        )

        assert "MarketData" in health.dependencies
        assert "BrokerAdapter" in health.dependencies


class TestSystemHealth:
    """Tests for SystemHealth dataclass."""

    def test_create_system_health(self):
        """Test creating system health."""
        components = {
            "MarketData": ComponentHealth(
                component_name="MarketData",
                status=ComponentStatus.HEALTHY,
                last_check=datetime.utcnow(),
            ),
            "Broker": ComponentHealth(
                component_name="Broker",
                status=ComponentStatus.HEALTHY,
                last_check=datetime.utcnow(),
            ),
        }

        health = SystemHealth(
            timestamp=datetime.utcnow(),
            environment=DeploymentEnvironment.PAPER,
            overall_status=ComponentStatus.HEALTHY,
            components=components,
            uptime_seconds=3600.0,
            warnings=[],
            errors=[],
        )

        assert health.environment == DeploymentEnvironment.PAPER
        assert health.overall_status == ComponentStatus.HEALTHY
        assert len(health.components) == 2

    def test_healthy_count(self):
        """Test healthy_count property."""
        components = {
            "Healthy1": ComponentHealth(
                component_name="Healthy1",
                status=ComponentStatus.HEALTHY,
                last_check=datetime.utcnow(),
            ),
            "Healthy2": ComponentHealth(
                component_name="Healthy2",
                status=ComponentStatus.STARTING,
                last_check=datetime.utcnow(),
            ),
            "Unhealthy": ComponentHealth(
                component_name="Unhealthy",
                status=ComponentStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
            ),
        }

        health = SystemHealth(
            timestamp=datetime.utcnow(),
            environment=DeploymentEnvironment.PAPER,
            overall_status=ComponentStatus.DEGRADED,
            components=components,
            uptime_seconds=1000.0,
            warnings=["Some warning"],
            errors=[],
        )

        assert health.healthy_count == 2
        assert health.total_count == 3

    def test_with_warnings_and_errors(self):
        """Test system health with warnings and errors."""
        health = SystemHealth(
            timestamp=datetime.utcnow(),
            environment=DeploymentEnvironment.PRODUCTION,
            overall_status=ComponentStatus.DEGRADED,
            components={},
            uptime_seconds=7200.0,
            warnings=["High latency detected", "Memory usage high"],
            errors=["Component X failed"],
        )

        assert len(health.warnings) == 2
        assert len(health.errors) == 1


class TestDeploymentConfig:
    """Tests for DeploymentConfig dataclass."""

    def test_create_development_config(self):
        """Test creating development config."""
        config = DeploymentConfig(
            environment=DeploymentEnvironment.DEVELOPMENT,
        )

        assert config.environment == DeploymentEnvironment.DEVELOPMENT
        assert config.enable_live_trading is False
        assert config.enable_paper_trading is True
        assert config.enable_backtesting is True

    def test_create_paper_config(self):
        """Test creating paper trading config."""
        config = DeploymentConfig(
            environment=DeploymentEnvironment.PAPER,
            enable_paper_trading=True,
            enable_websockets=True,
        )

        assert config.environment == DeploymentEnvironment.PAPER
        assert config.enable_paper_trading is True

    def test_create_production_config(self):
        """Test creating production config."""
        config = DeploymentConfig(
            environment=DeploymentEnvironment.PRODUCTION,
            enable_live_trading=True,
            enable_paper_trading=False,
        )

        assert config.environment == DeploymentEnvironment.PRODUCTION
        assert config.enable_live_trading is True
        assert config.enable_paper_trading is False

    def test_from_environment_development(self):
        """Test creating config from environment - development."""
        with patch.dict("os.environ", {}, clear=True):
            config = DeploymentConfig.from_environment()

        assert config.environment == DeploymentEnvironment.DEVELOPMENT

    def test_from_environment_paper(self):
        """Test creating config from environment - paper."""
        with patch.dict("os.environ", {"ORDINIS_ENVIRONMENT": "PAPER"}):
            config = DeploymentConfig.from_environment()

        assert config.environment == DeploymentEnvironment.PAPER

    def test_from_environment_live_trading(self):
        """Test from_environment with live trading enabled."""
        with patch.dict("os.environ", {"ORDINIS_LIVE_TRADING": "true"}):
            config = DeploymentConfig.from_environment()

        assert config.enable_live_trading is True

    def test_from_environment_invalid_env(self):
        """Test from_environment with invalid environment falls back."""
        with patch.dict("os.environ", {"ORDINIS_ENVIRONMENT": "INVALID"}):
            config = DeploymentConfig.from_environment()

        # Should fall back to DEVELOPMENT
        assert config.environment == DeploymentEnvironment.DEVELOPMENT


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    @pytest.fixture
    def config(self):
        """Create deployment config."""
        return DeploymentConfig(
            environment=DeploymentEnvironment.PAPER,
            health_check_interval_seconds=1,
        )

    @pytest.fixture
    def monitor(self, config):
        """Create health monitor."""
        return HealthMonitor(config)

    def test_init(self, monitor, config):
        """Test initialization."""
        assert monitor.config == config
        assert monitor._running is False

    def test_register_component(self, monitor):
        """Test registering a component."""
        def health_check():
            return ComponentHealth(
                component_name="test",
                status=ComponentStatus.HEALTHY,
                last_check=datetime.utcnow(),
            )

        monitor.register_component("test", health_check)

        assert "test" in monitor._components
        assert "test" in monitor._health_history
        assert monitor._failure_counts["test"] == 0

    def test_get_system_health_empty(self, monitor):
        """Test getting system health with no components."""
        health = monitor.get_system_health()

        assert health.environment == DeploymentEnvironment.PAPER
        assert len(health.components) == 0

    def test_get_system_health_with_components(self, monitor):
        """Test getting system health with components."""
        def healthy_check():
            return ComponentHealth(
                component_name="test",
                status=ComponentStatus.HEALTHY,
                last_check=datetime.utcnow(),
            )

        monitor.register_component("test", healthy_check)
        # Simulate health check
        monitor._health_history["test"].append(healthy_check())

        health = monitor.get_system_health()

        assert "test" in health.components
        assert health.components["test"].status == ComponentStatus.HEALTHY

    def test_on_unhealthy_callback(self, monitor):
        """Test registering unhealthy callback."""
        callback = MagicMock()
        monitor.on_unhealthy(callback)

        assert callback in monitor._on_unhealthy

    def test_on_recovery_callback(self, monitor):
        """Test registering recovery callback."""
        callback = MagicMock()
        monitor.on_recovery(callback)

        assert callback in monitor._on_recovery

    @pytest.mark.asyncio
    async def test_start_and_stop(self, monitor):
        """Test starting and stopping monitor."""
        await monitor.start()
        assert monitor._running is True

        await monitor.stop()
        assert monitor._running is False


class TestGracefulShutdownManager:
    """Tests for GracefulShutdownManager."""

    @pytest.fixture
    def config(self):
        """Create deployment config."""
        return DeploymentConfig(
            environment=DeploymentEnvironment.PAPER,
            shutdown_timeout_seconds=5,
        )

    @pytest.fixture
    def manager(self, config):
        """Create shutdown manager."""
        return GracefulShutdownManager(config)

    def test_init(self, manager, config):
        """Test initialization."""
        assert manager.config == config
        assert manager._is_shutting_down is False

    def test_register_handler(self, manager):
        """Test registering shutdown handler."""
        handler = MagicMock()
        manager.register_handler("test", handler, priority=50)

        assert len(manager._shutdown_handlers) == 1
        assert manager._shutdown_handlers[0] == (50, "test", handler)

    def test_register_handlers_sorted_by_priority(self, manager):
        """Test handlers are sorted by priority."""
        handler1 = MagicMock()
        handler2 = MagicMock()
        handler3 = MagicMock()

        manager.register_handler("high", handler1, priority=10)
        manager.register_handler("low", handler2, priority=90)
        manager.register_handler("mid", handler3, priority=50)

        priorities = [h[0] for h in manager._shutdown_handlers]
        assert priorities == [10, 50, 90]

    @pytest.mark.asyncio
    async def test_shutdown_calls_handlers(self, manager):
        """Test shutdown calls all handlers."""
        handler = MagicMock()
        manager.register_handler("test", handler)

        await manager.shutdown(reason="test")

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_async_handlers(self, manager):
        """Test shutdown with async handlers."""
        handler = AsyncMock()
        manager.register_handler("test", handler)

        await manager.shutdown(reason="test")

        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_sets_event(self, manager):
        """Test shutdown sets event."""
        await manager.shutdown(reason="test")

        assert manager._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_double_shutdown_ignored(self, manager):
        """Test double shutdown is ignored."""
        await manager.shutdown(reason="test1")
        # Second shutdown should be ignored
        await manager.shutdown(reason="test2")

        # Should still be shut down
        assert manager._is_shutting_down is True


class TestSystemIntegrator:
    """Tests for SystemIntegrator."""

    @pytest.fixture
    def config(self):
        """Create deployment config."""
        return DeploymentConfig(
            environment=DeploymentEnvironment.PAPER,
            startup_timeout_seconds=5,
        )

    @pytest.fixture
    def integrator(self, config):
        """Create system integrator."""
        return SystemIntegrator(config)

    def test_init(self, integrator, config):
        """Test initialization."""
        assert integrator.config == config
        assert integrator._started is False

    def test_register_component(self, integrator):
        """Test registering a component."""
        component = MagicMock()
        integrator.register_component("test", component)

        assert "test" in integrator._components
        assert integrator._components["test"]["instance"] == component

    def test_register_component_with_health_check(self, integrator):
        """Test registering component with health check."""
        component = MagicMock()

        def health_check():
            return ComponentHealth(
                component_name="test",
                status=ComponentStatus.HEALTHY,
                last_check=datetime.utcnow(),
            )

        integrator.register_component(
            "test",
            component,
            health_check=health_check,
        )

        assert "test" in integrator.health_monitor._components

    def test_get_component(self, integrator):
        """Test getting a component."""
        component = MagicMock()
        integrator.register_component("test", component)

        result = integrator.get_component("test")

        assert result == component

    def test_get_component_not_found(self, integrator):
        """Test getting nonexistent component."""
        result = integrator.get_component("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_startup(self, integrator):
        """Test startup."""
        startup_fn = MagicMock()
        component = MagicMock()
        integrator.register_component(
            "test",
            component,
            startup=startup_fn,
        )

        await integrator.startup()

        startup_fn.assert_called_once()
        assert integrator._started is True

    @pytest.mark.asyncio
    async def test_startup_async_functions(self, integrator):
        """Test startup with async functions."""
        startup_fn = AsyncMock()
        component = MagicMock()
        integrator.register_component(
            "test",
            component,
            startup=startup_fn,
        )

        await integrator.startup()

        startup_fn.assert_called_once()


class TestIntegrationTestRunner:
    """Tests for IntegrationTestRunner."""

    @pytest.fixture
    def integrator(self):
        """Create system integrator."""
        config = DeploymentConfig(environment=DeploymentEnvironment.PAPER)
        return SystemIntegrator(config)

    @pytest.fixture
    def runner(self, integrator):
        """Create test runner."""
        return IntegrationTestRunner(integrator)

    def test_init(self, runner, integrator):
        """Test initialization."""
        assert runner.integrator == integrator
        assert len(runner._tests) == 0

    def test_add_test(self, runner):
        """Test adding a test."""
        def test_fn():
            return True

        runner.add_test("test1", test_fn)

        assert len(runner._tests) == 1
        assert runner._tests[0][0] == "test1"

    @pytest.mark.asyncio
    async def test_run_tests_passing(self, runner):
        """Test running passing tests."""
        runner.add_test("test1", lambda: True)
        runner.add_test("test2", lambda: True)

        results = await runner.run_tests()

        assert results["passed"] == 2
        assert results["failed"] == 0
        assert results["all_passed"] is True

    @pytest.mark.asyncio
    async def test_run_tests_failing(self, runner):
        """Test running failing tests."""
        runner.add_test("test1", lambda: True)
        runner.add_test("test2", lambda: False)

        results = await runner.run_tests()

        assert results["passed"] == 1
        assert results["failed"] == 1
        assert results["all_passed"] is False

    @pytest.mark.asyncio
    async def test_run_tests_exception(self, runner):
        """Test running tests that raise exceptions."""

        def failing_test():
            raise ValueError("test error")

        runner.add_test("test1", failing_test)

        results = await runner.run_tests()

        assert results["failed"] == 1
        assert results["tests"][0]["error"] == "test error"

    @pytest.mark.asyncio
    async def test_run_async_tests(self, runner):
        """Test running async tests."""

        async def async_test():
            return True

        runner.add_test("async_test", async_test)

        results = await runner.run_tests()

        assert results["passed"] == 1

    def test_add_default_tests(self, runner):
        """Test adding default tests."""
        runner.add_default_tests()

        assert len(runner._tests) == 3


class TestCreateProductionSystem:
    """Tests for create_production_system factory."""

    def test_create_with_defaults(self):
        """Test creating with defaults."""
        with patch.dict("os.environ", {"ORDINIS_ENVIRONMENT": "PAPER"}):
            integrator = create_production_system()

        assert integrator is not None
        assert integrator.config.environment == DeploymentEnvironment.PAPER

    def test_create_with_custom_config(self):
        """Test creating with custom config."""
        config = DeploymentConfig(
            environment=DeploymentEnvironment.PRODUCTION,
            enable_live_trading=True,
        )

        integrator = create_production_system(config)

        assert integrator.config == config
        assert integrator.config.enable_live_trading is True
