"""
Comprehensive tests for plugin registry.

Tests cover:
- Plugin class registration and discovery
- Plugin instance creation and management
- Plugin lifecycle operations (initialize, shutdown, health checks)
- Error handling and edge cases
- Status and capability queries
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ordinis.plugins.base import (
    Plugin,
    PluginCapability,
    PluginConfig,
    PluginHealth,
    PluginStatus,
)
from ordinis.plugins.registry import PluginRegistry


class MockPlugin(Plugin):
    """Mock plugin for testing."""

    name = "mock_plugin"
    version = "1.0.0"
    description = "Mock plugin for testing"
    capabilities = [PluginCapability.READ, PluginCapability.WRITE]

    async def initialize(self) -> bool:
        """Initialize the mock plugin."""
        self._status = PluginStatus.READY
        return True

    async def shutdown(self) -> None:
        """Shutdown the mock plugin."""
        self._status = PluginStatus.STOPPED

    async def health_check(self) -> PluginHealth:
        """Perform health check."""
        return PluginHealth(
            status=self._status,
            last_check=datetime.utcnow(),
            latency_ms=10.5,
        )


class AnotherMockPlugin(Plugin):
    """Another mock plugin for testing multi-plugin scenarios."""

    name = "another_mock"
    version = "2.0.0"
    description = "Another mock plugin"
    capabilities = [PluginCapability.READ]

    async def initialize(self) -> bool:
        """Initialize the plugin."""
        self._status = PluginStatus.READY
        return True

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._status = PluginStatus.STOPPED

    async def health_check(self) -> PluginHealth:
        """Perform health check."""
        return PluginHealth(
            status=self._status,
            last_check=datetime.utcnow(),
            latency_ms=5.0,
        )


class TestRegistryBasics:
    """Test basic registry functionality."""

    @pytest.mark.unit
    def test_registry_singleton(self):
        """Test that global registry instance exists and is a singleton."""
        from ordinis.plugins.registry import registry as reg1
        from ordinis.plugins.registry import registry as reg2

        assert reg1 is reg2
        assert isinstance(reg1, PluginRegistry)

    @pytest.mark.unit
    def test_registry_initialization(self):
        """Test registry initializes with empty state."""
        registry = PluginRegistry()

        assert registry._plugins == {}
        assert registry._plugin_classes == {}
        assert registry.list_plugins() == []
        assert registry.list_available_classes() == []


class TestPluginClassRegistration:
    """Test plugin class registration."""

    @pytest.mark.unit
    def test_register_plugin_class(self):
        """Test registering a plugin class."""
        registry = PluginRegistry()

        registry.register_class(MockPlugin)

        assert "mock_plugin" in registry._plugin_classes
        assert registry._plugin_classes["mock_plugin"] == MockPlugin
        assert "mock_plugin" in registry.list_available_classes()

    @pytest.mark.unit
    def test_register_multiple_classes(self):
        """Test registering multiple plugin classes."""
        registry = PluginRegistry()

        registry.register_class(MockPlugin)
        registry.register_class(AnotherMockPlugin)

        assert len(registry.list_available_classes()) == 2
        assert "mock_plugin" in registry.list_available_classes()
        assert "another_mock" in registry.list_available_classes()

    @pytest.mark.unit
    def test_register_class_overwrite_warning(self):
        """Test that overwriting a plugin class logs a warning."""
        registry = PluginRegistry()

        registry.register_class(MockPlugin)

        with patch("ordinis.plugins.registry.logger") as mock_logger:
            registry.register_class(MockPlugin)
            mock_logger.warning.assert_called_once()
            assert "Overwriting" in mock_logger.warning.call_args[0][0]

    @pytest.mark.unit
    def test_list_available_classes_empty(self):
        """Test listing classes when none registered."""
        registry = PluginRegistry()

        assert registry.list_available_classes() == []


class TestPluginInstanceCreation:
    """Test plugin instance creation."""

    @pytest.mark.unit
    def test_create_plugin_instance(self):
        """Test creating a plugin instance from registered class."""
        registry = PluginRegistry()
        registry.register_class(MockPlugin)

        config = PluginConfig(name="test_instance")
        plugin = registry.create_plugin("mock_plugin", config)

        assert plugin is not None
        assert isinstance(plugin, MockPlugin)
        assert plugin.config.name == "test_instance"
        assert "test_instance" in registry._plugins

    @pytest.mark.unit
    def test_create_plugin_from_unregistered_class(self):
        """Test creating plugin from class that doesn't exist."""
        registry = PluginRegistry()

        config = PluginConfig(name="test_instance")
        plugin = registry.create_plugin("nonexistent", config)

        assert plugin is None
        assert "test_instance" not in registry._plugins

    @pytest.mark.unit
    def test_create_plugin_logs_error_for_missing_class(self):
        """Test that creating from missing class logs error."""
        registry = PluginRegistry()

        config = PluginConfig(name="test_instance")

        with patch("ordinis.plugins.registry.logger") as mock_logger:
            plugin = registry.create_plugin("nonexistent", config)

            assert plugin is None
            mock_logger.error.assert_called_once()
            assert "not found" in mock_logger.error.call_args[0][0]

    @pytest.mark.unit
    def test_create_multiple_instances_from_same_class(self):
        """Test creating multiple instances from same plugin class."""
        registry = PluginRegistry()
        registry.register_class(MockPlugin)

        config1 = PluginConfig(name="instance1")
        config2 = PluginConfig(name="instance2")

        plugin1 = registry.create_plugin("mock_plugin", config1)
        plugin2 = registry.create_plugin("mock_plugin", config2)

        assert plugin1 is not None
        assert plugin2 is not None
        assert plugin1 is not plugin2
        assert len(registry.list_plugins()) == 2

    @pytest.mark.unit
    def test_create_plugin_with_custom_config(self):
        """Test creating plugin with custom configuration."""
        registry = PluginRegistry()
        registry.register_class(MockPlugin)

        config = PluginConfig(
            name="custom_instance",
            enabled=False,
            api_key="test_key",
            timeout_seconds=60,
            extra={"custom_field": "value"},
        )
        plugin = registry.create_plugin("mock_plugin", config)

        assert plugin is not None
        assert plugin.config.enabled is False
        assert plugin.config.api_key == "test_key"
        assert plugin.config.timeout_seconds == 60
        assert plugin.config.extra["custom_field"] == "value"


class TestPluginRetrieval:
    """Test plugin retrieval operations."""

    @pytest.mark.unit
    def test_get_plugin_by_name(self):
        """Test retrieving plugin by name."""
        registry = PluginRegistry()
        registry.register_class(MockPlugin)

        config = PluginConfig(name="test_instance")
        created_plugin = registry.create_plugin("mock_plugin", config)

        retrieved_plugin = registry.get_plugin("test_instance")

        assert retrieved_plugin is not None
        assert retrieved_plugin is created_plugin

    @pytest.mark.unit
    def test_get_nonexistent_plugin(self):
        """Test getting a plugin that doesn't exist."""
        registry = PluginRegistry()

        plugin = registry.get_plugin("nonexistent")

        assert plugin is None

    @pytest.mark.unit
    def test_list_plugins_empty(self):
        """Test listing plugins when none exist."""
        registry = PluginRegistry()

        assert registry.list_plugins() == []

    @pytest.mark.unit
    def test_list_plugins_multiple(self):
        """Test listing multiple plugins."""
        registry = PluginRegistry()
        registry.register_class(MockPlugin)

        config1 = PluginConfig(name="plugin1")
        config2 = PluginConfig(name="plugin2")

        registry.create_plugin("mock_plugin", config1)
        registry.create_plugin("mock_plugin", config2)

        plugins = registry.list_plugins()

        assert len(plugins) == 2
        assert "plugin1" in plugins
        assert "plugin2" in plugins


class TestPluginLifecycle:
    """Test plugin lifecycle management."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_all_plugins(self):
        """Test initializing all registered plugins."""
        registry = PluginRegistry()
        registry.register_class(MockPlugin)
        registry.register_class(AnotherMockPlugin)

        config1 = PluginConfig(name="plugin1")
        config2 = PluginConfig(name="plugin2")

        registry.create_plugin("mock_plugin", config1)
        registry.create_plugin("another_mock", config2)

        results = await registry.initialize_all()

        assert len(results) == 2
        assert results["plugin1"] is True
        assert results["plugin2"] is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_all_with_failure(self):
        """Test initialize_all handles plugin failures gracefully."""
        registry = PluginRegistry()

        plugin_success = AsyncMock(spec=Plugin)
        plugin_success.initialize.return_value = True

        plugin_failure = AsyncMock(spec=Plugin)
        plugin_failure.initialize.return_value = False

        registry._plugins["success"] = plugin_success
        registry._plugins["failure"] = plugin_failure

        results = await registry.initialize_all()

        assert results["success"] is True
        assert results["failure"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_all_with_exception(self):
        """Test initialize_all handles exceptions during initialization."""
        registry = PluginRegistry()

        plugin = AsyncMock(spec=Plugin)
        plugin.initialize.side_effect = RuntimeError("Initialization error")

        registry._plugins["error_plugin"] = plugin

        results = await registry.initialize_all()

        assert results["error_plugin"] is False

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shutdown_all_plugins(self):
        """Test shutting down all plugins."""
        registry = PluginRegistry()

        plugin1 = AsyncMock(spec=Plugin)
        plugin2 = AsyncMock(spec=Plugin)

        registry._plugins["plugin1"] = plugin1
        registry._plugins["plugin2"] = plugin2

        await registry.shutdown_all()

        plugin1.shutdown.assert_called_once()
        plugin2.shutdown.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shutdown_all_with_exception(self):
        """Test shutdown_all handles exceptions gracefully."""
        registry = PluginRegistry()

        plugin_ok = AsyncMock(spec=Plugin)
        plugin_error = AsyncMock(spec=Plugin)
        plugin_error.shutdown.side_effect = RuntimeError("Shutdown error")

        registry._plugins["ok"] = plugin_ok
        registry._plugins["error"] = plugin_error

        await registry.shutdown_all()

        plugin_ok.shutdown.assert_called_once()
        plugin_error.shutdown.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_shutdown_all_empty_registry(self):
        """Test shutdown_all with no plugins."""
        registry = PluginRegistry()

        await registry.shutdown_all()

        assert len(registry._plugins) == 0


class TestHealthChecks:
    """Test health check functionality."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_all_plugins(self):
        """Test running health checks on all plugins."""
        registry = PluginRegistry()
        registry.register_class(MockPlugin)
        registry.register_class(AnotherMockPlugin)

        config1 = PluginConfig(name="plugin1")
        config2 = PluginConfig(name="plugin2")

        registry.create_plugin("mock_plugin", config1)
        registry.create_plugin("another_mock", config2)

        await registry.initialize_all()
        results = await registry.health_check_all()

        assert len(results) == 2
        assert "plugin1" in results
        assert "plugin2" in results
        assert isinstance(results["plugin1"], PluginHealth)
        assert isinstance(results["plugin2"], PluginHealth)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_with_exception(self):
        """Test health_check_all handles exceptions."""
        registry = PluginRegistry()

        plugin = AsyncMock(spec=Plugin)
        plugin.health_check.side_effect = RuntimeError("Health check failed")

        registry._plugins["error_plugin"] = plugin

        results = await registry.health_check_all()

        assert "error_plugin" in results
        health = results["error_plugin"]
        assert isinstance(health, PluginHealth)
        assert health.status == PluginStatus.ERROR
        assert health.last_error == "Health check failed"
        assert health.latency_ms == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_empty_registry(self):
        """Test health_check_all with no plugins."""
        registry = PluginRegistry()

        results = await registry.health_check_all()

        assert results == {}


class TestPluginQueries:
    """Test plugin query operations."""

    @pytest.mark.unit
    def test_get_plugins_by_status(self):
        """Test getting plugins filtered by status."""
        registry = PluginRegistry()

        plugin1 = MagicMock(spec=Plugin)
        plugin1.status = PluginStatus.READY

        plugin2 = MagicMock(spec=Plugin)
        plugin2.status = PluginStatus.ERROR

        plugin3 = MagicMock(spec=Plugin)
        plugin3.status = PluginStatus.READY

        registry._plugins["plugin1"] = plugin1
        registry._plugins["plugin2"] = plugin2
        registry._plugins["plugin3"] = plugin3

        ready_plugins = registry.get_plugins_by_status(PluginStatus.READY)

        assert len(ready_plugins) == 2
        assert plugin1 in ready_plugins
        assert plugin3 in ready_plugins

    @pytest.mark.unit
    def test_get_plugins_by_status_none_match(self):
        """Test get_plugins_by_status when no plugins match."""
        registry = PluginRegistry()

        plugin = MagicMock(spec=Plugin)
        plugin.status = PluginStatus.READY

        registry._plugins["plugin"] = plugin

        error_plugins = registry.get_plugins_by_status(PluginStatus.ERROR)

        assert error_plugins == []

    @pytest.mark.unit
    def test_get_plugins_by_capability(self):
        """Test getting plugins filtered by capability."""
        registry = PluginRegistry()

        plugin1 = MagicMock(spec=Plugin)
        plugin1.capabilities = [PluginCapability.READ, PluginCapability.WRITE]

        plugin2 = MagicMock(spec=Plugin)
        plugin2.capabilities = [PluginCapability.READ]

        plugin3 = MagicMock(spec=Plugin)
        plugin3.capabilities = [PluginCapability.STREAM]

        registry._plugins["plugin1"] = plugin1
        registry._plugins["plugin2"] = plugin2
        registry._plugins["plugin3"] = plugin3

        read_plugins = registry.get_plugins_by_capability("read")

        assert len(read_plugins) == 2
        assert plugin1 in read_plugins
        assert plugin2 in read_plugins

    @pytest.mark.unit
    def test_get_plugins_by_capability_none_match(self):
        """Test get_plugins_by_capability when no plugins match."""
        registry = PluginRegistry()

        plugin = MagicMock(spec=Plugin)
        plugin.capabilities = [PluginCapability.READ]

        registry._plugins["plugin"] = plugin

        write_plugins = registry.get_plugins_by_capability("write")

        assert write_plugins == []

    @pytest.mark.unit
    def test_get_plugins_by_capability_empty_registry(self):
        """Test get_plugins_by_capability with empty registry."""
        registry = PluginRegistry()

        plugins = registry.get_plugins_by_capability("read")

        assert plugins == []


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.unit
    def test_create_plugin_overwrites_existing_instance(self):
        """Test that creating plugin with same name overwrites existing."""
        registry = PluginRegistry()
        registry.register_class(MockPlugin)

        config1 = PluginConfig(name="same_name")
        config2 = PluginConfig(name="same_name", api_key="different_key")

        plugin1 = registry.create_plugin("mock_plugin", config1)
        plugin2 = registry.create_plugin("mock_plugin", config2)

        assert plugin1 is not plugin2
        assert registry.get_plugin("same_name") is plugin2
        assert len(registry.list_plugins()) == 1

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_initialize_all_empty_registry(self):
        """Test initialize_all with no plugins."""
        registry = PluginRegistry()

        results = await registry.initialize_all()

        assert results == {}

    @pytest.mark.unit
    def test_get_plugins_by_status_all_statuses(self):
        """Test get_plugins_by_status with each status value."""
        registry = PluginRegistry()

        for status in PluginStatus:
            plugin = MagicMock(spec=Plugin)
            plugin.status = status
            registry._plugins[status.value] = plugin

        for status in PluginStatus:
            plugins = registry.get_plugins_by_status(status)
            assert len(plugins) == 1
            assert plugins[0].status == status

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_lifecycle_integration(self):
        """Test full plugin lifecycle: register -> create -> init -> health -> shutdown."""
        registry = PluginRegistry()
        registry.register_class(MockPlugin)

        config = PluginConfig(name="lifecycle_test")
        plugin = registry.create_plugin("mock_plugin", config)

        assert plugin is not None
        assert plugin.status == PluginStatus.UNINITIALIZED

        init_results = await registry.initialize_all()
        assert init_results["lifecycle_test"] is True
        assert plugin.status == PluginStatus.READY

        health_results = await registry.health_check_all()
        assert health_results["lifecycle_test"].status == PluginStatus.READY

        await registry.shutdown_all()
        assert plugin.status == PluginStatus.STOPPED
