"""
Tests for plugin registry.

Tests cover:
- Plugin registration
- Plugin lookup
- Plugin lifecycle management
"""

from unittest.mock import AsyncMock

import pytest

from ordinis.plugins.base import Plugin, PluginConfig
from ordinis.plugins.registry import PluginRegistry


@pytest.mark.unit
def test_registry_singleton():
    """Test that global registry instance exists."""
    from ordinis.plugins.registry import registry as reg1
    from ordinis.plugins.registry import registry as reg2

    assert reg1 is reg2


@pytest.mark.unit
def test_registry_register_plugin_class():
    """Test registering a plugin class."""
    test_registry = PluginRegistry()

    # Create a mock plugin class
    class TestPlugin(Plugin):
        name = "test_plugin"
        version = "1.0.0"

        async def initialize(self) -> bool:
            return True

        async def shutdown(self) -> None:
            pass

        async def health_check(self):
            pass

    test_registry.register_class(TestPlugin)

    assert "test_plugin" in test_registry._plugin_classes

    # Create an instance
    config = PluginConfig(name="test_instance")
    plugin = test_registry.create_plugin("test_plugin", config)

    assert plugin is not None
    assert test_registry.get_plugin("test_instance") == plugin


@pytest.mark.unit
def test_registry_get_nonexistent_plugin():
    """Test getting a plugin that doesn't exist."""
    test_registry = PluginRegistry()

    plugin = test_registry.get_plugin("nonexistent")

    assert plugin is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registry_initialize_all():
    """Test initializing all plugins."""
    test_registry = PluginRegistry()
    plugin1 = AsyncMock(spec=Plugin)
    plugin1.name = "plugin1"
    plugin1.initialize.return_value = True

    plugin2 = AsyncMock(spec=Plugin)
    plugin2.name = "plugin2"
    plugin2.initialize.return_value = True

    # Add plugins directly to registry
    test_registry._plugins["plugin1"] = plugin1
    test_registry._plugins["plugin2"] = plugin2

    results = await test_registry.initialize_all()

    plugin1.initialize.assert_called_once()
    plugin2.initialize.assert_called_once()
    assert results["plugin1"] is True
    assert results["plugin2"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registry_shutdown_all():
    """Test shutting down all plugins."""
    test_registry = PluginRegistry()
    plugin = AsyncMock(spec=Plugin)
    plugin.name = "test_plugin"
    plugin.shutdown.return_value = None

    # Add plugin directly to registry
    test_registry._plugins["test_plugin"] = plugin

    await test_registry.shutdown_all()

    plugin.shutdown.assert_called_once()
