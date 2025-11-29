"""
Tests for plugin registry.

Tests cover:
- Plugin registration
- Plugin lookup
- Plugin lifecycle management
"""

from unittest.mock import AsyncMock

import pytest

from src.plugins.base import Plugin
from src.plugins.registry import PluginRegistry


@pytest.mark.unit
def test_registry_singleton():
    """Test that registry is a singleton."""
    registry1 = PluginRegistry()
    registry2 = PluginRegistry()

    assert registry1 is registry2


@pytest.mark.unit
def test_registry_register_plugin():
    """Test registering a plugin."""
    registry = PluginRegistry()
    plugin = AsyncMock(spec=Plugin)
    plugin.name = "test_plugin"

    registry.register("test_plugin", plugin)

    assert "test_plugin" in registry._plugins
    assert registry.get_plugin("test_plugin") == plugin


@pytest.mark.unit
def test_registry_get_nonexistent_plugin():
    """Test getting a plugin that doesn't exist."""
    registry = PluginRegistry()

    plugin = registry.get_plugin("nonexistent")

    assert plugin is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registry_initialize_all():
    """Test initializing all plugins."""
    registry = PluginRegistry()
    plugin1 = AsyncMock(spec=Plugin)
    plugin1.name = "plugin1"
    plugin1.initialize.return_value = None

    plugin2 = AsyncMock(spec=Plugin)
    plugin2.name = "plugin2"
    plugin2.initialize.return_value = None

    registry.register("plugin1", plugin1)
    registry.register("plugin2", plugin2)

    await registry.initialize_all()

    plugin1.initialize.assert_called_once()
    plugin2.initialize.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_registry_shutdown_all():
    """Test shutting down all plugins."""
    registry = PluginRegistry()
    plugin = AsyncMock(spec=Plugin)
    plugin.name = "test_plugin"
    plugin.shutdown.return_value = None

    registry.register("test_plugin", plugin)

    await registry.shutdown_all()

    plugin.shutdown.assert_called_once()
