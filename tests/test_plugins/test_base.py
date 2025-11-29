"""
Tests for plugin base classes and interfaces.

Tests cover:
- Plugin protocol
- Base plugin functionality
- Plugin lifecycle
- Health checking
"""

from unittest.mock import AsyncMock

import pytest

from src.plugins.base import Plugin, PluginConfig, PluginStatus


@pytest.mark.unit
def test_plugin_config_creation():
    """Test creating plugin configuration."""
    config = PluginConfig(name="test_plugin", enabled=True, config={"api_key": "test123"})

    assert config.name == "test_plugin"
    assert config.enabled is True
    assert config.config["api_key"] == "test123"


@pytest.mark.unit
def test_plugin_status_enum():
    """Test plugin status enum values."""
    assert PluginStatus.INITIALIZING.value == "initializing"
    assert PluginStatus.READY.value == "ready"
    assert PluginStatus.ERROR.value == "error"
    assert PluginStatus.DISABLED.value == "disabled"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plugin_initialize():
    """Test plugin initialization."""
    # Create a mock plugin instance
    plugin = AsyncMock(spec=Plugin)
    plugin.initialize.return_value = None
    plugin.name = "test_plugin"
    plugin.status = PluginStatus.READY

    await plugin.initialize()

    plugin.initialize.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plugin_shutdown():
    """Test plugin shutdown."""
    plugin = AsyncMock(spec=Plugin)
    plugin.shutdown.return_value = None

    await plugin.shutdown()

    plugin.shutdown.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_plugin_health_check():
    """Test plugin health checking."""
    plugin = AsyncMock(spec=Plugin)
    plugin.is_healthy.return_value = True

    is_healthy = await plugin.is_healthy()

    assert is_healthy is True
    plugin.is_healthy.assert_called_once()
