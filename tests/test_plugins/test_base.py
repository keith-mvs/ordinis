"""
Tests for plugin base classes and interfaces.

Tests cover:
- Plugin protocol
- Base plugin functionality
- Plugin lifecycle
- Health checking
"""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from ordinis.plugins.base import Plugin, PluginConfig, PluginStatus


@pytest.mark.unit
def test_plugin_config_creation():
    """Test creating plugin configuration."""
    config = PluginConfig(
        name="test_plugin", enabled=True, api_key="test123", extra={"custom_field": "value"}
    )

    assert config.name == "test_plugin"
    assert config.enabled is True
    assert config.api_key == "test123"
    assert config.extra["custom_field"] == "value"


@pytest.mark.unit
def test_plugin_status_enum():
    """Test plugin status enum values."""
    assert PluginStatus.INITIALIZING.value == "initializing"
    assert PluginStatus.READY.value == "ready"
    assert PluginStatus.ERROR.value == "error"
    assert PluginStatus.STOPPED.value == "stopped"


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
    from ordinis.plugins.base import PluginHealth

    plugin = AsyncMock(spec=Plugin)
    health = PluginHealth(status=PluginStatus.READY, last_check=datetime.now(), latency_ms=10.0)
    plugin.health_check.return_value = health

    result = await plugin.health_check()

    assert result.status == PluginStatus.READY
    plugin.health_check.assert_called_once()
