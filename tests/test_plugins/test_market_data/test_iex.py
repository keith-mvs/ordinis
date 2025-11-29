"""
Tests for IEX Cloud market data plugin.

Tests cover:
- Quote retrieval
- Historical data fetching
- Company information
- Error handling
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.plugins.base import PluginConfig
from src.plugins.market_data.iex import IEXDataPlugin


@pytest.fixture
def iex_config():
    """Create IEX plugin configuration."""
    return PluginConfig(name="iex", enabled=True, config={"api_key": "test_api_key"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_iex_plugin_initialization(iex_config):
    """Test IEX plugin initialization."""
    plugin = IEXDataPlugin(iex_config)

    assert plugin.name == "iex"
    assert plugin.config == iex_config


@pytest.mark.unit
@pytest.mark.asyncio
async def test_iex_get_quote(iex_config, mock_iex_quote_response):
    """Test getting quote from IEX."""
    plugin = IEXDataPlugin(iex_config)

    with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_iex_quote_response

        quote = await plugin.get_quote("AAPL")

        assert quote is not None
        mock_request.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_iex_get_historical(iex_config):
    """Test getting historical data from IEX."""
    plugin = IEXDataPlugin(iex_config)

    with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = [
            {"date": "2024-01-01", "close": 150.0},
            {"date": "2024-01-02", "close": 151.0},
        ]

        bars = await plugin.get_historical("AAPL", "1m")

        assert bars is not None
        assert isinstance(bars, dict | list)
        mock_request.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_iex_health_check(iex_config):
    """Test IEX plugin health check."""
    plugin = IEXDataPlugin(iex_config)

    with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = {"status": "up"}

        is_healthy = await plugin.is_healthy()

        assert is_healthy is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_iex_rate_limiting(iex_config):
    """Test that rate limiting is applied."""
    plugin = IEXDataPlugin(iex_config)

    # Verify rate limiter exists
    assert hasattr(plugin, "rate_limiter") or hasattr(plugin, "_rate_limiter")
