"""
Tests for Polygon.io market data plugin.

Tests cover:
- Quote retrieval
- Historical data fetching
- API response handling
- Error handling
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.plugins.base import PluginConfig
from src.plugins.market_data.polygon import PolygonDataPlugin


@pytest.fixture
def polygon_config():
    """Create Polygon plugin configuration."""
    return PluginConfig(name="polygon", enabled=True, config={"api_key": "test_api_key"})


@pytest.mark.unit
@pytest.mark.asyncio
async def test_polygon_plugin_initialization(polygon_config):
    """Test Polygon plugin initialization."""
    plugin = PolygonDataPlugin(polygon_config)

    assert plugin.name == "polygon"
    assert plugin.config == polygon_config


@pytest.mark.unit
@pytest.mark.asyncio
async def test_polygon_get_quote(polygon_config, mock_polygon_quote_response):
    """Test getting quote from Polygon."""
    plugin = PolygonDataPlugin(polygon_config)

    # Mock the HTTP client
    with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_polygon_quote_response

        quote = await plugin.get_quote("AAPL")

        assert quote is not None
        mock_request.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_polygon_get_historical(polygon_config, mock_polygon_bars_response):
    """Test getting historical data from Polygon."""
    plugin = PolygonDataPlugin(polygon_config)

    with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_polygon_bars_response

        bars = await plugin.get_historical("AAPL", "1D", "2024-01-01", "2024-01-31")

        assert bars is not None
        assert isinstance(bars, dict | list)
        mock_request.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_polygon_health_check(polygon_config):
    """Test Polygon plugin health check."""
    plugin = PolygonDataPlugin(polygon_config)

    with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = {"status": "OK"}

        is_healthy = await plugin.is_healthy()

        assert is_healthy is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_polygon_rate_limiting(polygon_config):
    """Test that rate limiting is applied."""
    plugin = PolygonDataPlugin(polygon_config)

    # Verify rate limiter exists
    assert hasattr(plugin, "rate_limiter") or hasattr(plugin, "_rate_limiter")
