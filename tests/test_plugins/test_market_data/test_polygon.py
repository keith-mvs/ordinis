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

from ordinis.adapters.market_data.polygon import PolygonDataPlugin
from ordinis.plugins.base import PluginConfig


@pytest.fixture
def polygon_config():
    """Create Polygon plugin configuration."""
    return PluginConfig(name="polygon", enabled=True, api_key="test_api_key")


@pytest.fixture
def mock_polygon_quote_response():
    """Mock Polygon quote response."""
    return {"ticker": "AAPL", "last": {"price": 150.0}, "lastQuote": {"bid": 149.95, "ask": 150.05}}


@pytest.fixture
def mock_polygon_bars_response():
    """Mock Polygon bars response."""
    return {
        "results": [
            {"t": 1704067200000, "o": 149.0, "h": 151.0, "l": 148.5, "c": 150.0, "v": 1000000}
        ]
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_polygon_plugin_initialization(polygon_config):
    """Test Polygon plugin initialization."""
    plugin = PolygonDataPlugin(polygon_config)

    assert plugin.name == "polygon"
    assert plugin.config == polygon_config


@pytest.mark.unit
@pytest.mark.asyncio
async def test_polygon_get_quote(polygon_config):
    """Test getting quote from Polygon."""
    plugin = PolygonDataPlugin(polygon_config)

    # Mock the HTTP client - Polygon makes 2 calls (trade + quote)
    with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
        mock_request.side_effect = [
            {"results": {"p": 150.0, "s": 100}},  # trade
            {"results": {"p": 149.95, "s": 50}},  # quote
        ]

        quote = await plugin.get_quote("AAPL")

        assert quote is not None
        assert quote["symbol"] == "AAPL"
        assert mock_request.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_polygon_get_historical(polygon_config, mock_polygon_bars_response):
    """Test getting historical data from Polygon."""
    from datetime import datetime

    plugin = PolygonDataPlugin(polygon_config)

    with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_polygon_bars_response

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 31)
        bars = await plugin.get_historical("AAPL", start, end, "1d")

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

        health = await plugin.health_check()

        assert health is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_polygon_rate_limiting(polygon_config):
    """Test that rate limiting is applied."""
    plugin = PolygonDataPlugin(polygon_config)

    # Verify rate limiter exists
    assert hasattr(plugin, "rate_limiter") or hasattr(plugin, "_rate_limiter")
