"""Comprehensive tests for Polygon.io market data plugin."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from plugins.base import PluginConfig, PluginStatus
from plugins.market_data.polygon import PolygonDataPlugin


@pytest.fixture
def polygon_config():
    """Create Polygon plugin configuration."""
    return PluginConfig(
        name="polygon",
        enabled=True,
        api_key="test_polygon_key_12345",
        timeout_seconds=30,
        extra={},
    )


class TestPolygonPluginInitialization:
    """Test Polygon plugin initialization."""

    @pytest.mark.asyncio
    async def test_plugin_attributes(self, polygon_config):
        """Test plugin has correct attributes."""
        plugin = PolygonDataPlugin(polygon_config)

        assert plugin.name == "polygon"
        assert plugin.version == "1.0.0"
        assert "Polygon.io" in plugin.description
        assert plugin.config == polygon_config

    @pytest.mark.asyncio
    async def test_base_url(self, polygon_config):
        """Test API base URL."""
        plugin = PolygonDataPlugin(polygon_config)

        assert plugin.BASE_URL == "https://api.polygon.io"

    @pytest.mark.asyncio
    async def test_initialize_success(self, polygon_config):
        """Test successful initialization."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"status": "OK", "resultsCount": 1}

            result = await plugin.initialize()

            assert result is True
            assert plugin.status == PluginStatus.READY
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_results_count(self, polygon_config):
        """Test initialization with resultsCount check."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            # Response without status but with resultsCount
            mock_request.return_value = {"resultsCount": 5}

            result = await plugin.initialize()

            assert result is True
            assert plugin.status == PluginStatus.READY

    @pytest.mark.asyncio
    async def test_initialize_api_key_invalid(self, polygon_config):
        """Test initialization with invalid API key."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"status": "ERROR", "error": "Invalid API key"}

            result = await plugin.initialize()

            assert result is False
            assert plugin.status == PluginStatus.ERROR

    @pytest.mark.asyncio
    async def test_initialize_network_error(self, polygon_config):
        """Test initialization with network error."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = aiohttp.ClientError("Network error")

            result = await plugin.initialize()

            assert result is False
            assert plugin.status in [PluginStatus.INITIALIZING, PluginStatus.ERROR]

    @pytest.mark.asyncio
    async def test_shutdown(self, polygon_config):
        """Test plugin shutdown."""
        plugin = PolygonDataPlugin(polygon_config)

        # Create mock session and websocket
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        mock_ws = AsyncMock()
        plugin._session = mock_session
        plugin._ws_connection = mock_ws

        await plugin.shutdown()

        mock_ws.close.assert_called_once()
        mock_session.close.assert_called_once()
        assert plugin.status == PluginStatus.STOPPED

    @pytest.mark.asyncio
    async def test_shutdown_without_websocket(self, polygon_config):
        """Test shutdown when no websocket connection exists."""
        plugin = PolygonDataPlugin(polygon_config)

        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        plugin._session = mock_session
        plugin._ws_connection = None

        await plugin.shutdown()

        mock_session.close.assert_called_once()
        assert plugin.status == PluginStatus.STOPPED


class TestPolygonHealthCheck:
    """Test Polygon plugin health checks."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, polygon_config):
        """Test health check when API is healthy."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"market": "open"}

            health = await plugin.health_check()

            assert health.status == PluginStatus.READY
            assert health.latency_ms is not None
            assert health.latency_ms >= 0
            assert "open" in health.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_market_closed(self, polygon_config):
        """Test health check when market is closed."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"market": "closed"}

            health = await plugin.health_check()

            assert health.status == PluginStatus.READY
            assert "closed" in health.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, polygon_config):
        """Test health check when API is down."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = aiohttp.ClientError("API down")

            health = await plugin.health_check()

            assert health.status == PluginStatus.ERROR
            assert health.last_error is not None
            assert len(health.last_error) > 0

    @pytest.mark.asyncio
    async def test_health_check_latency_calculation(self, polygon_config):
        """Test that latency is calculated correctly."""
        plugin = PolygonDataPlugin(polygon_config)

        async def slow_response(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms delay
            return {"market": "open"}

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = slow_response

            health = await plugin.health_check()

            assert health.latency_ms >= 40
            assert health.latency_ms < 200


class TestPolygonQuoteRetrieval:
    """Test quote retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_quote_success(self, polygon_config):
        """Test successful quote retrieval."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            # Polygon makes 2 calls: trade + quote
            mock_request.side_effect = [
                {"results": {"p": 175.50, "s": 100}},  # Last trade
                {"results": {"p": 175.45, "s": 50, "P": 175.55, "S": 60}},  # Last quote
            ]

            quote = await plugin.get_quote("AAPL")

            assert quote is not None
            assert quote["symbol"] == "AAPL"
            assert quote["last"] == 175.50
            assert quote["last_size"] == 100
            assert quote["bid"] == 175.45
            assert quote["ask"] == 175.55
            assert quote["source"] == "polygon"
            assert mock_request.call_count == 2

    @pytest.mark.asyncio
    async def test_get_quote_api_endpoints(self, polygon_config):
        """Test that correct API endpoints are called."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [
                {"results": {"p": 100.0}},
                {"results": {"p": 99.95, "P": 100.05}},
            ]

            await plugin.get_quote("TSLA")

            # Verify endpoint calls
            calls = mock_request.call_args_list
            assert "/v2/last/trade/TSLA" in str(calls[0])
            assert "/v2/last/nbbo/TSLA" in str(calls[1])

    @pytest.mark.asyncio
    async def test_get_quote_missing_fields(self, polygon_config):
        """Test quote with missing optional fields."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            # Minimal response
            mock_request.side_effect = [
                {"results": {}},  # Empty trade data
                {"results": {}},  # Empty quote data
            ]

            quote = await plugin.get_quote("AAPL")

            assert quote["symbol"] == "AAPL"
            assert quote["last"] is None
            assert quote["bid"] is None
            assert quote["source"] == "polygon"

    @pytest.mark.asyncio
    async def test_get_quote_multiple_symbols(self, polygon_config):
        """Test getting quotes for multiple symbols."""
        plugin = PolygonDataPlugin(polygon_config)

        symbols = ["AAPL", "GOOGL", "MSFT"]

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = [
                {"results": {"p": 100.0}},
                {"results": {"p": 99.95}},
            ] * 3

            for symbol in symbols:
                quote = await plugin.get_quote(symbol)
                assert quote is not None
                assert quote["symbol"] == symbol


class TestPolygonHistoricalData:
    """Test historical data retrieval."""

    @pytest.mark.asyncio
    async def test_get_historical_daily(self, polygon_config):
        """Test getting daily historical data."""
        plugin = PolygonDataPlugin(polygon_config)

        mock_data = {
            "results": [
                {"t": 1704067200000, "o": 150.0, "h": 152.0, "l": 149.0, "c": 151.0, "v": 1000000},
                {"t": 1704153600000, "o": 151.0, "h": 153.0, "l": 150.0, "c": 152.5, "v": 1200000},
                {"t": 1704240000000, "o": 152.5, "h": 155.0, "l": 152.0, "c": 154.0, "v": 1500000},
            ]
        }

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_data

            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 3)

            data = await plugin.get_historical("AAPL", start, end, "1d")

            assert data is not None
            assert isinstance(data, list)
            assert len(data) == 3
            assert data[0]["open"] == 150.0

    @pytest.mark.asyncio
    async def test_get_historical_intraday(self, polygon_config):
        """Test getting intraday historical data."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "results": [
                    {
                        "t": 1704067200000,
                        "o": 150.0,
                        "h": 150.5,
                        "l": 149.5,
                        "c": 150.25,
                        "v": 10000,
                    }
                ]
            }

            start = datetime(2024, 1, 1, 9, 30)
            end = datetime(2024, 1, 1, 16, 0)

            # Test different intraday timeframes
            for timeframe in ["1m", "5m", "15m", "1h"]:
                data = await plugin.get_historical("AAPL", start, end, timeframe)
                assert data is not None

    @pytest.mark.asyncio
    async def test_get_historical_timeframe_mapping(self, polygon_config):
        """Test timeframe mapping to Polygon format."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"results": []}

            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 31)

            # Test 1h timeframe
            await plugin.get_historical("AAPL", start, end, "1h")

            # Verify endpoint contains hour/1
            call_args = str(mock_request.call_args)
            assert "hour" in call_args or "1h" in call_args.lower()

    @pytest.mark.asyncio
    async def test_get_historical_weekly(self, polygon_config):
        """Test weekly historical data."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {
                "results": [
                    {
                        "t": 1704067200000,
                        "o": 150.0,
                        "h": 157.0,
                        "l": 148.0,
                        "c": 155.0,
                        "v": 5000000,
                    }
                ]
            }

            start = datetime(2024, 1, 1)
            end = datetime(2024, 12, 31)

            data = await plugin.get_historical("AAPL", start, end, "1w")

            assert len(data) == 1
            assert data[0]["open"] == 150.0
            assert data[0]["high"] == 157.0
            assert data[0]["low"] == 148.0
            assert data[0]["close"] == 155.0
            assert data[0]["volume"] == 5000000

    @pytest.mark.asyncio
    async def test_get_historical_date_formatting(self, polygon_config):
        """Test date formatting in request."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"results": []}

            start = datetime(2024, 1, 15)
            end = datetime(2024, 1, 20)

            await plugin.get_historical("AAPL", start, end, "1d")

            # Verify dates are formatted as YYYY-MM-DD
            call_args = str(mock_request.call_args)
            assert "2024-01-15" in call_args
            assert "2024-01-20" in call_args

    @pytest.mark.asyncio
    async def test_get_historical_empty_results(self, polygon_config):
        """Test handling of empty results."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"results": []}

            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 31)

            data = await plugin.get_historical("AAPL", start, end, "1d")

            assert data == []


class TestPolygonRateLimiting:
    """Test rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_rate_limit_retry(self, polygon_config):
        """Test automatic retry on rate limit."""
        plugin = PolygonDataPlugin(polygon_config)
        plugin._session = AsyncMock(spec=aiohttp.ClientSession)

        # Create mock responses
        rate_limit_response = AsyncMock()
        rate_limit_response.status = 429
        rate_limit_response.__aenter__ = AsyncMock(return_value=rate_limit_response)
        rate_limit_response.__aexit__ = AsyncMock()

        success_response = AsyncMock()
        success_response.status = 200
        success_response.json = AsyncMock(return_value={"status": "OK"})
        success_response.raise_for_status = MagicMock()
        success_response.__aenter__ = AsyncMock(return_value=success_response)
        success_response.__aexit__ = AsyncMock()

        # First call returns 429, second succeeds
        plugin._session.get.side_effect = [rate_limit_response, success_response]

        # Mock sleep to avoid actual delay
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await plugin._make_request("/test/endpoint")

            assert result == {"status": "OK"}
            assert plugin._session.get.call_count == 2


class TestPolygonErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_handle_http_error(self, polygon_config):
        """Test handling of HTTP errors."""
        plugin = PolygonDataPlugin(polygon_config)

        # Mock _make_request to raise HTTP error
        error = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=500,
            message="Server error",
        )

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = error

            with pytest.raises(aiohttp.ClientResponseError):
                await plugin.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_handle_timeout(self, polygon_config):
        """Test handling of timeout errors."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = TimeoutError()

            with pytest.raises((asyncio.TimeoutError, Exception)):
                await plugin.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_handle_network_error(self, polygon_config):
        """Test handling of network errors."""
        plugin = PolygonDataPlugin(polygon_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = aiohttp.ClientError("Network error")

            with pytest.raises((aiohttp.ClientError, Exception)):
                await plugin.get_quote("AAPL")


class TestPolygonRequestMethods:
    """Test internal request methods."""

    @pytest.mark.asyncio
    async def test_make_request_includes_api_key(self, polygon_config):
        """Test that API key is included in requests."""
        plugin = PolygonDataPlugin(polygon_config)
        plugin._session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"test": "data"})
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        plugin._session.get.return_value = mock_response

        await plugin._make_request("/test")

        # API key should be in params
        call_args = plugin._session.get.call_args
        assert call_args is not None
        assert "apiKey" in str(call_args) or polygon_config.api_key in str(call_args)

    @pytest.mark.asyncio
    async def test_make_request_constructs_url(self, polygon_config):
        """Test that request URL is constructed correctly."""
        plugin = PolygonDataPlugin(polygon_config)
        plugin._session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={})
        mock_response.raise_for_status = MagicMock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        plugin._session.get.return_value = mock_response

        await plugin._make_request("/v2/test/endpoint")

        # Verify URL construction
        call_args = plugin._session.get.call_args
        assert call_args is not None
        assert "https://api.polygon.io/v2/test/endpoint" in str(call_args)


class TestPolygonCapabilities:
    """Test plugin capabilities."""

    @pytest.mark.asyncio
    async def test_plugin_capabilities(self, polygon_config):
        """Test that plugin advertises correct capabilities."""
        from plugins.base import PluginCapability

        plugin = PolygonDataPlugin(polygon_config)

        assert PluginCapability.READ in plugin.capabilities
        assert PluginCapability.REALTIME in plugin.capabilities
        assert PluginCapability.HISTORICAL in plugin.capabilities
        assert PluginCapability.STREAM in plugin.capabilities
        assert PluginCapability.WRITE not in plugin.capabilities

    @pytest.mark.asyncio
    async def test_plugin_metadata(self, polygon_config):
        """Test plugin metadata."""
        plugin = PolygonDataPlugin(polygon_config)

        assert plugin.name == "polygon"
        assert plugin.version is not None
        assert plugin.description is not None
        assert len(plugin.description) > 0
        assert "polygon" in plugin.description.lower()
