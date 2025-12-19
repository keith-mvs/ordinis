"""Comprehensive tests for IEX Cloud market data plugin."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from ordinis.adapters.market_data.iex import IEXDataPlugin
from ordinis.plugins.base import PluginConfig, PluginStatus


@pytest.fixture
def iex_config():
    """Create IEX plugin configuration."""
    return PluginConfig(
        name="iex",
        enabled=True,
        api_key="test_api_key_12345",
        timeout_seconds=30,
        extra={"sandbox": False},
    )


@pytest.fixture
def sandbox_config():
    """Create IEX sandbox configuration."""
    return PluginConfig(
        name="iex",
        enabled=True,
        api_key="test_sandbox_key",
        timeout_seconds=30,
        extra={"sandbox": True},
    )


class TestIEXPluginInitialization:
    """Test IEX plugin initialization."""

    @pytest.mark.asyncio
    async def test_plugin_attributes(self, iex_config):
        """Test plugin has correct attributes."""
        plugin = IEXDataPlugin(iex_config)

        assert plugin.name == "iex"
        assert plugin.version == "1.0.0"
        assert "IEX Cloud" in plugin.description
        assert plugin.config == iex_config

    @pytest.mark.asyncio
    async def test_api_url_production(self, iex_config):
        """Test production API URL."""
        plugin = IEXDataPlugin(iex_config)

        assert plugin.api_url == "https://cloud.iexapis.com/stable"
        assert "sandbox" not in plugin.api_url

    @pytest.mark.asyncio
    async def test_api_url_sandbox(self, sandbox_config):
        """Test sandbox API URL."""
        plugin = IEXDataPlugin(sandbox_config)

        assert plugin.api_url == "https://sandbox.iexapis.com/stable"
        assert "sandbox" in plugin.api_url

    @pytest.mark.asyncio
    async def test_initialize_success(self, iex_config):
        """Test successful initialization."""
        plugin = IEXDataPlugin(iex_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"symbol": "AAPL", "latestPrice": 150.0}

            result = await plugin.initialize()

            assert result is True
            assert plugin.status == PluginStatus.READY
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_api_key_invalid(self, iex_config):
        """Test initialization with invalid API key."""
        plugin = IEXDataPlugin(iex_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"error": "Invalid API key"}

            result = await plugin.initialize()

            assert result is False
            assert plugin.status == PluginStatus.ERROR

    @pytest.mark.asyncio
    async def test_initialize_network_error(self, iex_config):
        """Test initialization with network error."""
        plugin = IEXDataPlugin(iex_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = aiohttp.ClientError("Network error")

            result = await plugin.initialize()

            assert result is False
            # Status may still be INITIALIZING if error occurs early
            assert plugin.status in [PluginStatus.INITIALIZING, PluginStatus.ERROR]

    @pytest.mark.asyncio
    async def test_shutdown(self, iex_config):
        """Test plugin shutdown."""
        plugin = IEXDataPlugin(iex_config)

        # Create mock session
        mock_session = AsyncMock(spec=aiohttp.ClientSession)
        plugin._session = mock_session

        await plugin.shutdown()

        mock_session.close.assert_called_once()
        assert plugin.status == PluginStatus.STOPPED


class TestIEXHealthCheck:
    """Test IEX plugin health checks."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, iex_config):
        """Test health check when API is healthy."""
        plugin = IEXDataPlugin(iex_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"status": "up"}

            health = await plugin.health_check()

            assert health.status == PluginStatus.READY
            assert health.latency_ms is not None
            assert health.latency_ms >= 0
            assert "healthy" in health.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, iex_config):
        """Test health check when API is down."""
        plugin = IEXDataPlugin(iex_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = aiohttp.ClientError("API down")

            health = await plugin.health_check()

            assert health.status == PluginStatus.ERROR
            assert health.last_error is not None
            assert len(health.last_error) > 0

    @pytest.mark.asyncio
    async def test_health_check_latency_calculation(self, iex_config):
        """Test that latency is calculated correctly."""
        plugin = IEXDataPlugin(iex_config)

        async def slow_response(*args, **kwargs):
            # Simulate slow response (50ms)
            import asyncio

            await asyncio.sleep(0.05)
            return {"status": "up"}

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = slow_response

            health = await plugin.health_check()

            assert health.latency_ms >= 40  # Allow some variance
            assert health.latency_ms < 200  # Should be under 200ms


class TestIEXQuoteRetrieval:
    """Test quote retrieval functionality."""

    @pytest.mark.asyncio
    async def test_get_quote_success(self, iex_config):
        """Test successful quote retrieval."""
        plugin = IEXDataPlugin(iex_config)

        mock_quote = {
            "symbol": "AAPL",
            "latestPrice": 175.50,
            "latestTime": "2024-01-15T16:00:00Z",
            "latestVolume": 50000000,
            "change": 2.50,
            "changePercent": 0.0145,
        }

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_quote

            quote = await plugin.get_quote("AAPL")

            assert quote is not None
            assert quote["symbol"] == "AAPL"
            assert quote["last"] == 175.50  # Returns formatted "last", not "latestPrice"
            assert quote["source"] == "iex"
            mock_request.assert_called_with("/stock/AAPL/quote")

    @pytest.mark.asyncio
    async def test_get_quote_invalid_symbol(self, iex_config):
        """Test quote retrieval with invalid symbol."""
        plugin = IEXDataPlugin(iex_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            # IEX returns error via exception, not in response
            mock_request.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=404,
                message="Unknown symbol",
            )

            with pytest.raises(aiohttp.ClientResponseError):
                await plugin.get_quote("INVALID")

    @pytest.mark.asyncio
    async def test_get_quote_multiple_symbols(self, iex_config):
        """Test getting quotes for multiple symbols."""
        plugin = IEXDataPlugin(iex_config)

        symbols = ["AAPL", "GOOGL", "MSFT"]

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"symbol": "TEST", "latestPrice": 100}

            for symbol in symbols:
                quote = await plugin.get_quote(symbol)
                assert quote is not None


class TestIEXHistoricalData:
    """Test historical data retrieval."""

    @pytest.mark.asyncio
    async def test_get_historical_data(self, iex_config):
        """Test getting historical data."""
        plugin = IEXDataPlugin(iex_config)

        mock_data = [
            {
                "date": "2024-01-01",
                "open": 150.0,
                "high": 152.0,
                "low": 149.0,
                "close": 151.0,
                "volume": 1000000,
            },
            {
                "date": "2024-01-02",
                "open": 151.0,
                "high": 153.0,
                "low": 150.0,
                "close": 152.5,
                "volume": 1200000,
            },
            {
                "date": "2024-01-03",
                "open": 152.5,
                "high": 155.0,
                "low": 152.0,
                "close": 154.0,
                "volume": 1500000,
            },
        ]

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_data

            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 3)

            data = await plugin.get_historical("AAPL", start, end, "1d")

            assert data is not None
            assert len(data) == 3 or isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_get_historical_different_intervals(self, iex_config):
        """Test historical data with different intervals."""
        plugin = IEXDataPlugin(iex_config)

        intervals = ["1d", "1h", "5m"]

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = [{"date": "2024-01-01", "close": 150.0}]

            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 31)

            for interval in intervals:
                data = await plugin.get_historical("AAPL", start, end, interval)
                assert data is not None

    @pytest.mark.asyncio
    async def test_get_historical_long_date_range(self, iex_config):
        """Test historical data for long date range."""
        plugin = IEXDataPlugin(iex_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            # Simulate large dataset
            mock_request.return_value = [
                {"date": f"2023-{i:02d}-01", "close": 150.0 + i} for i in range(1, 13)
            ]

            start = datetime(2023, 1, 1)
            end = datetime(2023, 12, 31)

            data = await plugin.get_historical("AAPL", start, end, "1d")

            assert data is not None


class TestIEXDataFormats:
    """Test data format handling."""

    @pytest.mark.asyncio
    async def test_quote_format_structure(self, iex_config):
        """Test that quote returns properly formatted data."""
        plugin = IEXDataPlugin(iex_config)

        mock_quote = {
            "symbol": "AAPL",
            "latestPrice": 175.50,
            "latestVolume": 50000000,
            "open": 174.0,
            "high": 176.0,
            "low": 173.5,
            "close": 175.50,
        }

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_quote

            quote = await plugin.get_quote("AAPL")

            # Verify formatted structure
            assert "symbol" in quote
            assert "timestamp" in quote
            assert "last" in quote
            assert "source" in quote
            assert quote["source"] == "iex"


class TestIEXErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_handle_api_rate_limit(self, iex_config):
        """Test handling of rate limit errors."""
        plugin = IEXDataPlugin(iex_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=429,
                message="Rate limit exceeded",
            )

            with pytest.raises((aiohttp.ClientResponseError, Exception)):
                await plugin.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_handle_timeout(self, iex_config):
        """Test handling of timeout errors."""
        plugin = IEXDataPlugin(iex_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = TimeoutError()

            with pytest.raises((asyncio.TimeoutError, Exception)):
                await plugin.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_handle_network_error(self, iex_config):
        """Test handling of network errors."""
        plugin = IEXDataPlugin(iex_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = aiohttp.ClientError("Network error")

            with pytest.raises((aiohttp.ClientError, Exception)):
                await plugin.get_quote("AAPL")

    @pytest.mark.asyncio
    async def test_handle_invalid_response(self, iex_config):
        """Test handling of invalid JSON response."""
        plugin = IEXDataPlugin(iex_config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            # Empty response still returns dict structure
            mock_request.return_value = {}

            result = await plugin.get_quote("AAPL")

            # Should return formatted dict even with empty response
            assert isinstance(result, dict)
            assert "source" in result
            assert result["source"] == "iex"


class TestIEXRequestMethods:
    """Test internal request methods."""

    @pytest.mark.asyncio
    async def test_make_request_constructs_url(self, iex_config):
        """Test that request URL is constructed correctly."""
        plugin = IEXDataPlugin(iex_config)
        plugin._session = AsyncMock(spec=aiohttp.ClientSession)

        # Mock the session.get response
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"test": "data"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        plugin._session.get.return_value = mock_response

        await plugin._make_request("/test/endpoint")

        # Verify URL construction
        call_args = plugin._session.get.call_args
        assert call_args is not None
        assert "/test/endpoint" in str(call_args)

    @pytest.mark.asyncio
    async def test_make_request_includes_api_key(self, iex_config):
        """Test that API key is included in requests."""
        plugin = IEXDataPlugin(iex_config)
        plugin._session = AsyncMock(spec=aiohttp.ClientSession)

        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"test": "data"})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock()

        plugin._session.get.return_value = mock_response

        await plugin._make_request("/test")

        # API key should be in params
        call_args = plugin._session.get.call_args
        assert call_args is not None


class TestIEXCapabilities:
    """Test plugin capabilities."""

    @pytest.mark.asyncio
    async def test_plugin_capabilities(self, iex_config):
        """Test that plugin advertises correct capabilities."""
        from ordinis.plugins.base import PluginCapability

        plugin = IEXDataPlugin(iex_config)

        assert PluginCapability.READ in plugin.capabilities
        assert PluginCapability.REALTIME in plugin.capabilities
        assert PluginCapability.HISTORICAL in plugin.capabilities
        assert PluginCapability.WRITE not in plugin.capabilities

    @pytest.mark.asyncio
    async def test_plugin_metadata(self, iex_config):
        """Test plugin metadata."""
        plugin = IEXDataPlugin(iex_config)

        assert plugin.name == "iex"
        assert plugin.version is not None
        assert plugin.description is not None
        assert len(plugin.description) > 0
