"""Tests for Alpha Vantage data plugin."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from ordinis.adapters.market_data.alphavantage import AlphaVantageDataPlugin
from ordinis.plugins.base import PluginCapability, PluginConfig, PluginStatus


class TestAlphaVantageDataPluginAttributes:
    """Tests for AlphaVantageDataPlugin class attributes."""

    def test_class_attributes(self):
        """Test plugin class attributes."""
        assert AlphaVantageDataPlugin.name == "alphavantage"
        assert AlphaVantageDataPlugin.version == "1.0.0"
        assert "Alpha Vantage" in AlphaVantageDataPlugin.description
        assert PluginCapability.REALTIME in AlphaVantageDataPlugin.capabilities
        assert PluginCapability.HISTORICAL in AlphaVantageDataPlugin.capabilities

    def test_base_url(self):
        """Test base URL is set correctly."""
        assert AlphaVantageDataPlugin.BASE_URL == "https://www.alphavantage.co/query"


class TestAlphaVantageDataPluginInit:
    """Tests for AlphaVantageDataPlugin initialization."""

    def test_init(self):
        """Test plugin initialization."""
        config = PluginConfig(name="alphavantage", api_key="test_key")
        plugin = AlphaVantageDataPlugin(config)

        assert plugin.config == config
        assert plugin._session is None
        assert plugin.status == PluginStatus.UNINITIALIZED

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        config = PluginConfig(name="alphavantage", api_key="test_key")
        plugin = AlphaVantageDataPlugin(config)

        mock_response = {"Global Quote": {"01. symbol": "AAPL"}}

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await plugin.initialize()

            assert result is True
            assert plugin.status == PluginStatus.READY
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_api_error(self):
        """Test initialization with API error."""
        config = PluginConfig(name="alphavantage", api_key="test_key")
        plugin = AlphaVantageDataPlugin(config)

        mock_response = {"Error Message": "Invalid API key"}

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await plugin.initialize()

            assert result is False
            assert plugin.status == PluginStatus.ERROR

    @pytest.mark.asyncio
    async def test_initialize_exception(self):
        """Test initialization handles exceptions."""
        config = PluginConfig(name="alphavantage", api_key="test_key")
        plugin = AlphaVantageDataPlugin(config)

        with patch.object(
            plugin, "_make_request", new_callable=AsyncMock
        ) as mock_request:
            mock_request.side_effect = Exception("Connection error")

            result = await plugin.initialize()

            assert result is False

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test plugin shutdown."""
        config = PluginConfig(name="alphavantage", api_key="test_key")
        plugin = AlphaVantageDataPlugin(config)

        # Mock the session
        mock_session = AsyncMock()
        plugin._session = mock_session

        await plugin.shutdown()

        mock_session.close.assert_called_once()
        assert plugin.status == PluginStatus.STOPPED


class TestAlphaVantageDataPluginQuote:
    """Tests for get_quote method."""

    @pytest.fixture
    def plugin(self):
        """Create plugin with mocked rate limiter."""
        config = PluginConfig(name="alphavantage", api_key="test_key")
        plugin = AlphaVantageDataPlugin(config)
        plugin._rate_limiter = MagicMock()
        plugin._rate_limiter.wait_for_token = AsyncMock()
        return plugin

    @pytest.mark.asyncio
    async def test_get_quote_success(self, plugin):
        """Test successful quote retrieval."""
        mock_response = {
            "Global Quote": {
                "01. symbol": "AAPL",
                "02. open": "149.00",
                "03. high": "151.00",
                "04. low": "148.50",
                "05. price": "150.25",
                "06. volume": "1000000",
                "07. latest trading day": "2024-01-15",
                "08. previous close": "149.50",
                "09. change": "0.75",
                "10. change percent": "0.50%",
            }
        }

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            quote = await plugin.get_quote("AAPL")

            assert quote["symbol"] == "AAPL"
            assert quote["last"] == 150.25
            assert quote["open"] == 149.00
            assert quote["high"] == 151.00
            assert quote["low"] == 148.50
            assert quote["volume"] == 1000000
            assert quote["source"] == "alphavantage"

    @pytest.mark.asyncio
    async def test_get_quote_empty_response(self, plugin):
        """Test quote retrieval with empty response."""
        mock_response = {"Global Quote": {}}

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            quote = await plugin.get_quote("AAPL")

            # Should return quote with defaults
            assert quote["symbol"] == "AAPL"
            assert quote["source"] == "alphavantage"


class TestAlphaVantageDataPluginHistorical:
    """Tests for get_historical method."""

    @pytest.fixture
    def plugin(self):
        """Create plugin with mocked rate limiter."""
        config = PluginConfig(name="alphavantage", api_key="test_key")
        plugin = AlphaVantageDataPlugin(config)
        plugin._rate_limiter = MagicMock()
        plugin._rate_limiter.wait_for_token = AsyncMock()
        return plugin

    @pytest.mark.asyncio
    async def test_get_historical_daily(self, plugin):
        """Test historical daily data retrieval."""
        mock_response = {
            "Time Series (Daily)": {
                "2024-01-15": {
                    "1. open": "149.00",
                    "2. high": "151.00",
                    "3. low": "148.50",
                    "4. close": "150.25",
                    "5. volume": "1000000",
                },
                "2024-01-14": {
                    "1. open": "148.00",
                    "2. high": "150.00",
                    "3. low": "147.50",
                    "4. close": "149.00",
                    "5. volume": "950000",
                },
            }
        }

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            start = datetime(2024, 1, 14)
            end = datetime(2024, 1, 15)
            bars = await plugin.get_historical("AAPL", start, end, "1d")

            assert len(bars) >= 1
            assert bars[0]["symbol"] == "AAPL"
            assert bars[0]["source"] == "alphavantage"

    @pytest.mark.asyncio
    async def test_get_historical_intraday(self, plugin):
        """Test historical intraday data retrieval."""
        mock_response = {
            "Time Series (5min)": {
                "2024-01-15 16:00:00": {
                    "1. open": "149.00",
                    "2. high": "149.50",
                    "3. low": "148.90",
                    "4. close": "149.25",
                    "5. volume": "50000",
                },
            }
        }

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            start = datetime(2024, 1, 15, 15, 0)
            end = datetime(2024, 1, 15, 16, 0)
            bars = await plugin.get_historical("AAPL", start, end, "5m")

            assert len(bars) == 1

    @pytest.mark.asyncio
    async def test_get_historical_empty(self, plugin):
        """Test handling of empty historical data."""
        mock_response = {"Time Series (Daily)": {}}

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 3)
            bars = await plugin.get_historical("AAPL", start, end, "1d")

            assert bars == []


class TestAlphaVantageDataPluginHealthCheck:
    """Tests for health_check method."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when API is responsive."""
        config = PluginConfig(name="alphavantage", api_key="test_key")
        plugin = AlphaVantageDataPlugin(config)

        mock_response = {"Global Quote": {"01. symbol": "AAPL"}}

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            health = await plugin.health_check()

            assert health.status == PluginStatus.READY
            assert health.latency_ms >= 0
            assert "healthy" in health.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_error(self):
        """Test health check when API errors."""
        config = PluginConfig(name="alphavantage", api_key="test_key")
        plugin = AlphaVantageDataPlugin(config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Connection error")

            health = await plugin.health_check()

            assert health.status == PluginStatus.ERROR
            assert health.error_count > 0


class TestAlphaVantageDataPluginCompany:
    """Tests for get_company method."""

    @pytest.fixture
    def plugin(self):
        """Create plugin with mocked rate limiter."""
        config = PluginConfig(name="alphavantage", api_key="test_key")
        plugin = AlphaVantageDataPlugin(config)
        plugin._rate_limiter = MagicMock()
        plugin._rate_limiter.wait_for_token = AsyncMock()
        return plugin

    @pytest.mark.asyncio
    async def test_get_company(self, plugin):
        """Test company info retrieval."""
        mock_response = {
            "Symbol": "AAPL",
            "Name": "Apple Inc",
            "Description": "Apple designs and manufactures...",
            "Sector": "Technology",
            "Industry": "Consumer Electronics",
            "Country": "USA",
            "Exchange": "NASDAQ",
        }

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            info = await plugin.get_company("AAPL")

            assert info["symbol"] == "AAPL"
            assert info["name"] == "Apple Inc"
            assert info["sector"] == "Technology"
