"""Tests for Finnhub data plugin."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ordinis.adapters.market_data.finnhub import FinnhubDataPlugin
from ordinis.plugins.base import PluginCapability, PluginConfig, PluginStatus


class TestFinnhubDataPluginAttributes:
    """Tests for FinnhubDataPlugin class attributes."""

    def test_class_attributes(self):
        """Test plugin class attributes."""
        assert FinnhubDataPlugin.name == "finnhub"
        assert FinnhubDataPlugin.version == "1.0.0"
        assert "Finnhub" in FinnhubDataPlugin.description
        assert PluginCapability.REALTIME in FinnhubDataPlugin.capabilities
        assert PluginCapability.HISTORICAL in FinnhubDataPlugin.capabilities

    def test_base_url(self):
        """Test base URL is set correctly."""
        assert "finnhub.io" in FinnhubDataPlugin.BASE_URL


class TestFinnhubDataPluginInit:
    """Tests for FinnhubDataPlugin initialization."""

    def test_init(self):
        """Test plugin initialization."""
        config = PluginConfig(name="finnhub", api_key="test_key")
        plugin = FinnhubDataPlugin(config)

        assert plugin.config == config
        assert plugin._session is None
        assert plugin.status == PluginStatus.UNINITIALIZED

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        config = PluginConfig(name="finnhub", api_key="test_key")
        plugin = FinnhubDataPlugin(config)

        mock_response = {"c": 150.25, "d": 0.5, "dp": 0.33}

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            result = await plugin.initialize()

            assert result is True
            assert plugin.status == PluginStatus.READY

    @pytest.mark.asyncio
    async def test_initialize_exception(self):
        """Test initialization handles exceptions."""
        config = PluginConfig(name="finnhub", api_key="test_key")
        plugin = FinnhubDataPlugin(config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Connection error")

            result = await plugin.initialize()

            assert result is False

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test plugin shutdown."""
        config = PluginConfig(name="finnhub", api_key="test_key")
        plugin = FinnhubDataPlugin(config)

        mock_session = AsyncMock()
        plugin._session = mock_session

        await plugin.shutdown()

        mock_session.close.assert_called_once()
        assert plugin.status == PluginStatus.STOPPED


class TestFinnhubDataPluginQuote:
    """Tests for get_quote method."""

    @pytest.fixture
    def plugin(self):
        """Create plugin with mocked rate limiter."""
        config = PluginConfig(name="finnhub", api_key="test_key")
        plugin = FinnhubDataPlugin(config)
        plugin._rate_limiter = MagicMock()
        plugin._rate_limiter.wait_for_token = AsyncMock()
        return plugin

    @pytest.mark.asyncio
    async def test_get_quote_success(self, plugin):
        """Test successful quote retrieval."""
        mock_response = {
            "c": 150.25,  # Current price
            "d": 0.75,    # Change
            "dp": 0.50,   # Percent change
            "h": 151.00,  # High
            "l": 148.50,  # Low
            "o": 149.00,  # Open
            "pc": 149.50, # Previous close
            "t": 1705344000,  # Timestamp
        }

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            quote = await plugin.get_quote("AAPL")

            assert quote["symbol"] == "AAPL"
            assert quote["last"] == 150.25
            assert quote["open"] == 149.00
            assert quote["high"] == 151.00
            assert quote["low"] == 148.50
            assert quote["source"] == "finnhub"

    @pytest.mark.asyncio
    async def test_get_quote_empty_data(self, plugin):
        """Test quote retrieval with minimal data."""
        mock_response = {"c": 0, "d": None, "dp": None}

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            quote = await plugin.get_quote("AAPL")

            # Should return quote with defaults
            assert quote["symbol"] == "AAPL"
            assert quote["source"] == "finnhub"


class TestFinnhubDataPluginHistorical:
    """Tests for get_historical method."""

    @pytest.fixture
    def plugin(self):
        """Create plugin with mocked rate limiter."""
        config = PluginConfig(name="finnhub", api_key="test_key")
        plugin = FinnhubDataPlugin(config)
        plugin._rate_limiter = MagicMock()
        plugin._rate_limiter.wait_for_token = AsyncMock()
        return plugin

    @pytest.mark.asyncio
    async def test_get_historical_success(self, plugin):
        """Test historical data retrieval."""
        mock_response = {
            "c": [150.0, 151.0, 152.0],  # Close
            "h": [151.0, 152.0, 153.0],  # High
            "l": [149.0, 150.0, 151.0],  # Low
            "o": [149.5, 150.5, 151.5],  # Open
            "v": [1000000, 1100000, 1200000],  # Volume
            "t": [1705257600, 1705344000, 1705430400],  # Timestamps
            "s": "ok",
        }

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            start = datetime(2024, 1, 14)
            end = datetime(2024, 1, 16)
            bars = await plugin.get_historical("AAPL", start, end, "1d")

            assert len(bars) == 3
            assert bars[0]["symbol"] == "AAPL"
            assert bars[0]["close"] == 150.0
            assert bars[0]["source"] == "finnhub"

    @pytest.mark.asyncio
    async def test_get_historical_empty(self, plugin):
        """Test handling of empty historical data."""
        mock_response = {"s": "no_data"}

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 3)
            bars = await plugin.get_historical("AAPL", start, end, "1d")

            assert bars == []


class TestFinnhubDataPluginHealthCheck:
    """Tests for health_check method."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when API is responsive."""
        config = PluginConfig(name="finnhub", api_key="test_key")
        plugin = FinnhubDataPlugin(config)

        mock_response = {"c": 150.25}

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            health = await plugin.health_check()

            assert health.status == PluginStatus.READY
            assert health.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_health_check_error(self):
        """Test health check when API errors."""
        config = PluginConfig(name="finnhub", api_key="test_key")
        plugin = FinnhubDataPlugin(config)

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = Exception("Connection error")

            health = await plugin.health_check()

            assert health.status == PluginStatus.ERROR


class TestFinnhubDataPluginCompany:
    """Tests for get_company method."""

    @pytest.fixture
    def plugin(self):
        """Create plugin with mocked rate limiter."""
        config = PluginConfig(name="finnhub", api_key="test_key")
        plugin = FinnhubDataPlugin(config)
        plugin._rate_limiter = MagicMock()
        plugin._rate_limiter.wait_for_token = AsyncMock()
        return plugin

    @pytest.mark.asyncio
    async def test_get_company(self, plugin):
        """Test company info retrieval."""
        mock_response = {
            "ticker": "AAPL",
            "name": "Apple Inc",
            "finnhubIndustry": "Technology",
            "country": "US",
            "exchange": "NASDAQ",
            "marketCapitalization": 2500000,
            "weburl": "https://www.apple.com",
        }

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            info = await plugin.get_company("AAPL")

            assert info["symbol"] == "AAPL"
            assert info["name"] == "Apple Inc"


class TestFinnhubDataPluginNews:
    """Tests for get_news method."""

    @pytest.fixture
    def plugin(self):
        """Create plugin with mocked rate limiter."""
        config = PluginConfig(name="finnhub", api_key="test_key")
        plugin = FinnhubDataPlugin(config)
        plugin._rate_limiter = MagicMock()
        plugin._rate_limiter.wait_for_token = AsyncMock()
        return plugin

    @pytest.mark.asyncio
    async def test_get_news(self, plugin):
        """Test news retrieval."""
        mock_response = [
            {
                "category": "technology",
                "datetime": 1705344000,
                "headline": "Apple announces new product",
                "id": 12345,
                "image": "https://example.com/image.jpg",
                "related": "AAPL",
                "source": "Reuters",
                "summary": "Apple has announced...",
                "url": "https://example.com/article",
            }
        ]

        with patch.object(plugin, "_make_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response

            news = await plugin.get_news("AAPL", limit=5)

            assert len(news) == 1
            assert news[0]["headline"] == "Apple announces new product"
            assert news[0]["source"] == "Reuters"
