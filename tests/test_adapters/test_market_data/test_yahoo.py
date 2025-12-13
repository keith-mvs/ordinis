"""Tests for Yahoo Finance data plugin."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from ordinis.plugins.base import PluginConfig, PluginStatus

# Skip all tests if yfinance not available
pytest.importorskip("yfinance")

from ordinis.adapters.market_data.yahoo import YahooDataPlugin  # noqa: E402


class TestYahooDataPluginAttributes:
    """Tests for YahooDataPlugin class attributes."""

    def test_class_attributes(self):
        """Test plugin class attributes."""
        assert YahooDataPlugin.name == "yahoo"
        assert YahooDataPlugin.version == "1.0.0"
        assert "free" in YahooDataPlugin.description.lower()

    def test_timeframe_map(self):
        """Test timeframe mapping."""
        assert YahooDataPlugin.TIMEFRAME_MAP["1m"] == "1m"
        assert YahooDataPlugin.TIMEFRAME_MAP["1d"] == "1d"
        assert YahooDataPlugin.TIMEFRAME_MAP["1wk"] == "1wk"


class TestYahooDataPluginInit:
    """Tests for YahooDataPlugin initialization."""

    def test_init(self):
        """Test plugin initialization."""
        config = PluginConfig(name="yahoo")
        plugin = YahooDataPlugin(config)

        assert plugin.config == config
        assert plugin._ticker_cache == {}
        assert plugin.status == PluginStatus.UNINITIALIZED

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        config = PluginConfig(name="yahoo")
        plugin = YahooDataPlugin(config)

        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {"symbol": "AAPL", "shortName": "Apple Inc."}
            mock_yf.Ticker.return_value = mock_ticker

            result = await plugin.initialize()

            assert result is True
            assert plugin.status == PluginStatus.READY

    @pytest.mark.asyncio
    async def test_initialize_failure(self):
        """Test initialization failure."""
        config = PluginConfig(name="yahoo")
        plugin = YahooDataPlugin(config)

        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {}  # Empty info indicates failure
            mock_yf.Ticker.return_value = mock_ticker

            result = await plugin.initialize()

            assert result is False
            assert plugin.status == PluginStatus.ERROR

    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test plugin shutdown."""
        config = PluginConfig(name="yahoo")
        plugin = YahooDataPlugin(config)

        # Add some tickers to cache
        plugin._ticker_cache["AAPL"] = MagicMock()

        await plugin.shutdown()

        assert plugin._ticker_cache == {}
        assert plugin.status == PluginStatus.STOPPED


class TestYahooDataPluginQuote:
    """Tests for get_quote method."""

    @pytest.fixture
    def plugin(self):
        """Create plugin with mocked rate limiter."""
        config = PluginConfig(name="yahoo", rate_limit_per_minute=60)
        plugin = YahooDataPlugin(config)
        plugin._rate_limiter.wait_for_token = AsyncMock()
        return plugin

    @pytest.mark.asyncio
    async def test_get_quote_success(self, plugin: YahooDataPlugin):
        """Test successful quote retrieval."""
        mock_fast_info = {
            "lastPrice": 150.25,
            "open": 149.00,
            "dayHigh": 151.00,
            "dayLow": 148.50,
            "previousClose": 149.50,
            "lastVolume": 1000000,
        }

        mock_info = {
            "symbol": "AAPL",
            "currentPrice": 150.25,
            "open": 149.00,
            "dayHigh": 151.00,
            "dayLow": 148.50,
            "previousClose": 149.50,
            "volume": 1000000,
            "bid": 150.20,
            "bidSize": 100,
            "ask": 150.30,
            "askSize": 200,
            "marketCap": 2500000000000,
            "fiftyTwoWeekHigh": 180.00,
            "fiftyTwoWeekLow": 120.00,
            "averageVolume": 50000000,
        }

        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.fast_info = mock_fast_info
            mock_ticker.info = mock_info
            mock_yf.Ticker.return_value = mock_ticker

            quote = await plugin.get_quote("AAPL")

            assert quote["symbol"] == "AAPL"
            assert quote["last"] == 150.25
            assert quote["open"] == 149.00
            assert quote["high"] == 151.00
            assert quote["low"] == 148.50
            assert quote["volume"] == 1000000
            assert quote["source"] == "yahoo"
            assert quote["change"] is not None
            assert quote["change_percent"] is not None

    @pytest.mark.asyncio
    async def test_get_quote_calculates_change(self, plugin: YahooDataPlugin):
        """Test that change is calculated correctly."""
        mock_fast_info = {"lastPrice": 150.0, "previousClose": 145.0}
        mock_info = {"previousClose": 145.0}

        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.fast_info = mock_fast_info
            mock_ticker.info = mock_info
            mock_yf.Ticker.return_value = mock_ticker

            quote = await plugin.get_quote("AAPL")

            assert quote["change"] == pytest.approx(5.0)
            assert quote["change_percent"] == pytest.approx((5.0 / 145.0) * 100)

    @pytest.mark.asyncio
    async def test_get_quote_caches_ticker(self, plugin: YahooDataPlugin):
        """Test that ticker objects are cached."""
        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.fast_info = {"lastPrice": 150.0}
            mock_ticker.info = {}
            mock_yf.Ticker.return_value = mock_ticker

            await plugin.get_quote("AAPL")
            await plugin.get_quote("AAPL")

            # Ticker should be created only once
            assert mock_yf.Ticker.call_count == 1


class TestYahooDataPluginHistorical:
    """Tests for get_historical method."""

    @pytest.fixture
    def plugin(self):
        """Create plugin with mocked rate limiter."""
        config = PluginConfig(name="yahoo", rate_limit_per_minute=60)
        plugin = YahooDataPlugin(config)
        plugin._rate_limiter.wait_for_token = AsyncMock()
        return plugin

    @pytest.mark.asyncio
    async def test_get_historical_success(self, plugin: YahooDataPlugin):
        """Test successful historical data retrieval."""
        # Create mock DataFrame
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "Open": [149.0, 150.0, 151.0],
                "High": [151.0, 152.0, 153.0],
                "Low": [148.0, 149.0, 150.0],
                "Close": [150.0, 151.0, 152.0],
                "Volume": [1000000, 1100000, 1200000],
            },
            index=dates,
        )

        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = df
            mock_yf.Ticker.return_value = mock_ticker

            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 3)
            bars = await plugin.get_historical("AAPL", start, end, "1d")

            assert len(bars) == 3
            assert bars[0]["symbol"] == "AAPL"
            assert bars[0]["open"] == 149.0
            assert bars[0]["close"] == 150.0
            assert bars[0]["source"] == "yahoo"

    @pytest.mark.asyncio
    async def test_get_historical_empty(self, plugin: YahooDataPlugin):
        """Test handling of empty historical data."""
        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = pd.DataFrame()
            mock_yf.Ticker.return_value = mock_ticker

            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 3)
            bars = await plugin.get_historical("AAPL", start, end, "1d")

            assert bars == []

    @pytest.mark.asyncio
    async def test_get_historical_timeframe_mapping(self, plugin: YahooDataPlugin):
        """Test that timeframe is mapped correctly."""
        df = pd.DataFrame(
            {
                "Open": [149.0],
                "High": [151.0],
                "Low": [148.0],
                "Close": [150.0],
                "Volume": [1000000],
            },
            index=pd.date_range("2024-01-01", periods=1, freq="h"),
        )

        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = df
            mock_yf.Ticker.return_value = mock_ticker

            start = datetime(2024, 1, 1)
            end = datetime(2024, 1, 2)
            await plugin.get_historical("AAPL", start, end, "1h")

            # Verify history was called with correct interval
            mock_ticker.history.assert_called_once()
            call_kwargs = mock_ticker.history.call_args[1]
            assert call_kwargs["interval"] == "1h"


class TestYahooDataPluginCompanyInfo:
    """Tests for get_company_info method."""

    @pytest.fixture
    def plugin(self):
        """Create plugin with mocked rate limiter."""
        config = PluginConfig(name="yahoo", rate_limit_per_minute=60)
        plugin = YahooDataPlugin(config)
        plugin._rate_limiter.wait_for_token = AsyncMock()
        return plugin

    @pytest.mark.asyncio
    async def test_get_company_info(self, plugin: YahooDataPlugin):
        """Test company info retrieval."""
        mock_info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "country": "United States",
            "website": "https://www.apple.com",
            "longBusinessSummary": "Apple designs and manufactures...",
            "fullTimeEmployees": 164000,
            "exchange": "NASDAQ",
            "currency": "USD",
        }

        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = mock_info
            mock_yf.Ticker.return_value = mock_ticker

            info = await plugin.get_company_info("AAPL")

            assert info["symbol"] == "AAPL"
            assert info["name"] == "Apple Inc."
            assert info["sector"] == "Technology"
            assert info["source"] == "yahoo"


class TestYahooDataPluginValidation:
    """Tests for validate_symbol method."""

    @pytest.fixture
    def plugin(self):
        """Create plugin with mocked rate limiter."""
        config = PluginConfig(name="yahoo", rate_limit_per_minute=60)
        plugin = YahooDataPlugin(config)
        return plugin

    @pytest.mark.asyncio
    async def test_validate_symbol_valid(self, plugin: YahooDataPlugin):
        """Test validation of valid symbol."""
        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {"symbol": "AAPL"}
            mock_yf.Ticker.return_value = mock_ticker

            result = await plugin.validate_symbol("AAPL")

            assert result is True

    @pytest.mark.asyncio
    async def test_validate_symbol_invalid(self, plugin: YahooDataPlugin):
        """Test validation of invalid symbol."""
        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = {}  # Empty info for invalid symbol
            mock_yf.Ticker.return_value = mock_ticker

            result = await plugin.validate_symbol("INVALID123")

            assert result is False

    @pytest.mark.asyncio
    async def test_validate_symbol_exception(self, plugin: YahooDataPlugin):
        """Test validation handles exceptions."""
        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.info = property(lambda self: (_ for _ in ()).throw(Exception("API Error")))
            mock_yf.Ticker.return_value = mock_ticker

            # Should return False on exception, not raise
            result = await plugin.validate_symbol("AAPL")

            assert result is False


class TestYahooDataPluginHealthCheck:
    """Tests for health_check method."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when API is responsive."""
        config = PluginConfig(name="yahoo")
        plugin = YahooDataPlugin(config)

        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_ticker = MagicMock()
            mock_ticker.fast_info = {"lastPrice": 500.0}
            mock_yf.Ticker.return_value = mock_ticker

            health = await plugin.health_check()

            assert health.status == PluginStatus.READY
            assert health.latency_ms > 0
            assert "healthy" in health.message.lower()

    @pytest.mark.asyncio
    async def test_health_check_error(self):
        """Test health check when API errors."""
        config = PluginConfig(name="yahoo")
        plugin = YahooDataPlugin(config)

        with patch("ordinis.adapters.market_data.yahoo.yf") as mock_yf:
            mock_yf.Ticker.side_effect = Exception("Connection error")

            health = await plugin.health_check()

            assert health.status == PluginStatus.ERROR
            assert health.error_count > 0
            assert "Connection error" in health.last_error
