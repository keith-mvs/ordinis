"""
Polygon.io / Massive market data plugin.

Provides real-time and historical market data from Polygon.io (now Massive).
Note: As of October 2025, Polygon.io rebranded to Massive.
Both api.polygon.io and api.massive.com are supported.
"""

import asyncio
from datetime import datetime
import logging
from typing import Any

import aiohttp

from ..base import (
    DataPlugin,
    PluginCapability,
    PluginConfig,
    PluginHealth,
    PluginStatus,
)

logger = logging.getLogger(__name__)


class PolygonDataPlugin(DataPlugin):
    """
    Polygon.io market data plugin.

    Provides:
    - Real-time quotes
    - Historical OHLCV data
    - Options chains
    - News
    - Reference data
    """

    name = "polygon"
    version = "1.0.0"
    description = "Polygon.io market data provider"
    capabilities = [
        PluginCapability.READ,
        PluginCapability.REALTIME,
        PluginCapability.HISTORICAL,
        PluginCapability.STREAM,
    ]

    BASE_URL = "https://api.polygon.io"

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self._session: aiohttp.ClientSession | None = None
        self._ws_connection = None

    async def initialize(self) -> bool:
        """Initialize the Polygon connection."""
        await self._set_status(PluginStatus.INITIALIZING)

        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

            # Validate API key with a test request
            test_result = await self._make_request("/v2/aggs/ticker/AAPL/prev")

            if test_result.get("status") == "OK" or test_result.get("resultsCount", 0) > 0:
                await self._set_status(PluginStatus.READY)
                logger.info("Polygon plugin initialized successfully")
                return True
            logger.error(f"Polygon API key validation failed: {test_result}")
            await self._set_status(PluginStatus.ERROR)
            return False

        except Exception as e:
            await self._handle_error(e)
            return False

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        if self._ws_connection:
            await self._ws_connection.close()

        if self._session:
            await self._session.close()

        await self._set_status(PluginStatus.STOPPED)
        logger.info("Polygon plugin shutdown complete")

    async def health_check(self) -> PluginHealth:
        """Check plugin health."""
        start_time = datetime.utcnow()

        try:
            # Simple health check request
            result = await self._make_request("/v1/marketstatus/now")
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            self._health = PluginHealth(
                status=PluginStatus.READY,
                last_check=datetime.utcnow(),
                latency_ms=latency,
                message=f"Market status: {result.get('market', 'unknown')}",
            )

        except Exception as e:
            self._health = PluginHealth(
                status=PluginStatus.ERROR,
                last_check=datetime.utcnow(),
                latency_ms=0,
                error_count=self._health.error_count + 1,
                last_error=str(e),
            )

        return self._health

    async def _make_request(self, endpoint: str, params: dict | None = None) -> dict[str, Any]:
        """Make an API request to Polygon."""
        await self._rate_limiter.wait_for_token()

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["apiKey"] = self.config.api_key

        async with self._session.get(url, params=params) as response:
            if response.status == 429:
                # Rate limited
                logger.warning("Polygon rate limit hit, backing off")
                await asyncio.sleep(60)
                return await self._make_request(endpoint, params)

            response.raise_for_status()
            return await response.json()

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Quote data including bid, ask, last price.
        """
        # Get last trade
        trade_data = await self._make_request(f"/v2/last/trade/{symbol}")

        # Get last quote
        quote_data = await self._make_request(f"/v2/last/nbbo/{symbol}")

        last_trade = trade_data.get("results", {})
        last_quote = quote_data.get("results", {})

        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "last": last_trade.get("p"),
            "last_size": last_trade.get("s"),
            "bid": last_quote.get("p"),
            "bid_size": last_quote.get("s"),
            "ask": last_quote.get("P"),
            "ask_size": last_quote.get("S"),
            "source": "polygon",
        }

    async def get_historical(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1d"
    ) -> list[dict[str, Any]]:
        """
        Get historical OHLCV data.

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime.
            end: End datetime.
            timeframe: Bar timeframe (1m, 5m, 15m, 1h, 1d).

        Returns:
            List of OHLCV bars.
        """
        # Map timeframe to Polygon format
        timeframe_map = {
            "1m": ("minute", 1),
            "5m": ("minute", 5),
            "15m": ("minute", 15),
            "30m": ("minute", 30),
            "1h": ("hour", 1),
            "1d": ("day", 1),
            "1w": ("week", 1),
        }

        unit, multiplier = timeframe_map.get(timeframe, ("day", 1))

        endpoint = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{unit}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

        params = {"adjusted": "true", "sort": "asc", "limit": 50000}

        result = await self._make_request(endpoint, params)

        bars = []
        for bar in result.get("results", []):
            bars.append(
                {
                    "symbol": symbol,
                    "timestamp": datetime.fromtimestamp(bar["t"] / 1000).isoformat(),
                    "open": bar["o"],
                    "high": bar["h"],
                    "low": bar["l"],
                    "close": bar["c"],
                    "volume": bar["v"],
                    "vwap": bar.get("vw"),
                    "transactions": bar.get("n"),
                    "source": "polygon",
                }
            )

        return bars

    async def get_previous_close(self, symbol: str) -> dict[str, Any]:
        """Get previous day's close data."""
        result = await self._make_request(f"/v2/aggs/ticker/{symbol}/prev")

        if result.get("results"):
            bar = result["results"][0]
            return {
                "symbol": symbol,
                "date": datetime.fromtimestamp(bar["t"] / 1000).strftime("%Y-%m-%d"),
                "open": bar["o"],
                "high": bar["h"],
                "low": bar["l"],
                "close": bar["c"],
                "volume": bar["v"],
                "vwap": bar.get("vw"),
                "source": "polygon",
            }

        return {}

    async def get_snapshot(self, symbol: str) -> dict[str, Any]:
        """Get a snapshot of current data for a symbol."""
        result = await self._make_request(f"/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}")

        if result.get("ticker"):
            ticker = result["ticker"]
            return {
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat(),
                "day": ticker.get("day", {}),
                "prev_day": ticker.get("prevDay", {}),
                "last_trade": ticker.get("lastTrade", {}),
                "last_quote": ticker.get("lastQuote", {}),
                "min": ticker.get("min", {}),
                "todays_change": ticker.get("todaysChange"),
                "todays_change_perc": ticker.get("todaysChangePerc"),
                "source": "polygon",
            }

        return {}

    async def get_market_status(self) -> dict[str, Any]:
        """Get current market status."""
        result = await self._make_request("/v1/marketstatus/now")

        return {
            "market": result.get("market"),
            "server_time": result.get("serverTime"),
            "exchanges": result.get("exchanges", {}),
            "currencies": result.get("currencies", {}),
            "source": "polygon",
        }

    async def get_ticker_details(self, symbol: str) -> dict[str, Any]:
        """Get detailed information about a ticker."""
        result = await self._make_request(f"/v3/reference/tickers/{symbol}")

        if result.get("results"):
            details = result["results"]
            return {
                "symbol": symbol,
                "name": details.get("name"),
                "market": details.get("market"),
                "locale": details.get("locale"),
                "primary_exchange": details.get("primary_exchange"),
                "type": details.get("type"),
                "currency": details.get("currency_name"),
                "cik": details.get("cik"),
                "composite_figi": details.get("composite_figi"),
                "share_class_figi": details.get("share_class_figi"),
                "market_cap": details.get("market_cap"),
                "sic_code": details.get("sic_code"),
                "sic_description": details.get("sic_description"),
                "homepage": details.get("homepage_url"),
                "source": "polygon",
            }

        return {}

    async def get_options_chain(
        self,
        symbol: str,
        expiration: str | None = None,
        strike_price: float | None = None,
        contract_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get options chain for an underlying.

        Args:
            symbol: Underlying ticker symbol.
            expiration: Expiration date (YYYY-MM-DD).
            strike_price: Filter by strike price.
            contract_type: 'call' or 'put'.

        Returns:
            Options chain data.
        """
        params = {"underlying_ticker": symbol, "limit": 250}

        if expiration:
            params["expiration_date"] = expiration
        if strike_price:
            params["strike_price"] = strike_price
        if contract_type:
            params["contract_type"] = contract_type

        result = await self._make_request("/v3/reference/options/contracts", params)

        contracts = []
        for contract in result.get("results", []):
            contracts.append(
                {
                    "ticker": contract.get("ticker"),
                    "underlying": contract.get("underlying_ticker"),
                    "contract_type": contract.get("contract_type"),
                    "strike_price": contract.get("strike_price"),
                    "expiration_date": contract.get("expiration_date"),
                    "shares_per_contract": contract.get("shares_per_contract"),
                    "exercise_style": contract.get("exercise_style"),
                    "primary_exchange": contract.get("primary_exchange"),
                }
            )

        return {
            "symbol": symbol,
            "contracts": contracts,
            "count": len(contracts),
            "source": "polygon",
        }

    async def get_news(self, symbol: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Get news articles."""
        params = {"limit": limit, "order": "desc"}

        if symbol:
            params["ticker"] = symbol

        result = await self._make_request("/v2/reference/news", params)

        articles = []
        for article in result.get("results", []):
            articles.append(
                {
                    "id": article.get("id"),
                    "title": article.get("title"),
                    "author": article.get("author"),
                    "published": article.get("published_utc"),
                    "article_url": article.get("article_url"),
                    "tickers": article.get("tickers", []),
                    "description": article.get("description"),
                    "keywords": article.get("keywords", []),
                    "source": article.get("publisher", {}).get("name"),
                    "image_url": article.get("image_url"),
                }
            )

        return articles
