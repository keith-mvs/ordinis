"""
Finnhub market data plugin.

Provides real-time and historical market data from Finnhub.
"""

from datetime import datetime
import logging
from typing import Any

import aiohttp

from ordinis.plugins.base import (
    DataPlugin,
    PluginCapability,
    PluginConfig,
    PluginHealth,
    PluginStatus,
)

logger = logging.getLogger(__name__)


class FinnhubDataPlugin(DataPlugin):
    """
    Finnhub market data plugin.

    Provides:
    - Real-time quotes
    - Historical candles (up to 30 years)
    - Company profiles
    - News and sentiment
    - Earnings and financials

    Free tier: 60 API calls/minute
    """

    name = "finnhub"
    version = "1.0.0"
    description = "Finnhub market data provider"
    capabilities = [
        PluginCapability.READ,
        PluginCapability.REALTIME,
        PluginCapability.HISTORICAL,
        PluginCapability.STREAM,
    ]

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self._session: aiohttp.ClientSession | None = None

    async def initialize(self) -> bool:
        """Initialize the Finnhub connection."""
        await self._set_status(PluginStatus.INITIALIZING)

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

            # Test API key with a simple request
            test_result = await self._make_request("/quote", {"symbol": "AAPL"})

            if "c" in test_result or "error" not in test_result:
                await self._set_status(PluginStatus.READY)
                logger.info("Finnhub plugin initialized successfully")
                return True
            logger.error(f"Finnhub API key validation failed: {test_result}")
            await self._set_status(PluginStatus.ERROR)
            return False

        except Exception as e:
            await self._handle_error(e)
            return False

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        if self._session:
            await self._session.close()

        await self._set_status(PluginStatus.STOPPED)
        logger.info("Finnhub plugin shutdown complete")

    async def health_check(self) -> PluginHealth:
        """Check plugin health."""
        start_time = datetime.utcnow()

        try:
            await self._make_request("/quote", {"symbol": "AAPL"})
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            self._health = PluginHealth(
                status=PluginStatus.READY,
                last_check=datetime.utcnow(),
                latency_ms=latency,
                message="Finnhub API healthy",
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

    async def _make_request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make an API request to Finnhub."""
        await self._rate_limiter.wait_for_token()

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}

        headers = {"X-Finnhub-Token": self.config.api_key}

        async with self._session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()

            # Check for error message
            if isinstance(data, dict) and "error" in data:
                raise ValueError(f"Finnhub API error: {data['error']}")

            return data

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """
        Get real-time quote for a symbol.

        Response fields:
        - c: Current price
        - d: Change
        - dp: Percent change
        - h: High price of the day
        - l: Low price of the day
        - o: Open price of the day
        - pc: Previous close price
        """
        result = await self._make_request("/quote", {"symbol": symbol})

        return {
            "symbol": symbol,
            "timestamp": datetime.utcnow().isoformat(),
            "last": result.get("c", 0),
            "open": result.get("o", 0),
            "high": result.get("h", 0),
            "low": result.get("l", 0),
            "previous_close": result.get("pc", 0),
            "change": result.get("d", 0),
            "change_percent": result.get("dp", 0),
            "source": "finnhub",
        }

    async def get_historical(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1d"
    ) -> list[dict[str, Any]]:
        """
        Get historical candle data.

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime.
            end: End datetime.
            timeframe: Resolution (1, 5, 15, 30, 60, D, W, M).

        Returns:
            List of OHLCV bars.
        """
        # Map timeframe to Finnhub resolution
        resolution_map = {
            "1m": "1",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "60m": "60",
            "1h": "60",
            "1d": "D",
            "1w": "W",
            "1M": "M",
        }
        resolution = resolution_map.get(timeframe, "D")

        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
        }

        result = await self._make_request("/stock/candle", params)

        bars = []
        if result.get("s") == "ok":
            timestamps = result.get("t", [])
            opens = result.get("o", [])
            highs = result.get("h", [])
            lows = result.get("l", [])
            closes = result.get("c", [])
            volumes = result.get("v", [])

            for i in range(len(timestamps)):
                bars.append(
                    {
                        "symbol": symbol,
                        "timestamp": datetime.fromtimestamp(timestamps[i]).isoformat(),
                        "open": opens[i],
                        "high": highs[i],
                        "low": lows[i],
                        "close": closes[i],
                        "volume": volumes[i],
                        "source": "finnhub",
                    }
                )

        return bars

    async def get_company(self, symbol: str) -> dict[str, Any]:
        """Get company profile."""
        result = await self._make_request("/stock/profile2", {"symbol": symbol})

        return {
            "symbol": result.get("ticker", symbol),
            "name": result.get("name"),
            "country": result.get("country"),
            "currency": result.get("currency"),
            "exchange": result.get("exchange"),
            "ipo": result.get("ipo"),
            "market_cap": result.get("marketCapitalization"),
            "shares_outstanding": result.get("shareOutstanding"),
            "logo": result.get("logo"),
            "phone": result.get("phone"),
            "website": result.get("weburl"),
            "industry": result.get("finnhubIndustry"),
            "source": "finnhub",
        }

    async def get_news(self, symbol: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Get news articles."""
        if symbol:
            # Company news
            from_date = datetime.utcnow().strftime("%Y-%m-%d")
            to_date = from_date
            result = await self._make_request(
                "/company-news", {"symbol": symbol, "from": from_date, "to": to_date}
            )
        else:
            # General market news
            result = await self._make_request("/news", {"category": "general"})

        articles = []
        for article in result[:limit]:
            articles.append(
                {
                    "id": article.get("id"),
                    "category": article.get("category"),
                    "datetime": article.get("datetime"),
                    "headline": article.get("headline"),
                    "image": article.get("image"),
                    "related": article.get("related"),
                    "source": article.get("source"),
                    "summary": article.get("summary"),
                    "url": article.get("url"),
                }
            )

        return articles

    async def get_earnings(self, symbol: str) -> dict[str, Any]:
        """Get earnings data."""
        result = await self._make_request("/stock/earnings", {"symbol": symbol})

        return {
            "symbol": symbol,
            "earnings": result,
            "source": "finnhub",
        }

    async def get_sentiment(self, symbol: str) -> dict[str, Any]:
        """Get social sentiment and news sentiment."""
        # News sentiment
        news_sentiment = await self._make_request("/news-sentiment", {"symbol": symbol})

        # Social sentiment (Reddit and Twitter)
        social_sentiment = await self._make_request("/stock/social-sentiment", {"symbol": symbol})

        return {
            "symbol": symbol,
            "news_sentiment": news_sentiment,
            "social_sentiment": social_sentiment,
            "source": "finnhub",
        }

    async def get_recommendation(self, symbol: str) -> dict[str, Any]:
        """Get analyst recommendation trends."""
        result = await self._make_request("/stock/recommendation", {"symbol": symbol})

        return {
            "symbol": symbol,
            "recommendations": result,
            "source": "finnhub",
        }

    async def get_price_target(self, symbol: str) -> dict[str, Any]:
        """Get price target consensus."""
        result = await self._make_request("/stock/price-target", {"symbol": symbol})

        return {
            "symbol": result.get("symbol", symbol),
            "target_high": result.get("targetHigh"),
            "target_low": result.get("targetLow"),
            "target_mean": result.get("targetMean"),
            "target_median": result.get("targetMedian"),
            "last_updated": result.get("lastUpdated"),
            "source": "finnhub",
        }

    async def search_symbols(self, query: str) -> list[dict[str, Any]]:
        """Search for symbols."""
        result = await self._make_request("/search", {"q": query})

        matches = []
        for match in result.get("result", []):
            matches.append(
                {
                    "symbol": match.get("symbol"),
                    "description": match.get("description"),
                    "display_symbol": match.get("displaySymbol"),
                    "type": match.get("type"),
                }
            )

        return matches
