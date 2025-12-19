"""
IEX Cloud market data plugin.

Provides backup/alternative market data from IEX Cloud.
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


class IEXDataPlugin(DataPlugin):
    """
    IEX Cloud market data plugin.

    Provides:
    - Real-time quotes
    - Historical data
    - Company information
    - Financial data
    """

    name = "iex"
    version = "1.0.0"
    description = "IEX Cloud market data provider"
    capabilities = [
        PluginCapability.READ,
        PluginCapability.REALTIME,
        PluginCapability.HISTORICAL,
    ]

    BASE_URL = "https://cloud.iexapis.com/stable"
    SANDBOX_URL = "https://sandbox.iexapis.com/stable"

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self._session: aiohttp.ClientSession | None = None
        self._use_sandbox = config.extra.get("sandbox", False)

    @property
    def api_url(self) -> str:
        """Get the appropriate API URL."""
        return self.SANDBOX_URL if self._use_sandbox else self.BASE_URL

    async def initialize(self) -> bool:
        """Initialize the IEX connection."""
        await self._set_status(PluginStatus.INITIALIZING)

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

            # Test API key with a simple request
            test_result = await self._make_request("/stock/AAPL/quote")

            if test_result.get("symbol") == "AAPL":
                await self._set_status(PluginStatus.READY)
                logger.info("IEX plugin initialized successfully")
                return True
            logger.error("IEX API key validation failed")
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
        logger.info("IEX plugin shutdown complete")

    async def health_check(self) -> PluginHealth:
        """Check plugin health."""
        start_time = datetime.utcnow()

        try:
            await self._make_request("/status")
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            self._health = PluginHealth(
                status=PluginStatus.READY,
                last_check=datetime.utcnow(),
                latency_ms=latency,
                message="IEX API healthy",
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

    async def _make_request(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Make an API request to IEX."""
        await self._rate_limiter.wait_for_token()

        url = f"{self.api_url}{endpoint}"
        params = params or {}
        params["token"] = self.config.api_key

        async with self._session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get real-time quote for a symbol."""
        result = await self._make_request(f"/stock/{symbol}/quote")

        return {
            "symbol": result.get("symbol"),
            "timestamp": datetime.utcnow().isoformat(),
            "last": result.get("latestPrice"),
            "last_size": result.get("latestVolume"),
            "bid": result.get("iexBidPrice"),
            "bid_size": result.get("iexBidSize"),
            "ask": result.get("iexAskPrice"),
            "ask_size": result.get("iexAskSize"),
            "open": result.get("open"),
            "high": result.get("high"),
            "low": result.get("low"),
            "close": result.get("close"),
            "previous_close": result.get("previousClose"),
            "change": result.get("change"),
            "change_percent": result.get("changePercent"),
            "volume": result.get("volume"),
            "avg_volume": result.get("avgTotalVolume"),
            "market_cap": result.get("marketCap"),
            "pe_ratio": result.get("peRatio"),
            "week_52_high": result.get("week52High"),
            "week_52_low": result.get("week52Low"),
            "source": "iex",
        }

    async def get_historical(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1d"
    ) -> list[dict[str, Any]]:
        """Get historical OHLCV data."""
        # IEX uses range-based endpoints
        days = (end - start).days

        if days <= 5:
            range_param = "5d"
        elif days <= 30:
            range_param = "1m"
        elif days <= 90:
            range_param = "3m"
        elif days <= 180:
            range_param = "6m"
        elif days <= 365:
            range_param = "1y"
        elif days <= 730:
            range_param = "2y"
        else:
            range_param = "5y"

        result = await self._make_request(f"/stock/{symbol}/chart/{range_param}")

        bars = []
        for bar in result:
            bar_date = datetime.strptime(bar["date"], "%Y-%m-%d")
            if start <= bar_date <= end:
                bars.append(
                    {
                        "symbol": symbol,
                        "timestamp": bar["date"],
                        "open": bar.get("open"),
                        "high": bar.get("high"),
                        "low": bar.get("low"),
                        "close": bar.get("close"),
                        "volume": bar.get("volume"),
                        "change": bar.get("change"),
                        "change_percent": bar.get("changePercent"),
                        "source": "iex",
                    }
                )

        return bars

    async def get_company(self, symbol: str) -> dict[str, Any]:
        """Get company information."""
        result = await self._make_request(f"/stock/{symbol}/company")

        return {
            "symbol": result.get("symbol"),
            "name": result.get("companyName"),
            "exchange": result.get("exchange"),
            "industry": result.get("industry"),
            "sector": result.get("sector"),
            "description": result.get("description"),
            "ceo": result.get("CEO"),
            "employees": result.get("employees"),
            "website": result.get("website"),
            "city": result.get("city"),
            "state": result.get("state"),
            "country": result.get("country"),
            "source": "iex",
        }

    async def get_financials(self, symbol: str, period: str = "quarterly") -> dict[str, Any]:
        """Get financial statements."""
        result = await self._make_request(f"/stock/{symbol}/financials", {"period": period})

        return {
            "symbol": symbol,
            "period": period,
            "financials": result.get("financials", []),
            "source": "iex",
        }

    async def get_stats(self, symbol: str) -> dict[str, Any]:
        """Get key statistics."""
        result = await self._make_request(f"/stock/{symbol}/stats")

        return {
            "symbol": symbol,
            "market_cap": result.get("marketcap"),
            "shares_outstanding": result.get("sharesOutstanding"),
            "float": result.get("float"),
            "avg_30_volume": result.get("avg30Volume"),
            "avg_10_volume": result.get("avg10Volume"),
            "employees": result.get("employees"),
            "ttm_eps": result.get("ttmEPS"),
            "ttm_dividend_rate": result.get("ttmDividendRate"),
            "dividend_yield": result.get("dividendYield"),
            "next_dividend_date": result.get("nextDividendDate"),
            "ex_dividend_date": result.get("exDividendDate"),
            "next_earnings_date": result.get("nextEarningsDate"),
            "pe_ratio": result.get("peRatio"),
            "beta": result.get("beta"),
            "week_52_high": result.get("week52high"),
            "week_52_low": result.get("week52low"),
            "week_52_change": result.get("week52change"),
            "source": "iex",
        }

    async def get_earnings(self, symbol: str, last: int = 4) -> dict[str, Any]:
        """Get earnings data."""
        result = await self._make_request(f"/stock/{symbol}/earnings", {"last": last})

        return {
            "symbol": symbol,
            "earnings": result.get("earnings", []),
            "source": "iex",
        }

    async def get_news(self, symbol: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Get news articles."""
        if symbol:
            endpoint = f"/stock/{symbol}/news/last/{limit}"
        else:
            endpoint = f"/news/last/{limit}"

        result = await self._make_request(endpoint)

        articles = []
        for article in result:
            articles.append(
                {
                    "datetime": article.get("datetime"),
                    "headline": article.get("headline"),
                    "source": article.get("source"),
                    "url": article.get("url"),
                    "summary": article.get("summary"),
                    "related": article.get("related"),
                    "image": article.get("image"),
                    "lang": article.get("lang"),
                    "has_paywall": article.get("hasPaywall"),
                }
            )

        return articles
