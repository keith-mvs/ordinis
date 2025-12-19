"""
Alpha Vantage market data plugin.

Provides real-time and historical market data from Alpha Vantage.
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


class AlphaVantageDataPlugin(DataPlugin):
    """
    Alpha Vantage market data plugin.

    Provides:
    - Real-time quotes
    - Historical OHLCV data (daily, intraday)
    - Company information
    - Technical indicators

    Free tier: 25 API calls/day (500/month with registration)
    """

    name = "alphavantage"
    version = "1.0.0"
    description = "Alpha Vantage market data provider"
    capabilities = [
        PluginCapability.READ,
        PluginCapability.REALTIME,
        PluginCapability.HISTORICAL,
    ]

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self._session: aiohttp.ClientSession | None = None

    async def initialize(self) -> bool:
        """Initialize the Alpha Vantage connection."""
        await self._set_status(PluginStatus.INITIALIZING)

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

            # Test API key with a simple request
            test_result = await self._make_request({"function": "GLOBAL_QUOTE", "symbol": "AAPL"})

            if "Global Quote" in test_result or "Error Message" not in test_result:
                await self._set_status(PluginStatus.READY)
                logger.info("Alpha Vantage plugin initialized successfully")
                return True
            logger.error(f"Alpha Vantage API key validation failed: {test_result}")
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
        logger.info("Alpha Vantage plugin shutdown complete")

    async def health_check(self) -> PluginHealth:
        """Check plugin health."""
        start_time = datetime.utcnow()

        try:
            await self._make_request({"function": "GLOBAL_QUOTE", "symbol": "AAPL"})
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000

            self._health = PluginHealth(
                status=PluginStatus.READY,
                last_check=datetime.utcnow(),
                latency_ms=latency,
                message="Alpha Vantage API healthy",
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

    async def _make_request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Make an API request to Alpha Vantage."""
        await self._rate_limiter.wait_for_token()

        params["apikey"] = self.config.api_key

        async with self._session.get(self.BASE_URL, params=params) as response:
            response.raise_for_status()
            data = await response.json()

            # Check for rate limit message
            if "Note" in data:
                logger.warning(f"Alpha Vantage rate limit note: {data['Note']}")
            if "Error Message" in data:
                raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")

            return data

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get real-time quote for a symbol."""
        result = await self._make_request({"function": "GLOBAL_QUOTE", "symbol": symbol})

        quote = result.get("Global Quote", {})

        def safe_float(value: Any, default: float = 0.0) -> float:
            """Safely convert value to float."""
            try:
                return float(value) if value else default
            except (ValueError, TypeError):
                return default

        def safe_int(value: Any, default: int = 0) -> int:
            """Safely convert value to int."""
            try:
                return int(value) if value else default
            except (ValueError, TypeError):
                return default

        return {
            "symbol": quote.get("01. symbol", symbol),
            "timestamp": datetime.utcnow().isoformat(),
            "last": safe_float(quote.get("05. price")),
            "open": safe_float(quote.get("02. open")),
            "high": safe_float(quote.get("03. high")),
            "low": safe_float(quote.get("04. low")),
            "previous_close": safe_float(quote.get("08. previous close")),
            "change": safe_float(quote.get("09. change")),
            "change_percent": safe_float(quote.get("10. change percent", "0%").rstrip("%")),
            "volume": safe_int(quote.get("06. volume")),
            "source": "alphavantage",
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
            timeframe: Bar timeframe (1m, 5m, 15m, 30m, 60m, 1d).

        Returns:
            List of OHLCV bars.
        """
        # Map timeframe to Alpha Vantage intervals
        if timeframe == "1d":
            result = await self._make_request(
                {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "full"}
            )
            time_series = result.get("Time Series (Daily)", {})
        else:
            # Intraday data
            interval_map = {
                "1m": "1min",
                "5m": "5min",
                "15m": "15min",
                "30m": "30min",
                "60m": "60min",
            }
            interval = interval_map.get(timeframe, "5min")

            result = await self._make_request(
                {
                    "function": "TIME_SERIES_INTRADAY",
                    "symbol": symbol,
                    "interval": interval,
                    "outputsize": "full",
                }
            )
            time_series = result.get(f"Time Series ({interval})", {})

        bars = []
        for timestamp_str, bar_data in time_series.items():
            bar_time = datetime.fromisoformat(timestamp_str.replace(" ", "T"))

            if start <= bar_time <= end:
                bars.append(
                    {
                        "symbol": symbol,
                        "timestamp": timestamp_str,
                        "open": float(bar_data.get("1. open", 0)),
                        "high": float(bar_data.get("2. high", 0)),
                        "low": float(bar_data.get("3. low", 0)),
                        "close": float(bar_data.get("4. close", 0)),
                        "volume": int(bar_data.get("5. volume", 0)),
                        "source": "alphavantage",
                    }
                )

        # Sort by timestamp ascending
        bars.sort(key=lambda x: x["timestamp"])
        return bars

    async def get_company(self, symbol: str) -> dict[str, Any]:
        """Get company information."""
        result = await self._make_request({"function": "OVERVIEW", "symbol": symbol})

        return {
            "symbol": result.get("Symbol", symbol),
            "name": result.get("Name"),
            "exchange": result.get("Exchange"),
            "currency": result.get("Currency"),
            "country": result.get("Country"),
            "sector": result.get("Sector"),
            "industry": result.get("Industry"),
            "description": result.get("Description"),
            "market_cap": result.get("MarketCapitalization"),
            "pe_ratio": result.get("PERatio"),
            "peg_ratio": result.get("PEGRatio"),
            "book_value": result.get("BookValue"),
            "dividend_per_share": result.get("DividendPerShare"),
            "dividend_yield": result.get("DividendYield"),
            "eps": result.get("EPS"),
            "revenue_per_share": result.get("RevenuePerShareTTM"),
            "profit_margin": result.get("ProfitMargin"),
            "operating_margin": result.get("OperatingMarginTTM"),
            "return_on_assets": result.get("ReturnOnAssetsTTM"),
            "return_on_equity": result.get("ReturnOnEquityTTM"),
            "revenue": result.get("RevenueTTM"),
            "gross_profit": result.get("GrossProfitTTM"),
            "diluted_eps": result.get("DilutedEPSTTM"),
            "quarterly_earnings_growth": result.get("QuarterlyEarningsGrowthYOY"),
            "quarterly_revenue_growth": result.get("QuarterlyRevenueGrowthYOY"),
            "analyst_target_price": result.get("AnalystTargetPrice"),
            "trailing_pe": result.get("TrailingPE"),
            "forward_pe": result.get("ForwardPE"),
            "price_to_sales": result.get("PriceToSalesRatioTTM"),
            "price_to_book": result.get("PriceToBookRatio"),
            "ev_to_revenue": result.get("EVToRevenue"),
            "ev_to_ebitda": result.get("EVToEBITDA"),
            "beta": result.get("Beta"),
            "week_52_high": result.get("52WeekHigh"),
            "week_52_low": result.get("52WeekLow"),
            "50_day_ma": result.get("50DayMovingAverage"),
            "200_day_ma": result.get("200DayMovingAverage"),
            "shares_outstanding": result.get("SharesOutstanding"),
            "source": "alphavantage",
        }

    async def get_earnings(self, symbol: str) -> dict[str, Any]:
        """Get earnings data."""
        result = await self._make_request({"function": "EARNINGS", "symbol": symbol})

        return {
            "symbol": symbol,
            "annual_earnings": result.get("annualEarnings", []),
            "quarterly_earnings": result.get("quarterlyEarnings", []),
            "source": "alphavantage",
        }

    async def search_symbols(self, keywords: str) -> list[dict[str, Any]]:
        """Search for symbols by keywords."""
        result = await self._make_request({"function": "SYMBOL_SEARCH", "keywords": keywords})

        matches = []
        for match in result.get("bestMatches", []):
            matches.append(
                {
                    "symbol": match.get("1. symbol"),
                    "name": match.get("2. name"),
                    "type": match.get("3. type"),
                    "region": match.get("4. region"),
                    "market_open": match.get("5. marketOpen"),
                    "market_close": match.get("6. marketClose"),
                    "timezone": match.get("7. timezone"),
                    "currency": match.get("8. currency"),
                    "match_score": match.get("9. matchScore"),
                }
            )

        return matches
