"""
Twelve Data market data plugin.

Provides real-time and historical market data from Twelve Data.
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


class TwelveDataPlugin(DataPlugin):
    """
    Twelve Data market data plugin.

    Provides:
    - Real-time quotes
    - Historical time series (intraday and daily)
    - Technical indicators
    - Multiple asset classes (stocks, forex, crypto, ETFs, indices)
    - Fundamental data

    Free tier: 800 API calls/day
    """

    name = "twelvedata"
    version = "1.0.0"
    description = "Twelve Data market data provider"
    capabilities = [
        PluginCapability.READ,
        PluginCapability.REALTIME,
        PluginCapability.HISTORICAL,
    ]

    BASE_URL = "https://api.twelvedata.com"

    def __init__(self, config: PluginConfig):
        super().__init__(config)
        self._session: aiohttp.ClientSession | None = None

    async def initialize(self) -> bool:
        """Initialize the Twelve Data connection."""
        await self._set_status(PluginStatus.INITIALIZING)

        try:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)

            # Test API key with a simple request
            test_result = await self._make_request("/quote", {"symbol": "AAPL"})

            if "price" in test_result or "status" not in test_result:
                await self._set_status(PluginStatus.READY)
                logger.info("Twelve Data plugin initialized successfully")
                return True
            logger.error(f"Twelve Data API key validation failed: {test_result}")
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
        logger.info("Twelve Data plugin shutdown complete")

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
                message="Twelve Data API healthy",
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
        """Make an API request to Twelve Data."""
        await self._rate_limiter.wait_for_token()

        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["apikey"] = self.config.api_key

        async with self._session.get(url, params=params) as response:
            response.raise_for_status()
            data = await response.json()

            # Check for error status
            if isinstance(data, dict):
                if data.get("status") == "error":
                    raise ValueError(f"Twelve Data API error: {data.get('message')}")
                if "code" in data and data["code"] >= 400:
                    raise ValueError(f"Twelve Data API error: {data.get('message')}")

            return data

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float."""
        try:
            return float(value) if value and value != "null" else default
        except (ValueError, TypeError):
            return default

    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Safely convert value to int."""
        try:
            return int(value) if value and value != "null" else default
        except (ValueError, TypeError):
            return default

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get real-time quote for a symbol."""
        result = await self._make_request("/quote", {"symbol": symbol})

        return {
            "symbol": result.get("symbol", symbol),
            "timestamp": datetime.utcnow().isoformat(),
            "last": self._safe_float(result.get("close")),
            "open": self._safe_float(result.get("open")),
            "high": self._safe_float(result.get("high")),
            "low": self._safe_float(result.get("low")),
            "previous_close": self._safe_float(result.get("previous_close")),
            "change": self._safe_float(result.get("change")),
            "change_percent": self._safe_float(result.get("percent_change")),
            "volume": self._safe_int(result.get("volume")),
            "source": "twelvedata",
        }

    async def get_historical(
        self, symbol: str, start: datetime, end: datetime, timeframe: str = "1d"
    ) -> list[dict[str, Any]]:
        """
        Get historical time series data.

        Args:
            symbol: Stock ticker symbol.
            start: Start datetime.
            end: End datetime.
            timeframe: Interval (1min, 5min, 15min, 30min, 45min, 1h, 2h, 4h, 1day, 1week, 1month).

        Returns:
            List of OHLCV bars.
        """
        # Map common timeframes to Twelve Data intervals
        interval_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1h",
            "1d": "1day",
            "1w": "1week",
        }
        interval = interval_map.get(timeframe, "1day")

        params = {
            "symbol": symbol,
            "interval": interval,
            "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
            "outputsize": "5000",
        }

        result = await self._make_request("/time_series", params)

        bars = []
        if "values" in result:
            for bar in result["values"]:
                bars.append(
                    {
                        "symbol": symbol,
                        "timestamp": bar.get("datetime"),
                        "open": self._safe_float(bar.get("open")),
                        "high": self._safe_float(bar.get("high")),
                        "low": self._safe_float(bar.get("low")),
                        "close": self._safe_float(bar.get("close")),
                        "volume": self._safe_int(bar.get("volume")),
                        "source": "twelvedata",
                    }
                )

        # Reverse to get chronological order (Twelve Data returns newest first)
        bars.reverse()
        return bars

    async def get_company(self, symbol: str) -> dict[str, Any]:
        """Get company profile."""
        result = await self._make_request("/profile", {"symbol": symbol})

        return {
            "symbol": result.get("symbol", symbol),
            "name": result.get("name"),
            "exchange": result.get("exchange"),
            "currency": result.get("currency"),
            "country": result.get("country"),
            "sector": result.get("sector"),
            "industry": result.get("industry"),
            "description": result.get("description"),
            "type": result.get("type"),
            "website": result.get("website"),
            "ceo": result.get("ceo"),
            "employees": result.get("employees"),
            "address": result.get("address"),
            "phone": result.get("phone"),
            "source": "twelvedata",
        }

    async def get_earnings(self, symbol: str) -> dict[str, Any]:
        """Get earnings data."""
        result = await self._make_request("/earnings", {"symbol": symbol})

        return {
            "symbol": symbol,
            "earnings": result.get("earnings", []),
            "source": "twelvedata",
        }

    async def get_dividends(self, symbol: str) -> dict[str, Any]:
        """Get dividend data."""
        result = await self._make_request("/dividends", {"symbol": symbol})

        return {
            "symbol": symbol,
            "dividends": result.get("dividends", []),
            "source": "twelvedata",
        }

    async def get_statistics(self, symbol: str) -> dict[str, Any]:
        """Get key statistics."""
        result = await self._make_request("/statistics", {"symbol": symbol})

        statistics = result.get("statistics", {})

        return {
            "symbol": symbol,
            "valuations_metrics": statistics.get("valuations_metrics", {}),
            "financials": statistics.get("financials", {}),
            "stock_statistics": statistics.get("stock_statistics", {}),
            "stock_price_summary": statistics.get("stock_price_summary", {}),
            "dividends_and_splits": statistics.get("dividends_and_splits", {}),
            "source": "twelvedata",
        }

    async def get_technical_indicator(
        self,
        symbol: str,
        indicator: str,
        interval: str = "1day",
        time_period: int = 14,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Get technical indicator values.

        Args:
            symbol: Stock ticker symbol.
            indicator: Indicator name (sma, ema, rsi, macd, bbands, etc.).
            interval: Time interval.
            time_period: Lookback period.
            **kwargs: Additional indicator-specific parameters.

        Returns:
            Technical indicator data.
        """
        params = {
            "symbol": symbol,
            "interval": interval,
            "time_period": time_period,
            **kwargs,
        }

        result = await self._make_request(f"/{indicator}", params)

        return {
            "symbol": symbol,
            "indicator": indicator,
            "values": result.get("values", []),
            "meta": result.get("meta", {}),
            "source": "twelvedata",
        }

    async def search_symbols(self, query: str) -> list[dict[str, Any]]:
        """Search for symbols."""
        result = await self._make_request("/symbol_search", {"symbol": query})

        matches = []
        for match in result.get("data", []):
            matches.append(
                {
                    "symbol": match.get("symbol"),
                    "name": match.get("instrument_name"),
                    "type": match.get("instrument_type"),
                    "exchange": match.get("exchange"),
                    "mic_code": match.get("mic_code"),
                    "currency": match.get("currency"),
                    "country": match.get("country"),
                }
            )

        return matches
