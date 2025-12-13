"""
Yahoo Finance market data plugin.

Provides free market data via the yfinance library.
"""

import asyncio
from datetime import datetime
import logging
from typing import Any

from ordinis.plugins.base import (
    DataPlugin,
    PluginCapability,
    PluginConfig,
    PluginHealth,
    PluginStatus,
)

logger = logging.getLogger(__name__)

# Optional import - yfinance may not be installed
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None


class YahooDataPlugin(DataPlugin):
    """
    Yahoo Finance market data plugin.

    Provides free market data using the yfinance library:
    - Real-time quotes (delayed ~15 minutes for free tier)
    - Historical OHLCV data (daily, intraday)
    - Company information
    - Dividends and splits

    No API key required.

    Note: Yahoo Finance has rate limits and may block excessive requests.
    Recommended to use caching wrapper for production use.

    Example:
        config = PluginConfig(name="yahoo")
        plugin = YahooDataPlugin(config)
        await plugin.initialize()
        quote = await plugin.get_quote("AAPL")
    """

    name = "yahoo"
    version = "1.0.0"
    description = "Yahoo Finance market data provider (free, no API key)"
    capabilities = [
        PluginCapability.READ,
        PluginCapability.REALTIME,
        PluginCapability.HISTORICAL,
    ]

    # Timeframe mapping from our standard to yfinance intervals
    TIMEFRAME_MAP = {
        "1m": "1m",
        "2m": "2m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "90m": "90m",
        "1d": "1d",
        "5d": "5d",
        "1wk": "1wk",
        "1mo": "1mo",
        "3mo": "3mo",
    }

    def __init__(self, config: PluginConfig):
        """Initialize the Yahoo Finance plugin.

        Args:
            config: Plugin configuration. No API key needed for Yahoo.
        """
        super().__init__(config)
        self._ticker_cache: dict[str, Any] = {}

    async def initialize(self) -> bool:
        """Initialize the Yahoo Finance connection."""
        await self._set_status(PluginStatus.INITIALIZING)

        if not YFINANCE_AVAILABLE:
            logger.error("yfinance package not installed. Run: pip install yfinance")
            await self._set_status(PluginStatus.ERROR)
            self._health.last_error = "yfinance package not installed"
            return False

        try:
            # Test with a simple request
            test_ticker = yf.Ticker("AAPL")
            info = await asyncio.to_thread(lambda: test_ticker.info)

            if info and "symbol" in info:
                await self._set_status(PluginStatus.READY)
                logger.info("Yahoo Finance plugin initialized successfully")
                return True

            logger.error("Yahoo Finance test request failed")
            await self._set_status(PluginStatus.ERROR)
            return False

        except Exception as e:
            await self._handle_error(e)
            return False

    async def shutdown(self) -> None:
        """Shutdown the plugin."""
        self._ticker_cache.clear()
        await self._set_status(PluginStatus.STOPPED)
        logger.info("Yahoo Finance plugin shutdown complete")

    async def health_check(self) -> PluginHealth:
        """Check plugin health."""
        start_time = datetime.utcnow()

        try:
            # Quick health check with SPY
            ticker = yf.Ticker("SPY")
            await asyncio.to_thread(lambda: ticker.fast_info)
            latency = max((datetime.utcnow() - start_time).total_seconds() * 1000, 0.1)

            self._health = PluginHealth(
                status=PluginStatus.READY,
                last_check=datetime.utcnow(),
                latency_ms=latency,
                message="Yahoo Finance API healthy",
            )

        except Exception as e:
            self._health = PluginHealth(
                status=PluginStatus.ERROR,
                last_check=datetime.utcnow(),
                latency_ms=0.0,
                error_count=self._health.error_count + 1,
                last_error=str(e),
            )

        return self._health

    async def get_quote(self, symbol: str) -> dict[str, Any]:
        """Get real-time quote for a symbol.

        Args:
            symbol: The ticker symbol (e.g., "AAPL", "MSFT").

        Returns:
            Quote data dictionary with price, volume, change, etc.
        """
        await self._rate_limiter.wait_for_token()

        try:
            ticker = self._get_ticker(symbol)

            # Use fast_info for quick quotes
            fast_info = await asyncio.to_thread(lambda: ticker.fast_info)

            # Get more detailed info if needed
            info = await asyncio.to_thread(lambda: ticker.info)

            quote = {
                "symbol": symbol.upper(),
                "timestamp": datetime.utcnow().isoformat(),
                "last": fast_info.get("lastPrice", info.get("currentPrice")),
                "open": info.get("open", fast_info.get("open")),
                "high": info.get("dayHigh", fast_info.get("dayHigh")),
                "low": info.get("dayLow", fast_info.get("dayLow")),
                "close": info.get("previousClose", fast_info.get("previousClose")),
                "previous_close": info.get("previousClose"),
                "volume": int(info.get("volume", fast_info.get("lastVolume", 0)) or 0),
                "change": None,  # Calculated below
                "change_percent": info.get("52WeekChange"),
                "bid": info.get("bid"),
                "bid_size": info.get("bidSize"),
                "ask": info.get("ask"),
                "ask_size": info.get("askSize"),
                "market_cap": info.get("marketCap"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "average_volume": info.get("averageVolume"),
                "source": "yahoo",
            }

            # Calculate change if we have current and previous close
            if quote["last"] and quote["previous_close"]:
                quote["change"] = quote["last"] - quote["previous_close"]
                quote["change_percent"] = (
                    (quote["change"] / quote["previous_close"]) * 100
                    if quote["previous_close"]
                    else 0
                )

            return quote

        except Exception as e:
            await self._handle_error(e)
            raise

    async def get_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> list[dict[str, Any]]:
        """Get historical OHLCV data.

        Args:
            symbol: The ticker symbol.
            start: Start datetime.
            end: End datetime.
            timeframe: Bar timeframe (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo).

        Returns:
            List of OHLCV bar dictionaries.

        Note:
            yfinance has limitations on intraday data:
            - 1m: Last 7 days only
            - 2m-90m: Last 60 days only
            - Daily and above: Full history available
        """
        await self._rate_limiter.wait_for_token()

        try:
            ticker = self._get_ticker(symbol)

            # Map timeframe to yfinance interval
            interval = self.TIMEFRAME_MAP.get(timeframe.lower(), "1d")

            # Fetch historical data
            df = await asyncio.to_thread(
                lambda: ticker.history(
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=True,
                    actions=False,
                )
            )

            if df.empty:
                logger.warning(f"No historical data for {symbol} from {start} to {end}")
                return []

            # Convert DataFrame to list of dicts
            bars = []
            for timestamp, row in df.iterrows():
                bar = {
                    "symbol": symbol.upper(),
                    "timestamp": timestamp.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                    "source": "yahoo",
                }
                bars.append(bar)

            return bars

        except Exception as e:
            await self._handle_error(e)
            raise

    async def get_company_info(self, symbol: str) -> dict[str, Any]:
        """Get company information.

        Args:
            symbol: The ticker symbol.

        Returns:
            Company info dictionary with name, sector, industry, etc.
        """
        await self._rate_limiter.wait_for_token()

        try:
            ticker = self._get_ticker(symbol)
            info = await asyncio.to_thread(lambda: ticker.info)

            return {
                "symbol": symbol.upper(),
                "name": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "country": info.get("country"),
                "website": info.get("website"),
                "description": info.get("longBusinessSummary"),
                "employees": info.get("fullTimeEmployees"),
                "exchange": info.get("exchange"),
                "currency": info.get("currency"),
                "source": "yahoo",
            }

        except Exception as e:
            await self._handle_error(e)
            raise

    async def get_dividends(self, symbol: str) -> list[dict[str, Any]]:
        """Get dividend history.

        Args:
            symbol: The ticker symbol.

        Returns:
            List of dividend records.
        """
        await self._rate_limiter.wait_for_token()

        try:
            ticker = self._get_ticker(symbol)
            dividends = await asyncio.to_thread(lambda: ticker.dividends)

            if dividends.empty:
                return []

            return [
                {
                    "symbol": symbol.upper(),
                    "date": date.isoformat(),
                    "amount": float(amount),
                    "source": "yahoo",
                }
                for date, amount in dividends.items()
            ]

        except Exception as e:
            await self._handle_error(e)
            raise

    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists.

        Args:
            symbol: The ticker symbol.

        Returns:
            True if symbol is valid and has data.
        """
        try:
            ticker = self._get_ticker(symbol)
            info = await asyncio.to_thread(lambda: ticker.info)
            # Check if we got real data (not just an empty dict)
            return bool(info and info.get("symbol"))
        except Exception:
            return False

    def _get_ticker(self, symbol: str) -> Any:
        """Get or create a yfinance Ticker object.

        Args:
            symbol: The ticker symbol.

        Returns:
            yfinance Ticker object.
        """
        symbol = symbol.upper()
        if symbol not in self._ticker_cache:
            self._ticker_cache[symbol] = yf.Ticker(symbol)
        return self._ticker_cache[symbol]
