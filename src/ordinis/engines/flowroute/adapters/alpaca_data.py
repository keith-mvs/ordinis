"""
Alpaca market data adapter for real-time and historical data.

Provides streaming bars, historical data, and quotes from Alpaca.
"""

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
import logging
import os
from typing import Any

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestBarRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame
import pandas as pd

logger = logging.getLogger(__name__)


class AlpacaMarketDataAdapter:
    """
    Alpaca market data adapter for real-time and historical data.

    Features:
    - Real-time bar streaming
    - Historical bars (intraday and daily)
    - Latest quotes and bars
    - Multiple timeframes support
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
    ):
        """
        Initialize Alpaca market data adapter.

        Args:
            api_key: Alpaca API key (defaults to env var ALPACA_API_KEY)
            api_secret: Alpaca API secret (defaults to env var ALPACA_API_SECRET)
        """
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Alpaca API credentials required. Set ALPACA_API_KEY and "
                "ALPACA_API_SECRET environment variables."
            )

        # Initialize historical data client
        self._data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
        )

        # Initialize streaming client (lazy initialization)
        self._stream: StockDataStream | None = None
        self._stream_running = False

    def get_latest_quote(self, symbol: str) -> dict[str, Any] | None:
        """
        Get latest quote for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'SPY')

        Returns:
            Quote dict with bid, ask, timestamp
        """
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self._data_client.get_stock_latest_quote(request)

            if symbol in quotes:
                quote = quotes[symbol]
                return {
                    "symbol": symbol,
                    "bid": float(quote.bid_price),
                    "ask": float(quote.ask_price),
                    "bid_size": int(quote.bid_size),
                    "ask_size": int(quote.ask_size),
                    "timestamp": quote.timestamp,
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return None

    def get_latest_bar(self, symbol: str) -> dict[str, Any] | None:
        """
        Get latest bar for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Bar dict with OHLCV data
        """
        try:
            request = StockLatestBarRequest(symbol_or_symbols=symbol)
            bars = self._data_client.get_stock_latest_bar(request)

            if symbol in bars:
                bar = bars[symbol]
                return {
                    "symbol": symbol,
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "trade_count": int(bar.trade_count) if hasattr(bar, "trade_count") else None,
                    "vwap": float(bar.vwap) if hasattr(bar, "vwap") else None,
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get latest bar for {symbol}: {e}")
            return None

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Get historical bars for a symbol.

        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1Hour', '1Day')
            start: Start datetime (defaults to 7 days ago)
            end: End datetime (defaults to now)
            limit: Maximum number of bars

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Parse timeframe
            from alpaca.data.timeframe import TimeFrameUnit

            tf_map = {
                "1Min": TimeFrame(1, TimeFrameUnit.Minute),
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
                "1Day": TimeFrame(1, TimeFrameUnit.Day),
            }
            timeframe_obj = tf_map.get(timeframe, TimeFrame(1, TimeFrameUnit.Minute))

            # Default start/end times
            if end is None:
                end = datetime.now()
            if start is None:
                start = end - timedelta(days=7)

            # Create request
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe_obj,
                start=start,
                end=end,
                limit=limit,
            )

            # Fetch bars
            bars = self._data_client.get_stock_bars(request)

            if symbol in bars:
                df = bars.df

                # Reset index to get timestamp as column
                if isinstance(df.index, pd.MultiIndex):
                    df = df.reset_index()

                # Rename columns to standard format
                df = df.rename(
                    columns={
                        "timestamp": "timestamp",
                        "open": "open",
                        "high": "high",
                        "low": "low",
                        "close": "close",
                        "volume": "volume",
                    }
                )

                return df

            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to get historical bars for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    async def stream_bars(
        self,
        symbols: list[str],
        on_bar: Callable[[dict[str, Any]], None],
    ) -> None:
        """
        Stream real-time bars for symbols.

        Args:
            symbols: List of symbols to stream
            on_bar: Callback function for each bar
        """
        try:
            # Initialize stream if not already
            if self._stream is None:
                self._stream = StockDataStream(
                    api_key=self.api_key,
                    secret_key=self.api_secret,
                )

            # Subscribe to bars
            async def handle_bar(bar):
                bar_dict = {
                    "symbol": bar.symbol,
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                }
                on_bar(bar_dict)

            self._stream.subscribe_bars(handle_bar, *symbols)

            # Run stream
            self._stream_running = True
            logger.info(f"Starting bar stream for {symbols}")
            await self._stream._run_forever()

        except Exception as e:
            logger.error(f"Stream error: {e}")
            self._stream_running = False
            raise

    async def stop_stream(self) -> None:
        """Stop the data stream."""
        if self._stream and self._stream_running:
            await self._stream.stop_ws()
            self._stream_running = False
            logger.info("Market data stream stopped")

    def get_price_history(
        self,
        symbol: str,
        periods: int = 100,
        timeframe: str = "1Min",
    ) -> list[float]:
        """
        Get recent closing prices for strategy calculations.

        Args:
            symbol: Stock symbol
            periods: Number of periods to fetch
            timeframe: Bar timeframe

        Returns:
            List of closing prices
        """
        end = datetime.now()
        start = end - timedelta(days=max(periods // 390 + 1, 2))  # Rough estimate

        df = self.get_historical_bars(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=periods,
        )

        if not df.empty and "close" in df.columns:
            return df["close"].tolist()

        return []

    def is_market_open(self) -> bool:
        """
        Check if market is currently open.

        Returns:
            True if market is open
        """
        # Simple check based on time (9:30 AM - 4:00 PM ET, Mon-Fri)
        # For production, use Alpaca's clock API
        now = datetime.now()

        # Weekend check
        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return False

        # Market hours check (simplified, assumes ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)

        return market_open <= now <= market_close


# Example usage
async def example_usage():
    """Example of using Alpaca market data adapter."""
    adapter = AlpacaMarketDataAdapter()

    # Get latest quote
    quote = adapter.get_latest_quote("SPY")
    print(f"SPY Quote: {quote}")

    # Get latest bar
    bar = adapter.get_latest_bar("SPY")
    print(f"SPY Bar: {bar}")

    # Get historical bars
    df = adapter.get_historical_bars(
        symbol="SPY",
        timeframe="1Min",
        start=datetime.now() - timedelta(hours=1),
    )
    print(f"Historical bars: {len(df)} bars")

    # Stream bars
    def on_bar(bar):
        print(f"New bar: {bar['symbol']} @ {bar['close']}")

    # Uncomment to test streaming
    # await adapter.stream_bars(["SPY"], on_bar)


if __name__ == "__main__":
    asyncio.run(example_usage())
