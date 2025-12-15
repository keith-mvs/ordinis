"""Live data pipeline integrating with market data providers."""

from abc import ABC, abstractmethod
import asyncio
from dataclasses import dataclass
from datetime import datetime

import pandas as pd


class DataProvider(ABC):
    """Abstract base for market data providers."""

    @abstractmethod
    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch historical bars.

        Args:
            symbol: Stock ticker
            timeframe: Timeframe ('1min', '5min', '1h', '1d', etc.)
            limit: Number of bars to fetch

        Returns:
            DataFrame with OHLCV data
        """

    @abstractmethod
    async def fetch_quote(self, symbol: str) -> dict:
        """Fetch current quote.

        Args:
            symbol: Stock ticker

        Returns:
            Dict with price, bid, ask, etc.
        """

    @abstractmethod
    async def subscribe_updates(self, symbols: list[str], callback) -> None:
        """Subscribe to real-time updates.

        Args:
            symbols: Symbols to track
            callback: Async callback(symbol, price, timestamp)
        """


@dataclass
class DataQualityMetrics:
    """Metrics for data quality monitoring.

    Attributes:
        missing_bars: Count of missing bars
        gap_days: Days with gaps
        outlier_count: Count of price outliers
        last_update: Last data timestamp
        quality_score: Overall quality 0-100
    """

    missing_bars: int = 0
    gap_days: int = 0
    outlier_count: int = 0
    last_update: datetime | None = None
    quality_score: float = 100.0


class AlphaVantageProvider(DataProvider):
    """Alpha Vantage data provider."""

    def __init__(self, api_key: str):
        """Initialize with API key."""
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch bars from Alpha Vantage."""
        # Placeholder implementation
        # In production, would call API and parse response
        df = pd.DataFrame(
            {
                "open": [100] * limit,
                "high": [101] * limit,
                "low": [99] * limit,
                "close": [100.5] * limit,
                "volume": [1000000] * limit,
            },
            index=pd.date_range(periods=limit, freq="D"),
        )
        return df

    async def fetch_quote(self, symbol: str) -> dict:
        """Fetch current quote."""
        # Placeholder
        return {
            "symbol": symbol,
            "price": 100.0,
            "bid": 99.95,
            "ask": 100.05,
            "timestamp": datetime.now(),
        }

    async def subscribe_updates(self, symbols: list[str], callback) -> None:
        """Subscribe to updates (Alpha Vantage doesn't support streaming)."""
        # Alpha Vantage doesn't support WebSocket; polling would go here
        for symbol in symbols:
            while True:
                quote = await self.fetch_quote(symbol)
                await callback(symbol, quote["price"], quote["timestamp"])
                await asyncio.sleep(1)  # Rate limited


class PolygonProvider(DataProvider):
    """Polygon.io data provider."""

    def __init__(self, api_key: str):
        """Initialize with API key."""
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch bars from Polygon."""
        # Placeholder implementation
        df = pd.DataFrame(
            {
                "open": [100] * limit,
                "high": [101] * limit,
                "low": [99] * limit,
                "close": [100.5] * limit,
                "volume": [1000000] * limit,
            },
            index=pd.date_range(periods=limit, freq="D"),
        )
        return df

    async def fetch_quote(self, symbol: str) -> dict:
        """Fetch current quote."""
        return {
            "symbol": symbol,
            "price": 100.0,
            "bid": 99.95,
            "ask": 100.05,
            "timestamp": datetime.now(),
        }

    async def subscribe_updates(self, symbols: list[str], callback) -> None:
        """Subscribe to WebSocket updates."""
        # Polygon supports WebSocket for real-time data
        # Placeholder for WebSocket connection


class DataQualityMonitor:
    """Monitors data quality and detects issues."""

    @staticmethod
    def check_quality(df: pd.DataFrame, symbol: str) -> DataQualityMetrics:
        """Check data quality.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol being checked

        Returns:
            DataQualityMetrics
        """
        metrics = DataQualityMetrics(last_update=datetime.now())

        if df.empty:
            metrics.quality_score = 0.0
            return metrics

        # Check for missing/NaN values
        missing = df.isnull().sum().sum()
        if missing > 0:
            metrics.missing_bars = missing

        # Check for gaps
        if isinstance(df.index, pd.DatetimeIndex):
            expected_bars = (df.index[-1] - df.index[0]).days
            actual_bars = len(df)
            if expected_bars > actual_bars:
                metrics.gap_days = expected_bars - actual_bars

        # Check for outliers (price > 10% change in one bar)
        returns = df["close"].pct_change().abs()
        outliers = (returns > 0.10).sum()
        metrics.outlier_count = outliers

        # Quality score
        quality = 100.0
        quality -= metrics.missing_bars * 5
        quality -= metrics.gap_days * 2
        quality -= metrics.outlier_count * 1
        metrics.quality_score = max(0.0, min(100.0, quality))

        return metrics


class ScheduledDataCollector:
    """Schedules and collects data from providers."""

    def __init__(self, provider: DataProvider, check_interval_sec: int = 60):
        """Initialize collector.

        Args:
            provider: Data provider
            check_interval_sec: Check interval in seconds
        """
        self.provider = provider
        self.check_interval = check_interval_sec
        self.symbols: list[str] = []
        self._running = False
        self._latest_data: dict[str, pd.DataFrame] = {}
        self._quality_metrics: dict[str, DataQualityMetrics] = {}

    async def add_symbols(self, symbols: list[str]):
        """Add symbols to track.

        Args:
            symbols: List of tickers
        """
        self.symbols.extend(symbols)

        # Initial fetch
        for symbol in symbols:
            try:
                df = await self.provider.fetch_bars(symbol)
                self._latest_data[symbol] = df
                metrics = DataQualityMonitor.check_quality(df, symbol)
                self._quality_metrics[symbol] = metrics
            except Exception as e:
                print(f"[data] Error fetching {symbol}: {e}")

    async def start_collection(self):
        """Start scheduled data collection."""
        self._running = True

        while self._running:
            try:
                # Collect data for all symbols
                for symbol in self.symbols:
                    try:
                        df = await self.provider.fetch_bars(symbol)
                        self._latest_data[symbol] = df

                        # Check quality
                        metrics = DataQualityMonitor.check_quality(df, symbol)
                        self._quality_metrics[symbol] = metrics

                        if metrics.quality_score < 80:
                            print(f"[data] ⚠ {symbol} quality: {metrics.quality_score:.0f}%")

                    except Exception as e:
                        print(f"[data] Error collecting {symbol}: {e}")

                await asyncio.sleep(self.check_interval)

            except Exception as e:
                print(f"[data] Collection error: {e}")
                await asyncio.sleep(self.check_interval)

    async def stop_collection(self):
        """Stop data collection."""
        self._running = False

    def get_latest(self, symbol: str) -> pd.DataFrame | None:
        """Get latest data for symbol."""
        return self._latest_data.get(symbol)

    def get_quality_metrics(self, symbol: str) -> DataQualityMetrics | None:
        """Get quality metrics for symbol."""
        return self._quality_metrics.get(symbol)

    def get_all_quality_report(self) -> dict[str, DataQualityMetrics]:
        """Get quality report for all symbols."""
        return dict(self._quality_metrics)


class LiveDataPipeline:
    """Complete live data pipeline with provider, collection, and storage."""

    def __init__(
        self,
        provider: DataProvider,
        storage_dir: str | None = None,
    ):
        """Initialize pipeline.

        Args:
            provider: Data provider
            storage_dir: Directory for storing data snapshots
        """
        self.provider = provider
        self.storage_dir = storage_dir or "data/live"
        self.collector = ScheduledDataCollector(provider)

    async def start(self, symbols: list[str]):
        """Start data pipeline.

        Args:
            symbols: Symbols to track
        """
        print(f"[pipeline] Starting live data collection for {len(symbols)} symbols...")
        await self.collector.add_symbols(symbols)
        await self.collector.start_collection()

    async def stop(self):
        """Stop data pipeline."""
        await self.collector.stop_collection()
        print("[pipeline] ✓ Data pipeline stopped")

    def get_latest_data(self, symbol: str) -> pd.DataFrame | None:
        """Get latest data for symbol."""
        return self.collector.get_latest(symbol)

    def get_quality_report(self) -> dict:
        """Get overall quality report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "symbols": {},
        }

        for symbol in self.collector.symbols:
            metrics = self.collector.get_quality_metrics(symbol)
            if metrics:
                report["symbols"][symbol] = {
                    "quality_score": metrics.quality_score,
                    "missing_bars": metrics.missing_bars,
                    "gap_days": metrics.gap_days,
                    "outlier_count": metrics.outlier_count,
                    "last_update": metrics.last_update.isoformat() if metrics.last_update else None,
                }

        return report
