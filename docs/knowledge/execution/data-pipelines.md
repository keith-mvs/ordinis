# Data Pipelines for Algorithmic Trading

## Overview

This document describes data pipeline architectures, patterns, and best practices for algorithmic trading systems. Reliable data pipelines are critical for accurate signal generation, backtesting, and live execution.

**Last Updated**: December 8, 2025

---

## 1. Pipeline Architecture

### 1.1 High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   SOURCES   │───>│  INGESTION  │───>│  PROCESSING │───>│   STORAGE   │  │
│  │  (External) │    │   Layer     │    │    Layer    │    │   Layer     │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │          │
│        │                  │                  │                  │          │
│        ▼                  ▼                  ▼                  ▼          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │ Market Data │    │ Validation  │    │ Feature Eng │    │ Time Series │  │
│  │ Broker APIs │    │ Deduping    │    │ Indicators  │    │ Cache/DB    │  │
│  │ Alt Data    │    │ Normalization│   │ ML Features │    │ Partitioned │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
│                              │                                              │
│                              ▼                                              │
│                    ┌─────────────────────┐                                  │
│                    │    CONSUMERS        │                                  │
│                    │  SignalCore Engine  │                                  │
│                    │  ProofBench         │                                  │
│                    │  Dashboard          │                                  │
│                    └─────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Pipeline Layers

| Layer | Responsibility | Components |
|-------|---------------|------------|
| **Ingestion** | Collect data from sources | API clients, websockets, file readers |
| **Validation** | Ensure data quality | Schema validation, range checks, deduplication |
| **Processing** | Transform and enrich | Normalization, feature engineering, indicators |
| **Storage** | Persist for retrieval | Time-series DB, cache, file storage |
| **Distribution** | Deliver to consumers | Pub/sub, queues, direct API |

---

## 2. Data Sources

### 2.1 Market Data Sources

| Source Type | Examples | Latency | Use Case |
|-------------|----------|---------|----------|
| **Real-time feeds** | Polygon, IEX, Alpaca | < 100ms | Live trading |
| **Delayed feeds** | Yahoo Finance, Alpha Vantage | 15+ min | Development, backtesting |
| **Historical data** | Quandl, EODData | N/A | Backtesting, research |
| **Broker APIs** | Alpaca, IB, TD | Varies | Execution, account data |

### 2.2 Alternative Data Sources

| Source | Data Type | Update Frequency |
|--------|-----------|------------------|
| **SEC EDGAR** | Filings (10-K, 10-Q, 8-K) | As filed |
| **News APIs** | Headlines, sentiment | Real-time |
| **Social media** | Twitter, Reddit sentiment | Real-time |
| **Economic data** | FRED (GDP, CPI, rates) | Scheduled releases |
| **Satellite/web** | Foot traffic, job postings | Daily/weekly |

### 2.3 Data Source Implementation

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import pandas as pd


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    cache_enabled: bool = True
    cache_ttl: int = 300  # seconds


class DataSource(ABC):
    """Abstract base class for data sources."""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        self._rate_limiter = RateLimiter(config.rate_limit)

    @abstractmethod
    def fetch_historical(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        pass

    @abstractmethod
    def fetch_realtime(self, symbol: str) -> dict:
        """Fetch real-time quote."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if data source is available."""
        pass


class YahooFinanceSource(DataSource):
    """Yahoo Finance data source implementation."""

    def fetch_historical(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data from Yahoo Finance.

        Args:
            symbol: Ticker symbol
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1d, 1h, 15m, etc.)

        Returns:
            DataFrame with OHLCV columns
        """
        import yfinance as yf

        self._rate_limiter.acquire()

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval
        )

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]

        # Add symbol column
        df["symbol"] = symbol

        return df

    def fetch_realtime(self, symbol: str) -> dict:
        """Fetch real-time quote (delayed for Yahoo)."""
        import yfinance as yf

        self._rate_limiter.acquire()

        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            "symbol": symbol,
            "price": info.get("regularMarketPrice"),
            "bid": info.get("bid"),
            "ask": info.get("ask"),
            "volume": info.get("regularMarketVolume"),
            "timestamp": datetime.utcnow(),
        }

    def health_check(self) -> bool:
        """Check Yahoo Finance availability."""
        try:
            import yfinance as yf
            ticker = yf.Ticker("AAPL")
            _ = ticker.info
            return True
        except Exception:
            return False
```

---

## 3. Ingestion Layer

### 3.1 Ingestion Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Batch ingestion** | Scheduled bulk data loads | Historical data, EOD processing |
| **Streaming ingestion** | Continuous real-time updates | Live trading, tick data |
| **Micro-batch** | Small frequent batches | Near real-time with aggregation |
| **Event-driven** | Triggered by external events | News, corporate actions |

### 3.2 Batch Ingestion

```python
from datetime import datetime, timedelta
from typing import List
import pandas as pd


class BatchIngestionPipeline:
    """
    Batch data ingestion for historical data.
    """

    def __init__(
        self,
        source: DataSource,
        storage: DataStorage,
        validator: DataValidator
    ):
        self.source = source
        self.storage = storage
        self.validator = validator

    def ingest_historical(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d"
    ) -> dict:
        """
        Ingest historical data for multiple symbols.

        Returns:
            Summary of ingestion results
        """
        results = {
            "success": [],
            "failed": [],
            "total_rows": 0,
        }

        for symbol in symbols:
            try:
                # Fetch data
                df = self.source.fetch_historical(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval
                )

                # Validate
                is_valid, errors = self.validator.validate(df)
                if not is_valid:
                    results["failed"].append({
                        "symbol": symbol,
                        "errors": errors
                    })
                    continue

                # Store
                rows_written = self.storage.write(
                    symbol=symbol,
                    data=df,
                    partition_key=interval
                )

                results["success"].append(symbol)
                results["total_rows"] += rows_written

            except Exception as e:
                results["failed"].append({
                    "symbol": symbol,
                    "errors": [str(e)]
                })

        return results


class ScheduledIngestion:
    """Schedule-based data ingestion."""

    SCHEDULES = {
        "market_open": "09:30",
        "market_close": "16:00",
        "eod_update": "17:00",
        "overnight": "06:00",
    }

    def __init__(self, pipeline: BatchIngestionPipeline):
        self.pipeline = pipeline

    def run_eod_update(self, symbols: List[str]):
        """
        End-of-day data update.
        Run after market close to get final daily bars.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)

        return self.pipeline.ingest_historical(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )
```

### 3.3 Streaming Ingestion

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional
import asyncio


@dataclass
class StreamConfig:
    """Configuration for streaming data."""
    buffer_size: int = 1000
    flush_interval: float = 1.0  # seconds
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10


class StreamingIngestionPipeline:
    """
    Real-time streaming data ingestion.
    """

    def __init__(
        self,
        config: StreamConfig,
        storage: DataStorage,
        on_message: Optional[Callable] = None
    ):
        self.config = config
        self.storage = storage
        self.on_message = on_message
        self._buffer = []
        self._running = False

    async def connect_websocket(self, url: str, symbols: list):
        """
        Connect to websocket stream.

        Example for Alpaca:
            url = "wss://stream.data.alpaca.markets/v2/iex"
        """
        import websockets
        import json

        reconnect_attempts = 0

        while reconnect_attempts < self.config.max_reconnect_attempts:
            try:
                async with websockets.connect(url) as ws:
                    # Subscribe to symbols
                    subscribe_msg = {
                        "action": "subscribe",
                        "trades": symbols,
                        "quotes": symbols,
                        "bars": symbols
                    }
                    await ws.send(json.dumps(subscribe_msg))

                    self._running = True
                    reconnect_attempts = 0

                    # Start buffer flush task
                    asyncio.create_task(self._flush_buffer_periodically())

                    # Process messages
                    async for message in ws:
                        data = json.loads(message)
                        await self._process_message(data)

            except websockets.ConnectionClosed:
                reconnect_attempts += 1
                await asyncio.sleep(self.config.reconnect_delay)

            except Exception as e:
                reconnect_attempts += 1
                await asyncio.sleep(self.config.reconnect_delay)

        self._running = False

    async def _process_message(self, data: dict):
        """Process incoming websocket message."""
        # Add timestamp if not present
        if "timestamp" not in data:
            data["timestamp"] = datetime.utcnow().isoformat()

        # Add to buffer
        self._buffer.append(data)

        # Callback for real-time processing
        if self.on_message:
            self.on_message(data)

        # Flush if buffer full
        if len(self._buffer) >= self.config.buffer_size:
            await self._flush_buffer()

    async def _flush_buffer(self):
        """Flush buffer to storage."""
        if not self._buffer:
            return

        data_to_write = self._buffer.copy()
        self._buffer.clear()

        await self.storage.write_batch(data_to_write)

    async def _flush_buffer_periodically(self):
        """Periodic buffer flush."""
        while self._running:
            await asyncio.sleep(self.config.flush_interval)
            await self._flush_buffer()
```

---

## 4. Validation Layer

### 4.1 Data Quality Checks

| Check Type | Description | Action on Failure |
|------------|-------------|-------------------|
| **Schema validation** | Column names, types | Reject record |
| **Range checks** | Price > 0, volume >= 0 | Flag or reject |
| **Completeness** | Required fields present | Reject record |
| **Timeliness** | Data freshness | Alert and proceed |
| **Consistency** | OHLC relationship | Flag for review |
| **Deduplication** | No duplicate records | Drop duplicates |

### 4.2 Validation Implementation

```python
from dataclasses import dataclass, field
from typing import List, Tuple
import pandas as pd
import numpy as np


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rows_checked: int = 0
    rows_failed: int = 0


class DataValidator:
    """
    Comprehensive data validation for market data.
    """

    # Expected OHLCV schema
    REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

    # Price sanity bounds
    MIN_PRICE = 0.0001
    MAX_PRICE = 1_000_000

    # Volume bounds
    MIN_VOLUME = 0
    MAX_VOLUME = 10_000_000_000  # 10 billion

    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Run all validations on DataFrame.

        Returns:
            (is_valid, list of error messages)
        """
        errors = []

        # Schema validation
        schema_errors = self._validate_schema(df)
        errors.extend(schema_errors)

        if schema_errors:
            return False, errors

        # Range validations
        range_errors = self._validate_ranges(df)
        errors.extend(range_errors)

        # OHLC consistency
        ohlc_errors = self._validate_ohlc_consistency(df)
        errors.extend(ohlc_errors)

        # Duplicate check
        dup_errors = self._validate_no_duplicates(df)
        errors.extend(dup_errors)

        # Timestamp continuity
        time_errors = self._validate_timestamps(df)
        errors.extend(time_errors)

        return len(errors) == 0, errors

    def _validate_schema(self, df: pd.DataFrame) -> List[str]:
        """Validate required columns exist."""
        errors = []
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns.str.lower())
        if missing:
            errors.append(f"Missing required columns: {missing}")
        return errors

    def _validate_ranges(self, df: pd.DataFrame) -> List[str]:
        """Validate price and volume ranges."""
        errors = []

        # Price checks
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                invalid_low = (df[col] < self.MIN_PRICE).sum()
                invalid_high = (df[col] > self.MAX_PRICE).sum()

                if invalid_low > 0:
                    errors.append(
                        f"{col} has {invalid_low} values below {self.MIN_PRICE}"
                    )
                if invalid_high > 0:
                    errors.append(
                        f"{col} has {invalid_high} values above {self.MAX_PRICE}"
                    )

        # Volume check
        if "volume" in df.columns:
            invalid_vol = (df["volume"] < self.MIN_VOLUME).sum()
            if invalid_vol > 0:
                errors.append(f"volume has {invalid_vol} negative values")

        return errors

    def _validate_ohlc_consistency(self, df: pd.DataFrame) -> List[str]:
        """Validate OHLC relationship: High >= Open, Close, Low; Low <= all."""
        errors = []

        # High should be highest
        high_violations = (
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["high"] < df["low"])
        ).sum()

        if high_violations > 0:
            errors.append(f"OHLC inconsistency: {high_violations} rows where high is not highest")

        # Low should be lowest
        low_violations = (
            (df["low"] > df["open"]) |
            (df["low"] > df["close"]) |
            (df["low"] > df["high"])
        ).sum()

        if low_violations > 0:
            errors.append(f"OHLC inconsistency: {low_violations} rows where low is not lowest")

        return errors

    def _validate_no_duplicates(self, df: pd.DataFrame) -> List[str]:
        """Check for duplicate timestamps."""
        errors = []

        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            errors.append(f"Found {dup_count} duplicate timestamps")

        return errors

    def _validate_timestamps(self, df: pd.DataFrame) -> List[str]:
        """Validate timestamp ordering and gaps."""
        errors = []

        if not df.index.is_monotonic_increasing:
            errors.append("Timestamps are not monotonically increasing")

        return errors


class DataCleaner:
    """
    Clean and repair data issues.
    """

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate timestamps, keeping last."""
        return df[~df.index.duplicated(keep="last")]

    @staticmethod
    def fix_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """Repair OHLC inconsistencies."""
        df = df.copy()

        # Ensure high is highest
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)

        # Ensure low is lowest
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)

        return df

    @staticmethod
    def fill_gaps(
        df: pd.DataFrame,
        freq: str = "1D",
        method: str = "ffill"
    ) -> pd.DataFrame:
        """Fill missing timestamps."""
        # Create complete date range
        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )

        # Reindex and fill
        df = df.reindex(full_range)

        if method == "ffill":
            df = df.fillna(method="ffill")
        elif method == "interpolate":
            df = df.interpolate()

        return df
```

---

## 5. Processing Layer

### 5.1 Data Transformation

```python
import pandas as pd
import numpy as np
from typing import Dict, Any


class DataTransformer:
    """
    Transform raw market data into analysis-ready format.
    """

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across data sources.
        """
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Adj Close": "adj_close",
            "Adj. Close": "adj_close",
        }

        df = df.rename(columns=column_mapping)
        df.columns = df.columns.str.lower()

        return df

    @staticmethod
    def adjust_for_splits(
        df: pd.DataFrame,
        split_factor: float
    ) -> pd.DataFrame:
        """
        Adjust historical prices for stock splits.
        """
        df = df.copy()

        df["open"] = df["open"] / split_factor
        df["high"] = df["high"] / split_factor
        df["low"] = df["low"] / split_factor
        df["close"] = df["close"] / split_factor
        df["volume"] = df["volume"] * split_factor

        return df

    @staticmethod
    def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various return measures.
        """
        df = df.copy()

        # Simple returns
        df["return"] = df["close"].pct_change()

        # Log returns (better for statistical analysis)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Cumulative returns
        df["cum_return"] = (1 + df["return"]).cumprod() - 1

        return df

    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features for seasonality analysis.
        """
        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            df["day_of_week"] = df.index.dayofweek
            df["month"] = df.index.month
            df["quarter"] = df.index.quarter
            df["year"] = df.index.year
            df["is_month_end"] = df.index.is_month_end
            df["is_quarter_end"] = df.index.is_quarter_end

        return df
```

### 5.2 Feature Engineering Pipeline

```python
class FeatureEngineeringPipeline:
    """
    Feature engineering for ML models.
    """

    def __init__(self, lookback_periods: list = None):
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 200]

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive feature set.
        """
        df = df.copy()

        # Price-based features
        df = self._add_price_features(df)

        # Volume features
        df = self._add_volume_features(df)

        # Technical indicators
        df = self._add_technical_features(df)

        # Statistical features
        df = self._add_statistical_features(df)

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-derived features."""
        # Range
        df["range"] = df["high"] - df["low"]
        df["range_pct"] = df["range"] / df["close"]

        # Body
        df["body"] = abs(df["close"] - df["open"])
        df["body_pct"] = df["body"] / df["close"]

        # Gap
        df["gap"] = df["open"] - df["close"].shift(1)
        df["gap_pct"] = df["gap"] / df["close"].shift(1)

        # Relative position
        df["close_position"] = (df["close"] - df["low"]) / df["range"]

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-derived features."""
        # Relative volume
        for period in self.lookback_periods:
            df[f"rel_volume_{period}"] = df["volume"] / df["volume"].rolling(period).mean()

        # Volume trend
        df["volume_trend"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()

        # Price-volume relationship
        df["pv_corr_20"] = df["close"].rolling(20).corr(df["volume"])

        return df

    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator features."""
        # Moving averages
        for period in self.lookback_periods:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
            df[f"ema_{period}"] = df["close"].ewm(span=period).mean()
            df[f"close_vs_sma_{period}"] = df["close"] / df[f"sma_{period}"] - 1

        # Volatility
        for period in [10, 20, 50]:
            df[f"volatility_{period}"] = df["return"].rolling(period).std() * np.sqrt(252)

        # RSI
        df["rsi_14"] = self._calculate_rsi(df["close"], 14)

        # MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        # Skewness and kurtosis
        df["skew_20"] = df["return"].rolling(20).skew()
        df["kurt_20"] = df["return"].rolling(20).kurt()

        # Z-score
        for period in [20, 50]:
            mean = df["close"].rolling(period).mean()
            std = df["close"].rolling(period).std()
            df[f"zscore_{period}"] = (df["close"] - mean) / std

        return df

    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
```

---

## 6. Storage Layer

### 6.1 Storage Options

| Storage Type | Use Case | Pros | Cons |
|--------------|----------|------|------|
| **CSV/Parquet** | Backtesting, small scale | Simple, portable | No querying |
| **SQLite** | Development, single user | Easy setup, SQL | Single writer |
| **PostgreSQL** | Production, multi-user | Full SQL, reliable | Setup overhead |
| **TimescaleDB** | Time-series optimized | Compression, speed | PostgreSQL extension |
| **InfluxDB** | Tick data, metrics | High write throughput | Different paradigm |
| **Arctic** | Quant research | Versioning, chunks | MongoDB dependency |

### 6.2 Storage Implementation

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
import pandas as pd


class DataStorage(ABC):
    """Abstract base class for data storage."""

    @abstractmethod
    def write(
        self,
        symbol: str,
        data: pd.DataFrame,
        partition_key: str = None
    ) -> int:
        """Write data to storage."""
        pass

    @abstractmethod
    def read(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Read data from storage."""
        pass

    @abstractmethod
    def list_symbols(self) -> List[str]:
        """List available symbols."""
        pass


class ParquetStorage(DataStorage):
    """
    Parquet-based storage with partitioning.
    Efficient for backtesting and batch processing.
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        symbol: str,
        data: pd.DataFrame,
        partition_key: str = None
    ) -> int:
        """
        Write data to partitioned parquet files.

        Directory structure:
            base_path/
            ├── AAPL/
            │   ├── 1d.parquet
            │   └── 1h.parquet
            └── MSFT/
                └── 1d.parquet
        """
        symbol_path = self.base_path / symbol
        symbol_path.mkdir(exist_ok=True)

        partition = partition_key or "default"
        file_path = symbol_path / f"{partition}.parquet"

        # Append if exists
        if file_path.exists():
            existing = pd.read_parquet(file_path)
            data = pd.concat([existing, data])
            data = data[~data.index.duplicated(keep="last")]
            data = data.sort_index()

        data.to_parquet(file_path, index=True)
        return len(data)

    def read(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        partition_key: str = "1d"
    ) -> pd.DataFrame:
        """Read data from parquet file."""
        file_path = self.base_path / symbol / f"{partition_key}.parquet"

        if not file_path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(file_path)

        # Filter by date if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df

    def list_symbols(self) -> List[str]:
        """List available symbols."""
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]


class CachingStorage(DataStorage):
    """
    In-memory caching layer over persistent storage.
    """

    def __init__(self, backend: DataStorage, max_cache_size: int = 100):
        self.backend = backend
        self.max_cache_size = max_cache_size
        self._cache = {}

    def write(
        self,
        symbol: str,
        data: pd.DataFrame,
        partition_key: str = None
    ) -> int:
        """Write to backend and update cache."""
        result = self.backend.write(symbol, data, partition_key)

        # Update cache
        cache_key = f"{symbol}:{partition_key or 'default'}"
        self._cache[cache_key] = data

        # Evict if too large
        if len(self._cache) > self.max_cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        return result

    def read(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        partition_key: str = "1d"
    ) -> pd.DataFrame:
        """Read from cache if available, else backend."""
        cache_key = f"{symbol}:{partition_key}"

        if cache_key in self._cache:
            df = self._cache[cache_key]
        else:
            df = self.backend.read(symbol, partition_key=partition_key)
            self._cache[cache_key] = df

        # Filter by date
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        return df

    def list_symbols(self) -> List[str]:
        return self.backend.list_symbols()
```

---

## 7. Pipeline Orchestration

### 7.1 Complete Pipeline Example

```python
class TradingDataPipeline:
    """
    Complete data pipeline for trading system.
    """

    def __init__(
        self,
        source: DataSource,
        storage: DataStorage,
        validator: DataValidator,
        transformer: DataTransformer,
        feature_engineer: FeatureEngineeringPipeline
    ):
        self.source = source
        self.storage = storage
        self.validator = validator
        self.transformer = transformer
        self.feature_engineer = feature_engineer

    def run_backfill(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """
        Backfill historical data for symbols.
        """
        results = {"success": [], "failed": [], "total_rows": 0}

        for symbol in symbols:
            try:
                # 1. Fetch
                raw_df = self.source.fetch_historical(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )

                # 2. Validate
                is_valid, errors = self.validator.validate(raw_df)
                if not is_valid:
                    results["failed"].append({"symbol": symbol, "errors": errors})
                    continue

                # 3. Transform
                df = self.transformer.normalize_columns(raw_df)
                df = self.transformer.calculate_returns(df)

                # 4. Engineer features
                df = self.feature_engineer.generate_features(df)

                # 5. Store
                rows = self.storage.write(symbol=symbol, data=df)
                results["success"].append(symbol)
                results["total_rows"] += rows

            except Exception as e:
                results["failed"].append({"symbol": symbol, "errors": [str(e)]})

        return results

    def get_analysis_ready_data(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Get fully processed data ready for analysis.
        """
        # Read from storage
        df = self.storage.read(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        # Ensure features are current
        if "rsi_14" not in df.columns:
            df = self.feature_engineer.generate_features(df)

        # Drop NaN rows from feature calculation
        df = df.dropna()

        return df
```

---

## 8. Best Practices

### 8.1 Data Quality

| Practice | Description |
|----------|-------------|
| **Validate at ingestion** | Catch issues early, before storage |
| **Log all anomalies** | Track data quality over time |
| **Version your data** | Enable reproducible research |
| **Document data sources** | Know provenance for debugging |
| **Monitor freshness** | Alert on stale data |

### 8.2 Performance

| Practice | Description |
|----------|-------------|
| **Partition by time** | Enable efficient range queries |
| **Use columnar formats** | Parquet/Arrow for analytics |
| **Cache hot data** | In-memory for frequent access |
| **Compress cold data** | Reduce storage costs |
| **Profile bottlenecks** | Measure before optimizing |

### 8.3 Reliability

| Practice | Description |
|----------|-------------|
| **Idempotent ingestion** | Safe to retry on failure |
| **Atomic writes** | No partial updates |
| **Backup critical data** | Disaster recovery |
| **Test with realistic data** | Catch edge cases |
| **Monitor end-to-end** | Alert on pipeline failures |

---

## 9. References

- Reis, J. & Housley, M. (2022): "Fundamentals of Data Engineering"
- Kleppmann, M. (2017): "Designing Data-Intensive Applications"
- Apache Arrow Documentation: https://arrow.apache.org/
- Parquet Format: https://parquet.apache.org/
- TimescaleDB Docs: https://docs.timescale.com/
