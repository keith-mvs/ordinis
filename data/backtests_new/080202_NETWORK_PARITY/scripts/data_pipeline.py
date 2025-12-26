#!/usr/bin/env python3
"""
Data Pipeline for Network Parity Optimization

Handles loading and processing of historical market data from massive flat files
with support for 1-minute and daily aggregation.

Author: Ordinis Quantitative Research
Version: 1.0.0
"""

import gzip
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import MASSIVE_DATA_DIR, BacktestingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class MarketData:
    """Container for loaded market data."""

    symbol: str
    bars: pd.DataFrame  # DatetimeIndex with OHLCV columns
    timeframe: str  # "1min", "5min", "1D"
    start_date: datetime | None = None
    end_date: datetime | None = None

    @property
    def n_bars(self) -> int:
        """Number of bars."""
        return len(self.bars)

    @property
    def returns(self) -> pd.Series:
        """Calculate log returns."""
        return np.log(self.bars["close"] / self.bars["close"].shift(1)).dropna()

    def to_daily(self) -> "MarketData":
        """Aggregate to daily bars."""
        if self.timeframe == "1D":
            return self

        daily = self.bars.resample("1D").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return MarketData(
            symbol=self.symbol,
            bars=daily,
            timeframe="1D",
            start_date=self.start_date,
            end_date=self.end_date,
        )


@dataclass
class DataPipelineResult:
    """Result of data loading operation."""

    data: dict[str, MarketData]  # symbol -> MarketData
    symbols_loaded: list[str] = field(default_factory=list)
    symbols_failed: list[str] = field(default_factory=list)
    total_bars: int = 0
    date_range: tuple[datetime | None, datetime | None] = (None, None)

    @property
    def n_symbols(self) -> int:
        """Number of successfully loaded symbols."""
        return len(self.symbols_loaded)

    def get_returns_matrix(self, timeframe: str = "1D") -> pd.DataFrame:
        """Get returns matrix for all symbols."""
        returns_dict = {}
        for symbol, market_data in self.data.items():
            if timeframe == "1D" and market_data.timeframe != "1D":
                market_data = market_data.to_daily()
            returns_dict[symbol] = market_data.returns

        return pd.DataFrame(returns_dict).dropna()


# =============================================================================
# DATA PIPELINE
# =============================================================================

class DataPipeline:
    """
    Data loading and processing pipeline.

    Loads market data from massive flat files and handles aggregation
    to different timeframes.
    """

    def __init__(
        self,
        data_dir: Path = MASSIVE_DATA_DIR,
        config: BacktestingConfig | None = None,
    ):
        """
        Initialize data pipeline.

        Args:
            data_dir: Directory containing massive flat files
            config: Backtesting configuration
        """
        self.data_dir = Path(data_dir)
        self.config = config or BacktestingConfig()
        self._file_cache: dict[str, pd.DataFrame] = {}

    def _load_gz_file(self, gz_path: Path) -> pd.DataFrame | None:
        """
        Load a gzipped CSV file with caching.

        Args:
            gz_path: Path to .csv.gz file

        Returns:
            DataFrame or None if failed
        """
        cache_key = str(gz_path)
        if cache_key in self._file_cache:
            return self._file_cache[cache_key]

        try:
            with gzip.open(gz_path, "rt") as f:
                df = pd.read_csv(f)
            self._file_cache[cache_key] = df
            return df
        except Exception as e:
            logger.warning(f"Failed to load {gz_path}: {e}")
            return None

    def get_available_dates(self) -> list[str]:
        """Get list of available data dates."""
        dates = []
        for gz_file in sorted(self.data_dir.glob("*.csv.gz")):
            # Extract date from filename (e.g., 2025-11-17.csv.gz)
            date_str = gz_file.stem.replace(".csv", "")
            dates.append(date_str)
        return dates

    def get_available_symbols(self, date: str | None = None) -> list[str]:
        """
        Get list of available symbols.

        Args:
            date: Optional specific date to check. If None, uses latest file.

        Returns:
            List of available ticker symbols
        """
        gz_files = sorted(self.data_dir.glob("*.csv.gz"))
        if not gz_files:
            return []

        # Use specified date or latest
        if date:
            target_file = self.data_dir / f"{date}.csv.gz"
            if not target_file.exists():
                logger.warning(f"File for date {date} not found")
                return []
        else:
            target_file = gz_files[-1]

        df = self._load_gz_file(target_file)
        if df is None or "ticker" not in df.columns:
            return []

        tickers = df["ticker"].dropna().unique().tolist()
        return sorted([t for t in tickers if isinstance(t, str)])

    def load_symbol(
        self,
        symbol: str,
        aggregate_mins: int = 5,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> MarketData | None:
        """
        Load all available data for a symbol.

        Args:
            symbol: Ticker symbol
            aggregate_mins: Aggregation period in minutes (default 5)
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            MarketData or None if insufficient data
        """
        dfs = []

        for gz_file in sorted(self.data_dir.glob("*.csv.gz")):
            # Date filtering
            file_date = gz_file.stem.replace(".csv", "")
            if start_date and file_date < start_date:
                continue
            if end_date and file_date > end_date:
                continue

            df = self._load_gz_file(gz_file)
            if df is None:
                continue

            # Filter for symbol
            sym_df = df[df["ticker"] == symbol].copy()
            if sym_df.empty:
                continue

            # Parse timestamps
            if "window_start" in sym_df.columns:
                sym_df["datetime"] = pd.to_datetime(
                    sym_df["window_start"], unit="ns", utc=True
                )
            elif "timestamp" in sym_df.columns:
                sym_df["datetime"] = pd.to_datetime(sym_df["timestamp"], utc=True)
            else:
                logger.warning(f"No timestamp column found for {symbol}")
                continue

            sym_df = sym_df.set_index("datetime")
            sym_df = sym_df[["open", "high", "low", "close", "volume"]]
            dfs.append(sym_df)

        if not dfs:
            logger.debug(f"No data found for {symbol}")
            return None

        # Combine all data
        combined = pd.concat(dfs).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]

        # Remove any rows with NaN or zero prices
        combined = combined[(combined["close"] > 0) & combined["close"].notna()]

        # Aggregate to specified timeframe
        if aggregate_mins > 1:
            combined = combined.resample(f"{aggregate_mins}min").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()

        # Minimum data requirement
        if len(combined) < 100:
            logger.debug(f"Insufficient data for {symbol}: {len(combined)} bars")
            return None

        timeframe = f"{aggregate_mins}min" if aggregate_mins > 1 else "1min"

        return MarketData(
            symbol=symbol,
            bars=combined,
            timeframe=timeframe,
            start_date=combined.index.min().to_pydatetime() if len(combined) > 0 else None,
            end_date=combined.index.max().to_pydatetime() if len(combined) > 0 else None,
        )

    def load_universe(
        self,
        symbols: list[str],
        aggregate_mins: int = 5,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> DataPipelineResult:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            aggregate_mins: Aggregation period in minutes
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            DataPipelineResult with loaded data
        """
        result = DataPipelineResult(data={})
        total_bars = 0
        min_date = None
        max_date = None

        for symbol in symbols:
            market_data = self.load_symbol(
                symbol,
                aggregate_mins=aggregate_mins,
                start_date=start_date,
                end_date=end_date,
            )

            if market_data is not None:
                result.data[symbol] = market_data
                result.symbols_loaded.append(symbol)
                total_bars += market_data.n_bars

                if market_data.start_date:
                    if min_date is None or market_data.start_date < min_date:
                        min_date = market_data.start_date
                if market_data.end_date:
                    if max_date is None or market_data.end_date > max_date:
                        max_date = market_data.end_date
            else:
                result.symbols_failed.append(symbol)

        result.total_bars = total_bars
        result.date_range = (min_date, max_date)

        logger.info(
            f"Loaded {len(result.symbols_loaded)}/{len(symbols)} symbols, "
            f"{total_bars:,} total bars"
        )

        return result

    def sample_periods(
        self,
        num_periods: int = 10,
        seed: int = 42,
    ) -> list[tuple[str, str]]:
        """
        Sample random 21-day periods from available data.

        For the specified sample years (2004, 2008, 2010, 2017, 2024),
        we use whatever data is available (currently 2025 data).

        Args:
            num_periods: Number of periods to sample
            seed: Random seed for reproducibility

        Returns:
            List of (start_date, end_date) tuples
        """
        available_dates = self.get_available_dates()
        if len(available_dates) < self.config.period_length_days:
            logger.warning(
                f"Only {len(available_dates)} dates available, "
                f"need {self.config.period_length_days}"
            )
            # Return the full available range
            if available_dates:
                return [(available_dates[0], available_dates[-1])]
            return []

        # Sample periods
        rng = np.random.default_rng(seed)
        max_start_idx = len(available_dates) - self.config.period_length_days

        periods = []
        for _ in range(num_periods):
            start_idx = rng.integers(0, max_start_idx + 1)
            end_idx = start_idx + self.config.period_length_days - 1
            periods.append((available_dates[start_idx], available_dates[end_idx]))

        return periods

    def clear_cache(self) -> None:
        """Clear the file cache to free memory."""
        self._file_cache.clear()
        logger.info("Data cache cleared")


def compute_correlation_matrix(
    returns: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 20,
) -> pd.DataFrame:
    """
    Compute correlation matrix from returns DataFrame.

    Args:
        returns: DataFrame with returns (columns are symbols)
        method: Correlation method (pearson, spearman, kendall)
        min_periods: Minimum periods for valid correlation

    Returns:
        Correlation matrix DataFrame
    """
    return returns.corr(method=method, min_periods=min_periods)


def compute_covariance_matrix(
    returns: pd.DataFrame,
    halflife: int | None = None,
    min_periods: int = 20,
) -> pd.DataFrame:
    """
    Compute covariance matrix with optional exponential decay.

    Args:
        returns: DataFrame with returns (columns are symbols)
        halflife: Halflife for exponential weighting (None for equal weight)
        min_periods: Minimum periods for valid covariance

    Returns:
        Covariance matrix DataFrame
    """
    if halflife is not None:
        # Exponentially weighted covariance
        return returns.ewm(halflife=halflife, min_periods=min_periods).cov().iloc[-len(returns.columns):]
    else:
        return returns.cov(min_periods=min_periods)


if __name__ == "__main__":
    # Test data pipeline
    pipeline = DataPipeline()

    print("Available dates:", pipeline.get_available_dates()[:5], "...")
    print("Available symbols:", pipeline.get_available_symbols()[:10], "...")

    # Load a single symbol
    test_symbols = ["RIOT", "MARA", "SOFI"]
    result = pipeline.load_universe(test_symbols, aggregate_mins=5)

    print(f"\nLoaded: {result.symbols_loaded}")
    print(f"Failed: {result.symbols_failed}")
    print(f"Total bars: {result.total_bars:,}")
    print(f"Date range: {result.date_range}")

    if result.n_symbols >= 2:
        # Test correlation matrix
        returns = result.get_returns_matrix("1D")
        print(f"\nReturns shape: {returns.shape}")

        corr = compute_correlation_matrix(returns)
        print(f"Correlation matrix:\n{corr}")
