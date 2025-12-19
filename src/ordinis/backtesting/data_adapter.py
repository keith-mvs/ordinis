"""Historical data adapters and loaders for backtesting."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DataAdapter:
    """Normalizes market data to common schema (OHLCV + fundamentals)."""

    def normalize_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data to OHLCV format with datetime index.

        Args:
            data: DataFrame with OHLCV columns (any case)

        Returns:
            Normalized DataFrame with lowercase OHLCV columns and datetime index

        Raises:
            ValueError: If required columns missing or data invalid
        """
        # Normalize column names
        cols_lower = {col.lower(): col for col in data.columns}
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(cols_lower.keys())

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Reorder to standard
        normalized = data[[cols_lower[col] for col in required]].copy()
        normalized.columns = required

        # Ensure datetime index
        if not isinstance(normalized.index, pd.DatetimeIndex):
            if "date" in cols_lower or "timestamp" in cols_lower:
                col_name = cols_lower.get("date") or cols_lower.get("timestamp")
                # Convert to UTC and remove timezone for consistent comparison
                normalized.index = pd.to_datetime(data[col_name], utc=True).dt.tz_localize(None)
            else:
                normalized.index = pd.to_datetime(normalized.index, utc=True).tz_localize(None)

        # Sort by timestamp
        normalized = normalized.sort_index()

        # Validate OHLC relationships
        invalid_high_low = normalized["high"] < normalized["low"]
        if invalid_high_low.any():
            print(f"Warning: Dropping {invalid_high_low.sum()} bars with High < Low")
            normalized = normalized[~invalid_high_low]

        invalid_ohlc = (normalized["high"] < normalized[["open", "close"]].max(axis=1)) | (
            normalized["low"] > normalized[["open", "close"]].min(axis=1)
        )
        if invalid_ohlc.any():
            print(f"Warning: Dropping {invalid_ohlc.sum()} bars with invalid OHLC relationships")
            normalized = normalized[~invalid_ohlc]

        return normalized

    def attach_fundamentals(self, data: pd.DataFrame, fundamentals: dict[str, Any]) -> pd.DataFrame:
        """Attach fundamental data snapshot.

        Args:
            data: OHLCV DataFrame
            fundamentals: Dict with PE, PB, DivYield, etc.

        Returns:
            DataFrame with fundamentals as columns (broadcast)
        """
        for key, value in fundamentals.items():
            data[key] = value

        return data


class HistoricalDataLoader:
    """Loads and caches historical data from CSV/Parquet."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize loader.

        Args:
            data_dir: Directory containing historical data files
        """
        self.data_dir = data_dir or Path("data/historical")
        self._cache: dict[str, pd.DataFrame] = {}
        self._adapter = DataAdapter()

    def load_symbol(
        self, symbol: str, start_date: str | None = None, end_date: str | None = None
    ) -> pd.DataFrame:
        """Load historical data for a symbol.

        Args:
            symbol: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Normalized OHLCV DataFrame

        Raises:
            FileNotFoundError: If data file not found
            ValueError: If data format invalid
        """
        # Try parquet first, then CSV
        # Check both {symbol}.ext and {symbol}_historical.ext
        # Search recursively in data_dir
        for ext in ["parquet", "csv"]:
            for pattern in [f"{symbol}.{ext}", f"{symbol}_historical.{ext}"]:
                # Try direct path first
                filepath = self.data_dir / pattern
                if filepath.exists():
                    return self._load_file(filepath, ext, start_date, end_date, symbol)

                # Try recursive search
                found_files = list(self.data_dir.rglob(pattern))
                if found_files:
                    return self._load_file(found_files[0], ext, start_date, end_date, symbol)

        raise FileNotFoundError(f"No data found for {symbol} in {self.data_dir}")

    def _load_file(
        self, filepath: Path, ext: str, start_date: str | None, end_date: str | None, symbol: str
    ) -> pd.DataFrame:
        """Helper to load and normalize file."""
        if ext == "parquet":
            data = pd.read_parquet(filepath)
        else:
            data = pd.read_csv(filepath)

        # Normalize
        data = self._adapter.normalize_ohlcv(data)

        # Filter date range
        if start_date:
            data = data[data.index >= pd.Timestamp(start_date)]
        if end_date:
            data = data[data.index <= pd.Timestamp(end_date)]

        self._cache[symbol] = data
        return data

    def load_batch(
        self,
        symbols: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load data for multiple symbols.

        Args:
            symbols: List of tickers
            start_date: Start date
            end_date: End date

        Returns:
            Dict of symbol -> DataFrame
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.load_symbol(symbol, start_date, end_date)
            except FileNotFoundError:
                # Skip missing symbols
                continue

        return data

    def get_cached(self, symbol: str) -> pd.DataFrame | None:
        """Get cached data (or None if not loaded)."""
        return self._cache.get(symbol)

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
