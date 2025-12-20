"""
Massive Historical Data Loader.

Loads real historical data exported through Massive.
NO SYNTHETIC DATA - for production backtesting only.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Base path for Massive-exported historical data
HISTORICAL_DATA_PATH = Path("data/historical")

# Available symbols from Massive exports
AVAILABLE_SYMBOLS = [
    "AAPL",
    "ABBV",
    "BAC",
    "COP",
    "CVX",
    "EOG",
    "GOOGL",
    "GS",
    "HD",
    "JNJ",
    "JPM",
    "MCD",
    "META",
    "MSFT",
    "MS",
    "NKE",
    "NVDA",
    "PFE",
    "SBUX",
    "SLB",
    "TMO",
    "UNH",
    "WFC",
    "WMT",
    "XOM",
]

# Sector mapping for portfolio analysis
SECTOR_MAP = {
    "AAPL": "Tech",
    "GOOGL": "Tech",
    "META": "Tech",
    "MSFT": "Tech",
    "NVDA": "Tech",
    "BAC": "Finance",
    "GS": "Finance",
    "JPM": "Finance",
    "MS": "Finance",
    "WFC": "Finance",
    "HD": "Consumer",
    "MCD": "Consumer",
    "NKE": "Consumer",
    "SBUX": "Consumer",
    "WMT": "Consumer",
    "ABBV": "Healthcare",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "TMO": "Healthcare",
    "UNH": "Healthcare",
    "COP": "Energy",
    "CVX": "Energy",
    "EOG": "Energy",
    "SLB": "Energy",
    "XOM": "Energy",
}


def load_massive_data(symbol: str, min_days: int = 252) -> pd.DataFrame | None:
    """
    Load historical data for a symbol from Massive exports.

    Args:
        symbol: Stock ticker symbol
        min_days: Minimum required days of data

    Returns:
        DataFrame with OHLCV data or None if unavailable
    """
    if symbol not in AVAILABLE_SYMBOLS:
        logger.warning(f"{symbol} not available in Massive exports")
        return None

    # Try CSV first (more common)
    csv_path = HISTORICAL_DATA_PATH / f"{symbol}_historical.csv"
    parquet_path = HISTORICAL_DATA_PATH / f"{symbol}.parquet"

    df = None

    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.rename(columns={"Date": "date"})
        df = df.set_index("date")
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
            df = df.set_index("date")

    if df is None:
        logger.warning(f"No data file found for {symbol}")
        return None

    # Standardize column names to lowercase
    df.columns = df.columns.str.lower()

    # Ensure required columns exist
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.warning(f"{symbol} missing columns: {missing}")
        return None

    # Sort by date
    df = df.sort_index()

    # Check minimum data requirement
    if len(df) < min_days:
        logger.warning(f"{symbol} has only {len(df)} days (need {min_days})")
        return None

    logger.info(f"Loaded {symbol}: {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

    return df


def load_universe(
    symbols: list[str] | None = None,
    min_days: int = 252,
) -> dict[str, pd.DataFrame]:
    """
    Load data for multiple symbols.

    Args:
        symbols: List of symbols to load (defaults to all available)
        min_days: Minimum required days per symbol

    Returns:
        Dict mapping symbol to DataFrame
    """
    if symbols is None:
        symbols = AVAILABLE_SYMBOLS

    universe = {}
    for symbol in symbols:
        df = load_massive_data(symbol, min_days)
        if df is not None:
            universe[symbol] = df

    logger.info(f"Loaded {len(universe)}/{len(symbols)} symbols")
    return universe


def get_common_date_range(universe: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Get the common date range across all symbols in universe."""
    if not universe:
        return pd.DatetimeIndex([])

    common_dates = None
    for symbol, df in universe.items():
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(set(df.index))

    return pd.DatetimeIndex(sorted(common_dates))


def build_returns_panel(
    universe: dict[str, pd.DataFrame],
    common_dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a panel of returns for all symbols on common dates."""
    if common_dates is None:
        common_dates = get_common_date_range(universe)

    returns = {}
    for symbol, df in universe.items():
        aligned = df.loc[df.index.intersection(common_dates), "close"]
        returns[symbol] = aligned.pct_change()

    return pd.DataFrame(returns).dropna()


def get_symbols_by_sector(sector: str) -> list[str]:
    """Get all symbols in a given sector."""
    return [sym for sym, sec in SECTOR_MAP.items() if sec == sector]


def get_sector_for_symbol(symbol: str) -> str:
    """Get sector for a symbol."""
    return SECTOR_MAP.get(symbol, "Unknown")


# Convenience function for quick loading
def quick_load(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Quick load for a list of symbols with default settings."""
    return load_universe(symbols, min_days=252)


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)

    print("Available symbols:", AVAILABLE_SYMBOLS)
    print(f"\nTotal: {len(AVAILABLE_SYMBOLS)} symbols")

    # Load all
    universe = load_universe()

    # Print summary
    for symbol, df in universe.items():
        print(f"  {symbol}: {len(df)} days, {df.index[0].date()} to {df.index[-1].date()}")
