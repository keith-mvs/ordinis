"""
Massive Data Loader.

Loads only real historical data exported through Massive.
NO SYNTHETIC DATA - production-quality backtests only.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Base path for historical data
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "historical"

# Available symbols from Massive export
MASSIVE_SYMBOLS = [
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

# Categorization for strategy selection
VOLATILE_SYMBOLS = ["NVDA", "META", "GOOGL", "MS", "GS"]  # Tech/Finance high beta
STABLE_SYMBOLS = ["JNJ", "PFE", "WMT", "MCD", "HD"]  # Consumer/Healthcare
ENERGY_SYMBOLS = ["XOM", "CVX", "COP", "EOG", "SLB"]  # Energy sector
FINANCE_SYMBOLS = ["JPM", "BAC", "GS", "MS", "WFC"]  # Financials
TECH_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]  # Technology


def load_symbol(symbol: str, min_days: int = 252) -> pd.DataFrame | None:
    """
    Load historical data for a single symbol.

    Returns None if symbol not available or insufficient data.
    """
    # Try CSV first (more complete)
    csv_path = DATA_DIR / f"{symbol}_historical.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            df = df.rename(columns={"Date": "date"})
            df = df.set_index("date")

            # Standardize column names
            df.columns = df.columns.str.lower()

            if len(df) >= min_days:
                logger.debug(f"Loaded {symbol}: {len(df)} days from CSV")
                return df
            logger.warning(f"{symbol}: Only {len(df)} days (need {min_days})")
            return None
        except Exception as e:
            logger.error(f"Error loading {symbol} CSV: {e}")

    # Try parquet
    parquet_path = DATA_DIR / f"{symbol}.parquet"
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            df.columns = df.columns.str.lower()

            if len(df) >= min_days:
                logger.debug(f"Loaded {symbol}: {len(df)} days from parquet")
                return df
            logger.warning(f"{symbol}: Only {len(df)} days (need {min_days})")
            return None
        except Exception as e:
            logger.error(f"Error loading {symbol} parquet: {e}")

    logger.warning(f"No data file found for {symbol}")
    return None


def load_universe(
    symbols: list[str] | None = None,
    min_days: int = 252,
) -> dict[str, pd.DataFrame]:
    """
    Load historical data for multiple symbols.

    Args:
        symbols: List of symbols to load. If None, loads all available.
        min_days: Minimum days required for inclusion.

    Returns:
        Dictionary of symbol -> DataFrame
    """
    if symbols is None:
        symbols = MASSIVE_SYMBOLS

    universe = {}

    for symbol in symbols:
        df = load_symbol(symbol, min_days)
        if df is not None:
            universe[symbol] = df

    logger.info(f"Loaded {len(universe)}/{len(symbols)} symbols with >= {min_days} days")
    return universe


def get_common_date_range(
    universe: dict[str, pd.DataFrame],
) -> pd.DatetimeIndex:
    """Get date range common to all symbols in universe."""
    if not universe:
        return pd.DatetimeIndex([])

    common_dates = None
    for symbol, df in universe.items():
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(set(df.index))

    return pd.DatetimeIndex(sorted(list(common_dates)))


def build_returns_matrix(
    universe: dict[str, pd.DataFrame],
    price_col: str = "close",
) -> pd.DataFrame:
    """Build matrix of returns for all symbols."""
    returns = {}

    for symbol, df in universe.items():
        if price_col in df.columns:
            returns[symbol] = df[price_col].pct_change()

    return pd.DataFrame(returns).dropna()


def get_symbol_stats(df: pd.DataFrame) -> dict:
    """Calculate basic statistics for a symbol."""
    returns = df["close"].pct_change().dropna()

    return {
        "days": len(df),
        "start_date": df.index.min(),
        "end_date": df.index.max(),
        "avg_return": returns.mean() * 252,
        "volatility": returns.std() * (252**0.5),
        "sharpe": (returns.mean() * 252) / (returns.std() * (252**0.5)) if returns.std() > 0 else 0,
        "min_price": df["close"].min(),
        "max_price": df["close"].max(),
        "current_price": df["close"].iloc[-1],
    }


if __name__ == "__main__":
    # Test the loader
    logging.basicConfig(level=logging.INFO)

    print("Available Massive symbols:", MASSIVE_SYMBOLS)
    print(f"\nTotal: {len(MASSIVE_SYMBOLS)} symbols")

    # Load all
    universe = load_universe()

    print(f"\nLoaded {len(universe)} symbols:")
    for symbol, df in universe.items():
        stats = get_symbol_stats(df)
        print(
            f"  {symbol}: {stats['days']} days, "
            f"Vol: {stats['volatility']*100:.1f}%, "
            f"Sharpe: {stats['sharpe']:.2f}"
        )
