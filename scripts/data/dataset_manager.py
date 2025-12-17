"""Comprehensive Dataset Manager for Trading Strategy Validation.

Generates and retrieves datasets spanning multiple sectors and timeframes:
- Synthetic data with realistic statistical properties
- Historical data covering 20+ years across sectors
- Rolling 2-3 month windows for walk-forward analysis
- Rich features: OHLCV, volume, volatility, macro indicators

Usage:
    python scripts/dataset_manager.py --mode historical --years 20 --output data/historical
    python scripts/dataset_manager.py --mode synthetic --sectors TECH,FINANCE --output data/synthetic
    python scripts/dataset_manager.py --mode combined --years 10 --window-months 3 --output data/combined
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class DatasetConfig:
    """Configuration for dataset generation and retrieval.

    Attributes:
        mode: 'synthetic', 'historical', or 'combined'
        years_back: Years of historical data (default: 20)
        window_months: Window size in months (default: 3)
        sectors: List of sectors to include
        symbols_per_sector: Number of symbols per sector
        start_date: Start date for historical data
        end_date: End date for historical data
        include_macro: Include macro indicators
        include_volatility: Include volatility measures
        output_format: 'csv', 'parquet', or 'json'
    """

    mode: str = "combined"
    years_back: int = 20
    window_months: int = 3
    sectors: list[str] = field(
        default_factory=lambda: ["TECH", "FINANCE", "HEALTHCARE", "ENERGY", "CONSUMER"]
    )
    symbols_per_sector: int = 5
    start_date: str | None = None
    end_date: str | None = None
    include_macro: bool = True
    include_volatility: bool = True
    output_format: str = "csv"


# Sector definitions with representative symbols
SECTOR_SYMBOLS = {
    "TECH": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "AMD", "INTC", "ORCL", "CRM"],
    "FINANCE": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "SCHW", "BLK"],
    "HEALTHCARE": ["UNH", "JNJ", "PFE", "ABBV", "TMO", "DHR", "ABT", "MRK", "LLY", "BMY"],
    "ENERGY": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "CONSUMER": ["WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "DG", "ROST"],
    "INDUSTRIAL": ["BA", "CAT", "GE", "UPS", "HON", "RTX", "LMT", "DE", "MMM", "FDX"],
    "MATERIALS": ["LIN", "APD", "ECL", "SHW", "DD", "NEM", "FCX", "NUE", "VMC", "MLM"],
    "REAL_ESTATE": ["AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "O", "WELL", "AVB", "EQR"],
    "UTILITIES": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "WEC", "ES"],
    "COMMUNICATION": ["T", "VZ", "TMUS", "CMCSA", "DIS", "NFLX", "CHTR", "EA", "ATVI", "TTWO"],
}

# Macro indicators to include
MACRO_INDICATORS = [
    "SPY",  # S&P 500 ETF (market benchmark)
    "^VIX",  # VIX (volatility index)
    "^TNX",  # 10-Year Treasury Yield
    "DXY",  # US Dollar Index
    "GLD",  # Gold ETF
    "USO",  # Oil ETF
]


# ═══════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════


def generate_synthetic_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    base_price: float = 100.0,
    volatility: float = 0.02,
    drift: float = 0.0001,
    sector: str = "TECH",
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic statistical properties.

    Uses geometric Brownian motion with sector-specific characteristics.

    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        base_price: Starting price
        volatility: Daily volatility (annualized / sqrt(252))
        drift: Daily drift (annualized / 252)
        sector: Sector for characteristic adjustments

    Returns:
        DataFrame with OHLCV data and datetime index
    """
    # Generate trading days
    days = pd.bdate_range(start=start_date, end=end_date)
    n_days = len(days)

    # Seed for reproducibility
    np.random.seed(hash(symbol + sector) % (2**32))

    # Sector-specific adjustments
    sector_vol_multipliers = {
        "TECH": 1.3,  # Higher volatility
        "FINANCE": 1.5,  # Highest volatility
        "HEALTHCARE": 0.9,  # Moderate volatility
        "ENERGY": 1.4,  # High volatility
        "CONSUMER": 0.8,  # Lower volatility
        "INDUSTRIAL": 1.0,  # Average volatility
        "MATERIALS": 1.2,  # Above-average volatility
        "REAL_ESTATE": 0.7,  # Low volatility
        "UTILITIES": 0.6,  # Lowest volatility
        "COMMUNICATION": 1.1,  # Above-average volatility
    }

    vol_multiplier = sector_vol_multipliers.get(sector, 1.0)
    adj_vol = volatility * vol_multiplier

    # Generate price path (geometric Brownian motion)
    returns = np.random.normal(drift, adj_vol, n_days)

    # Add autocorrelation for realism
    for i in range(1, n_days):
        returns[i] += 0.1 * returns[i - 1]  # Slight momentum

    # Calculate close prices
    close = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    intraday_vol = adj_vol * 0.5  # Intraday volatility
    open_prices = close * (1 + np.random.normal(0, intraday_vol * 0.3, n_days))
    high_prices = np.maximum(open_prices, close) * (
        1 + np.abs(np.random.normal(0, intraday_vol, n_days))
    )
    low_prices = np.minimum(open_prices, close) * (
        1 - np.abs(np.random.normal(0, intraday_vol, n_days))
    )

    # Ensure high >= open, close and low <= open, close
    high_prices = np.maximum(high_prices, np.maximum(open_prices, close))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, close))

    # Generate volume with clustering
    base_volume = 1000000
    volume_mult = np.exp(np.random.normal(0, 0.5, n_days))
    # Volume tends to be higher on volatile days
    vol_factor = 1 + 2 * np.abs(returns / adj_vol)
    volume = (base_volume * volume_mult * vol_factor).astype(int)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close,
            "volume": volume,
            "symbol": symbol,
            "sector": sector,
        },
        index=days,
    )

    return df


def generate_synthetic_dataset(
    config: DatasetConfig,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Generate complete synthetic dataset across sectors.

    Args:
        config: Dataset configuration
        output_dir: Output directory

    Returns:
        Dictionary of DataFrames by symbol
    """
    print("\nGenerating Synthetic Dataset")
    print(f"Years: {config.years_back}")
    print(f"Sectors: {', '.join(config.sectors)}")
    print(f"Symbols per sector: {config.symbols_per_sector}\n")

    # Date range
    end_date = datetime.now() if not config.end_date else datetime.fromisoformat(config.end_date)
    start_date = end_date - timedelta(days=365 * config.years_back)

    datasets = {}

    for sector in config.sectors:
        symbols = SECTOR_SYMBOLS.get(sector, [])[: config.symbols_per_sector]

        for symbol in symbols:
            print(f"Generating {symbol} ({sector})...")

            # Sector-specific base prices
            base_prices = {
                "TECH": 150,
                "FINANCE": 50,
                "HEALTHCARE": 100,
                "ENERGY": 70,
                "CONSUMER": 120,
                "INDUSTRIAL": 90,
                "MATERIALS": 80,
                "REAL_ESTATE": 110,
                "UTILITIES": 60,
                "COMMUNICATION": 55,
            }

            df = generate_synthetic_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                base_price=base_prices.get(sector, 100),
                sector=sector,
            )

            datasets[symbol] = df

            # Save individual file
            output_file = output_dir / f"{symbol}_synthetic.{config.output_format}"
            if config.output_format == "csv":
                df.to_csv(output_file)
            elif config.output_format == "parquet":
                df.to_parquet(output_file)
            elif config.output_format == "json":
                df.to_json(output_file, orient="records", date_format="iso")

    print(f"\nGenerated {len(datasets)} synthetic datasets")
    return datasets


# ═══════════════════════════════════════════════════════════════════════
# HISTORICAL DATA RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════


def fetch_historical_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    source: str = "yfinance",
) -> pd.DataFrame | None:
    """Fetch historical market data from external source.

    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        source: Data source ('yfinance', 'massive', 'alphavantage')

    Returns:
        DataFrame with OHLCV data or None if failed
    """
    try:
        if source == "yfinance":
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                print(f"  [WARN] No data for {symbol}")
                return None

            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            df = df.rename(columns={"adj close": "adj_close"})

            # Add metadata
            df["symbol"] = symbol

            return df

        print(f"  [WARN] Source '{source}' not implemented")
        return None

    except Exception as e:
        print(f"  [ERROR] Error fetching {symbol}: {e}")
        return None


def retrieve_historical_dataset(
    config: DatasetConfig,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Retrieve complete historical dataset across sectors.

    Args:
        config: Dataset configuration
        output_dir: Output directory

    Returns:
        Dictionary of DataFrames by symbol
    """
    print("\nRetrieving Historical Dataset")
    print(f"Years: {config.years_back}")
    print(f"Sectors: {', '.join(config.sectors)}")
    print(f"Symbols per sector: {config.symbols_per_sector}\n")

    # Date range
    end_date = datetime.now() if not config.end_date else datetime.fromisoformat(config.end_date)
    start_date = end_date - timedelta(days=365 * config.years_back)

    datasets = {}

    for sector in config.sectors:
        symbols = SECTOR_SYMBOLS.get(sector, [])[: config.symbols_per_sector]

        print(f"\n{sector} Sector:")
        for symbol in symbols:
            print(f"  Fetching {symbol}...", end=" ")

            df = fetch_historical_data(symbol, start_date, end_date)

            if df is not None:
                df["sector"] = sector
                datasets[symbol] = df
                print(f"[OK] {len(df)} bars")

                # Save individual file
                output_file = output_dir / f"{symbol}_historical.{config.output_format}"
                if config.output_format == "csv":
                    df.to_csv(output_file)
                elif config.output_format == "parquet":
                    df.to_parquet(output_file)
                elif config.output_format == "json":
                    df.to_json(output_file, orient="records", date_format="iso")

    # Fetch macro indicators
    if config.include_macro:
        print("\nMacro Indicators:")
        for symbol in MACRO_INDICATORS:
            print(f"  Fetching {symbol}...", end=" ")

            df = fetch_historical_data(symbol, start_date, end_date)

            if df is not None:
                df["sector"] = "MACRO"
                datasets[symbol] = df
                print(f"[OK] {len(df)} bars")

                output_file = (
                    output_dir / f"{symbol.replace('^', 'INDEX_')}_macro.{config.output_format}"
                )
                if config.output_format == "csv":
                    df.to_csv(output_file)

    print(f"\nRetrieved {len(datasets)} historical datasets")
    return datasets


# ═══════════════════════════════════════════════════════════════════════
# WINDOWING
# ═══════════════════════════════════════════════════════════════════════


def create_rolling_windows(
    df: pd.DataFrame,
    window_months: int = 3,
    step_months: int = 1,
) -> list[tuple[datetime, datetime, pd.DataFrame]]:
    """Create rolling windows from dataset.

    Args:
        df: Input DataFrame
        window_months: Window size in months
        step_months: Step size in months

    Returns:
        List of (start_date, end_date, window_df) tuples
    """
    windows = []

    start = df.index.min()
    end = df.index.max()

    current_start = start

    while current_start + pd.DateOffset(months=window_months) <= end:
        current_end = current_start + pd.DateOffset(months=window_months)

        window_df = df.loc[current_start:current_end]

        if len(window_df) > 0:
            windows.append((current_start, current_end, window_df))

        current_start += pd.DateOffset(months=step_months)

    return windows


def generate_windowed_dataset(
    datasets: dict[str, pd.DataFrame],
    config: DatasetConfig,
    output_dir: Path,
) -> None:
    """Generate windowed datasets for walk-forward analysis.

    Args:
        datasets: Dictionary of DataFrames by symbol
        config: Dataset configuration
        output_dir: Output directory
    """
    print(f"\nGenerating {config.window_months}-month windows...")

    window_dir = output_dir / "windows"
    window_dir.mkdir(parents=True, exist_ok=True)

    # Create windows for each symbol
    all_windows = {}

    for symbol, df in datasets.items():
        windows = create_rolling_windows(df, config.window_months)
        all_windows[symbol] = windows
        print(f"  {symbol}: {len(windows)} windows")

        # Save windows
        for i, (start, end, window_df) in enumerate(windows):
            window_file = (
                window_dir
                / f"{symbol}_window_{i:03d}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
            )
            window_df.to_csv(window_file)

    # Create metadata file
    metadata = []
    for symbol, windows in all_windows.items():
        for i, (start, end, window_df) in enumerate(windows):
            metadata.append(
                {
                    "symbol": symbol,
                    "window_id": i,
                    "start_date": start.isoformat(),
                    "end_date": end.isoformat(),
                    "num_bars": len(window_df),
                    "sector": window_df["sector"].iloc[0]
                    if "sector" in window_df.columns
                    else "UNKNOWN",
                }
            )

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_dir / "windows_metadata.csv", index=False)

    print(f"\nGenerated {len(metadata)} total windows")
    print("Metadata saved to: windows_metadata.csv")


# ═══════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility measures to dataset.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with volatility features
    """
    df = df.copy()

    # True range
    df["high_low"] = df["high"] - df["low"]
    df["high_close"] = np.abs(df["high"] - df["close"].shift())
    df["low_close"] = np.abs(df["low"] - df["close"].shift())
    df["true_range"] = df[["high_low", "high_close", "low_close"]].max(axis=1)

    # ATR (14-day)
    df["atr_14"] = df["true_range"].rolling(14).mean()

    # Historical volatility (20-day)
    df["returns"] = df["close"].pct_change()
    df["hvol_20"] = df["returns"].rolling(20).std() * np.sqrt(252)  # Annualized

    # Parkinson volatility (uses high-low range)
    df["parkinson_vol"] = np.sqrt((1 / (4 * np.log(2))) * ((np.log(df["high"] / df["low"])) ** 2))
    df["parkinson_vol_20"] = df["parkinson_vol"].rolling(20).mean() * np.sqrt(252)

    # Drop intermediate columns
    df = df.drop(columns=["high_low", "high_close", "low_close", "returns", "parkinson_vol"])

    return df


def enrich_dataset(
    datasets: dict[str, pd.DataFrame],
    config: DatasetConfig,
) -> dict[str, pd.DataFrame]:
    """Enrich datasets with additional features.

    Args:
        datasets: Dictionary of DataFrames
        config: Dataset configuration

    Returns:
        Enriched datasets
    """
    print("\nEnriching datasets with features...")

    enriched = {}

    for symbol, df in datasets.items():
        df_enriched = df.copy()

        # Add volatility features
        if config.include_volatility:
            df_enriched = add_volatility_features(df_enriched)

        enriched[symbol] = df_enriched

    print(f"  Added {len(df_enriched.columns) - len(df.columns)} features")

    return enriched


# ═══════════════════════════════════════════════════════════════════════
# METADATA GENERATION
# ═══════════════════════════════════════════════════════════════════════


def generate_metadata(
    datasets: dict[str, pd.DataFrame],
    config: DatasetConfig,
    output_dir: Path,
) -> pd.DataFrame:
    """Generate metadata for all datasets.

    Args:
        datasets: Dictionary of DataFrames
        config: Dataset configuration
        output_dir: Output directory

    Returns:
        Metadata DataFrame
    """
    metadata = []

    for symbol, df in datasets.items():
        meta = {
            "symbol": symbol,
            "sector": df["sector"].iloc[0] if "sector" in df.columns else "UNKNOWN",
            "start_date": df.index.min().isoformat(),
            "end_date": df.index.max().isoformat(),
            "num_bars": len(df),
            "num_features": len(df.columns),
            "avg_price": df["close"].mean(),
            "avg_volume": df["volume"].mean(),
            "volatility": df["close"].pct_change().std() * np.sqrt(252),
            "source": config.mode,
        }

        metadata.append(meta)

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_dir / "dataset_metadata.csv", index=False)

    print(f"\nMetadata generated for {len(metadata)} datasets")

    return metadata_df


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Comprehensive Dataset Manager")

    parser.add_argument(
        "--mode",
        choices=["synthetic", "historical", "combined"],
        default="combined",
        help="Dataset mode",
    )
    parser.add_argument("--years", type=int, default=20, help="Years of data")
    parser.add_argument("--window-months", type=int, default=3, help="Window size in months")
    parser.add_argument(
        "--sectors",
        type=str,
        default="TECH,FINANCE,HEALTHCARE,ENERGY,CONSUMER",
        help="Comma-separated sectors",
    )
    parser.add_argument("--symbols-per-sector", type=int, default=5, help="Symbols per sector")
    parser.add_argument("--output", type=str, default="data/datasets", help="Output directory")
    parser.add_argument(
        "--format", choices=["csv", "parquet", "json"], default="csv", help="Output format"
    )
    parser.add_argument("--no-macro", action="store_true", help="Exclude macro indicators")
    parser.add_argument("--no-volatility", action="store_true", help="Exclude volatility features")
    parser.add_argument("--no-windows", action="store_true", help="Skip windowing")

    args = parser.parse_args()

    # Create config
    config = DatasetConfig(
        mode=args.mode,
        years_back=args.years,
        window_months=args.window_months,
        sectors=args.sectors.split(","),
        symbols_per_sector=args.symbols_per_sector,
        include_macro=not args.no_macro,
        include_volatility=not args.no_volatility,
        output_format=args.format,
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("COMPREHENSIVE DATASET MANAGER")
    print("=" * 100)
    print(f"Mode: {config.mode}")
    print(f"Output: {output_dir}")
    print(f"Format: {config.output_format}")
    print("=" * 100)

    # Generate/retrieve datasets
    if config.mode == "synthetic":
        datasets = generate_synthetic_dataset(config, output_dir)
    elif config.mode == "historical":
        datasets = retrieve_historical_dataset(config, output_dir)
    else:  # combined
        print("\n=== SYNTHETIC DATA ===")
        synthetic_datasets = generate_synthetic_dataset(config, output_dir / "synthetic")

        print("\n\n=== HISTORICAL DATA ===")
        historical_datasets = retrieve_historical_dataset(config, output_dir / "historical")

        datasets = {**synthetic_datasets, **historical_datasets}

    # Enrich datasets
    datasets = enrich_dataset(datasets, config)

    # Generate windows
    if not args.no_windows:
        generate_windowed_dataset(datasets, config, output_dir)

    # Generate metadata
    generate_metadata(datasets, config, output_dir)

    print(f"\n{'=' * 100}")
    print("DATASET GENERATION COMPLETE")
    print(f"Total datasets: {len(datasets)}")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    main()
