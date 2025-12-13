"""
Parallelized Dataset Fetch - Max CPU Utilization

Fetches 101 symbols in parallel using all available CPU cores.
"""

import math
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from enhanced_dataset_config_v2 import (
    EXPANDED_SYMBOL_UNIVERSE as ENHANCED_SYMBOL_UNIVERSE,
    get_all_symbols_by_market_cap_v2 as get_all_symbols_by_market_cap,
)


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility features inline using vectorized operations for better performance."""
    # True Range - fully vectorized calculation (much faster than apply with axis=1)
    prev_close = df["close"].shift(1)

    # Calculate all three components vectorized
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - prev_close).abs()
    low_prev_close = (df["low"] - prev_close).abs()

    # Use numpy maximum for better performance than pd.concat
    df["true_range"] = np.maximum.reduce([
        high_low.values,
        high_prev_close.values,
        low_prev_close.values
    ])

    # ATR (14-day)
    df["atr_14"] = df["true_range"].rolling(window=14).mean()

    # Historical Volatility (20-day, annualized)
    df["hvol_20"] = df["close"].pct_change().rolling(window=20).std() * (252 ** 0.5)

    # Parkinson Volatility (20-day, annualized) - fully vectorized
    # Formula: σ = sqrt((1 / (4 * ln(2))) * mean(ln(high/low)²))
    high_low_ratio = df["high"] / df["low"]
    log_hl_sq = np.log(high_low_ratio) ** 2
    parkinson_variance = log_hl_sq.rolling(window=20).mean() / (4 * math.log(2))
    df["parkinson_vol_20"] = np.sqrt(parkinson_variance) * np.sqrt(252)

    return df


def fetch_symbol(args):
    """Fetch a single symbol (parallelizable)."""
    symbol, market_cap, start_date, end_date, output_dir, format = args

    try:
        # Determine output directory
        if market_cap == "LARGE":
            target_dir = output_dir / "historical" / "large_cap"
        elif market_cap == "MID":
            target_dir = output_dir / "historical" / "mid_cap"
        elif market_cap == "SMALL":
            target_dir = output_dir / "historical" / "small_cap"
        elif market_cap == "ETF":
            target_dir = output_dir / "historical" / "etfs"
        else:
            return (symbol, None, "Unknown market cap")

        target_dir.mkdir(parents=True, exist_ok=True)

        # Fetch from yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            return (symbol, None, "No data")

        # Standardize columns
        df.columns = [col.lower() for col in df.columns]
        df = df.rename(columns={"adj close": "adj_close"})
        df["symbol"] = symbol

        # Add volatility features
        df = add_volatility_features(df)

        # Save
        output_file = target_dir / f"{symbol}_historical.{format}"
        if format == "csv":
            df.to_csv(output_file)
        elif format == "parquet":
            df.to_parquet(output_file)

        return (symbol, len(df), None)

    except Exception as e:
        return (symbol, None, str(e))


def fetch_parallel(years=20, output_dir=Path("data"), format="csv", max_workers=None):
    """Fetch all symbols in parallel."""
    start_date = datetime.now() - timedelta(days=years * 365)
    end_date = datetime.now()

    # Get all symbols by market cap
    by_market_cap = get_all_symbols_by_market_cap()

    # Build task list
    tasks = []
    for market_cap, symbols in by_market_cap.items():
        for symbol in symbols:
            tasks.append((symbol, market_cap, start_date, end_date, output_dir, format))

    # Determine optimal worker count
    if max_workers is None:
        max_workers = mp.cpu_count()

    print("=" * 80)
    print("PARALLEL DATASET FETCH")
    print("=" * 80)
    print(f"Total Symbols: {len(tasks)}")
    print(f"CPU Cores: {mp.cpu_count()}")
    print(f"Workers: {max_workers}")
    print(f"Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"End: {end_date.strftime('%Y-%m-%d')}")
    print("=" * 80)

    # Track progress
    completed = 0
    successful = 0
    failed = 0
    results = []

    # Execute in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_symbol, task): task for task in tasks}

        for future in as_completed(futures):
            symbol, bars, error = future.result()
            completed += 1

            if error is None:
                successful += 1
                print(f"[{completed}/{len(tasks)}] {symbol}: OK ({bars} bars)")
                results.append({
                    "symbol": symbol,
                    "status": "SUCCESS",
                    "bars": bars,
                    "market_cap": futures[future][1],
                })
            else:
                failed += 1
                print(f"[{completed}/{len(tasks)}] {symbol}: FAILED ({error})")
                results.append({
                    "symbol": symbol,
                    "status": "FAILED",
                    "error": error,
                    "market_cap": futures[future][1],
                })

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total: {len(tasks)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {successful / len(tasks) * 100:.1f}%")

    # Save results summary
    results_df = pd.DataFrame(results)
    results_file = output_dir / "fetch_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved: {results_file}")

    # Generate metadata for successful fetches
    print("\nGenerating metadata...")
    metadata_rows = []

    for result in results:
        if result["status"] == "SUCCESS":
            symbol = result["symbol"]
            market_cap = result["market_cap"]

            # Find file
            if market_cap == "LARGE":
                file_path = output_dir / "historical" / "large_cap" / f"{symbol}_historical.{format}"
            elif market_cap == "MID":
                file_path = output_dir / "historical" / "mid_cap" / f"{symbol}_historical.{format}"
            elif market_cap == "SMALL":
                file_path = output_dir / "historical" / "small_cap" / f"{symbol}_historical.{format}"
            elif market_cap == "ETF":
                file_path = output_dir / "historical" / "etfs" / f"{symbol}_historical.{format}"
            else:
                continue

            if file_path.exists():
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)

                # Find metadata
                sector = None
                volatility = None
                bull_performer = None

                for group_data in ENHANCED_SYMBOL_UNIVERSE.values():
                    if symbol in group_data["symbols"]:
                        sector = group_data["sector"]
                        volatility = group_data.get("volatility")
                        bull_performer = group_data.get("bull_performer")
                        break

                metadata_rows.append({
                    "symbol": symbol,
                    "market_cap": market_cap,
                    "sector": sector,
                    "bull_performer": bull_performer,
                    "target_volatility": volatility,
                    "start_date": df.index[0] if len(df) > 0 else None,
                    "end_date": df.index[-1] if len(df) > 0 else None,
                    "num_bars": len(df),
                    "num_features": len(df.columns),
                    "avg_price": df["close"].mean() if "close" in df.columns else None,
                    "avg_volume": df["volume"].mean() if "volume" in df.columns else None,
                    "realized_volatility": df["close"].pct_change().std() * (252 ** 0.5) if "close" in df.columns else None,
                    "file_path": str(file_path.relative_to(output_dir)),
                })

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_file = output_dir / "enhanced_dataset_metadata.csv"
    metadata_df.to_csv(metadata_file, index=False)
    print(f"Metadata saved: {metadata_file} ({len(metadata_rows)} symbols)")

    print("=" * 80)
    print("FETCH COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel dataset fetch")
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--format", type=str, default="csv", choices=["csv", "parquet"])
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: all CPU cores)")

    args = parser.parse_args()

    fetch_parallel(
        years=args.years,
        output_dir=Path(args.output),
        format=args.format,
        max_workers=args.workers,
    )
