"""
Fetch Enhanced Dataset Collection

Fetches 100 stocks across all market caps and performance characteristics:
- 43 large cap
- 25 mid cap
- 20 small cap
- 13 ETFs

Organized by bull/bear market performance characteristics.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_manager import (
    enrich_dataset,
    fetch_historical_data,
)
from enhanced_dataset_config import (
    ENHANCED_SYMBOL_UNIVERSE,
    get_all_symbols_by_market_cap,
)
import pandas as pd


def fetch_enhanced_datasets(
    years: int = 20,
    output_dir: Path = Path("data"),
    format: str = "csv",
):
    """
    Fetch all enhanced datasets organized by market cap and characteristics.

    Args:
        years: Number of years of historical data
        output_dir: Base output directory
        format: Output format (csv or parquet)
    """
    start_date = datetime.now() - timedelta(days=years * 365)
    end_date = datetime.now()

    # Create subdirectories
    large_cap_dir = output_dir / "historical" / "large_cap"
    mid_cap_dir = output_dir / "historical" / "mid_cap"
    small_cap_dir = output_dir / "historical" / "small_cap"
    etf_dir = output_dir / "historical" / "etfs"

    for dir_path in [large_cap_dir, mid_cap_dir, small_cap_dir, etf_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Get symbols by market cap
    by_market_cap = get_all_symbols_by_market_cap()

    # Track statistics
    stats = {
        "total_attempted": 0,
        "total_successful": 0,
        "total_failed": 0,
        "by_market_cap": {},
    }

    print("=" * 80)
    print("FETCHING ENHANCED DATASET COLLECTION")
    print("=" * 80)
    print(f"Years: {years}")
    print(f"Start: {start_date.strftime('%Y-%m-%d')}")
    print(f"End: {end_date.strftime('%Y-%m-%d')}")
    print(f"Output: {output_dir}")
    print(f"Format: {format}")
    print("=" * 80)

    # Fetch by market cap category
    for cap_type, symbols in by_market_cap.items():
        if cap_type == "ETF":
            target_dir = etf_dir
        elif cap_type == "LARGE":
            target_dir = large_cap_dir
        elif cap_type == "MID":
            target_dir = mid_cap_dir
        elif cap_type == "SMALL":
            target_dir = small_cap_dir
        else:
            continue

        print(f"\n{cap_type} CAP ({len(symbols)} symbols):")
        print("-" * 80)

        cap_stats = {"attempted": 0, "successful": 0, "failed": 0}

        for symbol in sorted(symbols):
            stats["total_attempted"] += 1
            cap_stats["attempted"] += 1

            print(f"  Fetching {symbol}...", end=" ", flush=True)

            # Fetch data
            df = fetch_historical_data(symbol, start_date, end_date)

            if df is not None and not df.empty:
                # Enrich with features
                df = enrich_dataset(df)

                # Save file
                output_file = target_dir / f"{symbol}_historical.{format}"
                if format == "csv":
                    df.to_csv(output_file)
                elif format == "parquet":
                    df.to_parquet(output_file)

                print(f"[OK] {len(df)} bars")
                stats["total_successful"] += 1
                cap_stats["successful"] += 1
            else:
                print("[FAILED] No data")
                stats["total_failed"] += 1
                cap_stats["failed"] += 1

        stats["by_market_cap"][cap_type] = cap_stats

        print(f"  {cap_type}: {cap_stats['successful']}/{cap_stats['attempted']} successful")

    # Generate metadata
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Attempted: {stats['total_attempted']}")
    print(f"Total Successful: {stats['total_successful']}")
    print(f"Total Failed: {stats['total_failed']}")
    print(f"Success Rate: {stats['total_successful'] / stats['total_attempted'] * 100:.1f}%")

    print("\nBy Market Cap:")
    for cap_type, cap_stats in stats["by_market_cap"].items():
        success_rate = (
            cap_stats["successful"] / cap_stats["attempted"] * 100
            if cap_stats["attempted"] > 0
            else 0
        )
        print(
            f"  {cap_type}: {cap_stats['successful']}/{cap_stats['attempted']} ({success_rate:.1f}%)"
        )

    # Save metadata
    metadata_file = output_dir / "enhanced_dataset_metadata.csv"
    metadata_rows = []

    for cap_type, symbols in by_market_cap.items():
        if cap_type == "ETF":
            target_dir = etf_dir
        elif cap_type == "LARGE":
            target_dir = large_cap_dir
        elif cap_type == "MID":
            target_dir = mid_cap_dir
        elif cap_type == "SMALL":
            target_dir = small_cap_dir
        else:
            continue

        for symbol in sorted(symbols):
            file_path = target_dir / f"{symbol}_historical.{format}"
            if file_path.exists():
                if format == "csv":
                    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                else:
                    df = pd.read_parquet(file_path)

                # Find symbol metadata
                sector = None
                volatility = None
                bull_performer = None

                for group_data in ENHANCED_SYMBOL_UNIVERSE.values():
                    if symbol in group_data["symbols"]:
                        sector = group_data["sector"]
                        volatility = group_data.get("volatility")
                        bull_performer = group_data.get("bull_performer")
                        break

                metadata_rows.append(
                    {
                        "symbol": symbol,
                        "market_cap": cap_type,
                        "sector": sector,
                        "bull_performer": bull_performer,
                        "target_volatility": volatility,
                        "start_date": df.index[0] if len(df) > 0 else None,
                        "end_date": df.index[-1] if len(df) > 0 else None,
                        "num_bars": len(df),
                        "num_features": len(df.columns),
                        "avg_price": df["close"].mean() if "close" in df.columns else None,
                        "avg_volume": df["volume"].mean() if "volume" in df.columns else None,
                        "realized_volatility": df["close"].pct_change().std() * (252**0.5)
                        if "close" in df.columns
                        else None,
                        "file_path": str(file_path.relative_to(output_dir)),
                    }
                )

    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_csv(metadata_file, index=False)
    print(f"\nMetadata saved: {metadata_file}")

    print("=" * 80)
    print("DATASET FETCH COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch enhanced dataset collection")
    parser.add_argument("--years", type=int, default=20, help="Years of historical data")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument(
        "--format", type=str, default="csv", choices=["csv", "parquet"], help="Output format"
    )

    args = parser.parse_args()

    fetch_enhanced_datasets(
        years=args.years,
        output_dir=Path(args.output),
        format=args.format,
    )
