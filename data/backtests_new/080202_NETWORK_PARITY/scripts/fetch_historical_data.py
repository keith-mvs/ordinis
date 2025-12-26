#!/usr/bin/env python3
"""
Fetch Historical Data from Massive API for Network Parity Optimization

Downloads daily aggregates for sample periods across different market regimes:
- 2004: Post-dot-com recovery
- 2008: Financial crisis
- 2010: Flash crash period
- 2017: Low volatility bull market
- 2024: Recent market conditions

Uses the Massive REST API with daily aggregates.
"""

import asyncio
import gzip
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Massive API Configuration
MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY", "")
BASE_URL = "https://api.polygon.io/v2"  # Massive uses Polygon-compatible API

# Output directory
SCRIPTS_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPTS_DIR.parent
DATA_DIR = OUTPUT_DIR / "historical_data"

# Universe of small-cap stocks (from the Network Parity strategy)
SMALL_CAP_UNIVERSE = {
    "technology": ["RIOT", "MARA", "AI", "IONQ", "SOUN", "KULR"],
    "healthcare": ["BNGO", "SNDL", "TLRY", "CGC", "ACB", "XXII"],
    "energy_materials": ["PLUG", "FCEL", "BE", "CHPT", "BLNK", "EVGO"],
    "financials": ["SOFI", "HOOD", "AFRM", "UPST", "OPEN"],
    "consumer": ["GME", "AMC", "WKHS", "WISH", "CLOV"],
    "industrials": ["GOEV", "BITF", "CLSK", "CIFR", "WULF"],
}

# Alternative large-cap stocks for historical periods (many small caps didn't exist in 2004-2010)
HISTORICAL_UNIVERSE = {
    "technology": ["AAPL", "MSFT", "INTC", "CSCO", "ORCL", "IBM"],
    "healthcare": ["JNJ", "PFE", "MRK", "ABT", "AMGN", "GILD"],
    "energy_materials": ["XOM", "CVX", "SLB", "COP", "EOG", "OXY"],
    "financials": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "consumer": ["WMT", "HD", "MCD", "NKE", "SBUX", "TGT"],
    "industrials": ["GE", "CAT", "MMM", "HON", "UNP", "BA"],
}

# Sample periods for each market regime (21 trading days each)
SAMPLE_PERIODS = {
    "2004_recovery": {"start": "2004-06-01", "end": "2004-06-30"},
    "2008_crisis": {"start": "2008-09-15", "end": "2008-10-15"},
    "2010_flash_crash": {"start": "2010-05-01", "end": "2010-05-31"},
    "2017_bull": {"start": "2017-06-01", "end": "2017-06-30"},
    "2024_recent": {"start": "2024-06-01", "end": "2024-06-30"},
}


async def fetch_aggregates(
    session: aiohttp.ClientSession,
    ticker: str,
    from_date: str,
    to_date: str,
    timespan: str = "day",
    multiplier: int = 1,
) -> list[dict[str, Any]]:
    """Fetch aggregate bars from Massive API."""
    url = f"{BASE_URL}/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
    params = {
        "apiKey": MASSIVE_API_KEY,
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
    }

    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                if data.get("results"):
                    return data["results"]
                else:
                    logger.warning(f"No results for {ticker} ({from_date} to {to_date})")
                    return []
            elif response.status == 403:
                logger.error(f"Access denied for {ticker} - check API plan")
                return []
            else:
                text = await response.text()
                logger.error(f"Error fetching {ticker}: {response.status} - {text[:200]}")
                return []
    except Exception as e:
        logger.error(f"Exception fetching {ticker}: {e}")
        return []


async def fetch_period_data(
    session: aiohttp.ClientSession,
    period_name: str,
    start_date: str,
    end_date: str,
    use_historical_universe: bool = False,
) -> pd.DataFrame:
    """Fetch data for all symbols in a period."""
    universe = HISTORICAL_UNIVERSE if use_historical_universe else SMALL_CAP_UNIVERSE
    all_symbols = [sym for sector in universe.values() for sym in sector]

    logger.info(f"Fetching {period_name}: {start_date} to {end_date} ({len(all_symbols)} symbols)")

    all_data = []

    # Fetch in batches to avoid rate limiting
    batch_size = 5
    for i in range(0, len(all_symbols), batch_size):
        batch = all_symbols[i:i + batch_size]
        tasks = [
            fetch_aggregates(session, sym, start_date, end_date)
            for sym in batch
        ]
        results = await asyncio.gather(*tasks)

        for sym, bars in zip(batch, results):
            if bars:
                for bar in bars:
                    all_data.append({
                        "ticker": sym,
                        "timestamp": bar.get("t"),
                        "open": bar.get("o"),
                        "high": bar.get("h"),
                        "low": bar.get("l"),
                        "close": bar.get("c"),
                        "volume": bar.get("v"),
                        "vwap": bar.get("vw"),
                        "transactions": bar.get("n"),
                    })

        # Small delay to avoid rate limiting
        await asyncio.sleep(0.2)

    if all_data:
        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values(["ticker", "timestamp"])
        return df

    return pd.DataFrame()


async def main():
    """Main entry point for data fetching."""
    if not MASSIVE_API_KEY:
        logger.error("MASSIVE_API_KEY environment variable not set!")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MASSIVE HISTORICAL DATA FETCHER")
    logger.info("=" * 60)
    logger.info(f"Output directory: {DATA_DIR}")

    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=10)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Test API connection
        logger.info("Testing API connection...")
        test_data = await fetch_aggregates(session, "AAPL", "2024-01-02", "2024-01-05")
        if not test_data:
            logger.error("API test failed - cannot continue")
            sys.exit(1)
        logger.info(f"API test successful - got {len(test_data)} bars for AAPL")

        # Fetch each period
        for period_name, dates in SAMPLE_PERIODS.items():
            # Use historical universe for older periods (many small caps didn't exist)
            use_historical = period_name in ["2004_recovery", "2008_crisis", "2010_flash_crash"]

            df = await fetch_period_data(
                session,
                period_name,
                dates["start"],
                dates["end"],
                use_historical_universe=use_historical,
            )

            if not df.empty:
                # Save as compressed CSV (same format as massive flat files)
                output_file = DATA_DIR / f"{period_name}.csv.gz"
                df.to_csv(output_file, index=False, compression="gzip")
                logger.info(f"Saved {period_name}: {len(df)} rows to {output_file}")

                # Summary stats
                symbols_count = df["ticker"].nunique()
                date_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"
                logger.info(f"  → {symbols_count} symbols, {date_range}")
            else:
                logger.warning(f"No data for {period_name}")

        # Also fetch recent data (2025) using small-cap universe
        logger.info("\nFetching recent 2025 data...")
        recent_df = await fetch_period_data(
            session,
            "2025_current",
            "2025-11-01",
            "2025-12-20",
            use_historical_universe=False,
        )

        if not recent_df.empty:
            output_file = DATA_DIR / "2025_current.csv.gz"
            recent_df.to_csv(output_file, index=False, compression="gzip")
            logger.info(f"Saved 2025_current: {len(recent_df)} rows")

    logger.info("\n" + "=" * 60)
    logger.info("DATA FETCH COMPLETE")
    logger.info("=" * 60)

    # List all downloaded files
    files = list(DATA_DIR.glob("*.csv.gz"))
    logger.info(f"Downloaded {len(files)} period files:")
    for f in sorted(files):
        size_mb = f.stat().st_size / (1024 * 1024)
        logger.info(f"  {f.name}: {size_mb:.2f} MB")


if __name__ == "__main__":
    asyncio.run(main())
