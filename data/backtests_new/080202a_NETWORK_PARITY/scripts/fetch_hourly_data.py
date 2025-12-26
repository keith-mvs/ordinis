#!/usr/bin/env python3
"""
Fetch Hourly Historical Data for Short-Selling Optimization

Uses Polygon API to fetch 1-hour bars for modern periods (2019+).
Historical periods (2006-2015) may have limited/no hourly data available.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
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

# API Configuration
MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY", "")
BASE_URL = "https://api.polygon.io/v2"

# Output directory
SCRIPTS_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPTS_DIR.parent
DATA_DIR = OUTPUT_DIR / "historical_data_hourly"

# Modern small-caps (2019+) - these have hourly data
MODERN_SMALL_CAP = {
    "crypto_tech": ["RIOT", "MARA", "COIN", "BITF", "CLSK", "CIFR"],
    "growth_tech": ["AI", "IONQ", "SOUN", "KULR", "SMCI", "PLTR"],
    "cannabis": ["TLRY", "CGC", "ACB", "SNDL", "HEXO", "VFF"],
    "ev_energy": ["PLUG", "FCEL", "BLNK", "CHPT", "EVGO", "LCID"],
    "fintech": ["SOFI", "HOOD", "AFRM", "UPST", "OPEN", "LMND"],
    "meme": ["GME", "AMC", "WISH", "CLOV", "WKHS"],  # BBBY removed
}

# Periods with hourly data available (2019+)
HOURLY_PERIODS = {
    "2019_bull": {
        "start": "2019-11-01",
        "end": "2019-11-30",
        "description": "Late cycle bull market",
    },
    "2022_bear": {
        "start": "2022-06-01",
        "end": "2022-06-30",
        "description": "Inflation bear market",
    },
    "2023_rebound": {
        "start": "2023-05-01",
        "end": "2023-05-31",
        "description": "AI/Tech rebound",
    },
    "2024_recent": {
        "start": "2024-10-01",
        "end": "2024-10-31",
        "description": "Recent market (Oct 2024)",
    },
}


async def fetch_aggregates(
    session: aiohttp.ClientSession,
    ticker: str,
    from_date: str,
    to_date: str,
    timespan: str = "hour",
    multiplier: int = 1,
) -> list[dict[str, Any]]:
    """Fetch aggregate bars from API."""
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
                    logger.debug(f"No results for {ticker} ({from_date} to {to_date})")
                    return []
            elif response.status == 403:
                logger.warning(f"Access denied for {ticker}")
                return []
            else:
                logger.warning(f"Error fetching {ticker}: {response.status}")
                return []
    except Exception as e:
        logger.error(f"Exception fetching {ticker}: {e}")
        return []


async def fetch_period_data(
    session: aiohttp.ClientSession,
    period_name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch hourly data for all symbols in a period."""
    all_symbols = [sym for sector in MODERN_SMALL_CAP.values() for sym in sector]

    logger.info(f"Fetching {period_name} (hourly): {start_date} to {end_date} ({len(all_symbols)} symbols)")

    all_data = []

    # Fetch in batches
    batch_size = 5
    for i in range(0, len(all_symbols), batch_size):
        batch = all_symbols[i:i + batch_size]
        tasks = [
            fetch_aggregates(session, sym, start_date, end_date, timespan="hour")
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

        await asyncio.sleep(0.15)  # Rate limiting

    if all_data:
        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values(["ticker", "timestamp"])
        return df

    return pd.DataFrame()


async def main():
    """Main entry point for hourly data fetching."""
    if not MASSIVE_API_KEY:
        logger.error("MASSIVE_API_KEY environment variable not set!")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("HOURLY DATA FETCHER")
    logger.info("=" * 70)
    logger.info(f"Output directory: {DATA_DIR}")
    logger.info("Periods: 2019, 2022, 2023, 2024")
    logger.info("Resolution: 1-hour bars")

    timeout = aiohttp.ClientTimeout(total=120)
    connector = aiohttp.TCPConnector(limit=10)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Test API
        logger.info("\nTesting API connection...")
        test_data = await fetch_aggregates(session, "AAPL", "2024-01-02", "2024-01-05", timespan="hour")
        if not test_data:
            logger.error("API test failed!")
            sys.exit(1)
        logger.info(f"API test successful - got {len(test_data)} hourly bars")

        total_bars = 0

        # Fetch each period
        for period_name, config in HOURLY_PERIODS.items():
            df = await fetch_period_data(
                session,
                period_name,
                config["start"],
                config["end"],
            )

            if not df.empty:
                output_file = DATA_DIR / f"{period_name}.csv.gz"
                df.to_csv(output_file, index=False, compression="gzip")

                # Stats
                n_bars = len(df)
                total_bars += n_bars
                symbols_count = df["ticker"].nunique()
                hours_per_day = df.groupby(df["timestamp"].dt.date).size().mean()

                logger.info(f"Saved {period_name}:")
                logger.info(f"  {symbols_count} symbols, {n_bars} hourly bars")
                logger.info(f"  ~{hours_per_day:.1f} bars/day avg")
                logger.info(f"  File: {output_file}")
            else:
                logger.warning(f"No data for {period_name}")

    logger.info("\n" + "=" * 70)
    logger.info("HOURLY DATA FETCH COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total hourly bars: {total_bars}")

    # Save metadata
    metadata = {
        "periods": HOURLY_PERIODS,
        "timespan": "hour",
        "total_bars": total_bars,
        "fetch_timestamp": datetime.now().isoformat(),
    }
    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
