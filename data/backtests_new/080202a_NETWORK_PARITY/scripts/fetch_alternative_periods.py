#!/usr/bin/env python3
"""
Fetch Alternative Historical Periods for Short-Selling Optimization

Different years than the original test (2004, 2008, 2010, 2017, 2024).
Total duration: ~21 trading days across 6 periods.

Selected periods:
- 2006: Pre-crisis bull market (~4 days)
- 2012: European debt crisis recovery (~3 days)
- 2015: Mid-cycle volatility (~3 days)
- 2019: Late bull market (~4 days)
- 2022: Bear market / inflation (~4 days)
- 2023: Tech rebound (~3 days)
"""

import asyncio
import gzip
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

# Massive/Polygon API Configuration
MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY", "")
BASE_URL = "https://api.polygon.io/v2"

# Output directory
SCRIPTS_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPTS_DIR.parent
DATA_DIR = OUTPUT_DIR / "historical_data_v2"

# Universe - Small-cap proxies for ALL periods
# Historical small-caps that existed in older periods (2006-2015)
HISTORICAL_SMALL_CAP = {
    "technology": ["AMD", "MU", "AMAT", "LRCX", "MRVL", "SWKS"],  # Small-cap tech
    "healthcare": ["EXAS", "HZNP", "JAZZ", "NBIX", "TECH", "BIO"],  # Small biotech
    "energy": ["RRC", "AR", "CNX", "SM", "CDEV", "MTDR"],  # Small E&P
    "financials": ["SIVB", "SBNY", "WAL", "PACW", "ZION", "HBAN"],  # Regional banks
    "consumer": ["FIVE", "PLAY", "RH", "ETSY", "W", "BURL"],  # Small retail
    "industrials": ["GNRC", "PCAR", "MIDD", "TTC", "RBC", "SITE"],  # Small industrial
}

# Modern small-caps (2019+) - volatile meme/crypto/growth
MODERN_SMALL_CAP = {
    "crypto_tech": ["RIOT", "MARA", "COIN", "BITF", "CLSK", "CIFR"],
    "growth_tech": ["AI", "IONQ", "SOUN", "KULR", "SMCI", "PLTR"],
    "cannabis": ["TLRY", "CGC", "ACB", "SNDL", "HEXO", "VFF"],
    "ev_energy": ["PLUG", "FCEL", "BLNK", "CHPT", "EVGO", "LCID"],
    "fintech": ["SOFI", "HOOD", "AFRM", "UPST", "OPEN", "LMND"],
    "meme": ["GME", "AMC", "BBBY", "WISH", "CLOV", "WKHS"],
}

# Alternative sample periods (different from original: 2004, 2008, 2010, 2017, 2024)
# ~21 trading days PER period to have enough data for lookback calculations
ALTERNATIVE_PERIODS = {
    # 2006: Pre-crisis bull market - ~21 trading days
    "2006_bull": {
        "start": "2006-05-01",
        "end": "2006-05-31",
        "description": "Pre-crisis bull market",
        "universe": "historical"
    },
    # 2012: Post-European debt crisis recovery - ~21 trading days
    "2012_recovery": {
        "start": "2012-09-01",
        "end": "2012-09-30",
        "description": "European debt crisis recovery",
        "universe": "historical"
    },
    # 2015: Mid-cycle volatility (China concerns) - ~21 trading days
    "2015_volatility": {
        "start": "2015-08-01",
        "end": "2015-08-31",
        "description": "China-induced volatility",
        "universe": "historical"
    },
    # 2019: Late bull market - ~21 trading days
    "2019_bull": {
        "start": "2019-11-01",
        "end": "2019-11-30",
        "description": "Late cycle bull market",
        "universe": "modern"
    },
    # 2022: Bear market / inflation shock - ~21 trading days
    "2022_bear": {
        "start": "2022-06-01",
        "end": "2022-06-30",
        "description": "Inflation bear market",
        "universe": "modern"
    },
    # 2023: AI/Tech rebound - ~21 trading days
    "2023_rebound": {
        "start": "2023-05-01",
        "end": "2023-05-31",
        "description": "AI/Tech rebound",
        "universe": "modern"
    },
}


async def fetch_aggregates(
    session: aiohttp.ClientSession,
    ticker: str,
    from_date: str,
    to_date: str,
    timespan: str = "day",
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
                text = await response.text()
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
    universe_type: str,
) -> pd.DataFrame:
    """Fetch data for all symbols in a period."""
    universe = MODERN_SMALL_CAP if universe_type == "modern" else HISTORICAL_SMALL_CAP
    all_symbols = [sym for sector in universe.values() for sym in sector]

    logger.info(f"Fetching {period_name}: {start_date} to {end_date} ({len(all_symbols)} symbols)")

    all_data = []

    # Fetch in batches
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

        await asyncio.sleep(0.15)  # Rate limiting

    if all_data:
        df = pd.DataFrame(all_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.sort_values(["ticker", "timestamp"])
        return df

    return pd.DataFrame()


async def main():
    """Main entry point for alternative period data fetching."""
    if not MASSIVE_API_KEY:
        logger.error("MASSIVE_API_KEY environment variable not set!")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("ALTERNATIVE PERIODS DATA FETCHER")
    logger.info("=" * 70)
    logger.info(f"Output directory: {DATA_DIR}")
    logger.info("Periods: 2006, 2012, 2015, 2019, 2022, 2023")
    logger.info("Target: ~21 trading days total")

    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=10)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Test API
        logger.info("\nTesting API connection...")
        test_data = await fetch_aggregates(session, "AAPL", "2024-01-02", "2024-01-05")
        if not test_data:
            logger.error("API test failed!")
            sys.exit(1)
        logger.info(f"API test successful - got {len(test_data)} bars")

        total_trading_days = 0

        # Fetch each period
        for period_name, config in ALTERNATIVE_PERIODS.items():
            df = await fetch_period_data(
                session,
                period_name,
                config["start"],
                config["end"],
                config["universe"],
            )

            if not df.empty:
                output_file = DATA_DIR / f"{period_name}.csv.gz"
                df.to_csv(output_file, index=False, compression="gzip")

                # Count trading days
                trading_days = df["timestamp"].dt.date.nunique()
                total_trading_days += trading_days

                symbols_count = df["ticker"].nunique()
                date_range = f"{df['timestamp'].min().date()} to {df['timestamp'].max().date()}"

                logger.info(f"Saved {period_name}:")
                logger.info(f"  Description: {config['description']}")
                logger.info(f"  {symbols_count} symbols, {trading_days} trading days")
                logger.info(f"  Date range: {date_range}")
                logger.info(f"  File: {output_file}")
            else:
                logger.warning(f"No data for {period_name}")

    logger.info("\n" + "=" * 70)
    logger.info("ALTERNATIVE PERIODS FETCH COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total trading days across all periods: {total_trading_days}")

    # List downloaded files
    files = list(DATA_DIR.glob("*.csv.gz"))
    logger.info(f"\nDownloaded {len(files)} period files:")
    for f in sorted(files):
        size_kb = f.stat().st_size / 1024
        logger.info(f"  {f.name}: {size_kb:.1f} KB")

    # Save metadata
    metadata = {
        "periods": {k: {**v, "file": f"{k}.csv.gz"} for k, v in ALTERNATIVE_PERIODS.items()},
        "total_trading_days": total_trading_days,
        "fetch_timestamp": datetime.now().isoformat(),
    }
    with open(DATA_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
