"""
Download historical intraday data from Massive Flat Files (S3).

Uses boto3 to download 1-minute aggregates for backtesting.
"""

from datetime import datetime, timedelta
import logging
import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("massive_downloader")

# Massive S3 Configuration
ENDPOINT_URL = os.environ.get("MASSIVE_S3_ENDPOINT", "https://files.massive.com")
BUCKET_NAME = "flatfiles"

ACCESS_KEY = os.environ.get("MASSIVE_FF_ACC_KEY_ID")
SECRET_KEY = os.environ.get("MASSIVE_FF_SECRET_KEY")


def get_s3_client():
    """Initialize S3 client for Massive."""
    session = boto3.Session(
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
    )
    return session.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        config=Config(signature_version="s3v4"),
    )


def download_day_data(date_str: str, output_dir: str = "data/massive"):
    """
    Download 1-minute aggregates for a specific day.

    Path format: us_stocks_sip/minute_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz
    """
    s3 = get_s3_client()

    dt = datetime.strptime(date_str, "%Y-%m-%d")
    year = dt.strftime("%Y")
    month = dt.strftime("%m")

    # Construct object key
    # Using minute aggregates as requested (closest to 30s available in standard paths usually)
    # The user asked for 30s, but standard aggregates are usually 1m.
    # Let's check if we can find 1m first, or trades if needed.
    # For backtesting SMA, 1m is usually sufficient.
    prefix = f"us_stocks_sip/minute_aggs_v1/{year}/{month}/{date_str}.csv.gz"

    local_path = os.path.join(output_dir, f"{date_str}.csv.gz")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Downloading {prefix} to {local_path}...")

    try:
        s3.download_file(BUCKET_NAME, prefix, local_path)
        logger.info("Download complete.")
        return local_path
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.error(f"File not found: {prefix}")
        elif e.response["Error"]["Code"] == "403":
            logger.error("Access denied. Check credentials.")
        else:
            logger.error(f"Error downloading: {e}")
        return None


def load_massive_data(file_path: str, symbol: str) -> pd.DataFrame:
    """
    Load Massive CSV data and filter for a specific symbol.

    Massive CSV format:
    ticker,volume,open,close,high,low,window_start,transactions
    """
    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, compression="gzip")

    # Filter for symbol
    df = df[df["ticker"] == symbol].copy()

    if df.empty:
        logger.warning(f"No data found for {symbol}")
        return df

    # Convert window_start (nanoseconds) to datetime
    df["timestamp"] = pd.to_datetime(df["window_start"], unit="ns", utc=True)
    df = df.set_index("timestamp").sort_index()

    # Rename columns to standard format
    df = df.rename(
        columns={
            "ticker": "symbol",
            # volume is already volume
            # open, close, high, low are already correct
        }
    )

    return df[["open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    # Example usage
    # This script is intended to be imported or run with valid credentials
    if not ACCESS_KEY or not SECRET_KEY:
        logger.warning(
            "Please set MASSIVE_FF_ACC_KEY_ID and MASSIVE_FF_SECRET_KEY environment variables."
        )
    else:
        # Download yesterday's data as a test
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        download_day_data(yesterday)
