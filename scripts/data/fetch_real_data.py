"""Fetch real market data from Twelve Data API."""

import asyncio
import os
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import aiohttp
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


async def fetch_real_spy_data():
    """Fetch real SPY daily data from Twelve Data."""
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        print("[ERROR] NO TWELVEDATA_API_KEY in .env")
        return None

    url = f"https://api.twelvedata.com/time_series?symbol=SPY&interval=1day&outputsize=500&apikey={api_key}"

    print("Fetching real SPY data from Twelve Data...")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            data = await resp.json()

    if "values" not in data:
        print(f"[ERROR] API Error: {data}")
        return None

    values = data["values"]
    print(f"[OK] Fetched {len(values)} days of REAL SPY data")
    print(f"Date range: {values[-1]['datetime']} to {values[0]['datetime']}")

    # Show last 5 days
    print("\nLast 5 trading days (REAL DATA):")
    for bar in values[:5]:
        print(
            f"  {bar['datetime']}: "
            f"O=${float(bar['open']):.2f} "
            f"H=${float(bar['high']):.2f} "
            f"L=${float(bar['low']):.2f} "
            f"C=${float(bar['close']):.2f} "
            f"V={int(float(bar['volume'])):,}"
        )

    # Convert to DataFrame
    df = pd.DataFrame(values)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")
    df = df.rename(columns={"datetime": "date"})
    df.set_index("date", inplace=True)
    df["symbol"] = "SPY"

    # Convert to proper types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])

    # Save to CSV
    output_path = project_root / "data" / "real_spy_daily.csv"
    df.to_csv(output_path)

    start_price = df["close"].iloc[0]
    end_price = df["close"].iloc[-1]
    buy_hold_return = ((end_price - start_price) / start_price) * 100

    print(f"\n[SAVED] {len(df)} bars to {output_path}")
    print(f"Start: {df.index[0].date()} @ ${start_price:.2f}")
    print(f"End: {df.index[-1].date()} @ ${end_price:.2f}")
    print(f"Buy-Hold Return: {buy_hold_return:+.2f}%")

    return df


if __name__ == "__main__":
    asyncio.run(fetch_real_spy_data())
