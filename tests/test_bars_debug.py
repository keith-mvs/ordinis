"""Simple debug test for Alpaca historical bars."""

from datetime import datetime, timedelta
import os

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

api_key = os.getenv("ALPACA_API_KEY")
api_secret = os.getenv("ALPACA_SECRET_KEY")

print(f"API Key: {api_key[:10]}...")

client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)

try:
    # Try daily bars first (free tier)
    print("\n=== Testing DAILY bars ===")
    request_daily = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
    )

    bars_daily = client.get_stock_bars(request_daily)
    print(f"Daily bars response: {type(bars_daily)}")

    if bars_daily.data:
        print(f"Got {len(bars_daily.data)} symbols")
        for symbol in bars_daily.data:
            print(f"  Symbol: {symbol}, Bars: {len(bars_daily.data[symbol])}")
            if bars_daily.data[symbol]:
                print(f"  Last bar: {bars_daily.data[symbol][-1]}")

    # Try minute bars (may require subscription)
    print("\n=== Testing MINUTE bars ===")
    request = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=datetime.now() - timedelta(hours=2),
        end=datetime.now(),
        limit=10,
    )

    print("Requesting bars...")
    bars = client.get_stock_bars(request)
    print(f"Got bars: {type(bars)}")
    print(f"Keys: {bars.keys() if hasattr(bars, 'keys') else 'N/A'}")

    if "SPY" in bars:
        df = bars.df
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\n{df.tail()}")
    else:
        print("No SPY data in response")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback

    traceback.print_exc()
