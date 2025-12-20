"""
Generate realistic sample market data for testing.

This script creates synthetic but realistic OHLCV data that can be used
for backtesting without requiring API keys.
"""

from datetime import UTC, datetime
import sys

import numpy as np
import pandas as pd


def generate_realistic_prices(
    n_bars: int,
    base_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate realistic price series using geometric Brownian motion.

    Args:
        n_bars: Number of bars to generate
        base_price: Starting price
        volatility: Daily volatility (0.02 = 2%)
        trend: Daily drift (0.0001 = 0.01% per day)
        seed: Random seed for reproducibility

    Returns:
        Array of prices
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate returns using geometric Brownian motion
    dt = 1.0  # Daily timestep
    returns = np.random.normal(
        loc=trend * dt,
        scale=volatility * np.sqrt(dt),
        size=n_bars,
    )

    # Calculate cumulative prices
    prices = base_price * np.exp(np.cumsum(returns))

    return prices


def generate_ohlcv_from_close(
    close_prices: np.ndarray,
    volume_base: int = 1000000,
    volume_volatility: float = 0.3,
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data from close prices.

    Args:
        close_prices: Array of close prices
        volume_base: Average daily volume
        volume_volatility: Volume volatility (0.3 = 30%)

    Returns:
        DataFrame with OHLCV columns
    """
    n = len(close_prices)

    # Generate intraday volatility
    intraday_vol = np.random.uniform(0.005, 0.015, n)

    # Generate OHLC around close
    high = close_prices * (1 + intraday_vol)
    low = close_prices * (1 - intraday_vol)
    open_ = close_prices * (1 + np.random.uniform(-0.005, 0.005, n))

    # Ensure high >= open, close >= low
    high = np.maximum(high, np.maximum(open_, close_prices))
    low = np.minimum(low, np.minimum(open_, close_prices))

    # Generate volume with some correlation to price movement
    price_changes = np.abs(np.diff(close_prices, prepend=close_prices[0]))
    volume_factor = 1 + (price_changes / close_prices) * 5  # More volume on big moves

    volume = (
        np.random.lognormal(
            mean=np.log(volume_base),
            sigma=volume_volatility,
            size=n,
        )
        * volume_factor
    )

    volume = volume.astype(int)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close_prices,
            "volume": volume,
        }
    )


def generate_market_data(
    symbol: str = "SPY",
    start_date: str = "2023-01-01",
    end_date: str | None = None,
    market_regime: str = "trending_up",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate complete market data for backtesting.

    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        market_regime: Market regime type:
            - "trending_up": Upward trend
            - "trending_down": Downward trend
            - "sideways": Range-bound
            - "volatile": High volatility
        seed: Random seed for reproducibility

    Returns:
        DataFrame with datetime index and OHLCV columns
    """
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d")
    else:
        end = datetime.now(UTC)

    # Generate date range (trading days only - skip weekends)
    dates = pd.date_range(start=start, end=end, freq="B")  # B = business days
    n_bars = len(dates)

    # Set parameters based on market regime
    regime_params = {
        "trending_up": {
            "base_price": 100.0,
            "volatility": 0.015,
            "trend": 0.0005,  # ~13% annual
        },
        "trending_down": {
            "base_price": 100.0,
            "volatility": 0.018,
            "trend": -0.0003,  # -8% annual
        },
        "sideways": {
            "base_price": 100.0,
            "volatility": 0.012,
            "trend": 0.0,
        },
        "volatile": {
            "base_price": 100.0,
            "volatility": 0.035,
            "trend": 0.0001,
        },
    }

    params = regime_params.get(market_regime, regime_params["trending_up"])

    # Generate prices
    close_prices = generate_realistic_prices(
        n_bars=n_bars,
        base_price=params["base_price"],
        volatility=params["volatility"],
        trend=params["trend"],
        seed=seed,
    )

    # Generate OHLCV
    ohlcv = generate_ohlcv_from_close(close_prices)

    # Add timestamp index
    ohlcv.index = dates

    # Add symbol column
    ohlcv["symbol"] = symbol

    return ohlcv


def save_to_csv(df: pd.DataFrame, filename: str) -> None:
    """Save data to CSV file."""
    df.to_csv(filename)
    print(f"[OK] Saved {len(df)} bars to {filename}")


def main() -> int:
    """Generate sample data for multiple scenarios."""
    print("\n" + "=" * 60)
    print("SAMPLE MARKET DATA GENERATOR")
    print("=" * 60)

    # Scenario 1: SPY trending up (2023-2024)
    print("\n[1] Generating SPY trending up (2023-2024)...")
    spy_up = generate_market_data(
        symbol="SPY",
        start_date="2023-01-01",
        end_date="2024-11-30",
        market_regime="trending_up",
        seed=42,
    )
    save_to_csv(spy_up, "data/sample_spy_trending_up.csv")
    print(f"    Start: ${spy_up['close'].iloc[0]:.2f}")
    print(f"    End: ${spy_up['close'].iloc[-1]:.2f}")
    print(f"    Return: {((spy_up['close'].iloc[-1] / spy_up['close'].iloc[0]) - 1) * 100:.1f}%")

    # Scenario 2: QQQ volatile (2024)
    print("\n[2] Generating QQQ volatile (2024)...")
    qqq_volatile = generate_market_data(
        symbol="QQQ",
        start_date="2024-01-01",
        end_date="2024-11-30",
        market_regime="volatile",
        seed=123,
    )
    save_to_csv(qqq_volatile, "data/sample_qqq_volatile.csv")
    print(f"    Start: ${qqq_volatile['close'].iloc[0]:.2f}")
    print(f"    End: ${qqq_volatile['close'].iloc[-1]:.2f}")
    print(
        f"    Max drawdown: {((qqq_volatile['close'].cummax() - qqq_volatile['close']) / qqq_volatile['close'].cummax()).max() * 100:.1f}%"
    )

    # Scenario 3: XYZ sideways (2024)
    print("\n[3] Generating XYZ sideways (2024)...")
    xyz_sideways = generate_market_data(
        symbol="XYZ",
        start_date="2024-01-01",
        end_date="2024-11-30",
        market_regime="sideways",
        seed=456,
    )
    save_to_csv(xyz_sideways, "data/sample_xyz_sideways.csv")
    print(f"    Start: ${xyz_sideways['close'].iloc[0]:.2f}")
    print(f"    End: ${xyz_sideways['close'].iloc[-1]:.2f}")
    print(f"    Range: ${xyz_sideways['close'].min():.2f} - ${xyz_sideways['close'].max():.2f}")

    # Scenario 4: ABC trending down (2024)
    print("\n[4] Generating ABC trending down (2024)...")
    abc_down = generate_market_data(
        symbol="ABC",
        start_date="2024-01-01",
        end_date="2024-11-30",
        market_regime="trending_down",
        seed=789,
    )
    save_to_csv(abc_down, "data/sample_abc_trending_down.csv")
    print(f"    Start: ${abc_down['close'].iloc[0]:.2f}")
    print(f"    End: ${abc_down['close'].iloc[-1]:.2f}")
    print(
        f"    Return: {((abc_down['close'].iloc[-1] / abc_down['close'].iloc[0]) - 1) * 100:.1f}%"
    )

    print("\n" + "=" * 60)
    print("[SUCCESS] Sample data generated")
    print("=" * 60)
    print("\nFiles created:")
    print("  - data/sample_spy_trending_up.csv")
    print("  - data/sample_qqq_volatile.csv")
    print("  - data/sample_xyz_sideways.csv")
    print("  - data/sample_abc_trending_down.csv")
    print("\nUsage:")
    print("  import pandas as pd")
    print("  data = pd.read_csv('data/sample_spy_trending_up.csv', index_col=0, parse_dates=True)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
