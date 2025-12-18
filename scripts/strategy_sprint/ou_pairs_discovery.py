"""
Ornstein-Uhlenbeck Pairs Discovery.

Scans universe to find cointegrated pairs suitable for mean-reversion trading.
Uses Engle-Granger cointegration test and OU process fitting.
"""

import asyncio
from datetime import datetime
from itertools import combinations
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)

# Universe for pairs discovery
UNIVERSE = [
    # Tech pairs
    "AAPL",
    "MSFT",
    "GOOGL",
    "META",
    "AMZN",
    # Semiconductors
    "NVDA",
    "AMD",
    "INTC",
    "TSM",
    "AVGO",
    # Finance
    "JPM",
    "BAC",
    "GS",
    "MS",
    "C",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "OXY",
    # Consumer
    "KO",
    "PEP",
    "WMT",
    "COST",
    "TGT",
    # ETFs
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
]


def load_historical_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """Load historical price data."""
    cache_path = Path(f"data/historical/{symbol}_daily.parquet")

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    csv_path = Path(f"data/historical/{symbol}_daily.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

    return generate_synthetic_prices(symbol, days)


def generate_synthetic_prices(symbol: str, days: int = 500) -> pd.DataFrame:
    """Generate synthetic price data."""
    np.random.seed(hash(symbol) % 2**32)

    drift = np.random.uniform(-0.0001, 0.0003)
    volatility = np.random.uniform(0.012, 0.025)

    returns = np.random.normal(drift, volatility, days)
    base_price = np.random.uniform(50, 500)
    prices = base_price * np.exp(np.cumsum(returns))

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["high"] = prices * (1 + np.abs(np.random.normal(0, 0.01, days)))
    df["low"] = prices * (1 - np.abs(np.random.normal(0, 0.01, days)))
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["volume"] = np.random.randint(1_000_000, 50_000_000, days)

    return df


def adf_test(series: np.ndarray) -> tuple[float, float]:
    """
    Simple ADF test implementation.
    Returns (adf_stat, p_value).
    """
    n = len(series)
    if n < 20:
        return 0, 1.0

    # Difference the series
    diff = np.diff(series)
    lag = series[:-1]

    # Regression: diff = alpha + beta * lag + error
    X = np.column_stack([np.ones(len(lag)), lag])
    y = diff

    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta

        # Calculate t-statistic for beta[1]
        mse = np.sum(residuals**2) / (n - 2)
        var_beta = mse * np.linalg.inv(X.T @ X)
        se_beta1 = np.sqrt(var_beta[1, 1])

        adf_stat = beta[1] / se_beta1

        # Approximate p-value (critical values: -2.86 for 5%, -3.43 for 1%)
        if adf_stat < -3.43:
            p_value = 0.01
        elif adf_stat < -2.86:
            p_value = 0.05
        elif adf_stat < -2.57:
            p_value = 0.10
        else:
            p_value = 0.50

        return adf_stat, p_value

    except Exception:
        return 0, 1.0


def engle_granger_test(y1: np.ndarray, y2: np.ndarray) -> dict:
    """
    Engle-Granger cointegration test.
    Returns hedge ratio, ADF statistic, and p-value.
    """
    n = len(y1)

    # Step 1: Regress y1 on y2 to get hedge ratio
    X = np.column_stack([np.ones(n), y2])
    beta = np.linalg.lstsq(X, y1, rcond=None)[0]

    hedge_ratio = beta[1]
    intercept = beta[0]

    # Step 2: Calculate spread
    spread = y1 - hedge_ratio * y2 - intercept

    # Step 3: Test spread for stationarity
    adf_stat, p_value = adf_test(spread)

    return {
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "adf_stat": adf_stat,
        "p_value": p_value,
        "spread": spread,
    }


def fit_ou_process(spread: np.ndarray, dt: float = 1.0) -> dict:
    """
    Fit Ornstein-Uhlenbeck process to spread.
    dS = theta * (mu - S) * dt + sigma * dW
    """
    n = len(spread)
    if n < 20:
        return {"theta": 0, "mu": 0, "sigma": 0, "half_life": np.inf}

    # Regress change in spread on spread level
    S = spread[:-1]
    dS = np.diff(spread)

    X = np.column_stack([np.ones(len(S)), S])
    beta = np.linalg.lstsq(X, dS, rcond=None)[0]

    # OU parameters
    # dS = (a + b*S)dt -> theta = -b, mu = -a/b
    b = beta[1]
    a = beta[0]

    theta = -b / dt
    mu = -a / b if abs(b) > 1e-10 else spread.mean()

    # Residual volatility
    residuals = dS - X @ beta
    sigma = residuals.std() / np.sqrt(dt)

    # Half-life
    half_life = np.log(2) / theta if theta > 0 else np.inf

    return {
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "half_life": half_life,
    }


def scan_for_pairs(
    universe_data: dict[str, pd.DataFrame],
    p_threshold: float = 0.10,
    min_half_life: float = 1.0,
    max_half_life: float = 60.0,
) -> list[dict]:
    """Scan universe for cointegrated pairs."""
    symbols = list(universe_data.keys())
    pairs_found = []

    for sym1, sym2 in combinations(symbols, 2):
        df1 = universe_data[sym1]
        df2 = universe_data[sym2]

        # Align dates
        common = df1.index.intersection(df2.index)
        if len(common) < 252:
            continue

        y1 = df1.loc[common, "close"].values
        y2 = df2.loc[common, "close"].values

        # Cointegration test
        result = engle_granger_test(y1, y2)

        if result["p_value"] > p_threshold:
            continue

        # Fit OU process
        ou = fit_ou_process(result["spread"])

        # Filter by half-life
        if ou["half_life"] < min_half_life or ou["half_life"] > max_half_life:
            continue

        # Calculate spread statistics
        spread = result["spread"]
        spread_mean = spread.mean()
        spread_std = spread.std()
        current_z = (spread[-1] - spread_mean) / spread_std if spread_std > 0 else 0

        pairs_found.append(
            {
                "symbol_1": sym1,
                "symbol_2": sym2,
                "hedge_ratio": result["hedge_ratio"],
                "adf_stat": result["adf_stat"],
                "p_value": result["p_value"],
                "theta": ou["theta"],
                "half_life": ou["half_life"],
                "mu": ou["mu"],
                "sigma": ou["sigma"],
                "spread_mean": spread_mean,
                "spread_std": spread_std,
                "current_z": current_z,
                "obs": len(common),
            }
        )

    return pairs_found


def backtest_pair(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    hedge_ratio: float,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
) -> dict:
    """Backtest a pairs trading strategy."""
    common = df1.index.intersection(df2.index)

    y1 = df1.loc[common, "close"].values
    y2 = df2.loc[common, "close"].values

    spread = y1 - hedge_ratio * y2
    spread_mean = spread.mean()
    spread_std = spread.std()

    z_scores = (spread - spread_mean) / spread_std

    position = 0  # 1 = long spread, -1 = short spread
    returns = []
    trades = []

    for i in range(1, len(z_scores)):
        z = z_scores[i]
        z_prev = z_scores[i - 1]

        # Entry signals
        if position == 0:
            if z < -entry_z:
                position = 1  # Long spread (expect mean reversion up)
                trades.append({"date": common[i], "action": "LONG_SPREAD", "z": z})
            elif z > entry_z:
                position = -1  # Short spread (expect mean reversion down)
                trades.append({"date": common[i], "action": "SHORT_SPREAD", "z": z})

        # Exit signals
        elif position == 1:
            if z > -exit_z or z < -stop_z:
                position = 0
                trades.append({"date": common[i], "action": "EXIT_LONG", "z": z})
        elif position == -1:
            if z < exit_z or z > stop_z:
                position = 0
                trades.append({"date": common[i], "action": "EXIT_SHORT", "z": z})

        # Calculate return
        spread_return = (spread[i] - spread[i - 1]) / (
            abs(y1[i - 1]) + abs(hedge_ratio * y2[i - 1])
        )
        strat_return = position * spread_return
        returns.append(strat_return)

    returns = np.array(returns)

    total_return = np.sum(returns) * 100
    win_rate = np.sum(returns > 0) / np.sum(returns != 0) if np.sum(returns != 0) > 0 else 0
    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0

    return {
        "total_return": total_return,
        "win_rate": win_rate * 100,
        "sharpe": sharpe,
        "n_trades": len(trades),
        "trades": trades,
    }


async def run() -> dict:
    """Run OU pairs discovery."""
    logger.info("=" * 50)
    logger.info("ORNSTEIN-UHLENBECK PAIRS DISCOVERY")
    logger.info("=" * 50)

    # Load data
    logger.info(f"Loading data for {len(UNIVERSE)} symbols...")
    universe_data = {}
    for symbol in UNIVERSE:
        df = load_historical_data(symbol)
        if len(df) > 252:
            universe_data[symbol] = df

    logger.info(f"Loaded {len(universe_data)} symbols")

    # Scan for pairs
    n_pairs = len(list(combinations(universe_data.keys(), 2)))
    logger.info(f"Scanning {n_pairs} potential pairs...")

    pairs = scan_for_pairs(universe_data, p_threshold=0.10)

    logger.info(f"\nFound {len(pairs)} cointegrated pairs")

    # Sort by half-life (prefer moderate half-life)
    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values("p_value")

    # Top pairs
    logger.info("\n--- Top Cointegrated Pairs ---")
    for i, row in pairs_df.head(10).iterrows():
        logger.info(
            f"  {row['symbol_1']}-{row['symbol_2']}: "
            f"p={row['p_value']:.3f}, HL={row['half_life']:.1f}d, "
            f"Z={row['current_z']:+.2f}"
        )

    # Backtest top pairs
    backtest_results = []

    logger.info("\n--- Backtest Top Pairs ---")
    for _, row in pairs_df.head(5).iterrows():
        sym1, sym2 = row["symbol_1"], row["symbol_2"]

        bt = backtest_pair(
            universe_data[sym1],
            universe_data[sym2],
            row["hedge_ratio"],
        )

        logger.info(
            f"  {sym1}-{sym2}: Return={bt['total_return']:.1f}%, "
            f"Sharpe={bt['sharpe']:.2f}, Trades={bt['n_trades']}"
        )

        backtest_results.append(
            {
                "pair": f"{sym1}-{sym2}",
                **bt,
            }
        )

    # Current trading opportunities
    logger.info("\n--- Current Trading Opportunities ---")
    opportunities = pd.DataFrame()
    if len(pairs_df) > 0 and "current_z" in pairs_df.columns:
        opportunities = pairs_df[abs(pairs_df["current_z"]) > 1.5]

    if len(opportunities) > 0:
        for _, row in opportunities.iterrows():
            direction = "LONG spread" if row["current_z"] < 0 else "SHORT spread"
            logger.info(
                f"  {row['symbol_1']}-{row['symbol_2']}: "
                f"Z={row['current_z']:+.2f} -> {direction}"
            )
    else:
        logger.info("  No current opportunities (|Z| > 1.5)")

    # Save results
    output_dir = Path("artifacts/reports/strategy_sprint")
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(pairs_df) > 0:
        pairs_df.to_csv(output_dir / "ou_pairs_discovered.csv", index=False)

    bt_df = pd.DataFrame(backtest_results)
    if len(bt_df) > 0:
        bt_df.to_csv(output_dir / "ou_pairs_backtest.csv", index=False)

    return {
        "success": True,
        "summary": f"Found {len(pairs)} cointegrated pairs from {n_pairs} tested",
        "top_pairs": pairs_df.head(5)[["symbol_1", "symbol_2", "p_value", "half_life"]].to_dict(
            "records"
        )
        if len(pairs_df) > 0
        else [],
        "opportunities": opportunities[["symbol_1", "symbol_2", "current_z"]].to_dict("records")
        if len(opportunities) > 0
        else [],
    }


if __name__ == "__main__":
    asyncio.run(run())
