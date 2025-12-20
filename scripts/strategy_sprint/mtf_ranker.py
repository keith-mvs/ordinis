"""
MTF Momentum Universe Ranker.

Builds cross-sectional momentum rankings for a universe of stocks,
then applies stochastic timing for entry signals.
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


logger = logging.getLogger(__name__)

# Universe for momentum ranking
UNIVERSE = [
    # Tech
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "AMD",
    "TSLA",
    # Consumer
    "COST",
    "WMT",
    "HD",
    "NKE",
    # Finance
    "JPM",
    "BAC",
    "GS",
    "V",
    "MA",
    # Health
    "JNJ",
    "UNH",
    "PFE",
    "ABBV",
    # Energy
    "XOM",
    "CVX",
    # High vol
    "COIN",
    "DKNG",
    "RIVN",
]


def load_universe_data(symbols: list[str], days: int = 400) -> dict[str, pd.DataFrame]:
    """Load data for entire universe."""
    data = {}

    for symbol in symbols:
        df = load_symbol_data(symbol, days)
        if df is not None and len(df) > 260:
            data[symbol] = df

    return data


def load_symbol_data(symbol: str, days: int = 400) -> pd.DataFrame:
    """Load single symbol data."""
    cache_path = Path(f"data/historical/{symbol}_daily.parquet")

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    csv_path = Path(f"data/historical/{symbol}_daily.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

    # Generate synthetic
    return generate_synthetic_momentum(symbol, days)


def generate_synthetic_momentum(symbol: str, days: int = 400) -> pd.DataFrame:
    """Generate synthetic data with momentum characteristics."""
    np.random.seed(hash(symbol) % 2**32)

    # Momentum persistence factor (some stocks trend more)
    persistence = np.random.uniform(0.01, 0.05)
    volatility = np.random.uniform(0.015, 0.04)

    # Generate trending returns
    trend = np.random.choice([-1, 1]) * persistence
    noise = np.random.normal(0, volatility, days)

    # Add momentum (autocorrelation)
    returns = np.zeros(days)
    returns[0] = trend + noise[0]
    for i in range(1, days):
        # Mean reversion toward trend with momentum
        returns[i] = 0.3 * trend + 0.5 * returns[i - 1] * 0.1 + noise[i]

    base_price = np.random.uniform(50, 500)
    prices = base_price * np.exp(np.cumsum(returns))

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["high"] = prices * (1 + np.abs(np.random.normal(0, 0.01, days)))
    df["low"] = prices * (1 - np.abs(np.random.normal(0, 0.01, days)))
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["volume"] = np.random.randint(1_000_000, 50_000_000, days)

    df["high"] = df[["high", "close", "open"]].max(axis=1)
    df["low"] = df[["low", "close", "open"]].min(axis=1)

    return df


def calculate_momentum_12_1(prices: pd.Series) -> float:
    """Calculate 12-1 month momentum."""
    if len(prices) < 252:
        return np.nan

    # 12-month return minus 1-month return
    ret_12m = prices.iloc[-21] / prices.iloc[-252] - 1
    return ret_12m


def build_momentum_rankings(
    universe_data: dict[str, pd.DataFrame],
    as_of_date: datetime = None,
) -> pd.DataFrame:
    """Build cross-sectional momentum rankings."""
    if as_of_date is None:
        as_of_date = datetime.now()

    rankings = []

    for symbol, df in universe_data.items():
        # Filter to as_of_date
        df_filtered = df[df.index <= as_of_date]

        if len(df_filtered) < 252:
            continue

        mom = calculate_momentum_12_1(df_filtered["close"])

        if np.isnan(mom):
            continue

        # Get recent volatility
        returns = df_filtered["close"].pct_change().dropna()
        vol = returns.iloc[-21:].std() * np.sqrt(252)

        # Current price info
        current_price = df_filtered["close"].iloc[-1]

        rankings.append(
            {
                "symbol": symbol,
                "momentum_12_1": mom,
                "volatility": vol,
                "momentum_vol_adj": mom / vol if vol > 0 else 0,
                "price": current_price,
            }
        )

    rankings_df = pd.DataFrame(rankings)

    # Rank by momentum
    rankings_df["mom_rank"] = rankings_df["momentum_12_1"].rank(pct=True)
    rankings_df["mom_vol_rank"] = rankings_df["momentum_vol_adj"].rank(pct=True)

    # Sort by momentum
    rankings_df = rankings_df.sort_values("momentum_12_1", ascending=False)

    return rankings_df


def get_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
    """Calculate stochastic oscillator."""
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()

    k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)
    d = k.rolling(d_period).mean()

    return k, d


def identify_entry_signals(
    rankings_df: pd.DataFrame,
    universe_data: dict[str, pd.DataFrame],
    top_pct: float = 0.2,
    bottom_pct: float = 0.2,
) -> dict[str, dict]:
    """Identify entry signals for top/bottom momentum stocks."""
    signals = {}

    # Top momentum (long candidates)
    long_candidates = rankings_df[rankings_df["mom_rank"] >= (1 - top_pct)]

    # Bottom momentum (short candidates)
    short_candidates = rankings_df[rankings_df["mom_rank"] <= bottom_pct]

    for _, row in long_candidates.iterrows():
        symbol = row["symbol"]
        df = universe_data[symbol]

        k, d = get_stochastic(df)

        k_curr = k.iloc[-1]
        k_prev = k.iloc[-2]
        d_curr = d.iloc[-1]
        d_prev = d.iloc[-2]

        # Bullish crossover from oversold
        bullish_cross = k_curr > d_curr and k_prev <= d_prev
        oversold = k_curr < 30

        if bullish_cross or oversold:
            signals[symbol] = {
                "direction": "LONG",
                "momentum_rank": row["mom_rank"],
                "momentum": row["momentum_12_1"],
                "stoch_k": k_curr,
                "stoch_d": d_curr,
                "signal_type": "bullish_cross" if bullish_cross else "oversold",
                "strength": row["mom_vol_rank"],
            }

    for _, row in short_candidates.iterrows():
        symbol = row["symbol"]
        df = universe_data[symbol]

        k, d = get_stochastic(df)

        k_curr = k.iloc[-1]
        k_prev = k.iloc[-2]
        d_curr = d.iloc[-1]
        d_prev = d.iloc[-2]

        # Bearish crossover from overbought
        bearish_cross = k_curr < d_curr and k_prev >= d_prev
        overbought = k_curr > 70

        if bearish_cross or overbought:
            signals[symbol] = {
                "direction": "SHORT",
                "momentum_rank": row["mom_rank"],
                "momentum": row["momentum_12_1"],
                "stoch_k": k_curr,
                "stoch_d": d_curr,
                "signal_type": "bearish_cross" if bearish_cross else "overbought",
                "strength": 1 - row["mom_vol_rank"],
            }

    return signals


def backtest_momentum_ranker(
    universe_data: dict[str, pd.DataFrame],
    rebalance_freq: int = 21,  # Monthly
    top_n: int = 5,
    bottom_n: int = 5,
) -> dict:
    """Backtest momentum ranking strategy."""
    # Find common date range
    common_dates = None
    for symbol, df in universe_data.items():
        if common_dates is None:
            common_dates = set(df.index)
        else:
            common_dates = common_dates.intersection(set(df.index))

    common_dates = sorted(list(common_dates))

    if len(common_dates) < 300:
        return {"error": "Insufficient common dates"}

    # Start after 252 days for momentum calculation
    start_idx = 260

    portfolio_returns = []
    rebalance_dates = []
    holdings = {}

    for i in range(start_idx, len(common_dates), rebalance_freq):
        as_of = common_dates[i]

        # Build rankings as of this date
        rankings = build_momentum_rankings(universe_data, as_of)

        if len(rankings) < top_n + bottom_n:
            continue

        # Select top/bottom stocks
        longs = rankings.head(top_n)["symbol"].tolist()
        shorts = rankings.tail(bottom_n)["symbol"].tolist()

        rebalance_dates.append(as_of)

        # Calculate returns until next rebalance
        end_idx = min(i + rebalance_freq, len(common_dates) - 1)

        period_return = 0
        for symbol in longs:
            df = universe_data[symbol]
            df_period = df[(df.index >= as_of) & (df.index <= common_dates[end_idx])]
            if len(df_period) > 1:
                ret = df_period["close"].iloc[-1] / df_period["close"].iloc[0] - 1
                period_return += ret / top_n

        for symbol in shorts:
            df = universe_data[symbol]
            df_period = df[(df.index >= as_of) & (df.index <= common_dates[end_idx])]
            if len(df_period) > 1:
                ret = df_period["close"].iloc[-1] / df_period["close"].iloc[0] - 1
                period_return -= ret / bottom_n  # Short = negative

        portfolio_returns.append(
            {
                "date": as_of,
                "return": period_return,
                "longs": longs,
                "shorts": shorts,
            }
        )

    if not portfolio_returns:
        return {"error": "No portfolio returns"}

    returns_df = pd.DataFrame(portfolio_returns)

    total_return = (1 + returns_df["return"]).prod() - 1
    avg_return = returns_df["return"].mean()
    volatility = returns_df["return"].std() * np.sqrt(12)  # Annualized
    sharpe = avg_return * 12 / volatility if volatility > 0 else 0

    return {
        "total_return": total_return * 100,
        "avg_monthly_return": avg_return * 100,
        "volatility": volatility * 100,
        "sharpe": sharpe,
        "rebalances": len(returns_df),
        "returns_df": returns_df,
    }


async def run() -> dict:
    """Run MTF Momentum universe ranker analysis."""
    logger.info("=" * 50)
    logger.info("MTF MOMENTUM UNIVERSE RANKER")
    logger.info("=" * 50)

    # Load universe data
    logger.info(f"Loading data for {len(UNIVERSE)} symbols...")
    universe_data = load_universe_data(UNIVERSE)
    logger.info(f"Loaded {len(universe_data)} symbols with sufficient data")

    # Current rankings
    logger.info("\n--- Current Momentum Rankings ---")
    rankings = build_momentum_rankings(universe_data)

    logger.info("\nTop 10 Momentum:")
    for i, row in rankings.head(10).iterrows():
        logger.info(
            f"  {row['symbol']:6s} Mom: {row['momentum_12_1']*100:+6.1f}% "
            f"Rank: {row['mom_rank']:.2f} Vol: {row['volatility']*100:.1f}%"
        )

    logger.info("\nBottom 10 Momentum:")
    for i, row in rankings.tail(10).iterrows():
        logger.info(
            f"  {row['symbol']:6s} Mom: {row['momentum_12_1']*100:+6.1f}% "
            f"Rank: {row['mom_rank']:.2f} Vol: {row['volatility']*100:.1f}%"
        )

    # Entry signals
    logger.info("\n--- Current Entry Signals ---")
    signals = identify_entry_signals(rankings, universe_data)

    if signals:
        for symbol, sig in signals.items():
            logger.info(
                f"  {symbol:6s} {sig['direction']:5s} | "
                f"Mom: {sig['momentum']*100:+6.1f}% | "
                f"Stoch: {sig['stoch_k']:.0f}/{sig['stoch_d']:.0f} | "
                f"{sig['signal_type']}"
            )
    else:
        logger.info("  No entry signals currently")

    # Backtest
    logger.info("\n--- Momentum Backtest ---")
    backtest = backtest_momentum_ranker(universe_data, top_n=5, bottom_n=5)

    if "error" not in backtest:
        logger.info(f"  Total Return: {backtest['total_return']:.1f}%")
        logger.info(f"  Avg Monthly: {backtest['avg_monthly_return']:.2f}%")
        logger.info(f"  Volatility: {backtest['volatility']:.1f}%")
        logger.info(f"  Sharpe: {backtest['sharpe']:.2f}")
        logger.info(f"  Rebalances: {backtest['rebalances']}")
    else:
        logger.warning(f"  Backtest error: {backtest['error']}")

    # Save results
    output_dir = Path("artifacts/reports/strategy_sprint")
    output_dir.mkdir(parents=True, exist_ok=True)

    rankings.to_csv(output_dir / "mtf_current_rankings.csv", index=False)

    signals_df = pd.DataFrame([{"symbol": k, **v} for k, v in signals.items()])
    if len(signals_df) > 0:
        signals_df.to_csv(output_dir / "mtf_current_signals.csv", index=False)

    return {
        "success": True,
        "summary": f"Ranked {len(rankings)} symbols, {len(signals)} entry signals",
        "top_momentum": rankings.head(5)["symbol"].tolist(),
        "bottom_momentum": rankings.tail(5)["symbol"].tolist(),
        "current_signals": list(signals.keys()),
    }


if __name__ == "__main__":
    asyncio.run(run())
