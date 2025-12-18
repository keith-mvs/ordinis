"""
Mutual Information Signal Analysis.

Analyzes the predictive power of various technical signals using
mutual information to measure non-linear dependencies with future returns.
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logger = logging.getLogger(__name__)

# Symbols for analysis
SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD", "AMZN"]

# Signal lookbacks to test
SIGNAL_PARAMS = {
    "rsi": [7, 14, 21],
    "macd_hist": [(12, 26, 9)],
    "stoch": [14],
    "atr_ratio": [14],
    "bb_position": [20],
    "volume_ratio": [20],
    "momentum": [5, 10, 20],
    "roc": [5, 10, 20],
}


def load_historical_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """Load historical price data."""
    cache_path = Path(f"data/historical/{symbol}_daily.parquet")

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    csv_path = Path(f"data/historical/{symbol}_daily.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

    return generate_synthetic_data(symbol, days)


def generate_synthetic_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """Generate synthetic price data with realistic patterns."""
    np.random.seed(hash(symbol) % 2**32)

    drift = np.random.uniform(0.0001, 0.0005)
    volatility = np.random.uniform(0.012, 0.025)

    # Add some autocorrelation and mean reversion
    returns = np.zeros(days)
    for i in range(days):
        if i == 0:
            returns[i] = np.random.normal(drift, volatility)
        else:
            momentum = 0.1 * returns[i - 1]
            mean_rev = -0.02 * np.sum(returns[:i])
            returns[i] = drift + momentum + mean_rev + np.random.normal(0, volatility)

    base_price = np.random.uniform(100, 500)
    prices = base_price * np.exp(np.cumsum(returns))

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["high"] = prices * (1 + np.abs(np.random.normal(0, 0.012, days)))
    df["low"] = prices * (1 - np.abs(np.random.normal(0, 0.012, days)))
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["volume"] = np.random.randint(10_000_000, 100_000_000, days).astype(float)

    df["high"] = df[["high", "close", "open"]].max(axis=1)
    df["low"] = df[["low", "close", "open"]].min(axis=1)

    return df


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    """Calculate MACD histogram."""
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd - signal_line


def calculate_stoch(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate stochastic oscillator."""
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    return 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(abs(df["high"] - df["close"].shift(1)), abs(df["low"] - df["close"].shift(1))),
    )
    return tr.rolling(period).mean()


def calculate_bb_position(close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate position within Bollinger Bands (0-1)."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (close - lower) / (upper - lower + 1e-10)


def calculate_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical signals."""
    signals = pd.DataFrame(index=df.index)

    # RSI
    for period in SIGNAL_PARAMS["rsi"]:
        signals[f"rsi_{period}"] = calculate_rsi(df["close"], period)

    # MACD Histogram
    for fast, slow, sig in SIGNAL_PARAMS["macd_hist"]:
        signals["macd_hist"] = calculate_macd_hist(df["close"], fast, slow, sig)

    # Stochastic
    for period in SIGNAL_PARAMS["stoch"]:
        signals[f"stoch_{period}"] = calculate_stoch(df, period)

    # ATR Ratio (current ATR / SMA of ATR)
    for period in SIGNAL_PARAMS["atr_ratio"]:
        atr = calculate_atr(df, period)
        atr_sma = atr.rolling(50).mean()
        signals[f"atr_ratio_{period}"] = atr / (atr_sma + 1e-10)

    # BB Position
    for period in SIGNAL_PARAMS["bb_position"]:
        signals[f"bb_pos_{period}"] = calculate_bb_position(df["close"], period)

    # Volume Ratio
    for period in SIGNAL_PARAMS["volume_ratio"]:
        vol_sma = df["volume"].rolling(period).mean()
        signals[f"vol_ratio_{period}"] = df["volume"] / (vol_sma + 1e-10)

    # Momentum (rate of change normalized)
    for period in SIGNAL_PARAMS["momentum"]:
        signals[f"mom_{period}"] = df["close"].pct_change(period)

    # ROC
    for period in SIGNAL_PARAMS["roc"]:
        signals[f"roc_{period}"] = (df["close"] / df["close"].shift(period) - 1) * 100

    return signals


def discretize(series: pd.Series, n_bins: int = 10) -> pd.Series:
    """Discretize continuous series into bins."""
    series_clean = series.dropna()
    if len(series_clean) < n_bins:
        return pd.Series(index=series.index, dtype=float)

    # Use quantile-based binning
    bins = pd.qcut(series_clean, n_bins, labels=False, duplicates="drop")
    result = pd.Series(index=series.index, dtype=float)
    result.loc[series_clean.index] = bins
    return result


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate mutual information between two discrete arrays.
    MI(X, Y) = sum P(x,y) * log(P(x,y) / (P(x) * P(y)))
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask].astype(int)
    y = y[mask].astype(int)

    if len(x) < 10:
        return 0.0

    # Calculate joint and marginal distributions
    n = len(x)

    # Marginal probabilities
    x_vals, x_counts = np.unique(x, return_counts=True)
    y_vals, y_counts = np.unique(y, return_counts=True)

    p_x = x_counts / n
    p_y = y_counts / n

    # Joint probability
    mi = 0.0
    for i, xi in enumerate(x_vals):
        for j, yj in enumerate(y_vals):
            joint_count = np.sum((x == xi) & (y == yj))
            if joint_count == 0:
                continue

            p_xy = joint_count / n
            mi += p_xy * np.log(p_xy / (p_x[i] * p_y[j] + 1e-10) + 1e-10)

    return mi


def normalized_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate normalized mutual information (0-1)."""
    mi = mutual_information(x, y)

    # Calculate entropy
    def entropy(arr):
        arr = arr[~np.isnan(arr)].astype(int)
        vals, counts = np.unique(arr, return_counts=True)
        p = counts / len(arr)
        return -np.sum(p * np.log(p + 1e-10))

    h_x = entropy(x)
    h_y = entropy(y)

    if h_x == 0 or h_y == 0:
        return 0.0

    return 2 * mi / (h_x + h_y)


def analyze_signal_mi(
    signals: pd.DataFrame,
    returns: pd.Series,
    horizons: list[int] = [1, 5, 10, 20],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Analyze MI between signals and future returns."""
    results = []

    for col in signals.columns:
        signal = signals[col]
        signal_disc = discretize(signal, n_bins)

        for horizon in horizons:
            # Future returns
            future_ret = returns.shift(-horizon)
            ret_disc = discretize(future_ret, n_bins)

            # Calculate MI
            mi = normalized_mi(signal_disc.values, ret_disc.values)

            # Calculate correlation for comparison
            corr = signal.corr(future_ret)

            results.append(
                {
                    "signal": col,
                    "horizon": horizon,
                    "normalized_mi": mi,
                    "correlation": corr,
                    "mi_to_corr_ratio": mi / (abs(corr) + 1e-10),
                }
            )

    return pd.DataFrame(results)


async def run() -> dict:
    """Run MI signal analysis."""
    logger.info("=" * 50)
    logger.info("MUTUAL INFORMATION SIGNAL ANALYSIS")
    logger.info("=" * 50)

    all_results = []

    for symbol in SYMBOLS:
        logger.info(f"\n--- Analyzing {symbol} ---")

        df = load_historical_data(symbol)
        if len(df) < 300:
            logger.warning(f"  Insufficient data for {symbol}")
            continue

        # Calculate signals
        signals = calculate_all_signals(df)
        returns = df["close"].pct_change()

        # Analyze MI
        mi_results = analyze_signal_mi(signals, returns)
        mi_results["symbol"] = symbol

        all_results.append(mi_results)

        # Top signals by MI at 5-day horizon
        horizon_5 = mi_results[mi_results["horizon"] == 5].sort_values(
            "normalized_mi", ascending=False
        )

        logger.info(f"  Top signals (5-day horizon):")
        for _, row in horizon_5.head(5).iterrows():
            logger.info(
                f"    {row['signal']:15s} MI: {row['normalized_mi']:.4f} "
                f"Corr: {row['correlation']:+.3f}"
            )

    # Aggregate results
    if not all_results:
        return {"success": False, "error": "No results"}

    combined = pd.concat(all_results)

    # Average MI by signal across symbols
    signal_avg = (
        combined.groupby(["signal", "horizon"])
        .agg(
            {
                "normalized_mi": "mean",
                "correlation": "mean",
            }
        )
        .reset_index()
    )

    logger.info("\n" + "=" * 50)
    logger.info("AGGREGATE SIGNAL RANKINGS")
    logger.info("=" * 50)

    # Best signals for each horizon
    for horizon in [1, 5, 10, 20]:
        logger.info(f"\n--- Top Signals (Horizon {horizon}d) ---")
        horizon_data = signal_avg[signal_avg["horizon"] == horizon].sort_values(
            "normalized_mi", ascending=False
        )

        for _, row in horizon_data.head(5).iterrows():
            logger.info(
                f"  {row['signal']:15s} MI: {row['normalized_mi']:.4f} "
                f"Corr: {row['correlation']:+.3f}"
            )

    # Best overall signal (average across horizons)
    overall = (
        signal_avg.groupby("signal")
        .agg(
            {
                "normalized_mi": "mean",
                "correlation": lambda x: x.abs().mean(),
            }
        )
        .reset_index()
    )
    overall = overall.sort_values("normalized_mi", ascending=False)

    logger.info("\n--- Best Overall Signals ---")
    for _, row in overall.head(10).iterrows():
        logger.info(f"  {row['signal']:15s} Avg MI: {row['normalized_mi']:.4f}")

    # Non-linear signals (high MI but low correlation)
    overall["nonlinearity"] = overall["normalized_mi"] / (overall["correlation"] + 1e-10)
    nonlinear = overall.sort_values("nonlinearity", ascending=False)

    logger.info("\n--- Most Non-Linear Signals ---")
    for _, row in nonlinear.head(5).iterrows():
        logger.info(
            f"  {row['signal']:15s} MI: {row['normalized_mi']:.4f} "
            f"Corr: {row['correlation']:.3f}"
        )

    # Save results
    output_dir = Path("artifacts/reports/strategy_sprint")
    output_dir.mkdir(parents=True, exist_ok=True)

    combined.to_csv(output_dir / "mi_analysis_full.csv", index=False)
    signal_avg.to_csv(output_dir / "mi_analysis_summary.csv", index=False)
    overall.to_csv(output_dir / "mi_best_signals.csv", index=False)

    return {
        "success": True,
        "summary": f"Analyzed {len(signals.columns)} signals across {len(SYMBOLS)} symbols",
        "best_signals_5d": signal_avg[signal_avg["horizon"] == 5]
        .sort_values("normalized_mi", ascending=False)
        .head(5)["signal"]
        .tolist(),
        "most_nonlinear": nonlinear.head(3)["signal"].tolist(),
    }


if __name__ == "__main__":
    asyncio.run(run())
