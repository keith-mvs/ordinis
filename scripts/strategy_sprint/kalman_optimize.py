"""
Kalman Filter Q/R Parameter Optimization.

Grid searches for optimal process noise (Q) and measurement noise (R)
parameters per symbol to maximize Sharpe ratio.
"""

import asyncio
from datetime import datetime
from itertools import product
import logging
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


logger = logging.getLogger(__name__)

# Symbols for optimization
SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA", "SPY", "QQQ", "GOOGL"]

# Parameter grid
Q_VALUES = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]  # Process noise
R_VALUES = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02]  # Measurement noise
MOMENTUM_THRESHOLDS = [0.3, 0.5, 0.7]


def load_historical_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """Load historical price data."""
    cache_path = Path(f"data/historical/{symbol}_daily.parquet")

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    csv_path = Path(f"data/historical/{symbol}_daily.csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date"], index_col="date")

    return generate_synthetic_trending(symbol, days)


def generate_synthetic_trending(symbol: str, days: int = 500) -> pd.DataFrame:
    """Generate synthetic data with trends for Kalman testing."""
    np.random.seed(hash(symbol) % 2**32)

    # Multiple regime switches
    n_regimes = 4
    regime_lengths = np.random.multinomial(days - 20, [1 / n_regimes] * n_regimes) + 5

    returns = []
    for i, length in enumerate(regime_lengths):
        drift = np.random.choice([-0.001, 0, 0.0005, 0.001])
        vol = np.random.uniform(0.01, 0.025)
        regime_returns = np.random.normal(drift, vol, length)
        returns.extend(regime_returns)

    returns = np.array(returns[:days])

    base_price = np.random.uniform(100, 400)
    prices = base_price * np.exp(np.cumsum(returns))

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    df = pd.DataFrame(index=dates)
    df["close"] = prices
    df["high"] = prices * (1 + np.abs(np.random.normal(0, 0.01, days)))
    df["low"] = prices * (1 - np.abs(np.random.normal(0, 0.01, days)))
    df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
    df["volume"] = np.random.randint(5_000_000, 50_000_000, days)

    df["high"] = df[["high", "close", "open"]].max(axis=1)
    df["low"] = df[["low", "close", "open"]].min(axis=1)

    return df


class KalmanFilter:
    """Simple Kalman filter implementation for optimization."""

    def __init__(self, q: float = 0.001, r: float = 0.001):
        self.q = q  # Process noise
        self.r = r  # Measurement noise
        self.x = None  # State estimate
        self.p = 1.0  # Estimate covariance

    def update(self, measurement: float) -> tuple[float, float]:
        """Update filter with new measurement."""
        if self.x is None:
            self.x = measurement
            return self.x, 0.0

        # Predict
        x_pred = self.x
        p_pred = self.p + self.q

        # Kalman gain
        k = p_pred / (p_pred + self.r)

        # Update
        self.x = x_pred + k * (measurement - x_pred)
        self.p = (1 - k) * p_pred

        # Return filtered value and velocity estimate
        velocity = self.x - x_pred

        return self.x, velocity


def backtest_kalman(
    df: pd.DataFrame,
    q: float,
    r: float,
    momentum_threshold: float = 0.5,
) -> dict:
    """Backtest Kalman filter strategy."""
    kf = KalmanFilter(q=q, r=r)

    signals = []
    positions = []
    returns = []

    position = 0

    for i in range(len(df)):
        close = df["close"].iloc[i]

        filtered, velocity = kf.update(close)

        # Normalize velocity
        vel_norm = velocity / close if close > 0 else 0

        # Signal based on velocity
        if vel_norm > momentum_threshold * 0.01:
            signal = 1
        elif vel_norm < -momentum_threshold * 0.01:
            signal = -1
        else:
            signal = 0

        signals.append(signal)
        positions.append(position)

        # Calculate return
        if i > 0:
            daily_return = df["close"].iloc[i] / df["close"].iloc[i - 1] - 1
            strat_return = position * daily_return
            returns.append(strat_return)

        position = signal

    if not returns:
        return {"sharpe": 0, "total_return": 0, "trades": 0}

    returns = np.array(returns)

    total_return = (1 + returns).prod() - 1
    avg_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    sharpe = avg_return / volatility if volatility > 0 else 0

    # Count trades
    positions = np.array(positions)
    trades = np.sum(np.abs(np.diff(positions)) > 0)

    # Win rate
    winning = returns[returns > 0]
    win_rate = len(winning) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0

    return {
        "sharpe": sharpe,
        "total_return": total_return * 100,
        "avg_annual_return": avg_return * 100,
        "volatility": volatility * 100,
        "trades": trades,
        "win_rate": win_rate * 100,
    }


def optimize_symbol(
    symbol: str,
    df: pd.DataFrame,
    q_values: list[float],
    r_values: list[float],
    momentum_thresholds: list[float],
) -> dict:
    """Find optimal Q/R parameters for a symbol."""
    results = []

    for q, r, mom in product(q_values, r_values, momentum_thresholds):
        metrics = backtest_kalman(df, q, r, mom)

        results.append(
            {
                "q": q,
                "r": r,
                "momentum_threshold": mom,
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)

    # Find best by Sharpe
    best_idx = results_df["sharpe"].idxmax()
    best = results_df.iloc[best_idx]

    # Find Pareto-optimal (high Sharpe, low trades)
    results_df["efficiency"] = results_df["sharpe"] / (results_df["trades"] + 1)
    efficient_idx = results_df["efficiency"].idxmax()
    efficient = results_df.iloc[efficient_idx]

    return {
        "symbol": symbol,
        "best": {
            "q": best["q"],
            "r": best["r"],
            "momentum_threshold": best["momentum_threshold"],
            "sharpe": best["sharpe"],
            "return": best["total_return"],
            "trades": best["trades"],
        },
        "efficient": {
            "q": efficient["q"],
            "r": efficient["r"],
            "momentum_threshold": efficient["momentum_threshold"],
            "sharpe": efficient["sharpe"],
            "return": efficient["total_return"],
            "trades": efficient["trades"],
        },
        "grid_df": results_df,
    }


def analyze_q_r_relationship(grid_df: pd.DataFrame) -> dict:
    """Analyze relationship between Q/R and performance."""
    # Average Sharpe by Q
    q_analysis = grid_df.groupby("q")["sharpe"].mean().to_dict()

    # Average Sharpe by R
    r_analysis = grid_df.groupby("r")["sharpe"].mean().to_dict()

    # Q/R ratio analysis
    grid_df["q_r_ratio"] = grid_df["q"] / grid_df["r"]
    ratio_corr = grid_df["q_r_ratio"].corr(grid_df["sharpe"])

    return {
        "q_sharpe_map": q_analysis,
        "r_sharpe_map": r_analysis,
        "qr_ratio_correlation": ratio_corr,
    }


async def run() -> dict:
    """Run Kalman Q/R optimization."""
    logger.info("=" * 50)
    logger.info("KALMAN FILTER Q/R OPTIMIZATION")
    logger.info("=" * 50)

    logger.info(
        f"Testing {len(Q_VALUES)} Q values x {len(R_VALUES)} R values "
        f"x {len(MOMENTUM_THRESHOLDS)} momentum thresholds"
    )
    logger.info(
        f"Total combinations per symbol: {len(Q_VALUES) * len(R_VALUES) * len(MOMENTUM_THRESHOLDS)}"
    )

    all_results = []
    summary = []

    for symbol in SYMBOLS:
        logger.info(f"\n--- Optimizing {symbol} ---")

        df = load_historical_data(symbol)
        if len(df) < 300:
            logger.warning(f"  Insufficient data for {symbol}")
            continue

        result = optimize_symbol(symbol, df, Q_VALUES, R_VALUES, MOMENTUM_THRESHOLDS)

        best = result["best"]
        efficient = result["efficient"]

        logger.info(f"  Best (max Sharpe):")
        logger.info(
            f"    Q={best['q']:.4f}, R={best['r']:.4f}, "
            f"MomThresh={best['momentum_threshold']:.1f}"
        )
        logger.info(
            f"    Sharpe: {best['sharpe']:.2f}, Return: {best['return']:.1f}%, "
            f"Trades: {best['trades']}"
        )

        logger.info(f"  Efficient (Sharpe/Trade):")
        logger.info(f"    Q={efficient['q']:.4f}, R={efficient['r']:.4f}")
        logger.info(f"    Sharpe: {efficient['sharpe']:.2f}, Trades: {efficient['trades']}")

        # Analyze Q/R relationship
        analysis = analyze_q_r_relationship(result["grid_df"])
        logger.info(f"  Q/R ratio correlation with Sharpe: {analysis['qr_ratio_correlation']:.3f}")

        summary.append(
            {
                "symbol": symbol,
                "best_q": best["q"],
                "best_r": best["r"],
                "best_momentum": best["momentum_threshold"],
                "best_sharpe": best["sharpe"],
                "best_return": best["return"],
                "best_trades": best["trades"],
                "eff_q": efficient["q"],
                "eff_r": efficient["r"],
                "eff_sharpe": efficient["sharpe"],
                "eff_trades": efficient["trades"],
                "qr_ratio_corr": analysis["qr_ratio_correlation"],
            }
        )

        all_results.append(result)

    # Aggregate findings
    summary_df = pd.DataFrame(summary)

    logger.info("\n" + "=" * 50)
    logger.info("AGGREGATE FINDINGS")
    logger.info("=" * 50)

    # Most common best Q
    q_mode = summary_df["best_q"].mode().iloc[0] if len(summary_df) > 0 else 0.001
    r_mode = summary_df["best_r"].mode().iloc[0] if len(summary_df) > 0 else 0.001

    logger.info(f"Most common best Q: {q_mode}")
    logger.info(f"Most common best R: {r_mode}")
    logger.info(f"Average best Sharpe: {summary_df['best_sharpe'].mean():.2f}")
    logger.info(f"Average best Return: {summary_df['best_return'].mean():.1f}%")

    # Symbol-specific recommendations
    logger.info("\n--- Per-Symbol Recommendations ---")
    for _, row in summary_df.iterrows():
        logger.info(
            f"  {row['symbol']}: Q={row['best_q']:.4f}, R={row['best_r']:.4f} "
            f"(Sharpe: {row['best_sharpe']:.2f})"
        )

    # Save results
    output_dir = Path("artifacts/reports/strategy_sprint")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / "kalman_optimization_summary.csv", index=False)

    # Save detailed grid for best symbol
    if all_results:
        best_symbol_idx = summary_df["best_sharpe"].idxmax()
        best_symbol = summary_df.iloc[best_symbol_idx]["symbol"]
        all_results[best_symbol_idx]["grid_df"].to_csv(
            output_dir / f"kalman_grid_{best_symbol}.csv", index=False
        )

    return {
        "success": True,
        "summary": f"Optimized {len(summary)} symbols",
        "recommended_q": float(q_mode),
        "recommended_r": float(r_mode),
        "best_sharpe": float(summary_df["best_sharpe"].max()) if len(summary_df) > 0 else 0,
        "symbol_configs": summary_df.to_dict("records") if len(summary_df) > 0 else [],
    }


if __name__ == "__main__":
    asyncio.run(run())
