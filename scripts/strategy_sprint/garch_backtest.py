"""
GARCH Backtest on Volatile Symbols (Real Massive Data Only).

Runs comprehensive backtest of GARCH Breakout strategy on high-volatility
stocks from the Massive historical export. NO SYNTHETIC DATA.
"""

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from massive_data import VOLATILE_SYMBOLS

logger = logging.getLogger(__name__)

# High-volatility symbols from Massive data (based on historical vol)
SYMBOLS = VOLATILE_SYMBOLS  # NVDA, MS, BAC, META, WFC, EOG, SLB, GS


@dataclass
class GARCHConfig:
    """GARCH backtest configuration."""

    garch_lookback: int = 60
    realized_window: int = 20
    breakout_threshold: float = 1.5
    atr_stop_mult: float = 2.0
    atr_tp_mult: float = 3.0


def calculate_ewma_volatility(returns: pd.Series, span: int = 60) -> pd.Series:
    """Calculate EWMA volatility."""
    return returns.ewm(span=span).std()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def backtest_garch(
    df: pd.DataFrame,
    symbol: str,
    config: GARCHConfig,
) -> dict:
    """Run GARCH Breakout backtest on single symbol (simplified, no async)."""
    trades = []
    position = None

    start_idx = config.garch_lookback + 30

    # Pre-calculate volatility
    returns = df["close"].pct_change().dropna()

    for i in range(start_idx, len(df)):
        # Calculate EWMA volatility
        window_returns = returns.iloc[max(0, i - config.garch_lookback) : i]

        if len(window_returns) < 20:
            continue

        current_vol = window_returns.ewm(span=config.garch_lookback).std().iloc[-1]
        long_term_vol = window_returns.std()

        # Breakout ratio
        breakout_ratio = current_vol / long_term_vol if long_term_vol > 0 else 1.0

        current_price = df["close"].iloc[i]
        timestamp = df.index[i]

        # Check exits first
        if position is not None:
            exit_reason = None

            if position["direction"] == 1:
                if current_price <= position["stop_loss"]:
                    exit_reason = "stop_loss"
                elif current_price >= position["take_profit"]:
                    exit_reason = "take_profit"
            elif current_price >= position["stop_loss"]:
                exit_reason = "stop_loss"
            elif current_price <= position["take_profit"]:
                exit_reason = "take_profit"

            # Time-based exit (max 10 days)
            if position.get("bars_held", 0) >= 10:
                exit_reason = "time_exit"

            if exit_reason:
                pnl = (current_price - position["entry"]) * position["direction"]
                pnl_pct = pnl / position["entry"] * 100
                trades.append(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": timestamp,
                        "entry_price": position["entry"],
                        "exit_price": current_price,
                        "direction": position["direction"],
                        "pnl_pct": pnl_pct,
                        "exit_reason": exit_reason,
                        "bars_held": position.get("bars_held", 0),
                        "breakout_ratio": position.get("breakout_ratio", 0),
                    }
                )
                position = None
            else:
                position["bars_held"] = position.get("bars_held", 0) + 1

        # New entry: signal when breakout_ratio exceeds threshold
        if position is None and breakout_ratio > config.breakout_threshold:
            # Direction from recent trend
            recent_return = df["close"].iloc[i] / df["close"].iloc[i - 5] - 1
            direction = 1 if recent_return > 0 else -1

            # ATR for stop/target
            atr = (df["high"] - df["low"]).iloc[i - 14 : i].mean()

            position = {
                "entry": current_price,
                "entry_time": timestamp,
                "direction": direction,
                "stop_loss": current_price - direction * atr * config.atr_stop_mult,
                "take_profit": current_price + direction * atr * config.atr_tp_mult,
                "bars_held": 0,
                "breakout_ratio": breakout_ratio,
            }

    # Calculate metrics
    if not trades:
        return {
            "symbol": symbol,
            "trades": 0,
            "total_return": 0,
            "win_rate": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "sharpe": 0,
        }

    trades_df = pd.DataFrame(trades)
    winners = trades_df[trades_df["pnl_pct"] > 0]
    losers = trades_df[trades_df["pnl_pct"] <= 0]

    total_win = winners["pnl_pct"].sum() if len(winners) > 0 else 0
    total_loss = abs(losers["pnl_pct"].sum()) if len(losers) > 0 else 0

    # Calculate drawdown
    cumulative = trades_df["pnl_pct"].cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    max_dd = drawdown.max()

    # Sharpe approximation
    if trades_df["pnl_pct"].std() > 0:
        sharpe = (
            trades_df["pnl_pct"].mean()
            / trades_df["pnl_pct"].std()
            * np.sqrt(252 / trades_df["bars_held"].mean())
        )
    else:
        sharpe = 0

    return {
        "symbol": symbol,
        "trades": len(trades),
        "total_return": trades_df["pnl_pct"].sum(),
        "win_rate": len(winners) / len(trades) * 100 if trades else 0,
        "avg_win": winners["pnl_pct"].mean() if len(winners) > 0 else 0,
        "avg_loss": losers["pnl_pct"].mean() if len(losers) > 0 else 0,
        "profit_factor": total_win / total_loss if total_loss > 0 else float("inf"),
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "avg_bars_held": trades_df["bars_held"].mean(),
        "avg_breakout_ratio": trades_df["breakout_ratio"].mean(),
        "trades_df": trades_df,
    }


def run_parameter_sweep(
    df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """Sweep breakout threshold parameter."""
    results = []

    for threshold in [1.5, 1.75, 2.0, 2.25, 2.5, 3.0]:
        config = GARCHConfig(
            breakout_threshold=threshold,
            garch_lookback=252,
            realized_window=5,
        )

        result = backtest_garch(df, symbol, config)
        result["threshold"] = threshold
        results.append(result)

    return pd.DataFrame(results)


async def run() -> dict:
    """Run GARCH backtest analysis."""
    logger.info("=" * 50)
    logger.info("GARCH BREAKOUT BACKTEST")
    logger.info("=" * 50)

    all_results = []
    sweep_results = []

    for symbol in SYMBOLS:
        logger.info(f"\n--- {symbol} ---")

        # Load data
        df = load_historical_data(symbol)

        if len(df) < 300:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} bars")
            continue

        # Default backtest
        config = GARCHConfig()
        result = backtest_garch(df, symbol, config)

        logger.info(f"  Trades: {result['trades']}")
        logger.info(f"  Total Return: {result['total_return']:.2f}%")
        logger.info(f"  Win Rate: {result['win_rate']:.1f}%")
        logger.info(f"  Profit Factor: {result['profit_factor']:.2f}")
        logger.info(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
        logger.info(f"  Sharpe: {result['sharpe']:.2f}")

        all_results.append(result)

        # Parameter sweep
        sweep_df = run_parameter_sweep(df, symbol)
        sweep_df["symbol"] = symbol
        sweep_results.append(sweep_df)

    # Aggregate results
    results_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "trades_df"} for r in all_results]
    )

    # Save results
    output_dir = Path("artifacts/reports/strategy_sprint")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "garch_backtest_results.csv", index=False)

    if sweep_results:
        sweep_all = pd.concat(sweep_results, ignore_index=True)
        sweep_all.to_csv(output_dir / "garch_parameter_sweep.csv", index=False)

    # Find best threshold per symbol
    if sweep_results:
        sweep_all = pd.concat(sweep_results, ignore_index=True)
        best = sweep_all.loc[sweep_all.groupby("symbol")["total_return"].idxmax()]
        logger.info("\nBest threshold per symbol:")
        for _, row in best.iterrows():
            logger.info(
                f"  {row['symbol']}: threshold={row['threshold']}, return={row['total_return']:.2f}%"
            )

    return {
        "success": True,
        "summary": f"Tested {len(SYMBOLS)} symbols, avg return {results_df['total_return'].mean():.1f}%",
        "results": results_df.to_dict(),
    }


if __name__ == "__main__":
    asyncio.run(run())
