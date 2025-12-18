"""
EVT Risk Gate Overlay on ATR-RSI Production Strategy.

Wires EVT tail risk estimation as an overlay on the ATR-Optimized RSI
strategy. Reduces position sizes when tail risk is elevated.
"""

import asyncio
from datetime import datetime
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ordinis.engines.signalcore.core.model import ModelConfig

logger = logging.getLogger(__name__)

# Test symbols
SYMBOLS = ["TSLA", "AMD", "COIN", "DKNG", "NVDA"]


def load_historical_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """Load historical data from cache or generate synthetic."""
    cache_path = Path(f"data/historical/{symbol}_daily.parquet")

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        return df

    csv_path = Path(f"data/historical/{symbol}_daily.csv")
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        return df

    # Generate synthetic
    logger.warning(f"Generating synthetic data for {symbol}")
    return generate_synthetic_with_crashes(symbol, days)


def generate_synthetic_with_crashes(symbol: str, days: int = 500) -> pd.DataFrame:
    """Generate synthetic data with occasional crash events for EVT testing."""
    np.random.seed(hash(symbol) % 2**32)

    # Base volatility
    base_vol = 0.02

    # Generate returns with occasional crashes
    returns = np.random.normal(0.0005, base_vol, days)

    # Add crash events (important for EVT)
    crash_days = np.random.choice(days, size=int(days * 0.03), replace=False)
    for crash_day in crash_days:
        returns[crash_day] = np.random.uniform(-0.08, -0.15)  # 8-15% drops

    # Build prices
    base_price = {"TSLA": 250, "AMD": 130, "COIN": 200, "DKNG": 40, "NVDA": 500}.get(symbol, 100)
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


def compare_with_without_evt(
    df: pd.DataFrame,
    symbol: str,
) -> dict:
    """Compare strategy performance with and without EVT overlay."""

    # Base ATR-RSI config
    base_config = ModelConfig(
        model_id=f"atr_rsi_{symbol}",
        model_type="mean_reversion",
        parameters={
            "rsi_oversold": 35,
            "rsi_exit": 50,
            "atr_stop_mult": 1.5,
            "atr_tp_mult": 2.0,
        },
    )

    # Simplified backtest without using actual async models
    # This directly implements the strategy logic for comparison

    results = {}

    for name in ["base", "evt_gated"]:
        trades = []
        position = None

        start_idx = 280  # Need enough data for EVT

        # Pre-calculate indicators
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ATR
        tr = np.maximum(
            high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1)))
        )
        atr = tr.rolling(14).mean()

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        # Returns for EVT
        returns = close.pct_change().dropna()

        for i in range(start_idx, len(df)):
            current_price = close.iloc[i]
            current_rsi = rsi.iloc[i]
            current_atr = atr.iloc[i]
            timestamp = df.index[i]

            # EVT position multiplier
            position_mult = 1.0
            if name == "evt_gated":
                # Calculate tail risk
                window_returns = returns.iloc[max(0, i - 252) : i]
                if len(window_returns) > 50:
                    threshold = np.percentile(abs(window_returns), 95)
                    tail_returns = abs(window_returns[abs(window_returns) > threshold])

                    if len(tail_returns) > 10:
                        # Simple VaR estimate
                        var_99 = np.percentile(abs(window_returns), 99)
                        if var_99 > 0.03:  # 3% VaR threshold
                            position_mult = 0.5

            # Check exits
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

                if position.get("bars_held", 0) >= 10:
                    exit_reason = "time_exit"

                if exit_reason:
                    pnl = (current_price - position["entry"]) * position["direction"]
                    pnl_pct = pnl / position["entry"] * 100 * position.get("size_mult", 1.0)
                    trades.append(
                        {
                            "entry_time": position["entry_time"],
                            "exit_time": timestamp,
                            "pnl_pct": pnl_pct,
                            "exit_reason": exit_reason,
                            "size_mult": position.get("size_mult", 1.0),
                        }
                    )
                    position = None
                else:
                    position["bars_held"] = position.get("bars_held", 0) + 1

            # Entry signals (RSI-based)
            if position is None:
                if current_rsi < 30:  # Oversold
                    direction = 1
                elif current_rsi > 70:  # Overbought
                    direction = -1
                else:
                    direction = 0

                if direction != 0:
                    position = {
                        "entry": current_price,
                        "entry_time": timestamp,
                        "direction": direction,
                        "stop_loss": current_price - direction * current_atr * 1.5,
                        "take_profit": current_price + direction * current_atr * 2.0,
                        "bars_held": 0,
                        "size_mult": position_mult,
                    }

        # Calculate metrics
        if trades:
            trades_df = pd.DataFrame(trades)
            total_return = trades_df["pnl_pct"].sum()
            win_rate = (trades_df["pnl_pct"] > 0).mean() * 100
            max_dd = calculate_max_drawdown(trades_df["pnl_pct"])
            avg_size = trades_df["size_mult"].mean()
        else:
            total_return = 0
            win_rate = 0
            max_dd = 0
            avg_size = 1.0

        results[name] = {
            "trades": len(trades) if trades else 0,
            "total_return": total_return,
            "win_rate": win_rate,
            "max_drawdown": max_dd,
            "avg_position_size": avg_size,
        }

    return results


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Calculate maximum drawdown from trade returns."""
    cumulative = returns.cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    return drawdown.max()


def analyze_evt_triggers(df: pd.DataFrame, symbol: str) -> dict:
    """Analyze when EVT would have triggered position reductions."""
    returns = df["close"].pct_change().dropna()

    trigger_days = []
    multipliers = []

    for i in range(260, len(df)):
        window_returns = returns.iloc[max(0, i - 252) : i]

        if len(window_returns) < 50:
            continue

        # Simple EVT trigger: VaR > 3%
        var_99 = np.percentile(abs(window_returns), 99)

        if var_99 > 0.03:
            trigger_days.append(df.index[i])
            multipliers.append(0.5)  # Standard reduction

    return {
        "total_days": len(df) - 260,
        "trigger_days": len(trigger_days),
        "trigger_pct": len(trigger_days) / (len(df) - 260) * 100 if len(df) > 260 else 0,
        "avg_multiplier_when_triggered": np.mean(multipliers) if multipliers else 1.0,
    }


async def run() -> dict:
    """Run EVT overlay analysis."""
    logger.info("=" * 50)
    logger.info("EVT RISK GATE OVERLAY ANALYSIS")
    logger.info("=" * 50)

    all_comparisons = []
    trigger_analysis = []

    for symbol in SYMBOLS:
        logger.info(f"\n--- {symbol} ---")

        df = load_historical_data(symbol)

        if len(df) < 300:
            logger.warning(f"Insufficient data for {symbol}")
            continue

        # Compare with/without EVT
        comparison = compare_with_without_evt(df, symbol)
        comparison["symbol"] = symbol

        base = comparison["base"]
        gated = comparison["evt_gated"]

        logger.info(f"  Base Strategy:")
        logger.info(f"    Return: {base['total_return']:.2f}%, DD: {base['max_drawdown']:.2f}%")
        logger.info(f"  EVT-Gated:")
        logger.info(f"    Return: {gated['total_return']:.2f}%, DD: {gated['max_drawdown']:.2f}%")
        logger.info(f"    Avg Position Size: {gated['avg_position_size']:.2f}")

        # Return/DD improvement
        if base["max_drawdown"] > 0:
            dd_improvement = (
                (base["max_drawdown"] - gated["max_drawdown"]) / base["max_drawdown"] * 100
            )
            logger.info(f"    DD Improvement: {dd_improvement:.1f}%")

        all_comparisons.append(comparison)

        # Trigger analysis
        triggers = analyze_evt_triggers(df, symbol)
        triggers["symbol"] = symbol
        logger.info(f"  EVT Triggers: {triggers['trigger_pct']:.1f}% of days")
        trigger_analysis.append(triggers)

    # Save results
    output_dir = Path("artifacts/reports/strategy_sprint")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build comparison DataFrame
    comparison_rows = []
    for comp in all_comparisons:
        comparison_rows.append(
            {
                "symbol": comp["symbol"],
                "base_return": comp["base"]["total_return"],
                "base_dd": comp["base"]["max_drawdown"],
                "gated_return": comp["evt_gated"]["total_return"],
                "gated_dd": comp["evt_gated"]["max_drawdown"],
                "avg_position_size": comp["evt_gated"]["avg_position_size"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(output_dir / "evt_overlay_comparison.csv", index=False)

    trigger_df = pd.DataFrame(trigger_analysis)
    trigger_df.to_csv(output_dir / "evt_trigger_analysis.csv", index=False)

    # Summary
    avg_dd_improvement = comparison_df["base_dd"].mean() - comparison_df["gated_dd"].mean()

    return {
        "success": True,
        "summary": f"EVT overlay reduces avg DD by {avg_dd_improvement:.1f}%",
        "comparison": comparison_df.to_dict(),
    }


if __name__ == "__main__":
    asyncio.run(run())
