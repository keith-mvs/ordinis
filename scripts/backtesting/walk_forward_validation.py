#!/usr/bin/env python
"""
Walk-Forward Validation + Shorts + Drawdown Analysis

Tests strategy robustness by:
1. Training on older data, testing on newer data (walk-forward)
2. Adding short selling (RSI overbought)
3. Analyzing drawdowns and risk metrics

Usage:
    python scripts/backtesting/walk_forward_validation.py
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ordinis.engines.signalcore.features.technical import TechnicalIndicators
from ordinis.engines.signalcore.regime_detector import RegimeDetector

# ============================================================================
# BACKTEST ENGINE WITH SHORTS
# ============================================================================


@dataclass
class Trade:
    """Single trade record."""

    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl_pct: float
    exit_reason: str  # "stop", "tp", "signal"


def backtest_long_short(
    df: pd.DataFrame,
    rsi_oversold: int = 35,
    rsi_overbought: int = 65,
    rsi_exit_long: int = 50,
    rsi_exit_short: int = 50,
    atr_stop_mult: float = 1.5,
    atr_tp_mult: float = 2.0,
    enable_longs: bool = True,
    enable_shorts: bool = True,
) -> dict:
    """
    Backtest with both long and short positions.

    Long entry: RSI < rsi_oversold
    Short entry: RSI > rsi_overbought
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Compute indicators
    rsi = TechnicalIndicators.rsi(close, 14)
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(
        axis=1
    )
    atr = tr.rolling(14).mean()

    trades = []
    position = None
    entry_price = None
    stop_loss = None
    take_profit = None
    entry_idx = None

    for i in range(50, len(df)):
        curr_rsi = rsi.iloc[i]
        curr_price = close.iloc[i]
        curr_atr = atr.iloc[i]
        curr_time = df.index[i] if isinstance(df.index, pd.DatetimeIndex) else i

        if position is None:
            # Check for LONG entry
            if enable_longs and curr_rsi < rsi_oversold:
                position = "long"
                entry_price = curr_price
                stop_loss = entry_price - (atr_stop_mult * curr_atr)
                take_profit = entry_price + (atr_tp_mult * curr_atr)
                entry_idx = i
                entry_time = curr_time

            # Check for SHORT entry
            elif enable_shorts and curr_rsi > rsi_overbought:
                position = "short"
                entry_price = curr_price
                stop_loss = entry_price + (atr_stop_mult * curr_atr)
                take_profit = entry_price - (atr_tp_mult * curr_atr)
                entry_idx = i
                entry_time = curr_time

        elif position == "long":
            hit_stop = curr_price <= stop_loss
            hit_tp = curr_price >= take_profit
            exit_signal = curr_rsi > rsi_exit_long

            if hit_stop or hit_tp or exit_signal:
                pnl = (curr_price - entry_price) / entry_price * 100
                trades.append(
                    Trade(
                        symbol="",
                        direction="long",
                        entry_price=entry_price,
                        exit_price=curr_price,
                        entry_time=entry_time,
                        exit_time=curr_time,
                        pnl_pct=pnl,
                        exit_reason="stop" if hit_stop else ("tp" if hit_tp else "signal"),
                    )
                )
                position = None

        elif position == "short":
            hit_stop = curr_price >= stop_loss
            hit_tp = curr_price <= take_profit
            exit_signal = curr_rsi < rsi_exit_short

            if hit_stop or hit_tp or exit_signal:
                pnl = (entry_price - curr_price) / entry_price * 100
                trades.append(
                    Trade(
                        symbol="",
                        direction="short",
                        entry_price=entry_price,
                        exit_price=curr_price,
                        entry_time=entry_time,
                        exit_time=curr_time,
                        pnl_pct=pnl,
                        exit_reason="stop" if hit_stop else ("tp" if hit_tp else "signal"),
                    )
                )
                position = None

    return compute_stats(trades)


def compute_stats(trades: list[Trade]) -> dict:
    """Compute performance statistics from trades."""
    if not trades:
        return {
            "total_return": 0,
            "win_rate": 0,
            "total_trades": 0,
            "profit_factor": 0,
            "avg_trade": 0,
            "max_drawdown": 0,
            "long_trades": 0,
            "short_trades": 0,
            "trades": [],
        }

    pnls = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Drawdown calculation
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # Losing streaks
    streak = 0
    max_streak = 0
    for p in pnls:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    long_trades = [t for t in trades if t.direction == "long"]
    short_trades = [t for t in trades if t.direction == "short"]

    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 999

    return {
        "total_return": sum(pnls),
        "win_rate": len(wins) / len(pnls) * 100,
        "total_trades": len(trades),
        "profit_factor": pf,
        "avg_trade": np.mean(pnls),
        "max_drawdown": max_drawdown,
        "max_losing_streak": max_streak,
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_return": sum(t.pnl_pct for t in long_trades),
        "short_return": sum(t.pnl_pct for t in short_trades),
        "trades": trades,
    }


# ============================================================================
# DATA LOADING
# ============================================================================


def load_intraday_data(symbol: str, start_idx: int = 0, end_idx: int = -1) -> pd.DataFrame | None:
    """Load intraday data for specific date range."""
    base = Path("data/massive")
    files = sorted(base.glob("*.csv.gz"))

    if end_idx == -1:
        end_idx = len(files)

    files = files[start_idx:end_idx]

    if not files:
        return None

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df = df[df["ticker"] == symbol].copy()
        if len(df) > 0:
            dfs.append(df)

    if not dfs:
        return None

    result = pd.concat(dfs, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["window_start"], unit="ns")
    return result


def resample_to_5min(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to 5-minute bars."""
    df = df.set_index("timestamp")
    return (
        df.resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("=" * 80)
    print("🔬 WALK-FORWARD VALIDATION + SHORTS + DRAWDOWN ANALYSIS")
    print("=" * 80)

    symbols = ["COIN", "DKNG", "AMD", "TSLA", "PLTR", "ROKU", "AAL", "SOXS"]
    detector = RegimeDetector()

    # ========== PART 1: WALK-FORWARD VALIDATION ==========
    print("\n" + "=" * 80)
    print("📊 PART 1: WALK-FORWARD VALIDATION")
    print("   Train: Days 1-10 | Test: Days 11-21")
    print("=" * 80)

    walk_forward_results = []

    for symbol in symbols:
        # Load training data (first 10 days)
        df_train = load_intraday_data(symbol, start_idx=0, end_idx=10)
        # Load test data (last 11 days)
        df_test = load_intraday_data(symbol, start_idx=10, end_idx=21)

        if df_train is None or df_test is None:
            continue

        df_train_5m = resample_to_5min(df_train)
        df_test_5m = resample_to_5min(df_test)

        if len(df_train_5m) < 100 or len(df_test_5m) < 100:
            continue

        # Backtest on both periods
        train_result = backtest_long_short(df_train_5m, enable_shorts=False)
        test_result = backtest_long_short(df_test_5m, enable_shorts=False)

        train_ret = train_result["total_return"]
        test_ret = test_result["total_return"]

        status = "🟢" if test_ret > 0 else "🔴"
        degradation = ((train_ret - test_ret) / abs(train_ret) * 100) if train_ret != 0 else 0

        print(
            f"  {symbol}: Train={train_ret:+.1f}% | Test={test_ret:+.1f}% | {status} Degradation={degradation:.0f}%"
        )

        walk_forward_results.append(
            {
                "symbol": symbol,
                "train_return": train_ret,
                "test_return": test_ret,
                "degradation": degradation,
                "is_robust": test_ret > 0 and degradation < 50,
            }
        )

    robust_symbols = [r["symbol"] for r in walk_forward_results if r["is_robust"]]
    print(f"\n  ✅ Robust symbols: {robust_symbols}")

    # ========== PART 2: ADD SHORT SELLING ==========
    print("\n" + "=" * 80)
    print("📊 PART 2: LONG + SHORT COMPARISON")
    print("=" * 80)

    for symbol in robust_symbols[:5]:
        df = load_intraday_data(symbol, start_idx=10, end_idx=21)
        if df is None:
            continue

        df_5m = resample_to_5min(df)

        # Long only
        long_only = backtest_long_short(df_5m, enable_longs=True, enable_shorts=False)

        # Short only
        short_only = backtest_long_short(df_5m, enable_longs=False, enable_shorts=True)

        # Long + Short
        long_short = backtest_long_short(df_5m, enable_longs=True, enable_shorts=True)

        print(f"\n  {symbol}:")
        print(
            f"    Long Only:  {long_only['total_return']:+.1f}% | {long_only['total_trades']} trades"
        )
        print(
            f"    Short Only: {short_only['total_return']:+.1f}% | {short_only['total_trades']} trades"
        )
        print(
            f"    Long+Short: {long_short['total_return']:+.1f}% | {long_short['total_trades']} trades"
        )

    # ========== PART 3: DRAWDOWN ANALYSIS ==========
    print("\n" + "=" * 80)
    print("📊 PART 3: DRAWDOWN & RISK ANALYSIS")
    print("=" * 80)

    all_trades = []

    for symbol in robust_symbols:
        df = load_intraday_data(symbol, start_idx=0, end_idx=21)
        if df is None:
            continue

        df_5m = resample_to_5min(df)
        result = backtest_long_short(df_5m, enable_shorts=False)

        for t in result["trades"]:
            t.symbol = symbol
        all_trades.extend(result["trades"])

        print(f"\n  {symbol}:")
        print(f"    Total Return: {result['total_return']:+.1f}%")
        print(f"    Max Drawdown: {result['max_drawdown']:.1f}%")
        print(f"    Max Losing Streak: {result['max_losing_streak']} trades")
        print(f"    Profit Factor: {result['profit_factor']:.2f}")

    # Portfolio-level analysis
    if all_trades:
        pnls = [t.pnl_pct for t in all_trades]
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = np.max(drawdowns)

        # Find worst day
        trades_by_day = {}
        for t in all_trades:
            day = (
                str(t.exit_time)[:10] if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:10]
            )
            trades_by_day.setdefault(day, []).append(t.pnl_pct)

        daily_returns = {d: sum(pnls) for d, pnls in trades_by_day.items()}
        worst_day = min(daily_returns.items(), key=lambda x: x[1])
        best_day = max(daily_returns.items(), key=lambda x: x[1])

        print("\n  📈 PORTFOLIO SUMMARY:")
        print(f"    Total Trades: {len(all_trades)}")
        print(f"    Total Return: {sum(pnls):+.1f}%")
        print(f"    Max Drawdown: {max_dd:.1f}%")
        print(f"    Best Day: {best_day[0]} ({best_day[1]:+.1f}%)")
        print(f"    Worst Day: {worst_day[0]} ({worst_day[1]:+.1f}%)")
        print(f"    Sharpe Estimate: {np.mean(pnls) / np.std(pnls) * np.sqrt(252):.2f}")

    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("✅ VALIDATION COMPLETE")
    print("=" * 80)

    print(f"\n  Walk-Forward Robust: {len(robust_symbols)} symbols")
    print(f"  Symbols: {', '.join(robust_symbols)}")

    return robust_symbols


if __name__ == "__main__":
    main()
