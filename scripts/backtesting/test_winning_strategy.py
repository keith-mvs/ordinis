#!/usr/bin/env python
"""
Winning Strategy Test - Combines regime detection with optimized ATR-RSI.

This script:
1. Loads intraday data for multiple symbols
2. Runs regime detection to filter tradeable stocks
3. Applies ATR-optimized RSI strategy to tradeable stocks
4. Reports results

Usage:
    python scripts/backtesting/test_winning_strategy.py
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ordinis.engines.signalcore.models.atr_optimized_rsi import OPTIMIZED_CONFIGS, backtest
from ordinis.engines.signalcore.regime_detector import RegimeDetector


def load_intraday_data(symbol: str, days: int = 10) -> pd.DataFrame | None:
    """Load intraday data from massive directory."""
    base = Path("data/massive")
    files = sorted(base.glob("*.csv.gz"))[-days:]

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
    """Resample 1-min data to 5-min bars."""
    df = df.set_index("timestamp")
    df_5m = (
        df.resample("5min")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    return df_5m


def main():
    print("=" * 70)
    print("🎯 WINNING STRATEGY TEST")
    print("   Regime Detection + ATR-Optimized RSI")
    print("=" * 70)

    symbols = ["DKNG", "AMD", "COIN", "NET", "CRWD", "NVDA", "TSLA", "AAPL"]
    days = 10

    detector = RegimeDetector()

    results = []
    tradeable = []
    avoided = []

    print(f"\n📊 Loading {days} days of intraday data...\n")

    for symbol in symbols:
        df = load_intraday_data(symbol, days)
        if df is None:
            print(f"  {symbol}: No data")
            continue

        df_5m = resample_to_5min(df)

        if len(df_5m) < 100:
            print(f"  {symbol}: Insufficient data ({len(df_5m)} bars)")
            continue

        # Regime detection
        analysis = detector.analyze(df_5m, symbol=symbol, timeframe="5min")
        regime = analysis.regime
        recommendation = analysis.trade_recommendation

        if recommendation == "AVOID":
            avoided.append(
                {
                    "symbol": symbol,
                    "regime": regime.value,
                    "reason": analysis.reasoning[:50] + "...",
                }
            )
            print(f"  {symbol}: ❌ SKIP ({regime.value})")
            continue

        tradeable.append(symbol)
        print(f"  {symbol}: ✅ TRADE ({regime.value})")

        # Get optimized config or use default
        if symbol in OPTIMIZED_CONFIGS:
            cfg = OPTIMIZED_CONFIGS[symbol]
        else:
            cfg = OPTIMIZED_CONFIGS["DEFAULT"]

        # Run backtest
        result = backtest(
            df_5m,
            rsi_os=cfg.rsi_oversold,
            rsi_exit=cfg.rsi_exit,
            atr_stop_mult=cfg.atr_stop_mult,
            atr_tp_mult=cfg.atr_tp_mult,
        )

        results.append(
            {
                "symbol": symbol,
                "bars": len(df_5m),
                "regime": regime.value,
                **result,
            }
        )

    # Print results
    print("\n" + "=" * 70)
    print("📈 BACKTEST RESULTS (Tradeable Stocks Only)")
    print("=" * 70)

    total_return = 0
    total_trades = 0
    total_wins = 0

    for r in results:
        symbol = r["symbol"]
        ret = r["total_return"]
        wr = r["win_rate"]
        n = r["total_trades"]
        pf = r["profit_factor"]
        regime = r["regime"]

        total_return += ret
        total_trades += n
        total_wins += int(n * wr / 100)

        status = "🟢" if ret > 0 else "🔴"
        print(f"\n{status} {symbol} ({regime}):")
        print(f"   Return: {ret:+.1f}%")
        print(f"   Win Rate: {wr:.0f}%")
        print(f"   Trades: {n}")
        print(f"   Profit Factor: {pf:.2f}")

    # Portfolio summary
    print("\n" + "=" * 70)
    print("💼 PORTFOLIO SUMMARY")
    print("=" * 70)

    print(f"\n  Symbols Analyzed: {len(symbols)}")
    print(f"  Tradeable: {len(tradeable)} ({', '.join(tradeable)})")
    print(f"  Avoided: {len(avoided)}")

    for a in avoided:
        print(f"    - {a['symbol']}: {a['regime']}")

    print(f"\n  Total Return: {total_return:+.1f}%")
    print(f"  Total Trades: {total_trades}")

    if total_trades > 0:
        overall_wr = total_wins / total_trades * 100
        print(f"  Overall Win Rate: {overall_wr:.0f}%")
        print(f"  Avg Return/Trade: {total_return/total_trades:+.2f}%")

    # Verdict
    print("\n" + "=" * 70)
    if total_return > 0:
        print("✅ STRATEGY IS PROFITABLE!")
    else:
        print("❌ STRATEGY NEEDS IMPROVEMENT")
    print("=" * 70)


if __name__ == "__main__":
    main()
