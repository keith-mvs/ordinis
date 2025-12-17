#!/usr/bin/env python3
"""
Test Enhanced Strategies - Multi-Signal Confluence with ATR Stops.

Tests the new enhanced strategies that incorporate:
1. RSI + Stochastic confluence (reduces false signals)
2. ADX trend strength filter (regime awareness)
3. ATR-based adaptive stops (volatility-adjusted)

Goal: Find WINNING configurations after CRWD disaster (0% profitable).
"""

import asyncio
from datetime import datetime
import glob
import gzip
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models.multi_signal_confluence import MultiSignalConfluenceModel
from ordinis.engines.signalcore.regime_detector import RegimeDetector, regime_filter


def load_intraday_data(symbol: str, limit_files: int = 10) -> pd.DataFrame | None:
    """Load intraday data for symbol from massive directory."""
    massive_dir = project_root / "data" / "massive"
    pattern = str(massive_dir / "*.csv.gz")
    files = sorted(glob.glob(pattern))[-limit_files:]

    if not files:
        print(f"No data files found in {massive_dir}")
        return None

    all_data = []
    for f in files:
        try:
            with gzip.open(f, "rt") as gz:
                df = pd.read_csv(gz)
                if symbol in df["symbol"].unique():
                    sym_data = df[df["symbol"] == symbol].copy()
                    all_data.append(sym_data)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        return None

    combined = pd.concat(all_data, ignore_index=True)

    # Standardize columns
    combined.columns = [c.lower() for c in combined.columns]
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        combined = combined.sort_values("timestamp")

    return combined


def run_backtest_with_stops(
    model: MultiSignalConfluenceModel,
    data: pd.DataFrame,
    initial_capital: float = 10000.0,
) -> dict:
    """
    Run backtest with proper ATR-based stop handling.

    Returns dict with metrics including win rate, profit, etc.
    """
    capital = initial_capital
    position = 0  # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0

    trades = []
    equity_curve = [capital]

    model.reset_state()

    # Need enough data for indicators
    min_bars = model.config.min_data_points + 10

    for i in range(min_bars, len(data)):
        window = data.iloc[: i + 1].copy()
        current_bar = data.iloc[i]
        current_price = current_bar["close"]
        high = current_bar["high"]
        low = current_bar["low"]
        timestamp = datetime.now()

        # Check stops first if in position
        if position == 1:  # Long
            if low <= stop_loss:
                # Stopped out
                pnl = (stop_loss - entry_price) / entry_price
                capital *= 1 + pnl
                trades.append(
                    {
                        "entry": entry_price,
                        "exit": stop_loss,
                        "pnl_pct": pnl * 100,
                        "type": "long",
                        "exit_reason": "stop_loss",
                    }
                )
                position = 0
                model.reset_state()
            elif high >= take_profit:
                # Take profit hit
                pnl = (take_profit - entry_price) / entry_price
                capital *= 1 + pnl
                trades.append(
                    {
                        "entry": entry_price,
                        "exit": take_profit,
                        "pnl_pct": pnl * 100,
                        "type": "long",
                        "exit_reason": "take_profit",
                    }
                )
                position = 0
                model.reset_state()

        elif position == -1:  # Short
            if high >= stop_loss:
                # Stopped out
                pnl = (entry_price - stop_loss) / entry_price
                capital *= 1 + pnl
                trades.append(
                    {
                        "entry": entry_price,
                        "exit": stop_loss,
                        "pnl_pct": pnl * 100,
                        "type": "short",
                        "exit_reason": "stop_loss",
                    }
                )
                position = 0
                model.reset_state()
            elif low <= take_profit:
                # Take profit hit
                pnl = (entry_price - take_profit) / entry_price
                capital *= 1 + pnl
                trades.append(
                    {
                        "entry": entry_price,
                        "exit": take_profit,
                        "pnl_pct": pnl * 100,
                        "type": "short",
                        "exit_reason": "take_profit",
                    }
                )
                position = 0
                model.reset_state()

        # Generate signal if flat
        if position == 0:
            try:
                signal = asyncio.get_event_loop().run_until_complete(
                    model.generate("TEST", window, timestamp)
                )

                if signal and signal.signal_type.value == "entry":
                    if signal.direction.value == "long":
                        position = 1
                        entry_price = current_price
                        stop_loss = signal.metadata.get("stop_loss", entry_price * 0.98)
                        take_profit = signal.metadata.get("take_profit", entry_price * 1.04)
                    elif signal.direction.value == "short":
                        position = -1
                        entry_price = current_price
                        stop_loss = signal.metadata.get("stop_loss", entry_price * 1.02)
                        take_profit = signal.metadata.get("take_profit", entry_price * 0.96)
            except Exception:
                pass

        equity_curve.append(capital)

    # Close any open position at end
    if position != 0:
        final_price = data.iloc[-1]["close"]
        if position == 1:
            pnl = (final_price - entry_price) / entry_price
        else:
            pnl = (entry_price - final_price) / entry_price
        capital *= 1 + pnl
        trades.append(
            {
                "entry": entry_price,
                "exit": final_price,
                "pnl_pct": pnl * 100,
                "type": "long" if position == 1 else "short",
                "exit_reason": "end_of_data",
            }
        )

    # Calculate metrics
    total_return = (capital - initial_capital) / initial_capital * 100

    if trades:
        winning_trades = [t for t in trades if t["pnl_pct"] > 0]
        win_rate = len(winning_trades) / len(trades) * 100
        avg_win = np.mean([t["pnl_pct"] for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in trades if t["pnl_pct"] <= 0]
        avg_loss = np.mean([t["pnl_pct"] for t in losing_trades]) if losing_trades else 0

        # Stop/TP analysis
        stop_exits = len([t for t in trades if t["exit_reason"] == "stop_loss"])
        tp_exits = len([t for t in trades if t["exit_reason"] == "take_profit"])
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        stop_exits = 0
        tp_exits = 0

    return {
        "total_return": total_return,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "stop_exits": stop_exits,
        "tp_exits": tp_exits,
        "final_capital": capital,
        "trades": trades,
    }


def main():
    """Run enhanced strategy tests."""
    print("=" * 70)
    print("ENHANCED STRATEGY BACKTEST - Multi-Signal Confluence + ATR Stops")
    print("=" * 70)

    # Test stocks - include both good (DKNG) and bad (CRWD) performers
    symbols = ["DKNG", "NET", "AMD", "CRWD", "COIN"]

    # Parameter configurations to test
    configs = [
        # Conservative (tight stops)
        {
            "name": "Conservative",
            "rsi_oversold": 25,
            "rsi_overbought": 75,
            "stoch_oversold": 15,
            "stoch_overbought": 85,
            "adx_max_for_reversion": 25,
            "atr_stop_mult": 1.5,
            "atr_tp_mult": 2.5,
            "require_all_signals": True,
        },
        # Balanced
        {
            "name": "Balanced",
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "stoch_oversold": 20,
            "stoch_overbought": 80,
            "adx_max_for_reversion": 30,
            "atr_stop_mult": 2.0,
            "atr_tp_mult": 3.0,
            "require_all_signals": False,
        },
        # Aggressive (wider thresholds)
        {
            "name": "Aggressive",
            "rsi_oversold": 35,
            "rsi_overbought": 65,
            "stoch_oversold": 25,
            "stoch_overbought": 75,
            "adx_max_for_reversion": 35,
            "atr_stop_mult": 2.5,
            "atr_tp_mult": 4.0,
            "require_all_signals": False,
        },
        # Wide stops (for volatile stocks)
        {
            "name": "WideStops",
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "stoch_oversold": 20,
            "stoch_overbought": 80,
            "adx_max_for_reversion": 40,
            "atr_stop_mult": 3.0,
            "atr_tp_mult": 4.5,
            "require_all_signals": False,
        },
    ]

    # Initialize regime detector
    detector = RegimeDetector()

    all_results = []

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"TESTING: {symbol}")
        print("=" * 60)

        # Load data
        data = load_intraday_data(symbol, limit_files=15)
        if data is None or len(data) < 500:
            print(f"  Insufficient data for {symbol}")
            continue

        print(f"  Loaded {len(data)} bars")

        # Check regime
        should_trade, reason = regime_filter(data, "mean_reversion", symbol)
        print(f"  Regime filter: {'TRADE' if should_trade else 'SKIP'} - {reason}")

        # Get regime metrics
        try:
            metrics = detector.compute_metrics(data, symbol)
            regime = detector.classify_regime(metrics)
            print(f"  Regime: {regime.value.upper()}")
            print(f"  ADX: {metrics.adx:.1f} | Dir Change: {metrics.direction_change_rate:.1%}")
        except Exception as e:
            print(f"  Regime detection failed: {e}")
            regime = None

        # Test each configuration
        for cfg in configs:
            model_config = ModelConfig(
                name=f"MultiSignal_{cfg['name']}",
                version="1.0",
                parameters={k: v for k, v in cfg.items() if k != "name"},
            )

            model = MultiSignalConfluenceModel(model_config)

            try:
                result = run_backtest_with_stops(model, data)

                result["symbol"] = symbol
                result["config_name"] = cfg["name"]
                result["regime"] = regime.value if regime else "unknown"
                result["should_trade"] = should_trade

                all_results.append(result)

                # Print result
                status = "✓" if result["total_return"] > 0 else "✗"
                print(
                    f"  {status} {cfg['name']:12s}: {result['total_return']:+6.1f}% | "
                    f"Trades: {result['num_trades']:3d} | "
                    f"WinRate: {result['win_rate']:5.1f}% | "
                    f"SL: {result['stop_exits']} TP: {result['tp_exits']}"
                )

            except Exception as e:
                print(f"  Error with {cfg['name']}: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY - WINNING CONFIGURATIONS")
    print("=" * 70)

    winners = [r for r in all_results if r["total_return"] > 0]
    winners.sort(key=lambda x: x["total_return"], reverse=True)

    print(f"\nProfitable: {len(winners)}/{len(all_results)} configurations")

    if winners:
        print("\nTOP 10 WINNERS:")
        for i, w in enumerate(winners[:10], 1):
            print(
                f"  {i:2d}. {w['symbol']:5s} | {w['config_name']:12s} | "
                f"{w['total_return']:+6.1f}% | WR: {w['win_rate']:.0f}% | "
                f"Trades: {w['num_trades']}"
            )

    # Regime analysis
    print("\nRESULTS BY REGIME:")
    regime_results = {}
    for r in all_results:
        regime = r.get("regime", "unknown")
        if regime not in regime_results:
            regime_results[regime] = []
        regime_results[regime].append(r)

    for regime, results in sorted(regime_results.items()):
        profitable = len([r for r in results if r["total_return"] > 0])
        avg_return = np.mean([r["total_return"] for r in results])
        print(f"  {regime:15s}: {profitable}/{len(results)} profitable | Avg: {avg_return:+.1f}%")

    # Should-trade analysis
    print("\nREGIME FILTER EFFECTIVENESS:")
    trade_yes = [r for r in all_results if r["should_trade"]]
    trade_no = [r for r in all_results if not r["should_trade"]]

    if trade_yes:
        yes_profit = len([r for r in trade_yes if r["total_return"] > 0])
        yes_avg = np.mean([r["total_return"] for r in trade_yes])
        print(
            f"  Should Trade (YES): {yes_profit}/{len(trade_yes)} profitable | Avg: {yes_avg:+.1f}%"
        )

    if trade_no:
        no_profit = len([r for r in trade_no if r["total_return"] > 0])
        no_avg = np.mean([r["total_return"] for r in trade_no])
        print(f"  Should Trade (NO):  {no_profit}/{len(trade_no)} profitable | Avg: {no_avg:+.1f}%")


if __name__ == "__main__":
    main()
