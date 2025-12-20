"""
Enhanced Strategy Optimizer with Multi-Signal Confluence.

Tests all strategies including the new MultiSignalConfluenceModel with:
- RSI + Stochastic confluence
- ADX trend filtering
- ATR-based adaptive stops

This script focuses on finding WINNING configurations.
"""

import argparse
import asyncio
from datetime import datetime
import glob
import gzip
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models.momentum_breakout import MomentumBreakoutModel
from ordinis.engines.signalcore.models.multi_signal_confluence import MultiSignalConfluenceModel
from ordinis.engines.signalcore.models.rsi_volume_reversion import RSIVolumeReversionModel
from ordinis.engines.signalcore.models.trend_following import TrendFollowingModel
from ordinis.engines.signalcore.regime_detector import RegimeDetector, regime_filter


def load_intraday_data(symbol: str, date_str: str = "2024-11-13") -> pd.DataFrame:
    """Load intraday data from massive directory."""
    file_path = Path(f"data/massive/{date_str}.csv.gz")

    if not file_path.exists():
        # Try to find any file
        files = sorted(glob.glob("data/massive/*.csv.gz"))
        if files:
            file_path = Path(files[-1])
            print(f"Using file: {file_path}")
        else:
            raise FileNotFoundError("No data files found in data/massive/")

    with gzip.open(file_path, "rt") as f:
        df = pd.read_csv(f)

    # Filter for symbol
    df = df[df["symbol"] == symbol].copy()

    if len(df) == 0:
        raise ValueError(f"No data found for {symbol}")

    # Convert timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

    # Ensure required columns
    df.columns = df.columns.str.lower()

    return df


def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 1-min data to higher timeframes."""
    if timeframe == "1min":
        return df

    rule_map = {
        "5min": "5min",
        "15min": "15min",
        "30min": "30min",
        "1hour": "1h",
        "1H": "1h",
    }
    rule = rule_map.get(timeframe, timeframe)

    resampled = (
        df.resample(rule)
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

    return resampled


async def run_backtest(
    model,
    df: pd.DataFrame,
    symbol: str,
    initial_capital: float = 10000.0,
    position_size_pct: float = 0.95,
    stop_loss_pct: float = None,  # None = use ATR from signal
    take_profit_pct: float = None,  # None = use ATR from signal
    fixed_stop_loss_pct: float = 0.02,  # Fallback fixed stop
    fixed_take_profit_pct: float = 0.04,  # Fallback fixed TP
) -> dict:
    """Run backtest with ATR-aware stop management."""

    capital = initial_capital
    position = None  # (direction, entry_price, shares, stop_loss, take_profit)
    trades = []

    min_points = model.config.min_data_points

    for i in range(min_points, len(df)):
        window = df.iloc[: i + 1]
        current_bar = df.iloc[i]
        current_price = current_bar["close"]
        timestamp = df.index[i] if isinstance(df.index[i], datetime) else datetime.now()

        # Check stop loss / take profit if in position
        if position is not None:
            direction, entry_price, shares, stop_loss, take_profit = position

            if direction == "LONG":
                # Check stop loss (using low of bar)
                if current_bar["low"] <= stop_loss:
                    exit_price = stop_loss
                    pnl = (exit_price - entry_price) * shares
                    capital += pnl
                    trades.append(
                        {
                            "direction": direction,
                            "entry": entry_price,
                            "exit": exit_price,
                            "pnl": pnl,
                            "pnl_pct": (exit_price / entry_price - 1) * 100,
                            "reason": "stop_loss",
                        }
                    )
                    position = None
                    continue

                # Check take profit (using high of bar)
                if current_bar["high"] >= take_profit:
                    exit_price = take_profit
                    pnl = (exit_price - entry_price) * shares
                    capital += pnl
                    trades.append(
                        {
                            "direction": direction,
                            "entry": entry_price,
                            "exit": exit_price,
                            "pnl": pnl,
                            "pnl_pct": (exit_price / entry_price - 1) * 100,
                            "reason": "take_profit",
                        }
                    )
                    position = None
                    continue

            elif direction == "SHORT":
                # Check stop loss (using high of bar)
                if current_bar["high"] >= stop_loss:
                    exit_price = stop_loss
                    pnl = (entry_price - exit_price) * shares
                    capital += pnl
                    trades.append(
                        {
                            "direction": direction,
                            "entry": entry_price,
                            "exit": exit_price,
                            "pnl": pnl,
                            "pnl_pct": (entry_price / exit_price - 1) * 100,
                            "reason": "stop_loss",
                        }
                    )
                    position = None
                    continue

                # Check take profit (using low of bar)
                if current_bar["low"] <= take_profit:
                    exit_price = take_profit
                    pnl = (entry_price - exit_price) * shares
                    capital += pnl
                    trades.append(
                        {
                            "direction": direction,
                            "entry": entry_price,
                            "exit": exit_price,
                            "pnl": pnl,
                            "pnl_pct": (entry_price / exit_price - 1) * 100,
                            "reason": "take_profit",
                        }
                    )
                    position = None
                    continue

        # Generate signal
        try:
            signal = await model.generate(symbol, window, timestamp)
        except Exception as e:
            continue

        if signal is None:
            continue

        # Handle exit signals
        if signal.signal_type.value == "EXIT" and position is not None:
            direction, entry_price, shares, _, _ = position
            exit_price = current_price

            if direction == "LONG":
                pnl = (exit_price - entry_price) * shares
            else:
                pnl = (entry_price - exit_price) * shares

            capital += pnl
            trades.append(
                {
                    "direction": direction,
                    "entry": entry_price,
                    "exit": exit_price,
                    "pnl": pnl,
                    "pnl_pct": ((exit_price / entry_price - 1) * 100)
                    if direction == "LONG"
                    else ((entry_price / exit_price - 1) * 100),
                    "reason": "signal_exit",
                }
            )
            position = None
            continue

        # Handle entry signals
        if signal.signal_type.value == "ENTRY" and position is None:
            direction = signal.direction.value.upper()
            entry_price = current_price

            # Position size
            position_capital = capital * position_size_pct
            shares = position_capital / entry_price

            # Get stop/TP from signal metadata (ATR-based) or use fixed
            metadata = signal.metadata or {}

            if "stop_loss" in metadata:
                stop_loss = metadata["stop_loss"]
                take_profit = metadata.get(
                    "take_profit",
                    entry_price * (1 + fixed_take_profit_pct)
                    if direction == "LONG"
                    else entry_price * (1 - fixed_take_profit_pct),
                )
            elif direction == "LONG":
                stop_loss = entry_price * (1 - fixed_stop_loss_pct)
                take_profit = entry_price * (1 + fixed_take_profit_pct)
            else:
                stop_loss = entry_price * (1 + fixed_stop_loss_pct)
                take_profit = entry_price * (1 - fixed_take_profit_pct)

            position = (direction, entry_price, shares, stop_loss, take_profit)

    # Close any remaining position at last price
    if position is not None:
        direction, entry_price, shares, _, _ = position
        exit_price = df["close"].iloc[-1]

        if direction == "LONG":
            pnl = (exit_price - entry_price) * shares
        else:
            pnl = (entry_price - exit_price) * shares

        capital += pnl
        trades.append(
            {
                "direction": direction,
                "entry": entry_price,
                "exit": exit_price,
                "pnl": pnl,
                "pnl_pct": ((exit_price / entry_price - 1) * 100)
                if direction == "LONG"
                else ((entry_price / exit_price - 1) * 100),
                "reason": "end_of_data",
            }
        )

    # Calculate stats
    total_return = (capital / initial_capital - 1) * 100
    winning_trades = [t for t in trades if t["pnl"] > 0]
    losing_trades = [t for t in trades if t["pnl"] <= 0]

    win_rate = len(winning_trades) / len(trades) * 100 if trades else 0

    return {
        "total_return": total_return,
        "final_capital": capital,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "avg_win": np.mean([t["pnl_pct"] for t in winning_trades]) if winning_trades else 0,
        "avg_loss": np.mean([t["pnl_pct"] for t in losing_trades]) if losing_trades else 0,
        "trades": trades,
    }


def get_strategy_configs() -> list[tuple[str, type, dict]]:
    """Get all strategy configurations to test."""
    configs = []

    # =====================================================
    # 1. MULTI-SIGNAL CONFLUENCE (NEW - Our best hope!)
    # =====================================================

    # Standard confluence (2+ signals required)
    configs.append(
        (
            "confluence_std",
            MultiSignalConfluenceModel,
            {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "stoch_period": 14,
                "stoch_oversold": 20,
                "stoch_overbought": 80,
                "adx_max_for_reversion": 30,
                "atr_stop_mult": 2.0,
                "atr_tp_mult": 3.0,
                "require_all_signals": False,
                "enable_shorts": True,
                "enable_longs": True,
            },
        )
    )

    # Relaxed confluence (easier entry)
    configs.append(
        (
            "confluence_relaxed",
            MultiSignalConfluenceModel,
            {
                "rsi_period": 14,
                "rsi_oversold": 35,
                "rsi_overbought": 65,
                "stoch_period": 14,
                "stoch_oversold": 25,
                "stoch_overbought": 75,
                "adx_max_for_reversion": 35,
                "atr_stop_mult": 2.5,
                "atr_tp_mult": 4.0,
                "require_all_signals": False,
                "enable_shorts": True,
                "enable_longs": True,
            },
        )
    )

    # Strict confluence (all signals required)
    configs.append(
        (
            "confluence_strict",
            MultiSignalConfluenceModel,
            {
                "rsi_period": 14,
                "rsi_oversold": 25,
                "rsi_overbought": 75,
                "stoch_period": 14,
                "stoch_oversold": 15,
                "stoch_overbought": 85,
                "adx_max_for_reversion": 25,
                "atr_stop_mult": 1.5,
                "atr_tp_mult": 2.5,
                "require_all_signals": True,
                "enable_shorts": True,
                "enable_longs": True,
            },
        )
    )

    # Long-only confluence (for uptrending stocks)
    configs.append(
        (
            "confluence_long_only",
            MultiSignalConfluenceModel,
            {
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70,
                "stoch_period": 14,
                "stoch_oversold": 20,
                "stoch_overbought": 80,
                "adx_max_for_reversion": 30,
                "atr_stop_mult": 2.0,
                "atr_tp_mult": 3.0,
                "require_all_signals": False,
                "enable_shorts": False,
                "enable_longs": True,
            },
        )
    )

    # =====================================================
    # 2. RSI REVERSION (Relaxed for more trades)
    # =====================================================

    configs.append(
        (
            "rsi_relaxed",
            RSIVolumeReversionModel,
            {
                "rsi_period": 14,
                "oversold_threshold": 35,
                "overbought_threshold": 65,
                "exit_rsi": 50,
                "volume_mult": 1.0,  # No volume filter
                "trend_filter_period": 0,  # No trend filter
                "enable_shorts": True,
                "enable_longs": True,
            },
        )
    )

    configs.append(
        (
            "rsi_standard",
            RSIVolumeReversionModel,
            {
                "rsi_period": 14,
                "oversold_threshold": 30,
                "overbought_threshold": 70,
                "exit_rsi": 50,
                "volume_mult": 1.2,
                "trend_filter_period": 0,
                "enable_shorts": True,
                "enable_longs": True,
            },
        )
    )

    # =====================================================
    # 3. MOMENTUM BREAKOUT
    # =====================================================

    configs.append(
        (
            "momentum_std",
            MomentumBreakoutModel,
            {
                "breakout_period": 20,
                "volume_mult": 1.5,
                "atr_filter_mult": 1.0,
                "enable_shorts": True,
                "enable_longs": True,
            },
        )
    )

    configs.append(
        (
            "momentum_fast",
            MomentumBreakoutModel,
            {
                "breakout_period": 10,
                "volume_mult": 1.2,
                "atr_filter_mult": 0.8,
                "enable_shorts": True,
                "enable_longs": True,
            },
        )
    )

    # =====================================================
    # 4. TREND FOLLOWING
    # =====================================================

    configs.append(
        (
            "trend_std",
            TrendFollowingModel,
            {
                "fast_period": 10,
                "slow_period": 30,
                "signal_period": 5,
                "enable_shorts": True,
                "enable_longs": True,
            },
        )
    )

    configs.append(
        (
            "trend_fast",
            TrendFollowingModel,
            {
                "fast_period": 5,
                "slow_period": 15,
                "signal_period": 3,
                "enable_shorts": True,
                "enable_longs": True,
            },
        )
    )

    return configs


async def run_optimizer(
    symbols: list[str],
    timeframes: list[str],
    use_regime_filter: bool = True,
) -> pd.DataFrame:
    """Run optimization across all strategies, symbols, and timeframes."""

    configs = get_strategy_configs()
    results = []

    # Initialize regime detector
    detector = RegimeDetector()

    total_tests = len(configs) * len(symbols) * len(timeframes)
    test_num = 0

    print(f"\n{'='*80}")
    print("🚀 ENHANCED STRATEGY OPTIMIZER")
    print(f"   Strategies: {len(configs)}")
    print(f"   Symbols: {symbols}")
    print(f"   Timeframes: {timeframes}")
    print(f"   Regime Filter: {'ON' if use_regime_filter else 'OFF'}")
    print(f"   Total Tests: {total_tests}")
    print(f"{'='*80}\n")

    for symbol in symbols:
        print(f"\n📈 Loading {symbol}...")

        try:
            raw_df = load_intraday_data(symbol)
            print(f"   Loaded {len(raw_df)} rows")
        except Exception as e:
            print(f"   ❌ Failed to load {symbol}: {e}")
            continue

        for timeframe in timeframes:
            df = resample_data(raw_df, timeframe)
            print(f"\n   ⏱️ {timeframe}: {len(df)} bars")

            # Analyze regime
            if use_regime_filter:
                try:
                    analysis = detector.analyze(df, symbol, timeframe)
                    print(f"   📊 Regime: {analysis.regime.value} (ADX={analysis.metrics.adx:.1f})")
                except Exception as e:
                    print(f"   ⚠️ Could not analyze regime: {e}")
                    analysis = None

            for strategy_name, model_class, params in configs:
                test_num += 1

                # Check regime compatibility
                if use_regime_filter and analysis:
                    strategy_type = (
                        "rsi"
                        if "rsi" in strategy_name or "confluence" in strategy_name
                        else "momentum"
                        if "momentum" in strategy_name
                        else "trend"
                        if "trend" in strategy_name
                        else "unknown"
                    )

                    should_trade, reason = regime_filter(df, strategy_type, symbol)

                    if not should_trade:
                        print(
                            f"      [{test_num}/{total_tests}] {strategy_name}: SKIPPED - {reason}"
                        )
                        results.append(
                            {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "strategy": strategy_name,
                                "total_return": 0,
                                "num_trades": 0,
                                "win_rate": 0,
                                "skipped": True,
                                "skip_reason": reason,
                            }
                        )
                        continue

                # Create model
                config = ModelConfig(
                    name=strategy_name,
                    parameters=params,
                    min_data_points=max(50, params.get("breakout_period", 20) + 10),
                )

                try:
                    model = model_class(config)
                    result = await run_backtest(model, df, symbol)

                    # Color code results
                    if result["total_return"] > 5:
                        color = "🟢"
                    elif result["total_return"] > 0:
                        color = "🟡"
                    else:
                        color = "🔴"

                    print(
                        f"      [{test_num}/{total_tests}] {strategy_name}: {color} {result['total_return']:+.1f}% | {result['num_trades']} trades | {result['win_rate']:.0f}% win"
                    )

                    results.append(
                        {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "strategy": strategy_name,
                            "total_return": result["total_return"],
                            "num_trades": result["num_trades"],
                            "win_rate": result["win_rate"],
                            "winning_trades": result["winning_trades"],
                            "losing_trades": result["losing_trades"],
                            "avg_win": result["avg_win"],
                            "avg_loss": result["avg_loss"],
                            "skipped": False,
                        }
                    )

                except Exception as e:
                    print(f"      [{test_num}/{total_tests}] {strategy_name}: ❌ ERROR - {e}")
                    results.append(
                        {
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "strategy": strategy_name,
                            "total_return": 0,
                            "num_trades": 0,
                            "win_rate": 0,
                            "error": str(e),
                            "skipped": False,
                        }
                    )

    return pd.DataFrame(results)


def print_summary(results_df: pd.DataFrame):
    """Print summary of optimization results."""

    print(f"\n{'='*80}")
    print("📊 OPTIMIZATION SUMMARY")
    print(f"{'='*80}")

    # Filter out skipped/error results
    active = results_df[
        ~results_df.get("skipped", False) & ~results_df.get("error", "").notna()
    ].copy()

    if len(active) == 0:
        print("No active results to analyze.")
        return

    # Top performers
    print("\n🏆 TOP 10 PERFORMERS:")
    print("-" * 70)
    top = active.nlargest(10, "total_return")
    for _, row in top.iterrows():
        print(
            f"   {row['symbol']:5} | {row['timeframe']:5} | {row['strategy']:20} | {row['total_return']:+8.1f}% | {row['win_rate']:.0f}% win | {row['num_trades']} trades"
        )

    # Best by strategy
    print("\n📈 BEST RESULT BY STRATEGY:")
    print("-" * 70)
    for strategy in active["strategy"].unique():
        strat_df = active[active["strategy"] == strategy]
        best = strat_df.loc[strat_df["total_return"].idxmax()]
        if best["total_return"] > 0:
            color = "🟢"
        else:
            color = "🔴"
        print(
            f"   {color} {strategy:25} | {best['symbol']:5} {best['timeframe']:5} | {best['total_return']:+8.1f}%"
        )

    # Best by symbol
    print("\n📊 BEST RESULT BY SYMBOL:")
    print("-" * 70)
    for symbol in active["symbol"].unique():
        sym_df = active[active["symbol"] == symbol]
        best = sym_df.loc[sym_df["total_return"].idxmax()]
        profitable = (sym_df["total_return"] > 0).sum()
        total = len(sym_df)
        print(
            f"   {symbol:5} | Best: {best['strategy']:20} | {best['total_return']:+8.1f}% | {profitable}/{total} profitable configs"
        )

    # Overall stats
    profitable_configs = (active["total_return"] > 0).sum()
    total_configs = len(active)
    avg_return = active["total_return"].mean()

    print("\n📉 OVERALL STATS:")
    print(
        f"   Profitable Configs: {profitable_configs}/{total_configs} ({profitable_configs/total_configs*100:.1f}%)"
    )
    print(f"   Average Return: {avg_return:+.2f}%")
    print(f"   Best Return: {active['total_return'].max():+.2f}%")
    print(f"   Worst Return: {active['total_return'].min():+.2f}%")


async def main():
    parser = argparse.ArgumentParser(description="Enhanced Strategy Optimizer")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["DKNG", "NET", "COIN", "AMD", "CRWD"],
        help="Symbols to test",
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=["5min", "15min"], help="Timeframes to test"
    )
    parser.add_argument("--no-regime-filter", action="store_true", help="Disable regime filtering")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/enhanced_optimizer_results.csv",
        help="Output file for results",
    )

    args = parser.parse_args()

    # Run optimization
    results_df = await run_optimizer(
        symbols=args.symbols,
        timeframes=args.timeframes,
        use_regime_filter=not args.no_regime_filter,
    )

    # Print summary
    print_summary(results_df)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")

    # Return best result for quick check
    best = results_df.loc[results_df["total_return"].idxmax()]
    return best["total_return"]


if __name__ == "__main__":
    asyncio.run(main())
