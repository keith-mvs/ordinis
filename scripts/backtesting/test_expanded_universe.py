#!/usr/bin/env python
"""
Comprehensive Strategy Test - Expanded Universe

Tests:
1. Expanded stock universe (20+ symbols)
2. Intraday (5min) and Daily timeframes
3. Multi-signal confluence as secondary filter
4. Generates live trading configuration

Usage:
    python scripts/backtesting/test_expanded_universe.py
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from ordinis.engines.signalcore.features.technical import TechnicalIndicators
from ordinis.engines.signalcore.models.atr_optimized_rsi import OPTIMIZED_CONFIGS, backtest
from ordinis.engines.signalcore.regime_detector import RegimeDetector

# Expanded universe - high volume, liquid stocks
EXPANDED_UNIVERSE = [
    # Tech megacaps
    "NVDA",
    "TSLA",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    # High beta / momentum
    "AMD",
    "COIN",
    "DKNG",
    "NET",
    "CRWD",
    "PLTR",
    "ROKU",
    # ETFs
    "SPY",
    "QQQ",
    "TQQQ",
    "SOXL",
    "SOXS",
    # Financials/Other
    "JPM",
    "BAC",
    "F",
    "AAL",
    "PFE",
]


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


def load_daily_data(symbol: str) -> pd.DataFrame | None:
    """Load daily data from historical directory."""
    # Try multiple paths
    paths = [
        Path(f"data/historical/{symbol}.csv"),
        Path(f"data/historical/{symbol}_daily.csv"),
        Path(f"data/raw/{symbol}.csv"),
    ]

    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            # Standardize column names
            df.columns = df.columns.str.lower()
            if "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"])
            elif "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

    return None


def resample_to_timeframe(df: pd.DataFrame, timeframe: str = "5min") -> pd.DataFrame:
    """Resample data to specified timeframe."""
    df = df.set_index("timestamp")
    df_resampled = (
        df.resample(timeframe)
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
    return df_resampled


def compute_stochastic(high, low, close, period=14, smooth=3):
    """Compute Stochastic %K."""
    lowest = low.rolling(period).min()
    highest = high.rolling(period).max()
    raw_k = 100 * (close - lowest) / (highest - lowest + 1e-10)
    return raw_k.rolling(smooth).mean()


def check_confluence(df: pd.DataFrame, rsi_threshold: int = 35, stoch_threshold: int = 25) -> dict:
    """
    Check if RSI + Stochastic confluence would have improved results.

    Returns stats on how often both agree vs RSI-only.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    rsi = TechnicalIndicators.rsi(close, 14)
    stoch = compute_stochastic(high, low, close)

    rsi_signals = (rsi < rsi_threshold).sum()
    confluence_signals = ((rsi < rsi_threshold) & (stoch < stoch_threshold)).sum()

    filter_rate = 1 - (confluence_signals / rsi_signals) if rsi_signals > 0 else 0

    return {
        "rsi_signals": rsi_signals,
        "confluence_signals": confluence_signals,
        "filter_rate": filter_rate * 100,  # % of signals filtered out
    }


def backtest_with_confluence(
    df: pd.DataFrame,
    rsi_os: int = 35,
    stoch_os: int = 25,
    atr_stop_mult: float = 1.5,
    atr_tp_mult: float = 2.0,
) -> dict:
    """Backtest with RSI + Stochastic confluence filter."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    rsi = TechnicalIndicators.rsi(close, 14)
    stoch = compute_stochastic(high, low, close)

    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(
        axis=1
    )
    atr = tr.rolling(14).mean()

    trades = []
    position = None

    for i in range(50, len(df)):
        curr_rsi = rsi.iloc[i]
        curr_stoch = stoch.iloc[i]
        curr_price = close.iloc[i]
        curr_atr = atr.iloc[i]

        if position is None:
            # Confluence entry: both RSI AND Stochastic oversold
            if curr_rsi < rsi_os and curr_stoch < stoch_os:
                position = "long"
                entry_price = curr_price
                stop_loss = entry_price - (atr_stop_mult * curr_atr)
                take_profit = entry_price + (atr_tp_mult * curr_atr)

        elif position == "long":
            hit_stop = curr_price <= stop_loss
            hit_tp = curr_price >= take_profit
            # Exit when BOTH are above 50
            exit_signal = curr_rsi > 50 and curr_stoch > 50

            if hit_stop or hit_tp or exit_signal:
                trades.append((curr_price - entry_price) / entry_price * 100)
                position = None

    if not trades:
        return {"total_return": 0, "win_rate": 0, "total_trades": 0, "profit_factor": 0}

    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 999

    return {
        "total_return": sum(trades),
        "win_rate": len(wins) / len(trades) * 100,
        "total_trades": len(trades),
        "profit_factor": pf,
    }


def main():
    print("=" * 80)
    print("🎯 COMPREHENSIVE STRATEGY TEST - EXPANDED UNIVERSE")
    print("=" * 80)

    detector = RegimeDetector()

    # ========== PART 1: Expanded Universe Intraday ==========
    print("\n" + "=" * 80)
    print("📊 PART 1: EXPANDED UNIVERSE (Intraday 5min)")
    print("=" * 80)

    intraday_results = []

    for symbol in EXPANDED_UNIVERSE:
        df = load_intraday_data(symbol, days=10)
        if df is None or len(df) < 500:
            continue

        df_5m = resample_to_timeframe(df, "5min")

        if len(df_5m) < 100:
            continue

        # Regime detection
        analysis = detector.analyze(df_5m, symbol=symbol, timeframe="5min")
        regime = analysis.regime
        recommendation = analysis.trade_recommendation

        if recommendation == "AVOID":
            print(f"  {symbol}: ❌ SKIP ({regime.value})")
            continue

        # Use optimized config if available
        cfg = OPTIMIZED_CONFIGS.get(symbol, OPTIMIZED_CONFIGS["DEFAULT"])

        # RSI-only backtest
        result_rsi = backtest(
            df_5m,
            rsi_os=cfg.rsi_oversold,
            atr_stop_mult=cfg.atr_stop_mult,
            atr_tp_mult=cfg.atr_tp_mult,
        )

        # Confluence backtest
        result_conf = backtest_with_confluence(
            df_5m,
            rsi_os=cfg.rsi_oversold,
            stoch_os=25,
            atr_stop_mult=cfg.atr_stop_mult,
            atr_tp_mult=cfg.atr_tp_mult,
        )

        status = "🟢" if result_rsi["total_return"] > 0 else "🔴"
        print(
            f"  {symbol}: {status} RSI={result_rsi['total_return']:+.1f}% ({result_rsi['total_trades']}t) | Confluence={result_conf['total_return']:+.1f}% ({result_conf['total_trades']}t)"
        )

        intraday_results.append(
            {
                "symbol": symbol,
                "regime": regime.value,
                "rsi_return": result_rsi["total_return"],
                "rsi_trades": result_rsi["total_trades"],
                "rsi_wr": result_rsi["win_rate"],
                "rsi_pf": result_rsi["profit_factor"],
                "conf_return": result_conf["total_return"],
                "conf_trades": result_conf["total_trades"],
                "conf_wr": result_conf["win_rate"],
                "conf_pf": result_conf["profit_factor"],
            }
        )

    # Intraday summary
    if intraday_results:
        total_rsi = sum(r["rsi_return"] for r in intraday_results)
        total_conf = sum(r["conf_return"] for r in intraday_results)

        print("\n  📈 INTRADAY TOTALS:")
        print(
            f"     RSI-Only: {total_rsi:+.1f}% across {sum(r['rsi_trades'] for r in intraday_results)} trades"
        )
        print(
            f"     Confluence: {total_conf:+.1f}% across {sum(r['conf_trades'] for r in intraday_results)} trades"
        )

    # ========== PART 2: Daily Timeframe ==========
    print("\n" + "=" * 80)
    print("📊 PART 2: DAILY TIMEFRAME TEST")
    print("=" * 80)

    daily_results = []

    # Try to load daily data
    for symbol in ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMD"]:
        df = load_daily_data(symbol)
        if df is None:
            # Generate from intraday
            df_intra = load_intraday_data(symbol, days=30)
            if df_intra is not None:
                df = resample_to_timeframe(df_intra, "1D")
                if len(df) < 20:
                    continue
            else:
                continue
        else:
            df = df.set_index("timestamp") if "timestamp" in df.columns else df

        if len(df) < 50:
            print(f"  {symbol}: Insufficient daily data ({len(df)} bars)")
            continue

        # Ensure required columns
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            continue

        result = backtest(df, rsi_os=35, atr_stop_mult=1.5, atr_tp_mult=2.0)

        if result["total_trades"] > 0:
            status = "🟢" if result["total_return"] > 0 else "🔴"
            print(
                f"  {symbol}: {status} {result['total_return']:+.1f}% | {result['win_rate']:.0f}% WR | {result['total_trades']} trades"
            )
            daily_results.append({"symbol": symbol, **result})

    if daily_results:
        total_daily = sum(r["total_return"] for r in daily_results)
        print(f"\n  📈 DAILY TOTAL: {total_daily:+.1f}%")

    # ========== PART 3: Best Performers ==========
    print("\n" + "=" * 80)
    print("🏆 TOP PERFORMERS (by Return)")
    print("=" * 80)

    if intraday_results:
        sorted_results = sorted(intraday_results, key=lambda x: x["rsi_return"], reverse=True)

        print("\n  RSI-Only Strategy:")
        for i, r in enumerate(sorted_results[:5]):
            print(
                f"    {i+1}. {r['symbol']}: {r['rsi_return']:+.1f}% | WR {r['rsi_wr']:.0f}% | PF {r['rsi_pf']:.2f}"
            )

        print("\n  Confluence Strategy:")
        sorted_conf = sorted(intraday_results, key=lambda x: x["conf_return"], reverse=True)
        for i, r in enumerate(sorted_conf[:5]):
            print(
                f"    {i+1}. {r['symbol']}: {r['conf_return']:+.1f}% | WR {r['conf_wr']:.0f}% | PF {r['conf_pf']:.2f}"
            )

    # ========== PART 4: Generate Config ==========
    print("\n" + "=" * 80)
    print("⚙️  GENERATING LIVE TRADING CONFIG")
    print("=" * 80)

    # Find symbols with positive returns and good win rates
    tradeable_symbols = [
        r
        for r in intraday_results
        if r["rsi_return"] > 0 and r["rsi_wr"] >= 55 and r["rsi_pf"] >= 1.2
    ]

    config_path = Path("configs/strategies/atr_optimized_rsi.yaml")
    generate_live_config(tradeable_symbols, config_path)

    print(f"\n  ✅ Config saved to: {config_path}")
    print(f"  📋 Tradeable symbols: {len(tradeable_symbols)}")

    for r in tradeable_symbols:
        print(f"     • {r['symbol']}: {r['rsi_return']:+.1f}% | WR {r['rsi_wr']:.0f}%")


def generate_live_config(results: list, output_path: Path):
    """Generate YAML config for live trading."""
    import yaml

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build symbol configs
    symbol_configs = {}
    for r in results:
        symbol = r["symbol"]
        cfg = OPTIMIZED_CONFIGS.get(symbol, OPTIMIZED_CONFIGS["DEFAULT"])
        symbol_configs[symbol] = {
            "rsi_oversold": int(cfg.rsi_oversold),
            "rsi_exit": int(cfg.rsi_exit),
            "atr_stop_mult": float(cfg.atr_stop_mult),
            "atr_tp_mult": float(cfg.atr_tp_mult),
            "backtest_return": float(round(r["rsi_return"], 2)),
            "backtest_win_rate": float(round(r["rsi_wr"], 1)),
            "backtest_profit_factor": float(round(r["rsi_pf"], 2)),
        }

    config = {
        "strategy": {
            "name": "ATR-Optimized RSI Mean Reversion",
            "version": "1.0.0",
            "type": "mean_reversion",
            "description": "RSI oversold entries with ATR-based adaptive stops",
        },
        "global_params": {
            "rsi_period": 14,
            "atr_period": 14,
            "default_rsi_oversold": 35,
            "default_rsi_exit": 50,
            "default_atr_stop_mult": 1.5,
            "default_atr_tp_mult": 2.0,
        },
        "regime_filter": {
            "enabled": True,
            "avoid_regimes": ["quiet_choppy", "choppy"],
            "prefer_regimes": ["mean_reverting", "trending"],
        },
        "risk_management": {
            "max_position_size_pct": 5.0,
            "max_daily_loss_pct": 2.0,
            "max_concurrent_positions": 5,
            "use_atr_position_sizing": True,
        },
        "symbols": symbol_configs,
        "backtested_on": str(pd.Timestamp.now().date()),
    }

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    main()
