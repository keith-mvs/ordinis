#!/usr/bin/env python3
"""
Small-Cap Kalman Hybrid Aggressive Optimizer

Targets:
- Exclusive focus on small-cap stocks (<$25/share)
- Uses Massive 1-minute flat files for high-frequency signals
- GPU-accelerated Kalman filter and optimization
- NVIDIA Nemotron integration for parameter validation
- Target: >20% average return

Usage:
    python scripts/small_cap_kalman_optimizer.py --n_trials 50 --aggregate 5
"""

import asyncio
import argparse
import gzip
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except ImportError:
    GPU_AVAILABLE = False
    cp = np
    print("GPU: Not available, using CPU")

# NVIDIA API for Nemotron
import httpx

# =============================================================================
# CONFIGURATION
# =============================================================================

MASSIVE_DATA_DIR = Path(__file__).parent.parent / "data" / "massive"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "backtest_results" / "small_cap_kalman"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# NVIDIA API
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
NEMOTRON_MODEL = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
NEMOTRON_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

# Small-cap sectors with aggressive volatility focus
SMALL_CAP_SECTORS = {
    "crypto_mining": ["RIOT", "MARA", "CLSK", "BITF", "HUT", "CIFR", "WULF"],
    "biotech_speculative": ["BNGO", "SNDL", "TLRY", "CGC", "ACB", "XXII", "AGRX"],
    "ev_clean_energy": ["PLUG", "FCEL", "BE", "CHPT", "BLNK", "EVGO", "GOEV"],
    "fintech_disruptors": ["SOFI", "HOOD", "AFRM", "UPST", "OPEN", "LMND"],
    "retail_consumer": ["GME", "AMC", "BBIG", "WKHS", "WISH", "CLOV", "EXPR"],
    "industrial_materials": ["CLF", "X", "AA", "BTU", "ARCH", "HCC"],
    "tech_software": ["AI", "SOUN", "IONQ", "RGTI", "QUBT", "KULR"],
    "real_estate_reits": ["MPW", "AGNC", "NLY", "ABR", "TWO", "IVR"],
}

ALL_SMALL_CAPS = [s for sector in SMALL_CAP_SECTORS.values() for s in sector]


@dataclass
class AggressiveConfig:
    """Aggressive optimization config targeting >20% returns."""
    initial_capital: float = 100_000.0
    position_size_pct: float = 0.15  # Larger positions
    max_positions: int = 3  # Concentrated

    # Aggressive Kalman settings
    process_noise_q: float = 1e-4  # Fast adaptation
    observation_noise_r: float = 5e-3  # Less smoothing

    # Aggressive entry/exit
    residual_z_entry: float = 0.75  # Lower threshold = more trades
    residual_z_exit: float = 0.1  # Quick exits
    trend_slope_min: float = 0.0  # No trend filter

    # Tight risk management
    stop_loss_pct: float = 0.05  # 5% stop
    take_profit_pct: float = 0.08  # 8% TP
    trailing_stop_pct: float = 0.025  # 2.5% trailing

    # Timeframe
    aggregate_mins: int = 5  # 5-minute bars


# =============================================================================
# DATA LOADING
# =============================================================================

def load_massive_symbol(symbol: str, aggregate_mins: int = 5) -> pd.DataFrame | None:
    """Load all Massive data for a symbol and aggregate."""
    dfs = []

    for gz_file in sorted(MASSIVE_DATA_DIR.glob("*.csv.gz")):
        try:
            with gzip.open(gz_file, "rt") as f:
                df = pd.read_csv(f)

            sym_df = df[df["ticker"] == symbol].copy()
            if sym_df.empty:
                continue

            sym_df["datetime"] = pd.to_datetime(sym_df["window_start"], unit="ns", utc=True)
            sym_df = sym_df.set_index("datetime")
            sym_df = sym_df[["open", "high", "low", "close", "volume"]]
            dfs.append(sym_df)

        except Exception:
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    # Aggregate
    if aggregate_mins > 1:
        combined = combined.resample(f"{aggregate_mins}min").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

    if len(combined) < 100:
        return None

    return combined


def get_available_small_caps() -> list[tuple[str, float]]:
    """Get small-cap symbols available in Massive data with current prices."""
    # Read one file to get symbol list
    gz_files = list(MASSIVE_DATA_DIR.glob("*.csv.gz"))
    if not gz_files:
        return []

    # Get latest file
    latest = sorted(gz_files)[-1]

    with gzip.open(latest, "rt") as f:
        df = pd.read_csv(f)

    available = []
    for sym in ALL_SMALL_CAPS:
        sym_df = df[df["ticker"] == sym]
        if not sym_df.empty:
            last_price = sym_df["close"].iloc[-1]
            if last_price < 25:  # Filter under $25
                available.append((sym, last_price))

    return sorted(available, key=lambda x: x[1])


# =============================================================================
# GPU-ACCELERATED KALMAN FILTER
# =============================================================================

def kalman_filter_gpu(prices: np.ndarray, q: float, r: float) -> dict:
    """GPU-accelerated Kalman filter."""
    xp = cp if GPU_AVAILABLE else np
    n = len(prices)

    # Transfer to GPU
    prices_gpu = xp.asarray(prices, dtype=xp.float64)

    levels = xp.zeros(n, dtype=xp.float64)
    residuals = xp.zeros(n, dtype=xp.float64)
    variances = xp.zeros(n, dtype=xp.float64)

    # Initialize
    x = float(prices_gpu[0])
    p = 1.0

    for i in range(n):
        # Predict
        x_pred = x
        p_pred = p + q

        # Update
        k = p_pred / (p_pred + r)
        x = x_pred + k * (float(prices_gpu[i]) - x_pred)
        p = (1 - k) * p_pred

        levels[i] = x
        residuals[i] = prices_gpu[i] - x
        variances[i] = p

    # Compute derived signals
    slopes = xp.diff(levels, prepend=levels[0])

    # Rolling z-score (vectorized)
    lookback = 50
    residual_z = xp.zeros(n, dtype=xp.float64)

    for i in range(lookback, n):
        window = residuals[i-lookback:i]
        mean = xp.mean(window)
        std = xp.std(window) + 1e-10
        residual_z[i] = (residuals[i] - mean) / std

    # Transfer back to CPU
    if GPU_AVAILABLE:
        return {
            "levels": cp.asnumpy(levels),
            "residuals": cp.asnumpy(residuals),
            "residual_z": cp.asnumpy(residual_z),
            "slopes": cp.asnumpy(slopes),
            "variances": cp.asnumpy(variances),
        }
    return {
        "levels": levels,
        "residuals": residuals,
        "residual_z": residual_z,
        "slopes": slopes,
        "variances": variances,
    }


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute Average True Range."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]

    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    atr = np.zeros(n)
    if period <= n:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    return atr


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

@dataclass
class Trade:
    entry_date: datetime
    exit_date: datetime | None = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    direction: int = 0
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""


@dataclass
class BacktestResult:
    symbol: str
    config: dict
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0  # Better for volatile small-caps
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0
    avg_trade_pnl: float = 0.0
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "config": self.config,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "num_trades": self.num_trades,
            "avg_trade_pnl": self.avg_trade_pnl,
        }


def run_aggressive_backtest(
    df: pd.DataFrame,
    symbol: str,
    config: AggressiveConfig,
) -> BacktestResult:
    """Run aggressive backtest targeting high returns."""
    n = len(df)
    if n < 100:
        return BacktestResult(symbol=symbol, config={}, total_return=-1.0)

    prices = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # Kalman filter
    kalman = kalman_filter_gpu(prices, config.process_noise_q, config.observation_noise_r)
    atr = compute_atr(high, low, prices)

    # Trading simulation
    capital = config.initial_capital
    position = 0.0
    entry_price = 0.0
    entry_idx = 0
    max_price = 0.0

    trades: list[Trade] = []
    equity_curve = [capital]
    warmup = 60

    for i in range(warmup, n):
        current_price = prices[i]
        residual_z = kalman["residual_z"][i]
        slope = kalman["slopes"][i]
        current_atr = atr[i] if atr[i] > 0 else current_price * 0.02

        # Track equity
        if position != 0:
            unrealized = position * (current_price - entry_price)
            equity_curve.append(capital + unrealized)
        else:
            equity_curve.append(capital)

        # Exit logic
        if position != 0:
            if position > 0:
                pnl_pct = (current_price - entry_price) / entry_price
                max_price = max(max_price, current_price)
            else:
                pnl_pct = (entry_price - current_price) / entry_price
                max_price = min(max_price, current_price) if max_price > 0 else current_price

            exit_signal = False
            exit_reason = ""

            # Stop loss
            if pnl_pct < -config.stop_loss_pct:
                exit_signal = True
                exit_reason = "stop_loss"

            # Take profit
            elif pnl_pct >= config.take_profit_pct:
                exit_signal = True
                exit_reason = "take_profit"

            # Trailing stop
            elif pnl_pct > 0.02:
                if position > 0:
                    trailing = max_price * (1 - config.trailing_stop_pct)
                    if current_price <= trailing:
                        exit_signal = True
                        exit_reason = "trailing_stop"
                else:
                    trailing = max_price * (1 + config.trailing_stop_pct)
                    if current_price >= trailing:
                        exit_signal = True
                        exit_reason = "trailing_stop"

            # Mean reversion complete
            if not exit_signal:
                if (position > 0 and residual_z > -config.residual_z_exit) or \
                   (position < 0 and residual_z < config.residual_z_exit):
                    if pnl_pct > 0:
                        exit_signal = True
                        exit_reason = "mean_reversion"

            if exit_signal:
                pnl = position * (current_price - entry_price)
                capital += pnl
                trades.append(Trade(
                    entry_date=df.index[entry_idx],
                    exit_date=df.index[i],
                    entry_price=entry_price,
                    exit_price=current_price,
                    direction=1 if position > 0 else -1,
                    size=abs(position),
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=exit_reason,
                ))
                position = 0.0
                max_price = 0.0

        # Entry logic
        if position == 0:
            signal_long = residual_z < -config.residual_z_entry
            signal_short = residual_z > config.residual_z_entry

            if signal_long or signal_short:
                risk_amount = capital * config.position_size_pct
                shares = risk_amount / current_price

                if signal_long:
                    position = shares
                else:
                    position = -shares

                entry_price = current_price
                entry_idx = i
                max_price = current_price

    # Close remaining
    if position != 0:
        final_price = prices[-1]
        pnl = position * (final_price - entry_price)
        pnl_pct = (final_price - entry_price) / entry_price * np.sign(position)
        capital += pnl
        trades.append(Trade(
            entry_date=df.index[entry_idx],
            exit_date=df.index[-1],
            entry_price=entry_price,
            exit_price=final_price,
            direction=1 if position > 0 else -1,
            size=abs(position),
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason="end_of_data",
        ))
        equity_curve.append(capital)

    # Metrics
    equity_arr = np.array(equity_curve)
    total_return = (capital - config.initial_capital) / config.initial_capital

    if len(equity_arr) > 1:
        returns = np.diff(equity_arr) / equity_arr[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 78)  # 5-min bars

        # Sortino ratio - only penalize downside volatility
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1:
            downside_std = np.std(downside_returns)
            sortino = np.mean(returns) / (downside_std + 1e-10) * np.sqrt(252 * 78)
        else:
            sortino = sharpe * 2  # No downside = very good
    else:
        sharpe = 0.0
        sortino = 0.0

    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / (peak + 1e-10)
    max_dd = float(np.max(drawdown))

    if trades:
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]
        win_rate = len(winners) / len(trades)
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        avg_pnl = np.mean([t.pnl_pct for t in trades])
    else:
        win_rate = profit_factor = avg_pnl = 0

    return BacktestResult(
        symbol=symbol,
        config={
            "q": config.process_noise_q,
            "r": config.observation_noise_r,
            "z_entry": config.residual_z_entry,
            "z_exit": config.residual_z_exit,
            "stop_loss": config.stop_loss_pct,
            "take_profit": config.take_profit_pct,
        },
        total_return=total_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=len(trades),
        avg_trade_pnl=avg_pnl,
        trades=[{
            "pnl_pct": t.pnl_pct,
            "reason": t.exit_reason,
        } for t in trades[:50]],
        equity_curve=equity_curve[::max(1, len(equity_curve)//100)],
    )


# =============================================================================
# OPTIMIZATION
# =============================================================================

def run_optimization(
    df: pd.DataFrame,
    symbol: str,
    n_trials: int = 50,
) -> tuple[dict, BacktestResult, list]:
    """Run n_trials of random parameter search."""
    param_grid = {
        "process_noise_q": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
        "observation_noise_r": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "residual_z_entry": [0.5, 0.75, 1.0, 1.25, 1.5],
        "residual_z_exit": [0.05, 0.1, 0.15, 0.2, 0.3],
        "stop_loss_pct": [0.03, 0.05, 0.07, 0.10],
        "take_profit_pct": [0.05, 0.08, 0.10, 0.15, 0.20],
        "trailing_stop_pct": [0.015, 0.02, 0.025, 0.03],
        "position_size_pct": [0.10, 0.15, 0.20],
    }

    rng = np.random.default_rng(42)
    best_result = None
    best_params = None
    best_score = -np.inf
    all_results = []

    for trial in range(n_trials):
        config = AggressiveConfig(
            process_noise_q=rng.choice(param_grid["process_noise_q"]),
            observation_noise_r=rng.choice(param_grid["observation_noise_r"]),
            residual_z_entry=rng.choice(param_grid["residual_z_entry"]),
            residual_z_exit=rng.choice(param_grid["residual_z_exit"]),
            stop_loss_pct=rng.choice(param_grid["stop_loss_pct"]),
            take_profit_pct=rng.choice(param_grid["take_profit_pct"]),
            trailing_stop_pct=rng.choice(param_grid["trailing_stop_pct"]),
            position_size_pct=rng.choice(param_grid["position_size_pct"]),
        )

        result = run_aggressive_backtest(df, symbol, config)

        trial_data = {
            "trial": trial,
            "config": result.config,
            "return": result.total_return,
            "sharpe": result.sharpe_ratio,
            "sortino": result.sortino_ratio,
            "trades": result.num_trades,
        }
        all_results.append(trial_data)

        # Score: prioritize Sortino for volatile small-caps (downside-only volatility)
        if result.num_trades < 3:
            score = -np.inf
        elif result.max_drawdown > 0.35:
            score = -np.inf
        else:
            # Use Sortino ratio (better for volatile assets) + return emphasis
            score = result.total_return * 0.5 + result.sortino_ratio * 0.35 + result.win_rate * 0.15

        if score > best_score:
            best_score = score
            best_result = result
            best_params = result.config
            print(f"  [Trial {trial:2d}] NEW BEST: {result.total_return*100:+7.2f}%, "
                  f"Sortino={result.sortino_ratio:5.2f}, Sharpe={result.sharpe_ratio:5.2f}, "
                  f"WR={result.win_rate*100:4.1f}%, Trades={result.num_trades}")

    return best_params, best_result, all_results


# =============================================================================
# NEMOTRON INTEGRATION
# =============================================================================

async def analyze_with_nemotron(results: list[dict]) -> str:
    """Use Nemotron to analyze optimization results and suggest improvements."""
    if not NVIDIA_API_KEY:
        return "NVIDIA API key not set - skipping Nemotron analysis"

    # Prepare summary
    profitable = [r for r in results if r.get("return", 0) > 0]
    high_return = [r for r in results if r.get("return", 0) > 0.20]

    summary = f"""
Optimization Results Summary:
- Total trials: {len(results)}
- Profitable trials: {len(profitable)}
- Trials with >20% return: {len(high_return)}

Top 5 configurations by return:
"""
    sorted_results = sorted(results, key=lambda x: x.get("return", -999), reverse=True)
    for i, r in enumerate(sorted_results[:5]):
        summary += f"{i+1}. Return={r.get('return', 0)*100:.1f}%, Config={r.get('config', {})}\n"

    prompt = f"""You are a quantitative trading strategy optimizer. Analyze these Kalman filter optimization results for small-cap stocks and provide:

1. Pattern analysis: What parameter combinations produce the best results?
2. Risk assessment: Are the high-return strategies sustainable?
3. Recommendations: Specific parameter ranges to focus on for >20% returns

{summary}

Be concise and actionable. Focus on the mathematics and statistical patterns."""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                NEMOTRON_ENDPOINT,
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": NEMOTRON_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.3,
                },
            )
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except Exception as e:
        return f"Nemotron error: {e}"


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Small-Cap Kalman Optimizer")
    parser.add_argument("--n_trials", type=int, default=50, help="Optimization trials per symbol")
    parser.add_argument("--aggregate", type=int, default=5, help="Aggregate to N-minute bars")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols (default: all)")
    args = parser.parse_args()

    print("=" * 80)
    print("SMALL-CAP KALMAN HYBRID AGGRESSIVE OPTIMIZER")
    print(f"Target: >20% return on small-cap stocks (<$25)")
    print(f"Trials per symbol: {args.n_trials}")
    print(f"Aggregation: {args.aggregate}-minute bars")
    print("=" * 80)

    # Get available small-caps
    print("\nScanning Massive data for small-cap stocks...")
    available = get_available_small_caps()

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = [s[0] for s in available]

    print(f"Found {len(available)} small-cap symbols in Massive data")
    print(f"Testing: {len(symbols)} symbols")

    all_optimization_results = []
    symbol_results = {}

    for sym in symbols:
        print(f"\n{'='*60}")
        print(f"[{sym}]")
        print("=" * 60)

        df = load_massive_symbol(sym, aggregate_mins=args.aggregate)
        if df is None:
            print("  No data available")
            continue

        print(f"  Data: {len(df)} bars ({args.aggregate}-min)")

        # Run optimization
        best_params, best_result, trials = run_optimization(df, sym, n_trials=args.n_trials)

        if best_result and best_result.total_return > -1:
            symbol_results[sym] = best_result.to_dict()
            all_optimization_results.extend(trials)

            print(f"\n  BEST RESULT:")
            print(f"    Return: {best_result.total_return*100:+.2f}%")
            print(f"    Sortino: {best_result.sortino_ratio:.2f}")
            print(f"    Sharpe: {best_result.sharpe_ratio:.2f}")
            print(f"    MaxDD: {best_result.max_drawdown*100:.1f}%")
            print(f"    Win Rate: {best_result.win_rate*100:.1f}%")
            print(f"    Trades: {best_result.num_trades}")

    # Summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)

    if symbol_results:
        returns = [r["total_return"] for r in symbol_results.values()]
        avg_return = np.mean(returns)
        winners = len([r for r in returns if r > 0])
        high_return = len([r for r in returns if r > 0.20])

        print(f"Symbols tested: {len(symbol_results)}")
        print(f"Profitable: {winners}/{len(symbol_results)} ({100*winners/len(symbol_results):.1f}%)")
        print(f">20% return: {high_return}/{len(symbol_results)}")
        print(f"Average return: {avg_return*100:+.2f}%")

        # Best performers - sorted by Sortino (better for volatile small-caps)
        sorted_results = sorted(symbol_results.items(), key=lambda x: x[1]["sortino_ratio"], reverse=True)
        print("\nTop 5 by Sortino (downside-adjusted risk):")
        for sym, r in sorted_results[:5]:
            print(f"  {sym}: {r['total_return']*100:+.2f}%, Sortino={r['sortino_ratio']:.2f}, Sharpe={r['sharpe_ratio']:.2f}")

        # Also show top by return
        sorted_by_return = sorted(symbol_results.items(), key=lambda x: x[1]["total_return"], reverse=True)
        print("\nTop 5 by Return:")
        for sym, r in sorted_by_return[:5]:
            print(f"  {sym}: {r['total_return']*100:+.2f}%, Sortino={r['sortino_ratio']:.2f}")

        # Nemotron analysis
        print("\n" + "-" * 60)
        print("NEMOTRON ANALYSIS")
        print("-" * 60)
        analysis = await analyze_with_nemotron(all_optimization_results)
        print(analysis)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = OUTPUT_DIR / f"small_cap_optimization_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump({
                "timestamp": timestamp,
                "n_trials": args.n_trials,
                "aggregate_mins": args.aggregate,
                "symbols": list(symbol_results.keys()),
                "results": symbol_results,
                "all_trials": all_optimization_results[:500],  # Limit size
                "summary": {
                    "avg_return": avg_return,
                    "profitable_pct": winners / len(symbol_results),
                    "high_return_count": high_return,
                },
            }, f, indent=2, default=str)

        print(f"\nResults saved: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
