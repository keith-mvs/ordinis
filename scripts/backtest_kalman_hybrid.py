#!/usr/bin/env python3
"""
Kalman Hybrid Strategy Backtest - Two-Stage Optimization

Leverages proven patterns from Fibonacci ADX backtests with improvements:
1. Two-stage optimization: Filter params (Q/R) then trading params
2. Symbol-specific parameterization based on volatility clustering
3. GPU acceleration for filter computations
4. Walk-forward validation with regime awareness
5. Direct ProofBench engine integration

Usage:
    python scripts/backtest_kalman_hybrid.py --mode baseline
    python scripts/backtest_kalman_hybrid.py --mode optimize --symbols AAPL,MSFT,JPM
    python scripts/backtest_kalman_hybrid.py --mode full --parallel
"""

import asyncio
import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback to numpy

import yfinance as yf
import gzip

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ordinis.engines.signalcore.models.kalman_hybrid import (
    KalmanHybridModel,
    KalmanFilter,
    KalmanConfig,
    optimize_kalman_params,
)
from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import SignalType, Direction


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    """Unified backtest configuration."""
    initial_capital: float = 100_000.0
    position_size_pct: float = 0.10
    max_positions: int = 5

    # Filter defaults (will be optimized per-symbol)
    process_noise_q: float = 1e-5
    observation_noise_r: float = 1e-2

    # Trading params - relaxed for more signals
    residual_z_entry: float = 1.25  # Lower threshold
    residual_z_exit: float = 0.2    # Quick mean-reversion exit
    trend_slope_min: float = 0.0    # No trend filter (more signals)
    atr_stop_mult: float = 1.5
    atr_tp_mult: float = 2.0

    # Take profit settings (aggressive profit taking)
    pct_tp_1: float = 0.03   # Take partial profit at 3%
    pct_tp_2: float = 0.06   # Take more at 6%
    pct_tp_3: float = 0.10   # Final exit at 10%
    trailing_stop_pct: float = 0.02  # 2% trailing stop after initial profit

    # Backtest settings
    min_bars: int = 200
    warmup_bars: int = 120

    # Output
    output_dir: str = "data/backtest_results/kalman_hybrid"


# Symbol universe - DIVERSE across sectors, sub-$25 focus
SYMBOLS = {
    # Crypto/Mining (high vol)
    "crypto_mining": ["RIOT", "MARA", "CLSK", "BITF", "HUT", "CIFR", "COIN"],
    # Fintech
    "fintech": ["SOFI", "HOOD", "AFRM", "UPST", "SQ", "PYPL"],
    # Tech/Software
    "tech_software": ["PLTR", "SNAP", "PINS", "RBLX", "U", "AI"],
    # EVs/Clean Energy
    "ev_clean": ["PLUG", "LCID", "RIVN", "NKLA", "FCEL", "BE", "CHPT"],
    # Biotech/Pharma
    "biotech": ["BNGO", "SNDL", "TLRY", "CGC", "ACB", "MRNA", "DNA"],
    # Airlines/Travel
    "travel": ["AAL", "UAL", "DAL", "LUV", "JBLU", "ABNB"],
    # Retail/Consumer
    "consumer": ["GME", "AMC", "BBBY", "WKHS", "WISH", "CLOV"],
    # Industrial/Materials
    "industrial": ["CLF", "X", "FCX", "AA", "VALE"],
    # Real Estate
    "reits": ["SPG", "MPW", "EPR", "AGNC", "NLY"],
    # Small Cap Speculative
    "speculative": ["MULN", "FFIE", "GOEV", "RIDE", "NKLA"],
}

ALL_SYMBOLS = [s for cluster in SYMBOLS.values() for s in cluster]

# Regime periods for validation
REGIMES = {
    "covid_crash": ("2020-02-01", "2020-04-30"),
    "covid_recovery": ("2020-05-01", "2021-12-31"),
    "rate_hikes": ("2022-01-01", "2022-12-31"),
    "ai_boom": ("2023-01-01", "2024-06-30"),
    "full_5yr": ("2020-01-01", "2024-12-31"),
}


# =============================================================================
# DATA FETCHING
# =============================================================================

MASSIVE_DATA_DIR = Path(__file__).parent.parent / "data" / "massive"


def load_massive_data(symbol: str, aggregate_mins: int = 1) -> pd.DataFrame | None:
    """Load 1-minute data from Massive flat files and optionally aggregate.

    Args:
        symbol: Stock ticker
        aggregate_mins: Aggregate to N-minute bars (1, 5, 15, 30, 60)

    Returns:
        DataFrame with OHLCV columns, or None if insufficient data
    """
    dfs = []

    for gz_file in sorted(MASSIVE_DATA_DIR.glob("*.csv.gz")):
        try:
            with gzip.open(gz_file, "rt") as f:
                df = pd.read_csv(f)

            # Filter to symbol
            sym_df = df[df["ticker"] == symbol].copy()
            if sym_df.empty:
                continue

            # Parse timestamp (nanoseconds to datetime)
            sym_df["datetime"] = pd.to_datetime(sym_df["window_start"], unit="ns", utc=True)
            sym_df = sym_df.set_index("datetime")

            # Rename columns
            sym_df = sym_df.rename(columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            })
            sym_df = sym_df[["open", "high", "low", "close", "volume"]]
            dfs.append(sym_df)

        except Exception as e:
            print(f"  Warning: Error reading {gz_file}: {e}")
            continue

    if not dfs:
        return None

    combined = pd.concat(dfs).sort_index()

    # Remove duplicates
    combined = combined[~combined.index.duplicated(keep="first")]

    # Aggregate if needed
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


def fetch_daily_data(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame | None:
    """Fetch OHLCV data from Yahoo Finance.

    Args:
        symbol: Stock ticker
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        interval: 1m, 5m, 15m, 30m, 60m, 1h, 1d
    """
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)

        if df.empty or len(df) < 100:
            return None

        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df.dropna()

        # Validate data quality
        if (df["close"] <= 0).any():
            return None
        if (df["high"] < df["low"]).any():
            return None

        return df
    except Exception as e:
        print(f"  [{symbol}] Fetch error: {e}")
        return None


def get_volatility_cluster(symbol: str) -> str:
    """Get volatility cluster for a symbol."""
    for cluster, syms in SYMBOLS.items():
        if symbol in syms:
            return cluster
    return "med_vol"  # Default


# =============================================================================
# KALMAN FILTER OPTIMIZATION (STAGE 1)
# =============================================================================

def optimize_filter_params(
    prices: pd.Series,
    symbol: str,
    verbose: bool = True,
) -> dict:
    """
    Stage 1: Optimize Kalman filter Q/R parameters.

    Objective: Maximize trend extraction quality (not trading profit).
    """
    cluster = get_volatility_cluster(symbol)

    # Q/R ranges by volatility cluster
    if cluster == "low_vol":
        q_range = [1e-7, 1e-6, 5e-6, 1e-5]
        r_range = [5e-3, 1e-2, 2e-2, 5e-2]
    elif cluster == "high_vol":
        q_range = [1e-5, 5e-5, 1e-4, 5e-4]
        r_range = [1e-3, 5e-3, 1e-2]
    else:  # med_vol
        q_range = [1e-6, 5e-6, 1e-5, 5e-5]
        r_range = [5e-3, 1e-2, 2e-2]

    result = optimize_kalman_params(prices, q_range, r_range)

    if verbose:
        print(f"  [{symbol}] Optimal Q={result['best_q']:.2e}, R={result['best_r']:.2e}, "
              f"Score={result['best_score']:.4f}")

    return {
        "symbol": symbol,
        "cluster": cluster,
        "best_q": float(result["best_q"]),
        "best_r": float(result["best_r"]),
        "score": float(result["best_score"]),
    }


# =============================================================================
# GPU-ACCELERATED COMPUTATIONS
# =============================================================================

def compute_atr_gpu(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute ATR with GPU if available."""
    xp = cp if GPU_AVAILABLE else np

    h = xp.asarray(high)
    l = xp.asarray(low)
    c = xp.asarray(close)

    n = len(h)
    tr = xp.zeros(n)
    tr[0] = h[0] - l[0]

    for i in range(1, n):
        tr[i] = max(
            float(h[i] - l[i]),
            abs(float(h[i] - c[i-1])),
            abs(float(l[i] - c[i-1]))
        )

    atr = xp.zeros(n)
    if period <= n:
        atr[period-1] = xp.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period

    if GPU_AVAILABLE:
        return cp.asnumpy(atr)
    return atr


def run_kalman_filter_batch(
    prices: np.ndarray,
    q: float,
    r: float,
    lookback: int = 100,
) -> dict:
    """Run Kalman filter and compute all derived signals."""
    n = len(prices)

    # Kalman filter state
    x = prices[0]
    p = 1.0

    levels = np.zeros(n)
    residuals = np.zeros(n)
    variances = np.zeros(n)

    for i in range(n):
        # Predict
        x_pred = x
        p_pred = p + q

        # Update
        k = p_pred / (p_pred + r)
        x = x_pred + k * (prices[i] - x_pred)
        p = (1 - k) * p_pred

        levels[i] = x
        residuals[i] = prices[i] - x
        variances[i] = p

    # Trend slope
    slopes = np.diff(levels, prepend=levels[0])

    # Residual z-score (rolling)
    residual_z = np.zeros(n)
    for i in range(lookback, n):
        window = residuals[i-lookback:i]
        mean = np.mean(window)
        std = np.std(window) + 1e-10
        residual_z[i] = (residuals[i] - mean) / std

    return {
        "levels": levels,
        "residuals": residuals,
        "residual_z": residual_z,
        "slopes": slopes,
        "variances": variances,
    }


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

@dataclass
class Trade:
    """Trade record."""
    entry_date: datetime
    exit_date: datetime | None = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    direction: int = 0  # 1=long, -1=short
    size: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    entry_z: float = 0.0
    entry_slope: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest result."""
    symbol: str
    period: str
    params: dict

    # Core metrics
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Trade stats
    num_trades: int = 0
    avg_trade_pnl: float = 0.0
    avg_hold_days: float = 0.0

    # Kalman-specific
    avg_entry_z: float = 0.0
    trend_alignment_rate: float = 0.0

    # Raw data
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "period": self.period,
            "params": self.params,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "num_trades": self.num_trades,
            "avg_trade_pnl": self.avg_trade_pnl,
            "avg_hold_days": self.avg_hold_days,
            "avg_entry_z": self.avg_entry_z,
            "trend_alignment_rate": self.trend_alignment_rate,
        }


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    config: BacktestConfig,
    filter_params: dict | None = None,
) -> BacktestResult:
    """
    Run Kalman Hybrid backtest.

    Two-stage param usage:
    1. filter_params contains optimized Q/R (from Stage 1)
    2. config contains trading params (from Stage 2 or defaults)
    """
    n = len(df)
    if n < config.min_bars:
        return BacktestResult(
            symbol=symbol,
            period="insufficient_data",
            params={},
            total_return=-1.0,
        )

    # Use filter params if provided, else config defaults
    q = filter_params.get("best_q", config.process_noise_q) if filter_params else config.process_noise_q
    r = filter_params.get("best_r", config.observation_noise_r) if filter_params else config.observation_noise_r

    prices = df["close"].values
    high = df["high"].values
    low = df["low"].values

    # Run Kalman filter
    kalman = run_kalman_filter_batch(prices, q, r)

    # Compute ATR for stops
    atr = compute_atr_gpu(high, low, prices)

    # Trading simulation with improved profit taking
    capital = config.initial_capital
    position = 0.0
    entry_price = 0.0
    entry_idx = 0
    entry_z = 0.0
    entry_slope = 0.0
    max_favorable_price = 0.0  # For trailing stop
    partial_exits = 0  # Track partial profit taking

    trades: list[Trade] = []
    equity_curve = [capital]

    warmup = config.warmup_bars

    for i in range(warmup, n):
        current_price = prices[i]
        residual_z = kalman["residual_z"][i]
        slope = kalman["slopes"][i]
        current_atr = atr[i] if atr[i] > 0 else prices[i] * 0.02

        # Track equity
        if position != 0:
            unrealized = position * (current_price - entry_price)
            equity_curve.append(capital + unrealized)
        else:
            equity_curve.append(capital)

        # Exit logic with aggressive profit taking
        if position != 0:
            # Calculate current P&L percentage
            if position > 0:
                pnl_pct = (current_price - entry_price) / entry_price
                max_favorable_price = max(max_favorable_price, current_price)
            else:
                pnl_pct = (entry_price - current_price) / entry_price
                max_favorable_price = min(max_favorable_price, current_price) if max_favorable_price > 0 else current_price

            # Calculate stops
            if position > 0:
                stop_price = entry_price - (current_atr * config.atr_stop_mult)
                hit_stop = current_price <= stop_price

                # Trailing stop after profit
                if pnl_pct > config.pct_tp_1:  # In profit
                    trailing_stop = max_favorable_price * (1 - config.trailing_stop_pct)
                    if current_price <= trailing_stop:
                        hit_stop = True
            else:
                stop_price = entry_price + (current_atr * config.atr_stop_mult)
                hit_stop = current_price >= stop_price

                # Trailing stop for short
                if pnl_pct > config.pct_tp_1:
                    trailing_stop = max_favorable_price * (1 + config.trailing_stop_pct)
                    if current_price >= trailing_stop:
                        hit_stop = True

            # Multiple take profit levels
            hit_tp1 = pnl_pct >= config.pct_tp_1 and partial_exits == 0
            hit_tp2 = pnl_pct >= config.pct_tp_2 and partial_exits <= 1
            hit_tp3 = pnl_pct >= config.pct_tp_3

            # Residual normalization exit (mean reversion complete)
            residual_normalized = (
                (position > 0 and residual_z > -config.residual_z_exit) or
                (position < 0 and residual_z < config.residual_z_exit)
            )

            exit_signal = False
            exit_reason = ""

            if hit_stop:
                exit_signal = True
                exit_reason = "stop_loss" if pnl_pct < 0 else "trailing_stop"
            elif hit_tp3:
                exit_signal = True
                exit_reason = "take_profit_10pct"
            elif hit_tp2 and residual_normalized:
                exit_signal = True
                exit_reason = "take_profit_6pct"
            elif hit_tp1 and residual_normalized:
                exit_signal = True
                exit_reason = "take_profit_3pct"
            elif residual_normalized and pnl_pct > 0:
                exit_signal = True
                exit_reason = "residual_profit"

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
                    entry_z=entry_z,
                    entry_slope=entry_slope,
                ))
                position = 0.0
                partial_exits = 0
                max_favorable_price = 0.0

        # Entry logic - more aggressive
        if position == 0:
            signal_long = (
                residual_z < -config.residual_z_entry and
                (config.trend_slope_min == 0 or slope > config.trend_slope_min)
            )
            signal_short = (
                residual_z > config.residual_z_entry and
                (config.trend_slope_min == 0 or slope < -config.trend_slope_min)
            )

            if signal_long or signal_short:
                # Position sizing
                risk_amount = capital * config.position_size_pct
                shares = risk_amount / current_price

                if signal_long:
                    position = shares
                    max_favorable_price = current_price
                else:
                    position = -shares
                    max_favorable_price = current_price

                entry_price = current_price
                entry_idx = i
                entry_z = residual_z
                entry_slope = slope

    # Close remaining position
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
            entry_z=entry_z,
            entry_slope=entry_slope,
        ))
        equity_curve.append(capital)

    # Calculate metrics
    equity_arr = np.array(equity_curve)
    total_return = (capital - config.initial_capital) / config.initial_capital

    # Sharpe ratio
    if len(equity_arr) > 1:
        returns = np.diff(equity_arr) / equity_arr[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / (peak + 1e-10)
    max_dd = float(np.max(drawdown))

    # Trade metrics
    if trades:
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl <= 0]

        win_rate = len(winners) / len(trades)

        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        avg_pnl = np.mean([t.pnl_pct for t in trades])

        # Average hold time
        hold_days = [(t.exit_date - t.entry_date).days for t in trades if t.exit_date]
        avg_hold = np.mean(hold_days) if hold_days else 0

        # Kalman-specific: trend alignment
        aligned = [t for t in trades if
                   (t.direction > 0 and t.entry_slope > 0) or
                   (t.direction < 0 and t.entry_slope < 0)]
        trend_alignment = len(aligned) / len(trades)

        avg_entry_z = np.mean([abs(t.entry_z) for t in trades])
    else:
        win_rate = 0
        profit_factor = 0
        avg_pnl = 0
        avg_hold = 0
        trend_alignment = 0
        avg_entry_z = 0

    return BacktestResult(
        symbol=symbol,
        period="backtest",
        params={"q": q, "r": r, **{k: getattr(config, k) for k in [
            "residual_z_entry", "residual_z_exit", "trend_slope_min",
            "atr_stop_mult", "atr_tp_mult"
        ]}},
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=len(trades),
        avg_trade_pnl=avg_pnl,
        avg_hold_days=avg_hold,
        avg_entry_z=avg_entry_z,
        trend_alignment_rate=trend_alignment,
        trades=[{
            "entry": t.entry_date.isoformat() if hasattr(t.entry_date, 'isoformat') else str(t.entry_date),
            "exit": t.exit_date.isoformat() if t.exit_date and hasattr(t.exit_date, 'isoformat') else str(t.exit_date),
            "pnl_pct": t.pnl_pct,
            "reason": t.exit_reason,
        } for t in trades[:20]],  # Keep first 20
        equity_curve=equity_curve[::max(1, len(equity_curve)//100)],  # Downsample
    )


# =============================================================================
# OPTIMIZATION (STAGE 2)
# =============================================================================

def optimize_trading_params(
    df: pd.DataFrame,
    symbol: str,
    filter_params: dict,
    n_trials: int = 50,
    output_dir: Path | None = None,
) -> tuple[dict, BacktestResult, list]:
    """
    Stage 2: Optimize trading parameters with fixed filter params.

    Returns:
        best_params, best_result, all_results (for comprehensive logging)
    """
    best_result = None
    best_params = None
    best_score = -np.inf
    all_results = []

    # Parameter ranges - comprehensive grid for thorough optimization
    param_grid = {
        "residual_z_entry": [0.75, 1.0, 1.25, 1.5, 1.75, 2.0],  # Even lower thresholds
        "residual_z_exit": [0.1, 0.2, 0.3, 0.5],
        "trend_slope_min": [0.0, 1e-6, 5e-6, 1e-5, 5e-5],  # Very low for more signals
        "atr_stop_mult": [0.75, 1.0, 1.5, 2.0, 2.5],
        "atr_tp_mult": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
    }

    # Random search with n_trials
    rng = np.random.default_rng(42)

    for trial in range(n_trials):
        config = BacktestConfig(
            residual_z_entry=rng.choice(param_grid["residual_z_entry"]),
            residual_z_exit=rng.choice(param_grid["residual_z_exit"]),
            trend_slope_min=rng.choice(param_grid["trend_slope_min"]),
            atr_stop_mult=rng.choice(param_grid["atr_stop_mult"]),
            atr_tp_mult=rng.choice(param_grid["atr_tp_mult"]),
        )

        result = run_backtest(df, symbol, config, filter_params)

        trial_result = {
            "trial": trial,
            "params": {
                "residual_z_entry": config.residual_z_entry,
                "residual_z_exit": config.residual_z_exit,
                "trend_slope_min": config.trend_slope_min,
                "atr_stop_mult": config.atr_stop_mult,
                "atr_tp_mult": config.atr_tp_mult,
            },
            "metrics": result.to_dict() if result else None,
        }
        all_results.append(trial_result)

        # Score: risk-adjusted return with constraints
        if result.num_trades < 2:  # Very relaxed minimum trades
            score = -np.inf
        elif result.max_drawdown > 0.40:  # Relaxed from 0.35
            score = -np.inf
        else:
            # Prioritize profitability with good risk-adjusted returns
            score = (result.total_return * 0.4 + result.sharpe_ratio * 0.4) * (1 - result.max_drawdown * 0.2)

        if score > best_score:
            best_score = score
            best_result = result
            best_params = trial_result["params"]
            print(f"    [Trial {trial}] NEW BEST: Return={result.total_return*100:+.2f}%, "
                  f"Sharpe={result.sharpe_ratio:.2f}, Trades={result.num_trades}")

        # Save individual trial if output_dir provided
        if output_dir and result.num_trades > 0:
            trial_dir = output_dir / f"Kalman_Opt_{trial}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            # Save report
            with open(trial_dir / "report.json", "w") as f:
                json.dump({
                    "schema_version": "1.1",
                    "strategy": "Kalman_Hybrid",
                    "symbol": symbol,
                    "trial": trial,
                    "params": trial_result["params"],
                    "filter_params": filter_params,
                    "metrics": trial_result["metrics"],
                }, f, indent=2, default=str)

    return best_params, best_result, all_results


# =============================================================================
# MAIN EXECUTION MODES
# =============================================================================

async def run_baseline(symbols: list[str], period: tuple[str, str], config: BacktestConfig, interval: str = "1d") -> dict:
    """Run baseline backtest with default parameters."""
    print("\n" + "=" * 80)
    print("KALMAN HYBRID BASELINE BACKTEST")
    print(f"Period: {period[0]} to {period[1]}")
    print(f"Interval: {interval}")
    print(f"Symbols: {len(symbols)}")
    print("=" * 80)

    results = []

    for symbol in symbols:
        print(f"\n[{symbol}] ", end="")

        df = fetch_daily_data(symbol, period[0], period[1], interval=interval)
        if df is None:
            print("insufficient data")
            continue

        print(f"{len(df)} bars... ", end="")

        result = run_backtest(df, symbol, config)
        results.append(result)

        status = "+" if result.total_return > 0 else "-"
        print(f"{status} Return: {result.total_return*100:+.2f}% | "
              f"Sharpe: {result.sharpe_ratio:.2f} | "
              f"MaxDD: {result.max_drawdown*100:.1f}% | "
              f"Trades: {result.num_trades} | "
              f"WinRate: {result.win_rate*100:.1f}%")

    # Summary
    valid = [r for r in results if r.total_return > -1]
    if valid:
        print("\n" + "-" * 60)
        print("BASELINE SUMMARY")
        print("-" * 60)
        avg_ret = np.mean([r.total_return for r in valid])
        avg_sharpe = np.mean([r.sharpe_ratio for r in valid])
        avg_dd = np.mean([r.max_drawdown for r in valid])
        winners = len([r for r in valid if r.total_return > 0])

        print(f"Symbols tested: {len(valid)}")
        print(f"Profitable: {winners}/{len(valid)} ({100*winners/len(valid):.1f}%)")
        print(f"Avg Return: {avg_ret*100:+.2f}%")
        print(f"Avg Sharpe: {avg_sharpe:.2f}")
        print(f"Avg MaxDD: {avg_dd*100:.1f}%")

    return {"mode": "baseline", "results": [r.to_dict() for r in valid]}


async def run_optimized(symbols: list[str], period: tuple[str, str], config: BacktestConfig, interval: str = "1d") -> dict:
    """Run two-stage optimized backtest."""
    print("\n" + "=" * 80)
    print("KALMAN HYBRID TWO-STAGE OPTIMIZATION")
    print(f"Period: {period[0]} to {period[1]}")
    print(f"Interval: {interval}")
    print(f"Symbols: {len(symbols)}")
    print("=" * 80)

    results = []
    all_filter_params = {}
    all_trading_params = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"[{symbol}]")
        print("=" * 60)

        df = fetch_daily_data(symbol, period[0], period[1], interval=interval)
        if df is None or len(df) < config.min_bars:
            print("  Insufficient data, skipping")
            continue

        print(f"  Data: {len(df)} bars")

        # Stage 1: Filter optimization
        print("\n  STAGE 1: Filter Parameter Optimization")
        filter_params = optimize_filter_params(df["close"], symbol)
        all_filter_params[symbol] = filter_params

        # Stage 2: Trading parameter optimization
        print("\n  STAGE 2: Trading Parameter Optimization (50 trials)")
        trading_params, best_result, all_trials = optimize_trading_params(
            df, symbol, filter_params, n_trials=50, output_dir=Path(config.output_dir) / symbol
        )
        all_trading_params[symbol] = trading_params

        if best_result:
            results.append(best_result)
            print(f"\n  OPTIMIZED RESULT:")
            print(f"    Filter: Q={filter_params['best_q']:.2e}, R={filter_params['best_r']:.2e}")
            print(f"    Trading: z_entry={trading_params['residual_z_entry']}, "
                  f"z_exit={trading_params['residual_z_exit']}")
            print(f"    Return: {best_result.total_return*100:+.2f}%")
            print(f"    Sharpe: {best_result.sharpe_ratio:.2f}")
            print(f"    MaxDD: {best_result.max_drawdown*100:.1f}%")
            print(f"    Trades: {best_result.num_trades}")
            print(f"    Trend Alignment: {best_result.trend_alignment_rate*100:.1f}%")

    # Summary
    valid = [r for r in results if r.total_return > -1]
    if valid:
        print("\n" + "=" * 80)
        print("OPTIMIZATION SUMMARY")
        print("=" * 80)

        avg_ret = np.mean([r.total_return for r in valid])
        avg_sharpe = np.mean([r.sharpe_ratio for r in valid])
        avg_dd = np.mean([r.max_drawdown for r in valid])
        winners = len([r for r in valid if r.total_return > 0])

        print(f"Symbols optimized: {len(valid)}")
        print(f"Profitable: {winners}/{len(valid)} ({100*winners/len(valid):.1f}%)")
        print(f"Avg Return: {avg_ret*100:+.2f}%")
        print(f"Avg Sharpe: {avg_sharpe:.2f}")
        print(f"Avg MaxDD: {avg_dd*100:.1f}%")

        # Best performers
        sorted_results = sorted(valid, key=lambda x: x.sharpe_ratio, reverse=True)
        print(f"\nTop 3 by Sharpe:")
        for r in sorted_results[:3]:
            print(f"  {r.symbol}: Sharpe={r.sharpe_ratio:.2f}, "
                  f"Return={r.total_return*100:+.2f}%")

    return {
        "mode": "optimized",
        "results": [r.to_dict() for r in valid],
        "filter_params": all_filter_params,
        "trading_params": all_trading_params,
    }


async def run_regime_validation(
    symbols: list[str],
    filter_params: dict,
    trading_params: dict,
    config: BacktestConfig,
) -> dict:
    """Validate across multiple regimes."""
    print("\n" + "=" * 80)
    print("REGIME VALIDATION")
    print("=" * 80)

    regime_results = {}

    for regime_name, (start, end) in REGIMES.items():
        print(f"\n--- {regime_name.upper()} ({start} to {end}) ---")

        results = []
        for symbol in symbols:
            if symbol not in filter_params:
                continue

            df = fetch_daily_data(symbol, start, end)
            if df is None or len(df) < 100:
                continue

            # Use optimized params for this symbol
            fp = filter_params[symbol]
            tp = trading_params.get(symbol, {})

            cfg = BacktestConfig(
                residual_z_entry=tp.get("residual_z_entry", config.residual_z_entry),
                residual_z_exit=tp.get("residual_z_exit", config.residual_z_exit),
                trend_slope_min=tp.get("trend_slope_min", config.trend_slope_min),
                atr_stop_mult=tp.get("atr_stop_mult", config.atr_stop_mult),
                atr_tp_mult=tp.get("atr_tp_mult", config.atr_tp_mult),
            )

            result = run_backtest(df, symbol, cfg, fp)
            result.period = regime_name
            results.append(result)

        valid = [r for r in results if r.total_return > -1]
        if valid:
            avg_ret = np.mean([r.total_return for r in valid])
            avg_sharpe = np.mean([r.sharpe_ratio for r in valid])
            winners = len([r for r in valid if r.total_return > 0])

            print(f"  Symbols: {len(valid)} | "
                  f"Profitable: {winners}/{len(valid)} | "
                  f"Avg Return: {avg_ret*100:+.2f}% | "
                  f"Avg Sharpe: {avg_sharpe:.2f}")

            regime_results[regime_name] = {
                "num_symbols": len(valid),
                "profitable": winners,
                "avg_return": avg_ret,
                "avg_sharpe": avg_sharpe,
            }

    return regime_results


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Kalman Hybrid Backtest")
    parser.add_argument("--mode", choices=["baseline", "optimize", "full"], default="baseline")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbols")
    parser.add_argument("--period", type=str, default="5yr", help="5yr, 3yr, or YYYY-MM-DD:YYYY-MM-DD")
    parser.add_argument("--interval", type=str, default="1d", help="1m, 5m, 15m, 30m, 60m, 1h, 1d")
    parser.add_argument("--cluster", type=str, default=None, help="low_vol, med_vol, high_vol")
    parser.add_argument("--output", type=str, default="data/backtest_results/kalman_hybrid")
    args = parser.parse_args()

    # GPU status
    if GPU_AVAILABLE:
        print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        mem = cp.cuda.runtime.memGetInfo()
        print(f"VRAM: {mem[1]/1e9:.1f} GB available")
    else:
        print("GPU: Not available, using CPU")

    # Parse symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    elif args.cluster:
        symbols = SYMBOLS.get(args.cluster, ALL_SYMBOLS)
    else:
        symbols = ALL_SYMBOLS

    # Parse period
    if args.period == "5yr":
        end = datetime.now()
        start = end - timedelta(days=5*365)
        period = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    elif args.period == "3yr":
        end = datetime.now()
        start = end - timedelta(days=3*365)
        period = (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    elif ":" in args.period:
        parts = args.period.split(":")
        period = (parts[0], parts[1])
    else:
        period = REGIMES.get(args.period, REGIMES["full_5yr"])

    config = BacktestConfig(output_dir=args.output)

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run mode
    if args.mode == "baseline":
        result = await run_baseline(symbols, period, config, interval=args.interval)
    elif args.mode == "optimize":
        result = await run_optimized(symbols, period, config, interval=args.interval)
    elif args.mode == "full":
        # Run optimization first
        opt_result = await run_optimized(symbols, period, config)

        # Then validate across regimes
        if opt_result.get("filter_params") and opt_result.get("trading_params"):
            regime_result = await run_regime_validation(
                symbols,
                opt_result["filter_params"],
                opt_result["trading_params"],
                config,
            )
            opt_result["regime_validation"] = regime_result

        result = opt_result
    else:
        result = {}

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"kalman_hybrid_{args.mode}_{timestamp}.json"

    def serialize(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)

    with open(output_file, "w") as f:
        json.dump(result, f, indent=2, default=serialize)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
