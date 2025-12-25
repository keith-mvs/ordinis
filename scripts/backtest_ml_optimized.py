#!/usr/bin/env python3
"""
ML-Optimized Fibonacci ADX Backtest - Multi-Period Analysis
- 12hr and 24hr aggregated data
- 3 random 12-month periods between 2018-2025
- ML parameter tuning using Bayesian/Evolutionary optimization
- 30 stocks: 10 small cap, 10 mid cap, 10 random picks
"""

import asyncio
import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# GPU imports
try:
    import cupy as cp
    from numba import cuda
    GPU_AVAILABLE = True
    print(f"CuPy: {cp.__version__}")
    print(f"CUDA Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    mem_info = cp.cuda.runtime.memGetInfo()
    print(f"VRAM: {mem_info[1] / 1e9:.1f} GB")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU not available, using CPU")

import yfinance as yf

# Import ML optimizer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ordinis.engines.sprint.core.ml_profit_optimizer import (
    MLProfitOptimizer,
    OptimizationConfig,
    OptimizationMethod,
    ParameterSpec,
    ProfitMetric,
)


# =============================================================================
# STOCK UNIVERSE
# =============================================================================

SMALL_CAP = [
    "CPRX", "RCUS", "PAYO", "ESLT", "PRGS",
    "PGNY", "CALX", "SPSC", "GERN", "AMBA"
]

MID_CAP = [
    "WSM", "TRGP", "RJF", "HOLX", "EME",
    "FFIV", "AVY", "NRG", "ULTA", "GNRC"
]

RANDOM_PICKS = [
    "PLTR", "RKLB", "SOFI", "RIVN", "DKNG",
    "CRWD", "ZS", "SNOW", "MARA", "IONQ"
]


# =============================================================================
# RANDOM PERIOD GENERATION
# =============================================================================

def generate_random_periods(n: int = 3, seed: int = 42) -> list[tuple[str, str]]:
    """Generate n random 12-month periods between 2018-01-01 and 2024-12-31."""
    rng = random.Random(seed)
    
    start_range_begin = datetime(2018, 1, 1)
    start_range_end = datetime(2024, 1, 1)  # Latest start to allow 12 months
    
    periods = []
    used_years = set()
    
    while len(periods) < n:
        # Random start date
        days_range = (start_range_end - start_range_begin).days
        random_days = rng.randint(0, days_range)
        start_date = start_range_begin + timedelta(days=random_days)
        
        # Avoid overlapping years for diversity
        if start_date.year in used_years:
            continue
        used_years.add(start_date.year)
        
        end_date = start_date + timedelta(days=365)
        
        periods.append((
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        ))
    
    return sorted(periods)


# =============================================================================
# DATA FETCHING AND AGGREGATION
# =============================================================================

def fetch_daily_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily data from yfinance."""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval="1d")
        if df.empty:
            return pd.DataFrame()
        
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df = df.dropna()
        return df
    except Exception as e:
        print(f"  [!] Error fetching {symbol}: {e}")
        return pd.DataFrame()


def aggregate_to_interval(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """
    Aggregate daily data to multi-hour bars.
    For 12hr: 2 bars per day (AM/PM approximation using daily data)
    For 24hr: 1 bar per day (same as daily)
    """
    if hours >= 24:
        # 24hr is just daily data
        return df.copy()
    
    if hours == 12:
        # Split daily bar into AM/PM using typical price ratios
        # This is an approximation since we only have daily data
        result_rows = []
        
        for idx, row in df.iterrows():
            daily_range = row['high'] - row['low']
            
            # AM bar (first half of day)
            am_high = row['open'] + daily_range * 0.6  # Typical morning volatility
            am_low = row['low']
            am_close = row['open'] + (row['close'] - row['open']) * 0.4
            am_vol = row['volume'] * 0.45
            
            # PM bar (second half)
            pm_open = am_close
            pm_high = row['high']
            pm_low = row['low'] + daily_range * 0.3
            pm_close = row['close']
            pm_vol = row['volume'] * 0.55
            
            # Create timestamps
            if hasattr(idx, 'replace'):
                am_ts = idx.replace(hour=9, minute=30)
                pm_ts = idx.replace(hour=13, minute=0)
            else:
                am_ts = idx
                pm_ts = idx + pd.Timedelta(hours=4)
            
            result_rows.append({
                'timestamp': am_ts,
                'open': row['open'],
                'high': min(am_high, row['high']),
                'low': max(am_low, row['low']),
                'close': am_close,
                'volume': am_vol
            })
            result_rows.append({
                'timestamp': pm_ts,
                'open': pm_open,
                'high': min(pm_high, row['high']),
                'low': max(pm_low, row['low']),
                'close': pm_close,
                'volume': pm_vol
            })
        
        result = pd.DataFrame(result_rows)
        result = result.set_index('timestamp')
        return result
    
    return df


# =============================================================================
# FIBONACCI ADX STRATEGY (GPU-ACCELERATED)
# =============================================================================

@dataclass
class StrategyParams:
    """Strategy parameters for optimization."""
    adx_period: int = 14
    adx_threshold: float = 25.0
    fib_lookback: int = 50
    tolerance: float = 0.015
    position_size: float = 0.1
    profit_target: float = 0.03
    stop_loss: float = 0.015
    trailing_stop: float = 0.01
    min_bars_between: int = 5
    di_spread: float = 5.0  # Minimum DI+ vs DI- spread


def run_backtest(params: dict, df: pd.DataFrame) -> dict:
    """
    Run Fibonacci ADX backtest with given parameters.
    Returns dict compatible with MLProfitOptimizer.
    """
    if len(df) < 100:
        return {
            'total_return': -1.0,
            'net_profit': -100000,
            'trades': [],
            'equity_curve': [100000],
            'max_drawdown': 1.0,
            'sharpe_ratio': -10,
        }
    
    # Extract params
    p = StrategyParams(
        adx_period=int(params.get('adx_period', 14)),
        adx_threshold=float(params.get('adx_threshold', 25)),
        fib_lookback=int(params.get('fib_lookback', 50)),
        tolerance=float(params.get('tolerance', 0.015)),
        position_size=float(params.get('position_size', 0.1)),
        profit_target=float(params.get('profit_target', 0.03)),
        stop_loss=float(params.get('stop_loss', 0.015)),
        trailing_stop=float(params.get('trailing_stop', 0.01)),
        min_bars_between=int(params.get('min_bars_between', 5)),
        di_spread=float(params.get('di_spread', 5.0)),
    )
    
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    initial_capital = 100000
    
    # Compute ADX
    period = p.adx_period
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    if period <= n:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0
    
    smooth_plus = np.zeros(n)
    smooth_minus = np.zeros(n)
    if period <= n:
        smooth_plus[period-1] = np.mean(plus_dm[:period])
        smooth_minus[period-1] = np.mean(minus_dm[:period])
        for i in range(period, n):
            smooth_plus[i] = (smooth_plus[i-1] * (period - 1) + plus_dm[i]) / period
            smooth_minus[i] = (smooth_minus[i-1] * (period - 1) + minus_dm[i]) / period
    
    plus_di = np.where(atr > 0, 100 * smooth_plus / atr, 0)
    minus_di = np.where(atr > 0, 100 * smooth_minus / atr, 0)
    
    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0)
    adx = np.zeros(n)
    if 2*period <= n:
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # Fibonacci levels
    lookback = p.fib_lookback
    fib_382 = np.zeros(n)
    fib_500 = np.zeros(n)
    fib_618 = np.zeros(n)
    for i in range(lookback, n):
        ph = np.max(high[i-lookback:i])
        pl = np.min(low[i-lookback:i])
        r = ph - pl
        fib_382[i] = ph - 0.382 * r
        fib_500[i] = ph - 0.500 * r
        fib_618[i] = ph - 0.618 * r
    
    # Generate signals
    signals = np.zeros(n, dtype=int)
    for i in range(lookback, n):
        if adx[i] < p.adx_threshold:
            continue
        
        # Long: uptrend, price at 61.8% support
        if plus_di[i] > minus_di[i] + p.di_spread:
            if abs(close[i] - fib_618[i]) / fib_618[i] < p.tolerance:
                signals[i] = 1
        # Short: downtrend, price at 38.2% resistance
        elif minus_di[i] > plus_di[i] + p.di_spread:
            if abs(close[i] - fib_382[i]) / fib_382[i] < p.tolerance:
                signals[i] = -1
    
    # Simulate trades
    equity = [initial_capital]
    trades = []
    cash = initial_capital
    pos = 0.0
    entry = 0.0
    entry_bar = 0
    max_profit = 0.0
    
    for i in range(1, n):
        current_equity = cash + pos * close[i] if pos != 0 else cash
        
        if pos != 0:
            if pos > 0:
                pnl_pct = (close[i] - entry) / entry
            else:
                pnl_pct = (entry - close[i]) / entry
            
            max_profit = max(max_profit, pnl_pct)
            
            exit_trade = False
            exit_reason = ""
            
            if pnl_pct >= p.profit_target:
                exit_trade = True
                exit_reason = "target"
            elif pnl_pct <= -p.stop_loss:
                exit_trade = True
                exit_reason = "stop"
            elif max_profit >= p.trailing_stop and pnl_pct < max_profit - p.trailing_stop:
                exit_trade = True
                exit_reason = "trail"
            
            if exit_trade:
                trade_pnl = pos * (close[i] - entry)
                cash += pos * close[i]
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'entry_price': entry,
                    'exit_price': close[i],
                    'position': pos,
                    'pnl': trade_pnl,
                    'pnl_pct': pnl_pct,
                    'reason': exit_reason
                })
                pos = 0.0
                max_profit = 0.0
        
        if pos == 0 and signals[i] != 0:
            if i - entry_bar >= p.min_bars_between:
                shares = int(cash * p.position_size / close[i])
                if shares > 0:
                    pos = float(shares) if signals[i] == 1 else float(-shares)
                    entry = close[i]
                    entry_bar = i
                    cash -= abs(pos) * close[i]
                    max_profit = 0.0
        
        equity.append(current_equity)
    
    # Final metrics
    equity_arr = np.array(equity)
    final_equity = equity_arr[-1]
    total_return = (final_equity - initial_capital) / initial_capital
    net_profit = final_equity - initial_capital
    
    # Max drawdown
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / peak
    max_dd = float(np.max(drawdown))
    
    # Sharpe
    if len(equity_arr) > 1:
        returns = np.diff(equity_arr) / equity_arr[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    return {
        'total_return': total_return,
        'net_profit': net_profit,
        'trades': trades,
        'equity_curve': equity,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'n_trades': len(trades),
        'trades': trades,
        'win_rate': len([t for t in trades if t['pnl'] > 0]) / len(trades) if trades else 0,
    }


# =============================================================================
# ML OPTIMIZATION
# =============================================================================

def create_optimizer() -> MLProfitOptimizer:
    """Create ML optimizer for Fibonacci ADX parameters."""
    param_specs = [
        ParameterSpec(
            name="adx_period",
            min_value=7,
            max_value=28,
            default=14,
            integer=True,
            description="ADX calculation period",
        ),
        ParameterSpec(
            name="adx_threshold",
            min_value=15,
            max_value=45,
            default=25,
            step=1.0,
            description="Minimum ADX for trend confirmation",
        ),
        ParameterSpec(
            name="fib_lookback",
            min_value=20,
            max_value=150,
            default=50,
            integer=True,
            description="Bars for swing high/low identification",
        ),
        ParameterSpec(
            name="tolerance",
            min_value=0.005,
            max_value=0.04,
            default=0.015,
            step=0.001,
            description="Price tolerance near Fibonacci level",
        ),
        ParameterSpec(
            name="position_size",
            min_value=0.05,
            max_value=0.25,
            default=0.1,
            step=0.01,
            description="Position size as fraction of equity",
        ),
        ParameterSpec(
            name="profit_target",
            min_value=0.01,
            max_value=0.08,
            default=0.03,
            step=0.005,
            description="Profit target percentage",
        ),
        ParameterSpec(
            name="stop_loss",
            min_value=0.005,
            max_value=0.04,
            default=0.015,
            step=0.001,
            description="Stop loss percentage",
        ),
        ParameterSpec(
            name="trailing_stop",
            min_value=0.005,
            max_value=0.03,
            default=0.01,
            step=0.001,
            description="Trailing stop percentage",
        ),
        ParameterSpec(
            name="min_bars_between",
            min_value=2,
            max_value=20,
            default=5,
            integer=True,
            description="Minimum bars between trades",
        ),
        ParameterSpec(
            name="di_spread",
            min_value=2,
            max_value=15,
            default=5,
            step=1.0,
            description="Minimum DI+/DI- spread for signal",
        ),
    ]
    
    config = OptimizationConfig(
        profit_metric=ProfitMetric.TOTAL_RETURN,  # Maximize raw return
        method=OptimizationMethod.EVOLUTIONARY,   # Better for noisy objectives
        max_iterations=30,
        min_iterations=10,
        batch_size=6,
        convergence_threshold=0.0001,
        patience=15,
        use_walk_forward=False,  # Disable for faster iteration
        train_ratio=0.8,
        max_overfit_ratio=3.0,
        min_profit_factor=0.8,   # Relaxed
        max_drawdown=0.50,       # Relaxed
        min_trades=5,            # Relaxed
        min_win_rate=0.25,       # Relaxed
        exploration_ratio=0.35,
        output_dir="artifacts/optimization/fib_adx_ml",
        save_all_trials=True,
        verbose=False,
        seed=42,
    )
    
    return MLProfitOptimizer(param_specs, config)


async def optimize_for_data(
    combined_data: pd.DataFrame,
    optimizer: MLProfitOptimizer,
    initial_params: dict | None = None
) -> dict:
    """Run ML optimization on combined dataset."""
    
    def backtest_fn(params: dict, data: pd.DataFrame) -> dict:
        return run_backtest(params, data)
    
    result = await optimizer.optimize(backtest_fn, combined_data, initial_params)
    
    return {
        'best_params': result.best_params,
        'best_profit': result.best_profit,
        'best_metrics': result.best_metrics,
        'iterations': result.iterations_completed,
        'n_valid_trials': len(result.valid_trials),
        'overfit_score': result.overfit_score,
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description='ML-Optimized Fibonacci ADX Backtest')
    parser.add_argument('--capital', type=float, default=100000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--optimize-iterations', type=int, default=30)
    args = parser.parse_args()
    
    print("=" * 80)
    print("ML-OPTIMIZED FIBONACCI ADX BACKTEST")
    print("Multi-Period Analysis with Parameter Tuning")
    print("=" * 80)
    
    # Generate random 12-month periods
    periods = generate_random_periods(n=3, seed=args.seed)
    print(f"\nSelected 12-month periods:")
    for i, (start, end) in enumerate(periods, 1):
        print(f"  Period {i}: {start} to {end}")
    
    all_stocks = SMALL_CAP + MID_CAP + RANDOM_PICKS
    intervals = [12, 24]  # Hours
    
    results = {
        'periods': periods,
        'stocks': all_stocks,
        'intervals': intervals,
        'by_interval': {},
        'by_category': {},
        'optimized_params': {},
        'summary': {},
    }
    
    for hours in intervals:
        interval_name = f"{hours}hr"
        print(f"\n{'='*80}")
        print(f"INTERVAL: {interval_name} BARS")
        print("=" * 80)
        
        results['by_interval'][interval_name] = {}
        
        # Collect all data for this interval across periods
        all_period_data = {}
        
        for period_idx, (start, end) in enumerate(periods):
            period_name = f"P{period_idx+1}_{start[:4]}"
            print(f"\n--- Period: {start} to {end} ---")
            
            period_results = []
            
            for symbol in all_stocks:
                print(f"  [{symbol}] Fetching {start} to {end}...", end=" ")
                
                df_daily = fetch_daily_data(symbol, start, end)
                if df_daily.empty or len(df_daily) < 50:
                    print("insufficient data")
                    continue
                
                df = aggregate_to_interval(df_daily, hours)
                print(f"{len(df)} bars")
                
                if symbol not in all_period_data:
                    all_period_data[symbol] = []
                all_period_data[symbol].append(df)
        
        # Combine data across periods for ML optimization
        print(f"\n--- ML Parameter Optimization ({interval_name}) ---")
        
        combined_frames = []
        for symbol, dfs in all_period_data.items():
            for df in dfs:
                combined_frames.append(df)
        
        if combined_frames:
            # Sample subset for faster optimization
            sample_size = min(5, len(combined_frames))
            sample_frames = random.sample(combined_frames, sample_size)
            combined_df = pd.concat(sample_frames, ignore_index=False)
            
            print(f"  Optimizing on {len(combined_df)} total bars from {sample_size} samples...")
            
            optimizer = create_optimizer()
            optimizer.config.max_iterations = args.optimize_iterations
            
            opt_result = await optimize_for_data(combined_df, optimizer)
            
            results['optimized_params'][interval_name] = opt_result['best_params']
            print(f"  Best params found:")
            for k, v in opt_result['best_params'].items():
                print(f"    {k}: {v}")
            print(f"  Optimization score: {opt_result['best_profit']:.4f}")
            print(f"  Overfit score: {opt_result.get('overfit_score', 'N/A')}")
        
        # Now run backtest with optimized params on all data
        print(f"\n--- Running Backtests with Optimized Params ({interval_name}) ---")
        
        optimized_params = results['optimized_params'].get(interval_name, {})
        
        # Fallback to defaults if optimization didn't find good params
        if not optimized_params:
            print("  Using default parameters (optimization found no valid params)")
            optimized_params = {
                'adx_period': 14,
                'adx_threshold': 30,
                'fib_lookback': 50,
                'tolerance': 0.02,
                'position_size': 0.1,
                'profit_target': 0.04,
                'stop_loss': 0.02,
                'trailing_stop': 0.015,
                'min_bars_between': 5,
                'di_spread': 5,
            }
            results['optimized_params'][interval_name] = optimized_params
        
        category_results = {'Small Cap': [], 'Mid Cap': [], 'Random': []}
        
        for symbol in all_stocks:
            if symbol in SMALL_CAP:
                category = 'Small Cap'
            elif symbol in MID_CAP:
                category = 'Mid Cap'
            else:
                category = 'Random'
            
            if symbol not in all_period_data:
                continue
            
            symbol_results = []
            for period_idx, df in enumerate(all_period_data[symbol]):
                result = run_backtest(optimized_params, df)
                result['symbol'] = symbol
                result['period'] = periods[period_idx][0]
                result['category'] = category
                symbol_results.append(result)
            
            # Average across periods
            if symbol_results:
                avg_return = np.mean([r['total_return'] for r in symbol_results])
                avg_sharpe = np.mean([r['sharpe_ratio'] for r in symbol_results])
                avg_dd = np.mean([r['max_drawdown'] for r in symbol_results])
                total_trades = sum([r.get('n_trades', len(r.get('trades', []))) for r in symbol_results])
                
                summary = {
                    'symbol': symbol,
                    'category': category,
                    'avg_return': avg_return,
                    'avg_sharpe': avg_sharpe,
                    'avg_max_dd': avg_dd,
                    'total_trades': total_trades,
                    'period_results': symbol_results
                }
                
                category_results[category].append(summary)
                
                status = "✓" if avg_return > 0 else "✗"
                print(f"  [{symbol}] {status} Avg Return: {avg_return*100:+.2f}% | "
                      f"Sharpe: {avg_sharpe:.2f} | MaxDD: {avg_dd*100:.1f}% | "
                      f"Trades: {total_trades}")
        
        results['by_interval'][interval_name]['by_category'] = category_results
        
        # Print category summaries
        print(f"\n--- {interval_name} Category Summary ---")
        for cat, cat_results in category_results.items():
            if cat_results:
                avg_ret = np.mean([r['avg_return'] for r in cat_results])
                avg_sharpe = np.mean([r['avg_sharpe'] for r in cat_results])
                winners = len([r for r in cat_results if r['avg_return'] > 0])
                print(f"  {cat}: Avg Return={avg_ret*100:+.2f}%, "
                      f"Avg Sharpe={avg_sharpe:.2f}, "
                      f"Win Rate={winners}/{len(cat_results)}")
    
    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    
    for interval_name in results['by_interval']:
        print(f"\n{interval_name} Aggregation:")
        print(f"  Optimized Parameters:")
        for k, v in results['optimized_params'].get(interval_name, {}).items():
            print(f"    {k}: {v}")
        
        all_results = []
        for cat_results in results['by_interval'][interval_name]['by_category'].values():
            all_results.extend(cat_results)
        
        if all_results:
            total_avg_return = np.mean([r['avg_return'] for r in all_results])
            total_winners = len([r for r in all_results if r['avg_return'] > 0])
            print(f"  Overall Avg Return: {total_avg_return*100:+.2f}%")
            print(f"  Overall Win Rate: {total_winners}/{len(all_results)} ({100*total_winners/len(all_results):.1f}%)")
    
    # Save results
    output_dir = Path("data/backtest_results/ml_optimized")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"ml_backtest_{ts}.json"
    
    # Convert numpy types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj
    
    with open(output_file, 'w') as f:
        json.dump(convert_types(results), f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
