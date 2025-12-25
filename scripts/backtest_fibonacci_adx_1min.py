#!/usr/bin/env python3
"""
GPU-Accelerated Fibonacci ADX Backtest - 1-Minute Data
Uses yfinance for data (no API key required)
"""

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Try GPU imports
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

# Stock selections
SMALL_CAP_STOCKS = [
    # Small caps across sectors
    "CPRX",   # Healthcare - Catalyst Pharma
    "RCUS",   # Healthcare - Arcus Biosciences  
    "PAYO",   # Financials - Payoneer
    "ESLT",   # Industrials - Elbit Systems
    "CEIX",   # Energy - CONSOL Energy
    "PRGS",   # Technology - Progress Software
    "PGNY",   # Healthcare - Progyny
    "CALX",   # Technology - Calix
    "SPSC",   # Technology - SPS Commerce
    "APOG",   # Industrials - Apogee Enterprises
]

MID_CAP_STOCKS = [
    # One from each GICS sector
    "WSM",    # Consumer Discretionary - Williams-Sonoma
    "SIRI",   # Communication Services - Sirius XM
    "TRGP",   # Energy - Targa Resources
    "RJF",    # Financials - Raymond James
    "HOLX",   # Healthcare - Hologic
    "EME",    # Industrials - EMCOR Group
    "FFIV",   # Technology - F5 Networks
    "AVY",    # Materials - Avery Dennison
    "NRG",    # Utilities - NRG Energy
    "ULTA",   # Consumer Staples - Ulta Beauty
]

RANDOM_STOCKS = [
    # Surprise picks - mix of interesting names
    "PLTR",   # Palantir - AI/Defense
    "RKLB",   # Rocket Lab - Space
    "SOFI",   # SoFi - Fintech
    "RIVN",   # Rivian - EV
    "DKNG",   # DraftKings - Gaming
    "CRWD",   # CrowdStrike - Cybersecurity
    "ZS",     # Zscaler - Cloud Security
    "SNOW",   # Snowflake - Data Cloud
    "MARA",   # Marathon Digital - Crypto Mining
    "IONQ",   # IonQ - Quantum Computing
]


def fetch_1min_data(symbol: str, days: int = 7) -> pd.DataFrame:
    """Fetch 1-minute data from yfinance (max 7 days for 1min)"""
    try:
        ticker = yf.Ticker(symbol)
        # yfinance allows max 7 days for 1-min data
        df = ticker.history(period=f"{days}d", interval="1m")
        if df.empty:
            print(f"  [!] No data for {symbol}")
            return pd.DataFrame()
        
        # Standardize columns
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df = df.dropna()
        return df
    except Exception as e:
        print(f"  [!] Error fetching {symbol}: {e}")
        return pd.DataFrame()


# GPU Kernels
if GPU_AVAILABLE:
    @cuda.jit
    def compute_tr_kernel(high, low, close, tr, n):
        """True Range on GPU"""
        i = cuda.grid(1)
        if i > 0 and i < n:
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, max(hc, lc))
        elif i == 0:
            tr[i] = high[i] - low[i]

    @cuda.jit
    def compute_dm_kernel(high, low, plus_dm, minus_dm, n):
        """Directional Movement on GPU"""
        i = cuda.grid(1)
        if i > 0 and i < n:
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            else:
                plus_dm[i] = 0.0
                
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            else:
                minus_dm[i] = 0.0
        elif i == 0:
            plus_dm[i] = 0.0
            minus_dm[i] = 0.0

    @cuda.jit
    def ema_kernel(data, result, alpha, n):
        """EMA using Wilder smoothing on GPU"""
        # Single thread for sequential EMA
        if cuda.grid(1) == 0:
            result[0] = data[0]
            for i in range(1, n):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]

    @cuda.jit
    def signal_kernel(close, adx, plus_di, minus_di, fib_382, fib_618, signals, n, adx_threshold):
        """Generate trading signals on GPU"""
        i = cuda.grid(1)
        if i >= 50 and i < n:  # Need warmup period
            # Long signal: ADX strong, +DI > -DI, price near fib support
            if adx[i] > adx_threshold and plus_di[i] > minus_di[i]:
                if close[i] <= fib_618[i] * 1.02 and close[i] >= fib_618[i] * 0.98:
                    signals[i] = 1  # Long
            # Short signal: ADX strong, -DI > +DI, price near fib resistance
            elif adx[i] > adx_threshold and minus_di[i] > plus_di[i]:
                if close[i] >= fib_382[i] * 0.98 and close[i] <= fib_382[i] * 1.02:
                    signals[i] = -1  # Short

    @cuda.jit
    def simulate_trades_kernel(close, signals, equity, positions, n, position_size):
        """Simulate trades on GPU"""
        if cuda.grid(1) == 0:
            pos = 0.0
            entry_price = 0.0
            cash = equity[0]
            
            for i in range(1, n):
                # Check for exit
                if pos != 0:
                    pnl = (close[i] - entry_price) * pos
                    # Simple exit: 2% profit or 1% loss
                    if pnl / (abs(pos) * entry_price) > 0.02 or pnl / (abs(pos) * entry_price) < -0.01:
                        cash += pos * close[i]
                        pos = 0.0
                        entry_price = 0.0
                
                # Check for entry
                if pos == 0 and signals[i] != 0:
                    shares = int(cash * position_size / close[i])
                    if shares > 0:
                        if signals[i] == 1:  # Long
                            pos = float(shares)
                        else:  # Short
                            pos = float(-shares)
                        entry_price = close[i]
                        cash -= abs(pos) * close[i]
                
                # Track equity
                if pos != 0:
                    equity[i] = cash + pos * close[i]
                else:
                    equity[i] = cash
                positions[i] = pos


def run_gpu_backtest(df: pd.DataFrame, initial_capital: float = 100000) -> dict:
    """Run full backtest on GPU"""
    n = len(df)
    
    # Transfer to GPU
    t0 = time.perf_counter()
    high_gpu = cp.asarray(df['high'].values, dtype=cp.float64)
    low_gpu = cp.asarray(df['low'].values, dtype=cp.float64)
    close_gpu = cp.asarray(df['close'].values, dtype=cp.float64)
    t_transfer = time.perf_counter() - t0
    
    # Compute True Range
    t0 = time.perf_counter()
    tr_gpu = cp.zeros(n, dtype=cp.float64)
    threads = 256
    blocks = (n + threads - 1) // threads
    compute_tr_kernel[blocks, threads](high_gpu, low_gpu, close_gpu, tr_gpu, n)
    cuda.synchronize()
    
    # Compute DM
    plus_dm_gpu = cp.zeros(n, dtype=cp.float64)
    minus_dm_gpu = cp.zeros(n, dtype=cp.float64)
    compute_dm_kernel[blocks, threads](high_gpu, low_gpu, plus_dm_gpu, minus_dm_gpu, n)
    cuda.synchronize()
    
    # Compute smoothed values (14-period Wilder)
    period = 14
    alpha = 1.0 / period
    
    atr_gpu = cp.zeros(n, dtype=cp.float64)
    smoothed_plus_dm = cp.zeros(n, dtype=cp.float64)
    smoothed_minus_dm = cp.zeros(n, dtype=cp.float64)
    
    ema_kernel[1, 1](tr_gpu, atr_gpu, alpha, n)
    ema_kernel[1, 1](plus_dm_gpu, smoothed_plus_dm, alpha, n)
    ema_kernel[1, 1](minus_dm_gpu, smoothed_minus_dm, alpha, n)
    cuda.synchronize()
    
    # Compute DI
    plus_di_gpu = cp.zeros(n, dtype=cp.float64)
    minus_di_gpu = cp.zeros(n, dtype=cp.float64)
    
    # Avoid division by zero
    atr_safe = cp.where(atr_gpu > 0, atr_gpu, 1.0)
    plus_di_gpu = 100.0 * smoothed_plus_dm / atr_safe
    minus_di_gpu = 100.0 * smoothed_minus_dm / atr_safe
    
    # Compute DX and ADX
    di_sum = plus_di_gpu + minus_di_gpu
    di_sum_safe = cp.where(di_sum > 0, di_sum, 1.0)
    dx_gpu = 100.0 * cp.abs(plus_di_gpu - minus_di_gpu) / di_sum_safe
    
    adx_gpu = cp.zeros(n, dtype=cp.float64)
    ema_kernel[1, 1](dx_gpu, adx_gpu, alpha, n)
    cuda.synchronize()
    t_adx = time.perf_counter() - t0
    
    # Compute Fibonacci levels (rolling 50-period high/low)
    t0 = time.perf_counter()
    lookback = 50
    fib_382_gpu = cp.zeros(n, dtype=cp.float64)
    fib_618_gpu = cp.zeros(n, dtype=cp.float64)
    
    # Use CuPy for rolling max/min
    for i in range(lookback, n):
        period_high = cp.max(high_gpu[i-lookback:i])
        period_low = cp.min(low_gpu[i-lookback:i])
        range_val = period_high - period_low
        fib_382_gpu[i] = period_high - 0.382 * range_val
        fib_618_gpu[i] = period_high - 0.618 * range_val
    
    cuda.synchronize()
    t_fib = time.perf_counter() - t0
    
    # Generate signals
    t0 = time.perf_counter()
    signals_gpu = cp.zeros(n, dtype=cp.int32)
    signal_kernel[blocks, threads](
        close_gpu, adx_gpu, plus_di_gpu, minus_di_gpu,
        fib_382_gpu, fib_618_gpu, signals_gpu, n, 25.0
    )
    cuda.synchronize()
    t_signals = time.perf_counter() - t0
    
    # Simulate trades
    t0 = time.perf_counter()
    equity_gpu = cp.zeros(n, dtype=cp.float64)
    equity_gpu[0] = initial_capital
    positions_gpu = cp.zeros(n, dtype=cp.float64)
    
    simulate_trades_kernel[1, 1](
        close_gpu, signals_gpu, equity_gpu, positions_gpu, n, 0.1
    )
    cuda.synchronize()
    t_sim = time.perf_counter() - t0
    
    # Copy results back
    t0 = time.perf_counter()
    equity = cp.asnumpy(equity_gpu)
    signals = cp.asnumpy(signals_gpu)
    t_copy = time.perf_counter() - t0
    
    # Compute metrics on GPU
    t0 = time.perf_counter()
    final_equity = float(equity[-1]) if equity[-1] > 0 else float(equity[equity > 0][-1]) if len(equity[equity > 0]) > 0 else initial_capital
    total_return = (final_equity - initial_capital) / initial_capital * 100
    
    # Returns for Sharpe
    equity_valid = equity[equity > 0]
    if len(equity_valid) > 1:
        returns = np.diff(equity_valid) / equity_valid[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 390)  # 390 mins/day
    else:
        sharpe = 0.0
    
    # Max drawdown
    running_max = np.maximum.accumulate(equity_valid) if len(equity_valid) > 0 else np.array([initial_capital])
    drawdown = (running_max - equity_valid) / running_max if len(equity_valid) > 0 else np.array([0])
    max_dd = float(np.max(drawdown)) * 100 if len(drawdown) > 0 else 0.0
    
    signal_count = int(np.sum(np.abs(signals)))
    t_metrics = time.perf_counter() - t0
    
    return {
        'final_equity': final_equity,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'signals': signal_count,
        'bars': n,
        'timing': {
            'transfer': t_transfer,
            'adx': t_adx,
            'fibonacci': t_fib,
            'signals': t_signals,
            'simulation': t_sim,
            'copy': t_copy,
            'metrics': t_metrics
        }
    }


def run_cpu_backtest(df: pd.DataFrame, initial_capital: float = 100000) -> dict:
    """Fallback CPU backtest"""
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    # Simple ADX calculation
    period = 14
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smoothed TR (ATR)
    atr = np.zeros(n)
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    # +DM, -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0
    
    # Smoothed DM
    smooth_plus = np.zeros(n)
    smooth_minus = np.zeros(n)
    smooth_plus[period-1] = np.mean(plus_dm[:period])
    smooth_minus[period-1] = np.mean(minus_dm[:period])
    for i in range(period, n):
        smooth_plus[i] = (smooth_plus[i-1] * (period - 1) + plus_dm[i]) / period
        smooth_minus[i] = (smooth_minus[i-1] * (period - 1) + minus_dm[i]) / period
    
    # DI
    plus_di = np.where(atr > 0, 100 * smooth_plus / atr, 0)
    minus_di = np.where(atr > 0, 100 * smooth_minus / atr, 0)
    
    # DX and ADX
    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 0, 100 * np.abs(plus_di - minus_di) / di_sum, 0)
    adx = np.zeros(n)
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # Fibonacci levels
    lookback = 50
    fib_382 = np.zeros(n)
    fib_618 = np.zeros(n)
    for i in range(lookback, n):
        ph = np.max(high[i-lookback:i])
        pl = np.min(low[i-lookback:i])
        r = ph - pl
        fib_382[i] = ph - 0.382 * r
        fib_618[i] = ph - 0.618 * r
    
    # Signals
    signals = np.zeros(n, dtype=int)
    for i in range(lookback, n):
        if adx[i] > 25:
            if plus_di[i] > minus_di[i] and abs(close[i] - fib_618[i]) / fib_618[i] < 0.02:
                signals[i] = 1
            elif minus_di[i] > plus_di[i] and abs(close[i] - fib_382[i]) / fib_382[i] < 0.02:
                signals[i] = -1
    
    # Simulate
    equity = np.zeros(n)
    equity[0] = initial_capital
    cash = initial_capital
    pos = 0.0
    entry = 0.0
    
    for i in range(1, n):
        if pos != 0:
            pnl = (close[i] - entry) * pos
            if pnl / (abs(pos) * entry) > 0.02 or pnl / (abs(pos) * entry) < -0.01:
                cash += pos * close[i]
                pos = 0.0
        
        if pos == 0 and signals[i] != 0:
            shares = int(cash * 0.1 / close[i])
            if shares > 0:
                pos = float(shares) if signals[i] == 1 else float(-shares)
                entry = close[i]
                cash -= abs(pos) * close[i]
        
        equity[i] = cash + pos * close[i] if pos != 0 else cash
    
    final_eq = equity[-1] if equity[-1] > 0 else initial_capital
    returns_pct = (final_eq - initial_capital) / initial_capital * 100
    
    eq_valid = equity[equity > 0]
    if len(eq_valid) > 1:
        rets = np.diff(eq_valid) / eq_valid[:-1]
        sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252 * 390)
    else:
        sharpe = 0.0
    
    running_max = np.maximum.accumulate(eq_valid)
    dd = (running_max - eq_valid) / running_max
    max_dd = float(np.max(dd)) * 100
    
    return {
        'final_equity': final_eq,
        'total_return': returns_pct,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'signals': int(np.sum(np.abs(signals))),
        'bars': n
    }


def main():
    parser = argparse.ArgumentParser(description='Fibonacci ADX Backtest - 1min Data')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--days', type=int, default=7, help='Days of 1-min data (max 7)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("GPU-ACCELERATED FIBONACCI ADX BACKTEST - 1 MINUTE DATA")
    print("=" * 70)
    print(f"\nFetching data from yfinance (no API key required)")
    print(f"Period: {args.days} days of 1-minute bars\n")
    
    all_symbols = {
        'Small Cap': SMALL_CAP_STOCKS,
        'Mid Cap': MID_CAP_STOCKS,
        'Random Picks': RANDOM_STOCKS
    }
    
    all_results = {}
    total_bars = 0
    total_time = 0
    
    for category, symbols in all_symbols.items():
        print("=" * 70)
        print(f"  {category.upper()} STOCKS")
        print("=" * 70)
        
        category_results = []
        
        for symbol in symbols:
            print(f"\n[{symbol}] Fetching 1-min data...")
            df = fetch_1min_data(symbol, args.days)
            
            if df.empty or len(df) < 100:
                print(f"  [!] Insufficient data for {symbol}, skipping")
                continue
            
            print(f"  [{symbol}] {len(df)} bars loaded")
            
            if GPU_AVAILABLE:
                print(f"  [{symbol}] Running GPU backtest...")
                t0 = time.perf_counter()
                result = run_gpu_backtest(df, args.capital)
                elapsed = time.perf_counter() - t0
            else:
                print(f"  [{symbol}] Running CPU backtest...")
                t0 = time.perf_counter()
                result = run_cpu_backtest(df, args.capital)
                elapsed = time.perf_counter() - t0
            
            result['symbol'] = symbol
            result['category'] = category
            result['elapsed'] = elapsed
            category_results.append(result)
            
            total_bars += result['bars']
            total_time += elapsed
            
            # Print result
            print(f"  [{symbol}] Return: {result['total_return']:+.2f}% | "
                  f"Sharpe: {result['sharpe_ratio']:.2f} | "
                  f"MaxDD: {result['max_drawdown']:.2f}% | "
                  f"Signals: {result['signals']}")
        
        all_results[category] = category_results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)
    
    for category, results in all_results.items():
        if not results:
            continue
        avg_return = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_dd = np.mean([r['max_drawdown'] for r in results])
        print(f"\n{category}:")
        print(f"  Avg Return: {avg_return:+.2f}%")
        print(f"  Avg Sharpe: {avg_sharpe:.2f}")
        print(f"  Avg MaxDD:  {avg_dd:.2f}%")
    
    print("\n" + "=" * 70)
    print("OVERALL STATISTICS")
    print("=" * 70)
    all_flat = [r for cat in all_results.values() for r in cat]
    if all_flat:
        print(f"Total Symbols Tested: {len(all_flat)}")
        print(f"Total Bars Processed: {total_bars:,}")
        print(f"Total Processing Time: {total_time:.2f}s")
        print(f"Throughput: {total_bars / total_time:,.0f} bars/second")
        
        winners = [r for r in all_flat if r['total_return'] > 0]
        print(f"\nWin Rate: {len(winners)}/{len(all_flat)} ({100*len(winners)/len(all_flat):.1f}%)")
        
        if winners:
            best = max(all_flat, key=lambda x: x['total_return'])
            worst = min(all_flat, key=lambda x: x['total_return'])
            print(f"Best:  {best['symbol']} ({best['category']}) +{best['total_return']:.2f}%")
            print(f"Worst: {worst['symbol']} ({worst['category']}) {worst['total_return']:.2f}%")
    
    # Save results
    output_dir = Path("data/backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = output_dir / f"1min_backtest_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'initial_capital': args.capital,
                'days': args.days,
                'interval': '1m',
                'gpu_used': GPU_AVAILABLE
            },
            'results': all_results,
            'summary': {
                'total_symbols': len(all_flat) if all_flat else 0,
                'total_bars': total_bars,
                'total_time': total_time,
                'win_rate': len(winners) / len(all_flat) if all_flat else 0
            }
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
