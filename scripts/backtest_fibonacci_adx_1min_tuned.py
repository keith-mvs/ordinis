#!/usr/bin/env python3
"""
GPU-Accelerated Fibonacci ADX Backtest - 1-Minute Data (TUNED)
Properly calibrated parameters for intraday trading
"""

import argparse
import json
import time
from datetime import datetime
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

# =============================================================================
# INTRADAY-TUNED PARAMETERS
# =============================================================================
PARAMS = {
    'adx_period': 14,
    'adx_threshold': 35,           # Higher threshold for intraday noise
    'fib_lookback': 390,           # Full trading day (~6.5 hours)
    'min_bars_between_trades': 30, # Wait 30 mins between trades
    'position_size': 0.05,         # 5% per trade (reduced)
    'profit_target': 0.005,        # 0.5% profit target
    'stop_loss': 0.003,            # 0.3% stop loss
    'trailing_stop': 0.002,        # 0.2% trailing stop after 0.3% profit
    'volume_filter': True,         # Only trade during high volume
}

# Stock selections
SMALL_CAP_STOCKS = [
    "CPRX", "RCUS", "PAYO", "ESLT", "PRGS",
    "PGNY", "CALX", "SPSC", "APOG", "GERN"
]

MID_CAP_STOCKS = [
    "WSM", "SIRI", "TRGP", "RJF", "HOLX",
    "EME", "FFIV", "AVY", "NRG", "ULTA"
]

RANDOM_STOCKS = [
    "PLTR", "RKLB", "SOFI", "RIVN", "DKNG",
    "CRWD", "ZS", "SNOW", "MARA", "IONQ"
]


def fetch_1min_data(symbol: str, days: int = 7) -> pd.DataFrame:
    """Fetch 1-minute data from yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{days}d", interval="1m")
        if df.empty:
            print(f"  [!] No data for {symbol}")
            return pd.DataFrame()
        
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df = df.dropna()
        return df
    except Exception as e:
        print(f"  [!] Error fetching {symbol}: {e}")
        return pd.DataFrame()


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
        if cuda.grid(1) == 0:
            result[0] = data[0]
            for i in range(1, n):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]

    @cuda.jit
    def intraday_signal_kernel(
        close, volume, adx, plus_di, minus_di, 
        fib_382, fib_618, fib_500, avg_volume,
        signals, n, adx_threshold, vol_mult
    ):
        """Generate signals with intraday filters"""
        i = cuda.grid(1)
        lookback = 390  # Full day
        
        if i >= lookback and i < n:
            # Volume filter - only trade above average volume
            if volume[i] < avg_volume[i] * vol_mult:
                return
            
            # ADX must be strong (trending)
            if adx[i] < adx_threshold:
                return
            
            # Long: Strong uptrend, price at 61.8% support
            if plus_di[i] > minus_di[i] + 5:  # Clear directional bias
                if close[i] <= fib_618[i] * 1.005 and close[i] >= fib_618[i] * 0.995:
                    signals[i] = 1
            
            # Short: Strong downtrend, price at 38.2% resistance
            elif minus_di[i] > plus_di[i] + 5:
                if close[i] >= fib_382[i] * 0.995 and close[i] <= fib_382[i] * 1.005:
                    signals[i] = -1

    @cuda.jit
    def intraday_simulate_kernel(
        close, signals, equity, positions, trades,
        n, position_size, profit_target, stop_loss, trailing_pct,
        min_bars_between
    ):
        """Simulate trades with proper risk management"""
        if cuda.grid(1) == 0:
            pos = 0.0
            entry_price = 0.0
            cash = equity[0]
            last_trade_bar = -min_bars_between
            max_profit = 0.0
            trade_count = 0
            
            for i in range(1, n):
                # Check for exit if in position
                if pos != 0:
                    if pos > 0:
                        pnl_pct = (close[i] - entry_price) / entry_price
                    else:
                        pnl_pct = (entry_price - close[i]) / entry_price
                    
                    # Track max profit for trailing stop
                    if pnl_pct > max_profit:
                        max_profit = pnl_pct
                    
                    # Exit conditions
                    exit_trade = False
                    
                    # Hit profit target
                    if pnl_pct >= profit_target:
                        exit_trade = True
                    # Hit stop loss
                    elif pnl_pct <= -stop_loss:
                        exit_trade = True
                    # Trailing stop (after reaching 0.3% profit)
                    elif max_profit >= 0.003 and pnl_pct < max_profit - trailing_pct:
                        exit_trade = True
                    
                    if exit_trade:
                        cash += pos * close[i]
                        pos = 0.0
                        entry_price = 0.0
                        max_profit = 0.0
                        trade_count += 1
                
                # Check for entry
                if pos == 0 and signals[i] != 0:
                    # Enforce minimum bars between trades
                    if i - last_trade_bar >= min_bars_between:
                        shares = int(cash * position_size / close[i])
                        if shares > 0:
                            if signals[i] == 1:
                                pos = float(shares)
                            else:
                                pos = float(-shares)
                            entry_price = close[i]
                            cash -= abs(pos) * close[i]
                            last_trade_bar = i
                            max_profit = 0.0
                
                # Track equity
                if pos != 0:
                    equity[i] = cash + pos * close[i]
                else:
                    equity[i] = cash
                positions[i] = pos
            
            trades[0] = trade_count


def run_gpu_backtest(df: pd.DataFrame, initial_capital: float = 100000) -> dict:
    """Run intraday-tuned backtest on GPU"""
    n = len(df)
    params = PARAMS
    
    # Transfer to GPU
    t0 = time.perf_counter()
    high_gpu = cp.asarray(df['high'].values, dtype=cp.float64)
    low_gpu = cp.asarray(df['low'].values, dtype=cp.float64)
    close_gpu = cp.asarray(df['close'].values, dtype=cp.float64)
    volume_gpu = cp.asarray(df['volume'].values, dtype=cp.float64)
    t_transfer = time.perf_counter() - t0
    
    # Compute TR and DM
    t0 = time.perf_counter()
    tr_gpu = cp.zeros(n, dtype=cp.float64)
    threads = 256
    blocks = (n + threads - 1) // threads
    compute_tr_kernel[blocks, threads](high_gpu, low_gpu, close_gpu, tr_gpu, n)
    cuda.synchronize()
    
    plus_dm_gpu = cp.zeros(n, dtype=cp.float64)
    minus_dm_gpu = cp.zeros(n, dtype=cp.float64)
    compute_dm_kernel[blocks, threads](high_gpu, low_gpu, plus_dm_gpu, minus_dm_gpu, n)
    cuda.synchronize()
    
    # Smoothed values
    period = params['adx_period']
    alpha = 1.0 / period
    
    atr_gpu = cp.zeros(n, dtype=cp.float64)
    smoothed_plus_dm = cp.zeros(n, dtype=cp.float64)
    smoothed_minus_dm = cp.zeros(n, dtype=cp.float64)
    
    ema_kernel[1, 1](tr_gpu, atr_gpu, alpha, n)
    ema_kernel[1, 1](plus_dm_gpu, smoothed_plus_dm, alpha, n)
    ema_kernel[1, 1](minus_dm_gpu, smoothed_minus_dm, alpha, n)
    cuda.synchronize()
    
    # DI and ADX
    atr_safe = cp.where(atr_gpu > 0, atr_gpu, 1.0)
    plus_di_gpu = 100.0 * smoothed_plus_dm / atr_safe
    minus_di_gpu = 100.0 * smoothed_minus_dm / atr_safe
    
    di_sum = plus_di_gpu + minus_di_gpu
    di_sum_safe = cp.where(di_sum > 0, di_sum, 1.0)
    dx_gpu = 100.0 * cp.abs(plus_di_gpu - minus_di_gpu) / di_sum_safe
    
    adx_gpu = cp.zeros(n, dtype=cp.float64)
    ema_kernel[1, 1](dx_gpu, adx_gpu, alpha, n)
    cuda.synchronize()
    t_adx = time.perf_counter() - t0
    
    # Fibonacci levels (full day lookback)
    t0 = time.perf_counter()
    lookback = params['fib_lookback']
    fib_382_gpu = cp.zeros(n, dtype=cp.float64)
    fib_618_gpu = cp.zeros(n, dtype=cp.float64)
    fib_500_gpu = cp.zeros(n, dtype=cp.float64)
    
    for i in range(lookback, n):
        period_high = cp.max(high_gpu[i-lookback:i])
        period_low = cp.min(low_gpu[i-lookback:i])
        range_val = period_high - period_low
        fib_382_gpu[i] = period_high - 0.382 * range_val
        fib_500_gpu[i] = period_high - 0.500 * range_val
        fib_618_gpu[i] = period_high - 0.618 * range_val
    
    cuda.synchronize()
    t_fib = time.perf_counter() - t0
    
    # Average volume (rolling)
    t0 = time.perf_counter()
    avg_volume_gpu = cp.zeros(n, dtype=cp.float64)
    vol_lookback = 60  # 1 hour average
    for i in range(vol_lookback, n):
        avg_volume_gpu[i] = cp.mean(volume_gpu[i-vol_lookback:i])
    
    # Generate signals
    signals_gpu = cp.zeros(n, dtype=cp.int32)
    intraday_signal_kernel[blocks, threads](
        close_gpu, volume_gpu, adx_gpu, plus_di_gpu, minus_di_gpu,
        fib_382_gpu, fib_618_gpu, fib_500_gpu, avg_volume_gpu,
        signals_gpu, n, params['adx_threshold'], 0.8  # 80% of avg volume min
    )
    cuda.synchronize()
    t_signals = time.perf_counter() - t0
    
    # Simulate trades
    t0 = time.perf_counter()
    equity_gpu = cp.zeros(n, dtype=cp.float64)
    equity_gpu[0] = initial_capital
    positions_gpu = cp.zeros(n, dtype=cp.float64)
    trades_gpu = cp.zeros(1, dtype=cp.int32)
    
    intraday_simulate_kernel[1, 1](
        close_gpu, signals_gpu, equity_gpu, positions_gpu, trades_gpu,
        n, params['position_size'], params['profit_target'],
        params['stop_loss'], params['trailing_stop'],
        params['min_bars_between_trades']
    )
    cuda.synchronize()
    t_sim = time.perf_counter() - t0
    
    # Copy results
    t0 = time.perf_counter()
    equity = cp.asnumpy(equity_gpu)
    signals = cp.asnumpy(signals_gpu)
    num_trades = int(cp.asnumpy(trades_gpu)[0])
    t_copy = time.perf_counter() - t0
    
    # Metrics
    t0 = time.perf_counter()
    equity_valid = equity[equity > 0]
    if len(equity_valid) == 0:
        equity_valid = np.array([initial_capital])
    
    final_equity = float(equity_valid[-1])
    total_return = (final_equity - initial_capital) / initial_capital * 100
    
    if len(equity_valid) > 1:
        returns = np.diff(equity_valid) / equity_valid[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 390)
    else:
        sharpe = 0.0
    
    running_max = np.maximum.accumulate(equity_valid)
    drawdown = (running_max - equity_valid) / running_max
    max_dd = float(np.max(drawdown)) * 100
    
    signal_count = int(np.sum(np.abs(signals)))
    t_metrics = time.perf_counter() - t0
    
    return {
        'final_equity': final_equity,
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'signals': signal_count,
        'trades': num_trades,
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
    """CPU fallback with same logic"""
    n = len(df)
    params = PARAMS
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    volume = df['volume'].values
    
    # ADX
    period = params['adx_period']
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
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
    adx[2*period-1] = np.mean(dx[period:2*period])
    for i in range(2*period, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    # Fibonacci
    lookback = params['fib_lookback']
    fib_382 = np.zeros(n)
    fib_618 = np.zeros(n)
    for i in range(lookback, n):
        ph = np.max(high[i-lookback:i])
        pl = np.min(low[i-lookback:i])
        r = ph - pl
        fib_382[i] = ph - 0.382 * r
        fib_618[i] = ph - 0.618 * r
    
    # Volume average
    vol_lookback = 60
    avg_vol = np.zeros(n)
    for i in range(vol_lookback, n):
        avg_vol[i] = np.mean(volume[i-vol_lookback:i])
    
    # Signals
    signals = np.zeros(n, dtype=int)
    for i in range(lookback, n):
        if volume[i] < avg_vol[i] * 0.8:
            continue
        if adx[i] < params['adx_threshold']:
            continue
        
        if plus_di[i] > minus_di[i] + 5:
            if abs(close[i] - fib_618[i]) / fib_618[i] < 0.005:
                signals[i] = 1
        elif minus_di[i] > plus_di[i] + 5:
            if abs(close[i] - fib_382[i]) / fib_382[i] < 0.005:
                signals[i] = -1
    
    # Simulate
    equity = np.zeros(n)
    equity[0] = initial_capital
    cash = initial_capital
    pos = 0.0
    entry = 0.0
    last_trade = -params['min_bars_between_trades']
    max_profit = 0.0
    num_trades = 0
    
    for i in range(1, n):
        if pos != 0:
            pnl_pct = (close[i] - entry) / entry * (1 if pos > 0 else -1)
            max_profit = max(max_profit, pnl_pct)
            
            exit_trade = False
            if pnl_pct >= params['profit_target']:
                exit_trade = True
            elif pnl_pct <= -params['stop_loss']:
                exit_trade = True
            elif max_profit >= 0.003 and pnl_pct < max_profit - params['trailing_stop']:
                exit_trade = True
            
            if exit_trade:
                cash += pos * close[i]
                pos = 0.0
                max_profit = 0.0
                num_trades += 1
        
        if pos == 0 and signals[i] != 0:
            if i - last_trade >= params['min_bars_between_trades']:
                shares = int(cash * params['position_size'] / close[i])
                if shares > 0:
                    pos = float(shares) if signals[i] == 1 else float(-shares)
                    entry = close[i]
                    cash -= abs(pos) * close[i]
                    last_trade = i
                    max_profit = 0.0
        
        equity[i] = cash + pos * close[i] if pos != 0 else cash
    
    eq_valid = equity[equity > 0]
    if len(eq_valid) == 0:
        eq_valid = np.array([initial_capital])
    
    final_eq = eq_valid[-1]
    ret_pct = (final_eq - initial_capital) / initial_capital * 100
    
    if len(eq_valid) > 1:
        rets = np.diff(eq_valid) / eq_valid[:-1]
        sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(252 * 390)
    else:
        sharpe = 0.0
    
    rm = np.maximum.accumulate(eq_valid)
    dd = (rm - eq_valid) / rm
    max_dd = float(np.max(dd)) * 100
    
    return {
        'final_equity': final_eq,
        'total_return': ret_pct,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'signals': int(np.sum(np.abs(signals))),
        'trades': num_trades,
        'bars': n
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', type=float, default=100000)
    parser.add_argument('--days', type=int, default=7)
    args = parser.parse_args()
    
    print("=" * 70)
    print("GPU FIBONACCI ADX BACKTEST - INTRADAY TUNED")
    print("=" * 70)
    print(f"\nParameters (optimized for 1-min data):")
    for k, v in PARAMS.items():
        print(f"  {k}: {v}")
    print()
    
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
        print(f"  {category.upper()}")
        print("=" * 70)
        
        cat_results = []
        
        for symbol in symbols:
            print(f"\n[{symbol}] Fetching...")
            df = fetch_1min_data(symbol, args.days)
            
            if df.empty or len(df) < 400:
                print(f"  [!] Insufficient data, skipping")
                continue
            
            print(f"  [{symbol}] {len(df)} bars")
            
            if GPU_AVAILABLE:
                t0 = time.perf_counter()
                result = run_gpu_backtest(df, args.capital)
                elapsed = time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                result = run_cpu_backtest(df, args.capital)
                elapsed = time.perf_counter() - t0
            
            result['symbol'] = symbol
            result['category'] = category
            result['elapsed'] = elapsed
            cat_results.append(result)
            
            total_bars += result['bars']
            total_time += elapsed
            
            status = "✓" if result['total_return'] > 0 else "✗"
            print(f"  [{symbol}] {status} Return: {result['total_return']:+.2f}% | "
                  f"Sharpe: {result['sharpe_ratio']:.2f} | "
                  f"MaxDD: {result['max_drawdown']:.2f}% | "
                  f"Trades: {result.get('trades', result['signals'])}")
        
        all_results[category] = cat_results
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for category, results in all_results.items():
        if not results:
            continue
        avg_ret = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_dd = np.mean([r['max_drawdown'] for r in results])
        winners = len([r for r in results if r['total_return'] > 0])
        print(f"\n{category}:")
        print(f"  Avg Return: {avg_ret:+.2f}%")
        print(f"  Avg Sharpe: {avg_sharpe:.2f}")
        print(f"  Avg MaxDD:  {avg_dd:.2f}%")
        print(f"  Win Rate:   {winners}/{len(results)}")
    
    all_flat = [r for cat in all_results.values() for r in cat]
    if all_flat:
        print("\n" + "-" * 70)
        winners = [r for r in all_flat if r['total_return'] > 0]
        print(f"\nOverall Win Rate: {len(winners)}/{len(all_flat)} ({100*len(winners)/len(all_flat):.1f}%)")
        print(f"Total Bars: {total_bars:,}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Throughput: {total_bars / total_time:,.0f} bars/sec")
        
        if winners:
            best = max(all_flat, key=lambda x: x['total_return'])
            print(f"\nBest:  {best['symbol']} +{best['total_return']:.2f}%")
        
        worst = min(all_flat, key=lambda x: x['total_return'])
        print(f"Worst: {worst['symbol']} {worst['total_return']:.2f}%")
    
    # Save
    output_dir = Path("data/backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = output_dir / f"1min_tuned_backtest_{ts}.json"
    
    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': ts,
            'params': PARAMS,
            'config': {'capital': args.capital, 'days': args.days, 'gpu': GPU_AVAILABLE},
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\nResults: {out_file}")


if __name__ == "__main__":
    main()
