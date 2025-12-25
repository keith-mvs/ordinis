"""
GPU-Accelerated Fibonacci ADX Strategy Backtest.

Uses CuPy + Numba CUDA for full GPU acceleration.
All indicator calculations and signal generation run on GPU.

Usage:
    python scripts/backtest_fibonacci_adx_gpu.py --symbols AAPL MSFT NVDA --capital 100000
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# GPU imports
import cupy as cp
from numba import cuda, jit, prange

print(f"CuPy: {cp.__version__}")
print(f"CUDA Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
print(f"VRAM: {cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / 1e9:.1f} GB")


# =============================================================================
# GPU KERNELS - Run directly on CUDA cores
# =============================================================================

@cuda.jit
def compute_adx_kernel(high, low, close, adx_out, plus_di_out, minus_di_out, period):
    """CUDA kernel for ADX calculation."""
    i = cuda.grid(1)
    n = len(close)
    
    if i < period or i >= n:
        return
    
    # Calculate True Range and Directional Movement
    sum_tr = 0.0
    sum_plus_dm = 0.0
    sum_minus_dm = 0.0
    
    for j in range(i - period + 1, i + 1):
        if j > 0:
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]), 
                     abs(low[j] - close[j-1]))
            
            up_move = high[j] - high[j-1]
            down_move = low[j-1] - low[j]
            
            plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0
            
            sum_tr += tr
            sum_plus_dm += plus_dm
            sum_minus_dm += minus_dm
    
    if sum_tr > 0:
        plus_di = 100.0 * sum_plus_dm / sum_tr
        minus_di = 100.0 * sum_minus_dm / sum_tr
        
        plus_di_out[i] = plus_di
        minus_di_out[i] = minus_di
        
        di_sum = plus_di + minus_di
        if di_sum > 0:
            dx = 100.0 * abs(plus_di - minus_di) / di_sum
            adx_out[i] = dx


@cuda.jit
def compute_fibonacci_levels_kernel(high, low, close, swing_high, swing_low, 
                                     fib_382, fib_500, fib_618, lookback):
    """CUDA kernel for Fibonacci level calculation."""
    i = cuda.grid(1)
    n = len(close)
    
    if i < lookback or i >= n:
        return
    
    # Find swing high/low in lookback window
    max_high = high[i - lookback]
    min_low = low[i - lookback]
    
    for j in range(i - lookback, i + 1):
        if high[j] > max_high:
            max_high = high[j]
        if low[j] < min_low:
            min_low = low[j]
    
    swing_high[i] = max_high
    swing_low[i] = min_low
    
    # Calculate Fibonacci levels
    range_val = max_high - min_low
    fib_382[i] = min_low + range_val * 0.382
    fib_500[i] = min_low + range_val * 0.500
    fib_618[i] = min_low + range_val * 0.618


@cuda.jit
def generate_signals_kernel(close, adx, plus_di, minus_di, 
                            fib_382, fib_500, fib_618, swing_high, swing_low,
                            signals, stop_loss, take_profit,
                            adx_threshold, di_threshold, tolerance):
    """CUDA kernel for signal generation - runs on GPU cores."""
    i = cuda.grid(1)
    n = len(close)
    
    if i < 50 or i >= n:
        return
    
    price = close[i]
    
    # Check ADX threshold
    if adx[i] < adx_threshold:
        signals[i] = 0
        return
    
    # Determine trend direction from DI
    is_uptrend = plus_di[i] > minus_di[i]
    di_diff = abs(plus_di[i] - minus_di[i])
    
    if di_diff < di_threshold:
        signals[i] = 0
        return
    
    # Check if price is near Fibonacci levels
    range_val = swing_high[i] - swing_low[i]
    tol = range_val * tolerance
    
    signal = 0
    sl = 0.0
    tp = 0.0
    
    if is_uptrend:
        # Long signals at Fib levels
        if abs(price - fib_382[i]) < tol:
            signal = 1
            sl = fib_500[i] * 0.995
            tp = swing_high[i]
        elif abs(price - fib_500[i]) < tol:
            signal = 1
            sl = fib_618[i] * 0.995
            tp = swing_high[i]
        elif abs(price - fib_618[i]) < tol:
            signal = 1
            sl = swing_low[i] * 0.98
            tp = swing_high[i]
    else:
        # Short signals at Fib levels
        if abs(price - fib_382[i]) < tol:
            signal = -1
            sl = swing_high[i] * 1.02
            tp = swing_low[i]
        elif abs(price - fib_500[i]) < tol:
            signal = -1
            sl = fib_382[i] * 1.005
            tp = swing_low[i]
        elif abs(price - fib_618[i]) < tol:
            signal = -1
            sl = fib_500[i] * 1.005
            tp = swing_low[i]
    
    signals[i] = signal
    stop_loss[i] = sl
    take_profit[i] = tp


@cuda.jit
def simulate_trades_kernel(close, high, low, signals, stop_loss, take_profit,
                           equity_curve, initial_capital, position_size):
    """CUDA kernel for trade simulation - vectorized."""
    # This runs as a single thread since it's sequential
    n = len(close)
    
    equity = initial_capital
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    pos_stop = 0.0
    pos_tp = 0.0
    pos_qty = 0.0
    
    for i in range(n):
        equity_curve[i] = equity
        
        if position != 0:
            # Check stop-loss / take-profit
            exit_price = 0.0
            
            if position == 1:  # Long
                if low[i] <= pos_stop:
                    exit_price = pos_stop
                elif high[i] >= pos_tp:
                    exit_price = pos_tp
            else:  # Short
                if high[i] >= pos_stop:
                    exit_price = pos_stop
                elif low[i] <= pos_tp:
                    exit_price = pos_tp
            
            if exit_price > 0:
                # Close position
                if position == 1:
                    pnl = (exit_price - entry_price) * pos_qty
                else:
                    pnl = (entry_price - exit_price) * pos_qty
                equity += pnl
                position = 0
        
        # Check for new signal
        if position == 0 and signals[i] != 0:
            # Open position
            position = signals[i]
            entry_price = close[i]
            pos_stop = stop_loss[i]
            pos_tp = take_profit[i]
            pos_qty = (equity * position_size) / entry_price
        
        equity_curve[i] = equity


# =============================================================================
# GPU BACKTEST ENGINE
# =============================================================================

class GPUFibonacciADXBacktest:
    """Full GPU-accelerated backtest engine."""
    
    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        di_threshold: float = 20.0,
        swing_lookback: int = 20,
        tolerance: float = 0.01,
        position_size: float = 0.10,
    ):
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.di_threshold = di_threshold
        self.swing_lookback = swing_lookback
        self.tolerance = tolerance
        self.position_size = position_size
        
        # CUDA config
        self.threads_per_block = 256
    
    def run(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        initial_capital: float = 100000.0,
    ) -> dict:
        """Run full backtest on GPU."""
        n = len(close)
        
        print(f"\n[GPU] Transferring {n} bars to GPU VRAM...")
        start = time.time()
        
        # Transfer data to GPU
        d_high = cuda.to_device(high.astype(np.float64))
        d_low = cuda.to_device(low.astype(np.float64))
        d_close = cuda.to_device(close.astype(np.float64))
        
        # Allocate GPU memory for indicators
        d_adx = cuda.device_array(n, dtype=np.float64)
        d_plus_di = cuda.device_array(n, dtype=np.float64)
        d_minus_di = cuda.device_array(n, dtype=np.float64)
        d_swing_high = cuda.device_array(n, dtype=np.float64)
        d_swing_low = cuda.device_array(n, dtype=np.float64)
        d_fib_382 = cuda.device_array(n, dtype=np.float64)
        d_fib_500 = cuda.device_array(n, dtype=np.float64)
        d_fib_618 = cuda.device_array(n, dtype=np.float64)
        d_signals = cuda.device_array(n, dtype=np.int32)
        d_stop_loss = cuda.device_array(n, dtype=np.float64)
        d_take_profit = cuda.device_array(n, dtype=np.float64)
        d_equity = cuda.device_array(n, dtype=np.float64)
        
        # Initialize to zero
        d_adx[:] = 0
        d_signals[:] = 0
        
        transfer_time = time.time() - start
        print(f"[GPU] Transfer complete: {transfer_time:.3f}s")
        
        # Configure CUDA grid
        blocks = (n + self.threads_per_block - 1) // self.threads_per_block
        
        # =====================================================================
        # PHASE 1: Compute ADX on GPU
        # =====================================================================
        print("[GPU] Computing ADX indicators...")
        start = time.time()
        
        compute_adx_kernel[blocks, self.threads_per_block](
            d_high, d_low, d_close, 
            d_adx, d_plus_di, d_minus_di, 
            self.adx_period
        )
        cuda.synchronize()
        
        adx_time = time.time() - start
        print(f"[GPU] ADX computed: {adx_time:.3f}s")
        
        # =====================================================================
        # PHASE 2: Compute Fibonacci levels on GPU
        # =====================================================================
        print("[GPU] Computing Fibonacci levels...")
        start = time.time()
        
        compute_fibonacci_levels_kernel[blocks, self.threads_per_block](
            d_high, d_low, d_close,
            d_swing_high, d_swing_low,
            d_fib_382, d_fib_500, d_fib_618,
            self.swing_lookback
        )
        cuda.synchronize()
        
        fib_time = time.time() - start
        print(f"[GPU] Fibonacci computed: {fib_time:.3f}s")
        
        # =====================================================================
        # PHASE 3: Generate signals on GPU
        # =====================================================================
        print("[GPU] Generating signals...")
        start = time.time()
        
        generate_signals_kernel[blocks, self.threads_per_block](
            d_close, d_adx, d_plus_di, d_minus_di,
            d_fib_382, d_fib_500, d_fib_618, d_swing_high, d_swing_low,
            d_signals, d_stop_loss, d_take_profit,
            self.adx_threshold, self.di_threshold, self.tolerance
        )
        cuda.synchronize()
        
        signal_time = time.time() - start
        print(f"[GPU] Signals generated: {signal_time:.3f}s")
        
        # =====================================================================
        # PHASE 4: Simulate trades on GPU (sequential but on GPU memory)
        # =====================================================================
        print("[GPU] Simulating trades...")
        start = time.time()
        
        # Trade simulation is sequential, run on single GPU thread
        simulate_trades_kernel[1, 1](
            d_close, d_high, d_low, d_signals, d_stop_loss, d_take_profit,
            d_equity, initial_capital, self.position_size
        )
        cuda.synchronize()
        
        sim_time = time.time() - start
        print(f"[GPU] Simulation complete: {sim_time:.3f}s")
        
        # =====================================================================
        # PHASE 5: Copy results back to CPU
        # =====================================================================
        print("[GPU] Copying results to CPU...")
        start = time.time()
        
        equity_curve = d_equity.copy_to_host()
        signals = d_signals.copy_to_host()
        adx = d_adx.copy_to_host()
        
        copy_time = time.time() - start
        print(f"[GPU] Copy complete: {copy_time:.3f}s")
        
        # Calculate metrics using CuPy (stays on GPU)
        print("[GPU] Computing metrics on GPU...")
        start = time.time()
        
        equity_gpu = cp.asarray(equity_curve)
        returns_gpu = cp.diff(equity_gpu) / equity_gpu[:-1]
        
        final_equity = float(equity_gpu[-1])
        total_return = (final_equity - initial_capital) / initial_capital * 100
        
        mean_ret = float(cp.mean(returns_gpu))
        std_ret = float(cp.std(returns_gpu))
        sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
        
        # Max drawdown on CPU (accumulate not supported on GPU)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = float(np.max(drawdown)) * 100
        
        metrics_time = time.time() - start
        print(f"[GPU] Metrics computed: {metrics_time:.3f}s")
        
        # Count trades
        signal_changes = np.diff(signals)
        num_trades = np.sum(signal_changes != 0)
        num_signals = np.sum(signals != 0)
        
        return {
            "equity_curve": equity_curve,
            "signals": signals,
            "adx": adx,
            "metrics": {
                "initial_capital": initial_capital,
                "final_equity": final_equity,
                "total_return_pct": total_return,
                "sharpe_ratio": sharpe,
                "max_drawdown_pct": max_dd,
                "num_signals": int(num_signals),
                "num_trades": int(num_trades),
            },
            "timing": {
                "transfer": transfer_time,
                "adx": adx_time,
                "fibonacci": fib_time,
                "signals": signal_time,
                "simulation": sim_time,
                "copy": copy_time,
                "metrics": metrics_time,
                "total": transfer_time + adx_time + fib_time + signal_time + sim_time + copy_time + metrics_time,
            },
        }


def load_data(symbol: str, data_dir: Path) -> pd.DataFrame | None:
    """Load historical data."""
    csv_path = data_dir / f"{symbol}_historical.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
        df.columns = df.columns.str.lower()
        return df
    return None


def main():
    parser = argparse.ArgumentParser(description="GPU Fibonacci ADX Backtest")
    parser.add_argument("--symbols", nargs="+", default=["AAPL"])
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--output", type=str, default="data/backtest_results")
    args = parser.parse_args()
    
    data_dir = Path(__file__).parent.parent / "data" / "historical"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GPU-ACCELERATED FIBONACCI ADX BACKTEST")
    print("=" * 70)
    
    engine = GPUFibonacciADXBacktest()
    
    all_results = {}
    total_bars = 0
    total_time = 0
    
    for symbol in args.symbols:
        print(f"\n{'='*70}")
        print(f"Processing {symbol}...")
        print("=" * 70)
        
        df = load_data(symbol, data_dir)
        if df is None:
            print(f"No data for {symbol}")
            continue
        
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        
        print(f"Data: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
        
        start = time.time()
        result = engine.run(high, low, close, args.capital)
        elapsed = time.time() - start
        
        total_bars += len(df)
        total_time += elapsed
        
        metrics = result["metrics"]
        timing = result["timing"]
        
        print(f"\n{'RESULTS':^70}")
        print("-" * 70)
        print(f"Final Equity:      ${metrics['final_equity']:>15,.2f}")
        print(f"Total Return:      {metrics['total_return_pct']:>15.2f}%")
        print(f"Sharpe Ratio:      {metrics['sharpe_ratio']:>15.2f}")
        print(f"Max Drawdown:      {metrics['max_drawdown_pct']:>15.2f}%")
        print(f"Signals Generated: {metrics['num_signals']:>15}")
        
        print(f"\n{'GPU TIMING':^70}")
        print("-" * 70)
        print(f"Data Transfer:     {timing['transfer']:>15.4f}s")
        print(f"ADX Computation:   {timing['adx']:>15.4f}s")
        print(f"Fibonacci Levels:  {timing['fibonacci']:>15.4f}s")
        print(f"Signal Generation: {timing['signals']:>15.4f}s")
        print(f"Trade Simulation:  {timing['simulation']:>15.4f}s")
        print(f"Results Copy:      {timing['copy']:>15.4f}s")
        print(f"Metrics Calc:      {timing['metrics']:>15.4f}s")
        print(f"TOTAL GPU TIME:    {timing['total']:>15.4f}s")
        
        all_results[symbol] = metrics
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Bars Processed: {total_bars:,}")
    print(f"Total GPU Time:       {total_time:.2f}s")
    print(f"Throughput:           {total_bars/total_time:,.0f} bars/second")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = output_dir / f"gpu_backtest_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
