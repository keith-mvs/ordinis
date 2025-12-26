#!/usr/bin/env python3
"""
MAXIMUM CPU/GPU UTILIZATION OPTIMIZER v2

Launches parallel optimization to max out all system resources.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
import logging
import json
import time
import sys

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: pip install optuna")
    sys.exit(1)

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔥 CUDA: {torch.version.cuda}")
        DEVICE = torch.device('cuda')
    else:
        print("⚠️  CUDA not available - CPU mode")
        DEVICE = torch.device('cpu')
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False
    DEVICE = None
    print("⚠️  PyTorch not found - using NumPy only")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def generate_market_data(symbol: str, n_days: int = 1000) -> pd.DataFrame:
    """Generate realistic market data."""
    np.random.seed(hash(symbol) % 2**31)
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    base_price = np.random.uniform(15, 45)
    volatility = np.random.uniform(0.02, 0.04)
    trend = np.random.uniform(-0.001, 0.002)
    
    returns = np.random.normal(trend, volatility, n_days)
    prices = base_price * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'High': prices * (1 + np.random.uniform(0, 0.03, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.03, 0, n_days)),
        'Close': prices,
        'Volume': np.random.randint(100_000, 5_000_000, n_days),
    }, index=dates)


def gpu_compute(data: np.ndarray) -> float:
    """GPU-intensive matrix operations."""
    if HAS_TORCH and HAS_CUDA:
        tensor = torch.tensor(data, device=DEVICE, dtype=torch.float32)
        # Heavy matrix ops
        for _ in range(100):
            tensor = torch.mm(tensor, tensor.T)
            tensor = torch.relu(tensor)
            tensor = tensor / tensor.max().clamp(min=1e-8)
        return tensor.sum().cpu().item()
    else:
        # CPU fallback - still intensive
        for _ in range(50):
            data = data @ data.T
            data = np.maximum(data, 0)
            data = data / (data.max() + 1e-8)
        return float(np.sum(data))


def calculate_features(prices: pd.Series, lookback: int) -> tuple:
    """Calculate technical features - CPU intensive."""
    close = prices.values
    n = len(close)
    
    # Fixed-size feature matrix
    n_features = 10
    features = np.zeros((n, n_features))
    
    # Returns
    returns = np.zeros(n)
    returns[1:] = (close[1:] - close[:-1]) / close[:-1]
    features[:, 0] = returns
    
    # Moving averages ratios
    for i, window in enumerate([10, 20, 50], start=1):
        ma = pd.Series(close).rolling(window, min_periods=1).mean().values
        features[:, i] = close / (ma + 1e-8) - 1
    
    # Volatility
    for i, window in enumerate([10, 20, 50], start=4):
        vol = pd.Series(returns).rolling(window, min_periods=1).std().fillna(0).values
        features[:, i] = vol
    
    # Momentum
    for i, period in enumerate([5, 10, 20], start=7):
        mom = np.zeros(n)
        mom[period:] = (close[period:] - close[:-period]) / close[:-period]
        features[:, i] = mom
    
    # GPU operation on feature matrix
    sample_size = min(200, n)
    gpu_score = gpu_compute(features[-sample_size:, :])
    
    return features, gpu_score


def backtest_strategy(data: dict, params: dict) -> dict:
    """Run intensive backtest."""
    cash = 100_000.0
    initial = cash
    positions = {}
    trades = []
    equity = []
    
    mi_lookback = params['mi_lookback']
    forward_period = params['forward_period']
    threshold = params['threshold']
    
    all_dates = sorted(set.union(*[set(df.index) for df in data.values()]))
    
    for date in all_dates[mi_lookback:]:
        # Calculate signals with intensive features
        signals = {}
        for symbol, df in data.items():
            if date not in df.index:
                continue
            
            hist = df.loc[:date, 'Close']
            if len(hist) < mi_lookback:
                continue
            
            # CPU/GPU intensive feature calculation
            features, gpu_score = calculate_features(hist, mi_lookback)
            
            # MI proxy
            returns = hist.pct_change().dropna()
            autocorr = returns[-mi_lookback:].autocorr(1) if len(returns) >= mi_lookback else 0
            mi_score = abs(autocorr if not np.isnan(autocorr) else 0) * (gpu_score / 1e6)
            
            if mi_score > threshold:
                signals[symbol] = mi_score
        
        # Update positions
        for sym in list(positions.keys()):
            if sym in data and date in data[sym].index:
                positions[sym]['value'] = positions[sym]['shares'] * data[sym].loc[date, 'Close']
        
        portfolio = cash + sum(p['value'] for p in positions.values())
        equity.append(portfolio)
        
        # Enter positions
        if signals and len(positions) < 5:
            for sym, score in sorted(signals.items(), key=lambda x: -x[1])[:5-len(positions)]:
                if sym not in positions:
                    price = data[sym].loc[date, 'Close']
                    size = portfolio * 0.15
                    shares = int(size / price)
                    if shares > 0 and cash >= shares * price:
                        positions[sym] = {
                            'shares': shares,
                            'entry': price,
                            'date': date,
                            'value': shares * price
                        }
                        cash -= shares * price
        
        # Exit positions
        for sym in list(positions.keys()):
            if sym in data and date in data[sym].index:
                if (date - positions[sym]['date']).days >= forward_period:
                    price = data[sym].loc[date, 'Close']
                    pnl = positions[sym]['shares'] * (price - positions[sym]['entry'])
                    cash += positions[sym]['shares'] * price
                    trades.append(pnl)
                    del positions[sym]
    
    # Metrics
    if not equity:
        return {'total_return': 0, 'sharpe': 0, 'max_dd': 1, 'win_rate': 0, 'trades': 0}
    
    equity = np.array(equity)
    returns = np.diff(equity) / equity[:-1]
    
    total_return = (equity[-1] - initial) / initial
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    cummax = np.maximum.accumulate(equity)
    dd = (equity - cummax) / cummax
    max_dd = abs(dd.min())
    
    win_rate = sum(1 for t in trades if t > 0) / len(trades) if trades else 0
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'trades': len(trades)
    }


def worker(worker_id: int, n_trials: int, data: dict, queue: mp.Queue):
    """Worker process."""
    print(f"Worker {worker_id} starting {n_trials} trials", flush=True)
    
    def objective(trial):
        params = {
            'mi_lookback': trial.suggest_int('mi_lookback', 50, 200),
            'forward_period': trial.suggest_int('forward_period', 1, 10),
            'threshold': trial.suggest_float('threshold', 0.0001, 0.01),
        }
        
        result = backtest_strategy(data, params)
        
        penalty = 0.0
        if result['sharpe'] < 0.8:
            penalty += (0.8 - result['sharpe']) * 50
        if result['max_dd'] > 0.35:
            penalty += (result['max_dd'] - 0.35) * 50
        
        fitness = result['total_return'] - penalty
        
        queue.put({'worker': worker_id, 'trial': trial.number, 'fitness': fitness})
        return fitness
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=worker_id))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    queue.put({
        'worker': worker_id,
        'done': True,
        'best': study.best_value,
        'params': study.best_params
    })
    print(f"Worker {worker_id} done: {study.best_value:.4f}", flush=True)


def main():
    print("=" * 80)
    print("🔥 MAX CPU/GPU UTILIZATION OPTIMIZER 🔥")
    print("=" * 80)
    
    n_workers = mp.cpu_count()
    print(f"\n💻 Launching {n_workers} parallel workers")
    
    # Generate data
    print(f"📊 Generating data for 20 stocks...")
    data = {f"STK{i:02d}": generate_market_data(f"STK{i:02d}", 1000) for i in range(20)}
    print(f"✓ {sum(len(d) for d in data.values()):,} data points")
    
    # Launch workers
    print(f"\n🚀 STARTING OPTIMIZATION - CHECK htop/nvidia-smi!")
    
    manager = mp.Manager()
    queue = manager.Queue()
    
    trials_per = 25
    procs = []
    for i in range(n_workers):
        p = mp.Process(target=worker, args=(i, trials_per, data, queue))
        p.start()
        procs.append(p)
        print(f"  Worker {i} started (PID {p.pid})")
    
    # Monitor
    done = 0
    trials = 0
    results = []
    
    while done < n_workers:
        try:
            msg = queue.get(timeout=60)
            if msg.get('done'):
                done += 1
                results.append(msg)
                print(f"\n✓ Worker {msg['worker']} complete: {msg['best']:.4f}")
            else:
                trials += 1
                if trials % 20 == 0:
                    print(f"  Progress: {trials}/{n_workers * trials_per} trials")
        except:
            print(".", end="", flush=True)
    
    for p in procs:
        p.join()
    
    # Results
    print("\n" + "=" * 80)
    print("🏆 RESULTS")
    print("=" * 80)
    
    best = max(results, key=lambda x: x['best'])
    print(f"\nBest Score: {best['best']:.4f}")
    print(f"Best Params: {best['params']}")
    
    # Save
    out = Path("artifacts/optimization/max_util")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"result_{datetime.now():%Y%m%d_%H%M%S}.json", 'w') as f:
        json.dump({'best': best, 'all': results}, f, indent=2, default=str)
    
    print(f"\n💾 Saved to {out}")
    print("=" * 80)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
