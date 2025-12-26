#!/usr/bin/env python3
"""
MAXIMUM CPU/GPU UTILIZATION OPTIMIZER

Launches parallel optimization jobs to max out all system resources.
Uses multiprocessing for CPU parallelization and GPU for model inference.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Reduce TF spam

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import multiprocessing as mp
from functools import partial
import logging
import json
import time

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("ERROR: pip install optuna")
    exit(1)

try:
    import torch
    HAS_TORCH = True
    if torch.cuda.is_available():
        print(f"🔥 GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(f"🔥 CUDA VERSION: {torch.version.cuda}")
        print(f"🔥 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not found - CPU only mode")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(processName)-12s] %(message)s'
)
logger = logging.getLogger(__name__)


def generate_market_data(symbol: str, n_days: int = 1000) -> pd.DataFrame:
    """Generate realistic market data with trends and volatility."""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
    
    base_price = np.random.uniform(15, 45)
    volatility = np.random.uniform(0.02, 0.04)
    trend = np.random.uniform(-0.001, 0.002)
    
    returns = np.random.normal(trend, volatility, n_days)
    prices = base_price * (1 + returns).cumprod()
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'High': prices * (1 + np.random.uniform(0, 0.03, n_days)),
        'Low': prices * (1 + np.random.uniform(-0.03, 0, n_days)),
        'Close': prices,
        'Volume': np.random.randint(100_000, 5_000_000, n_days),
    })
    
    return df.set_index('Date')


def gpu_intensive_operation(data: np.ndarray, device: str = 'cuda'):
    """Perform GPU-intensive matrix operations."""
    if not HAS_TORCH or not torch.cuda.is_available():
        # CPU fallback - still intensive
        for _ in range(100):
            result = np.linalg.svd(data @ data.T)
        return np.mean(result[1])
    
    # GPU operations
    tensor = torch.tensor(data, device=device, dtype=torch.float32)
    
    # Intensive operations to max GPU utilization
    for _ in range(50):
        result = torch.mm(tensor, tensor.T)
        result = torch.svd(result)[1]
        result = torch.fft.fft(result.cpu()).abs()
    
    return result.mean().item()


def calculate_features(prices: pd.DataFrame, lookback: int) -> np.ndarray:
    """Calculate features with CPU-intensive operations."""
    close = prices['Close'].values
    
    # Multiple technical indicators
    features = []
    
    # Returns at multiple scales
    for period in [1, 5, 10, 20]:
        returns = np.diff(close, n=period) / close[:-period]
        features.append(np.pad(returns, (period, 0), constant_values=0))
    
    # Moving averages
    for window in [10, 20, 50, 100, 200]:
        ma = pd.Series(close).rolling(window).mean().fillna(0).values
        features.append(ma)
    
    # Volatility
    for window in [10, 20, 50]:
        vol = pd.Series(close).pct_change().rolling(window).std().fillna(0).values
        features.append(vol)
    
    # RSI
    for period in [14, 28]:
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.fillna(50).values)
    
    feature_matrix = np.vstack(features).T
    
    # GPU operation on features
    gpu_score = gpu_intensive_operation(feature_matrix[-500:, :])
    
    return feature_matrix, gpu_score


def backtest_strategy(data: dict, params: dict, use_gpu: bool = True) -> dict:
    """Backtest with intensive CPU/GPU usage."""
    portfolio_value = 100_000.0
    cash = portfolio_value
    positions = {}
    trades = []
    equity_curve = []
    
    mi_lookback = params['mi_lookback']
    forward_period = params['forward_period']
    threshold = params['threshold']
    
    # Get date range
    all_dates = sorted(set.union(*[set(df.index) for df in data.values()]))
    
    for i, date in enumerate(all_dates[mi_lookback:]):
        # CPU-intensive feature calculation
        signals = {}
        for symbol, df in data.items():
            if date not in df.index:
                continue
            
            historical = df.loc[:date]
            
            # Calculate intensive features
            features, gpu_score = calculate_features(historical, mi_lookback)
            
            # MI estimation via mutual information
            returns = historical['Close'].pct_change().dropna()
            if len(returns) < mi_lookback:
                continue
            
            recent_returns = returns[-mi_lookback:]
            autocorr = recent_returns.autocorr(1)
            
            mi_score = abs(autocorr) * gpu_score if not np.isnan(autocorr) else 0
            
            if mi_score > threshold:
                signals[symbol] = {'score': mi_score}
        
        # Update portfolio
        for symbol in list(positions.keys()):
            if symbol in data and date in data[symbol].index:
                current_price = data[symbol].loc[date, 'Close']
                positions[symbol]['value'] = positions[symbol]['shares'] * current_price
        
        portfolio_value = cash + sum(p['value'] for p in positions.values())
        equity_curve.append({'date': date, 'equity': portfolio_value})
        
        # Trade logic
        if len(signals) > 0 and len(positions) < 5:
            sorted_signals = sorted(signals.items(), key=lambda x: x[1]['score'], reverse=True)
            
            for symbol, signal in sorted_signals[:5 - len(positions)]:
                if symbol not in positions:
                    price = data[symbol].loc[date, 'Close']
                    position_size = portfolio_value * 0.15
                    shares = int(position_size / price)
                    
                    if shares > 0 and cash >= shares * price:
                        positions[symbol] = {
                            'shares': shares,
                            'entry_price': price,
                            'entry_date': date,
                            'value': shares * price,
                        }
                        cash -= shares * price
                        trades.append({'date': date, 'action': 'BUY', 'symbol': symbol})
        
        # Exit logic
        for symbol in list(positions.keys()):
            if symbol in data and date in data[symbol].index:
                hold_days = (date - positions[symbol]['entry_date']).days
                if hold_days >= forward_period:
                    price = data[symbol].loc[date, 'Close']
                    shares = positions[symbol]['shares']
                    cash += shares * price
                    pnl = shares * (price - positions[symbol]['entry_price'])
                    trades.append({'date': date, 'action': 'SELL', 'pnl': pnl})
                    del positions[symbol]
    
    # Metrics
    equity_df = pd.DataFrame(equity_curve).set_index('date')
    returns = equity_df['equity'].pct_change().dropna()
    
    total_return = (portfolio_value - 100_000) / 100_000
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if len(returns) > 1 else 0
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    closed_trades = [t for t in trades if t['action'] == 'SELL']
    winning = [t for t in closed_trades if t.get('pnl', 0) > 0]
    win_rate = len(winning) / len(closed_trades) if closed_trades else 0
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'trades': len(closed_trades),
    }


def optimize_worker(worker_id: int, n_trials: int, data: dict, queue: mp.Queue):
    """Worker process for parallel optimization."""
    logger.info(f"Worker {worker_id} starting with {n_trials} trials")
    
    def objective(trial):
        params = {
            'mi_lookback': trial.suggest_int('mi_lookback', 50, 250),
            'forward_period': trial.suggest_int('forward_period', 1, 10),
            'threshold': trial.suggest_float('threshold', 0.1, 0.5),
        }
        
        result = backtest_strategy(data, params, use_gpu=True)
        
        # Constraint penalties
        penalty = 0.0
        if result['sharpe'] < 0.8:
            penalty += (0.8 - result['sharpe']) * 50
        if result['max_dd'] > 0.35:
            penalty += (result['max_dd'] - 0.35) * 50
        
        fitness = result['total_return'] - penalty
        
        queue.put({
            'worker': worker_id,
            'trial': trial.number,
            'fitness': fitness,
            'params': params,
            'result': result,
        })
        
        return fitness
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42 + worker_id),
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    queue.put({
        'worker': worker_id,
        'status': 'COMPLETE',
        'best_value': study.best_value,
        'best_params': study.best_params,
    })
    
    logger.info(f"Worker {worker_id} complete - Best: {study.best_value:.4f}")


def main():
    print("=" * 80)
    print("🔥 MAXIMUM CPU/GPU UTILIZATION OPTIMIZER 🔥")
    print("=" * 80)
    
    # System info
    cpu_count = mp.cpu_count()
    print(f"\n💻 CPU Cores: {cpu_count}")
    print(f"🔄 Launching {cpu_count} parallel workers")
    
    if HAS_TORCH and torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🎮 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Generate data
    print(f"\n📊 Generating market data for 20 stocks...")
    data = {
        f"STOCK{i:02d}": generate_market_data(f"STOCK{i:02d}", n_days=1000)
        for i in range(1, 21)
    }
    print(f"✓ Generated {sum(len(df) for df in data.values()):,} total data points")
    
    # Launch parallel optimization
    print(f"\n🚀 LAUNCHING {cpu_count} PARALLEL OPTIMIZATIONS")
    print(f"⚡ This will MAX OUT your CPU and GPU")
    print(f"⏱️  Check htop/nvidia-smi to see utilization\n")
    
    manager = mp.Manager()
    queue = manager.Queue()
    
    trials_per_worker = 20
    processes = []
    
    for i in range(cpu_count):
        p = mp.Process(
            target=optimize_worker,
            args=(i, trials_per_worker, data, queue)
        )
        p.start()
        processes.append(p)
        print(f"✓ Worker {i} started (PID {p.pid})")
    
    # Monitor progress
    print("\n" + "=" * 80)
    print("MONITORING OPTIMIZATION PROGRESS")
    print("=" * 80)
    
    results = []
    completed = 0
    total_trials = 0
    
    while completed < cpu_count:
        try:
            msg = queue.get(timeout=1)
            
            if msg.get('status') == 'COMPLETE':
                completed += 1
                print(f"\n✓ Worker {msg['worker']} COMPLETE - Best: {msg['best_value']:.4f}")
                results.append(msg)
            else:
                total_trials += 1
                if total_trials % 50 == 0:
                    print(f"   Progress: {total_trials}/{cpu_count * trials_per_worker} trials", end='\r')
        except:
            pass
    
    # Wait for all processes
    for p in processes:
        p.join()
    
    # Final results
    print("\n\n" + "=" * 80)
    print("🏆 OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    best_result = max(results, key=lambda x: x['best_value'])
    
    print(f"\n🥇 BEST OVERALL RESULT:")
    print(f"   Worker: {best_result['worker']}")
    print(f"   Fitness: {best_result['best_value']:.4f}")
    print(f"\n   Best Parameters:")
    for k, v in best_result['best_params'].items():
        print(f"      {k:20s}: {v}")
    
    # Save results
    output_dir = Path("artifacts/optimization/max_util")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'best_result': best_result,
            'all_results': results,
            'total_trials': total_trials,
            'workers': cpu_count,
        }, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
