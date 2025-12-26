#!/usr/bin/env python3
"""
MAXIMUM GPU/CPU UTILIZATION OPTIMIZER v3

Single process but with heavy GPU operations.
Runs in foreground so you can see it.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import json
import time
import sys

# Check GPU
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim_torch
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        DEVICE = torch.device('cuda')
        print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🔥 CUDA: {torch.version.cuda}")
        print(f"🔥 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        DEVICE = torch.device('cpu')
        print("⚠️  No CUDA - CPU mode")
except ImportError:
    HAS_CUDA = False
    DEVICE = None
    print("❌ PyTorch not available")
    sys.exit(1)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("❌ pip install optuna")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)


class HeavyNeuralNet(nn.Module):
    """Large neural network for GPU saturation."""
    def __init__(self, input_dim=100, hidden_dim=2048, n_layers=6):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def generate_data(n_symbols=20, n_days=1000):
    """Generate market data."""
    data = {}
    for i in range(n_symbols):
        symbol = f"STK{i:02d}"
        np.random.seed(i)
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        base = np.random.uniform(15, 45)
        vol = np.random.uniform(0.02, 0.04)
        returns = np.random.normal(0.0005, vol, n_days)
        prices = base * (1 + returns).cumprod()
        data[symbol] = pd.DataFrame({
            'Close': prices,
            'Volume': np.random.randint(100_000, 5_000_000, n_days)
        }, index=dates)
    return data


def gpu_heavy_features(prices: np.ndarray, params: dict) -> torch.Tensor:
    """Compute features with HEAVY GPU usage."""
    n = len(prices)
    
    # Create large feature matrix
    features = np.zeros((n, 100))
    
    for i in range(100):
        window = (i % 50) + 5
        if i < 25:
            features[:, i] = pd.Series(prices).rolling(window, min_periods=1).mean().values
        elif i < 50:
            features[:, i] = pd.Series(prices).pct_change().rolling(window, min_periods=1).std().fillna(0).values
        elif i < 75:
            features[:, i] = pd.Series(prices).rolling(window, min_periods=1).max().values / prices - 1
        else:
            features[:, i] = prices / pd.Series(prices).rolling(window, min_periods=1).min().values - 1
    
    # Move to GPU
    X = torch.tensor(features, dtype=torch.float32, device=DEVICE)
    
    # HEAVY GPU operations - matrix multiplications
    for _ in range(params.get('gpu_iterations', 50)):
        X = torch.mm(X.T, X)  # 100x100 matmul
        X = X / X.max().clamp(min=1e-8)
        X = torch.relu(X)
    
    # SVD - very GPU intensive
    U, S, V = torch.linalg.svd(X)
    
    return S.sum().cpu().item()


def train_model_on_gpu(data_dict: dict, params: dict) -> float:
    """Train neural network on GPU - INTENSIVE."""
    # Prepare training data
    all_features = []
    all_targets = []
    
    lookback = params['lookback']
    
    for symbol, df in data_dict.items():
        prices = df['Close'].values
        n = len(prices)
        
        for i in range(lookback, n - 1):
            # Features: normalized price history
            feat = prices[i-lookback:i] / prices[i] - 1
            all_features.append(feat)
            
            # Target: next day return
            target = (prices[i+1] - prices[i]) / prices[i]
            all_targets.append(target)
    
    if len(all_features) < 100:
        return 0.0
    
    X = torch.tensor(np.array(all_features), dtype=torch.float32, device=DEVICE)
    y = torch.tensor(np.array(all_targets), dtype=torch.float32, device=DEVICE).unsqueeze(1)
    
    # Create model
    model = HeavyNeuralNet(
        input_dim=lookback,
        hidden_dim=params.get('hidden_dim', 1024),
        n_layers=params.get('n_layers', 4)
    ).to(DEVICE)
    
    optimizer_nn = optim_torch.Adam(model.parameters(), lr=params.get('lr', 0.001))
    criterion = nn.MSELoss()
    
    # Train - GPU INTENSIVE
    epochs = params.get('epochs', 50)
    batch_size = params.get('batch_size', 512)
    
    model.train()
    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(X.shape[0])
        X = X[perm]
        y = y[perm]
        
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, X.shape[0], batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            optimizer_nn.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer_nn.step()
            
            total_loss += loss.item()
            n_batches += 1
    
    # Evaluate - backtest
    model.eval()
    with torch.no_grad():
        preds = model(X).cpu().numpy().flatten()
    
    targets = y.cpu().numpy().flatten()
    
    # Simulate trading
    portfolio = 100_000.0
    equity = [portfolio]
    
    for i in range(len(preds)):
        if preds[i] > params['threshold']:
            # Long
            ret = targets[i]
            portfolio *= (1 + ret * 0.5)  # 50% position
        elif preds[i] < -params['threshold']:
            # Short
            ret = targets[i]
            portfolio *= (1 - ret * 0.5)
        equity.append(portfolio)
    
    returns = np.diff(equity) / equity[:-1]
    total_return = (portfolio - 100_000) / 100_000
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    # Max drawdown
    cum = np.array(equity)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = abs(dd.min())
    
    return total_return, sharpe, max_dd


def run_optimization(data: dict, n_trials: int = 100):
    """Run optimization with HEAVY GPU usage."""
    
    def objective(trial):
        params = {
            'lookback': trial.suggest_int('lookback', 20, 100),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [512, 1024, 2048]),
            'n_layers': trial.suggest_int('n_layers', 2, 6),
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'epochs': trial.suggest_int('epochs', 20, 100),
            'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024]),
            'threshold': trial.suggest_float('threshold', 0.001, 0.01),
            'gpu_iterations': trial.suggest_int('gpu_iterations', 20, 100),
        }
        
        # Heavy GPU features
        for symbol, df in data.items():
            gpu_heavy_features(df['Close'].values, params)
        
        # Train neural network
        total_return, sharpe, max_dd = train_model_on_gpu(data, params)
        
        # Fitness
        penalty = 0.0
        if sharpe < 1.0:
            penalty += (1.0 - sharpe) * 50
        if max_dd > 0.3:
            penalty += (max_dd - 0.3) * 100
        
        fitness = total_return - penalty
        
        trial.set_user_attr('return', total_return)
        trial.set_user_attr('sharpe', sharpe)
        trial.set_user_attr('max_dd', max_dd)
        
        return fitness
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print(f"\n🚀 Starting {n_trials} trials - WATCH nvidia-smi!")
    print("=" * 60)
    
    for i in range(n_trials):
        trial = study.ask()
        
        start = time.time()
        try:
            value = objective(trial)
            study.tell(trial, value)
            elapsed = time.time() - start
            
            ret = trial.user_attrs.get('return', 0) * 100
            sharpe = trial.user_attrs.get('sharpe', 0)
            
            print(f"Trial {i+1:3d}/{n_trials}: Return={ret:6.2f}%, Sharpe={sharpe:5.2f}, Best={study.best_value:.4f} ({elapsed:.1f}s)")
            
        except Exception as e:
            study.tell(trial, float('-inf'))
            print(f"Trial {i+1:3d}/{n_trials}: FAILED - {e}")
    
    return study


def main():
    print("=" * 70)
    print("🔥 MAXIMUM GPU UTILIZATION OPTIMIZER v3 🔥")
    print("=" * 70)
    
    if not HAS_CUDA:
        print("❌ CUDA required for this optimizer")
        sys.exit(1)
    
    # Generate data
    print("\n📊 Generating market data for 20 stocks...")
    data = generate_data(n_symbols=20, n_days=1000)
    print(f"✓ {sum(len(d) for d in data.values()):,} data points")
    
    # Warm up GPU
    print("\n🔥 Warming up GPU...")
    warmup = torch.randn(5000, 5000, device=DEVICE)
    for _ in range(10):
        warmup = torch.mm(warmup, warmup.T)
        warmup = warmup / warmup.max()
    del warmup
    torch.cuda.empty_cache()
    print("✓ GPU ready")
    
    # Run optimization
    print("\n" + "=" * 70)
    print("🚀 OPTIMIZATION STARTING - CHECK nvidia-smi FOR GPU UTILIZATION!")
    print("=" * 70)
    
    study = run_optimization(data, n_trials=50)
    
    # Results
    print("\n" + "=" * 70)
    print("🏆 OPTIMIZATION COMPLETE")
    print("=" * 70)
    
    best = study.best_trial
    print(f"\nBest Fitness: {best.value:.4f}")
    print(f"Best Return: {best.user_attrs['return']*100:.2f}%")
    print(f"Best Sharpe: {best.user_attrs['sharpe']:.2f}")
    print(f"Best Max DD: {best.user_attrs['max_dd']*100:.2f}%")
    
    print("\nBest Parameters:")
    for k, v in best.params.items():
        print(f"  {k:15s}: {v}")
    
    # Save
    out_dir = Path("artifacts/optimization/gpu_max")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'best_value': best.value,
        'best_params': best.params,
        'best_metrics': {
            'return': best.user_attrs['return'],
            'sharpe': best.user_attrs['sharpe'],
            'max_dd': best.user_attrs['max_dd'],
        },
        'n_trials': len(study.trials),
        'timestamp': datetime.now().isoformat(),
    }
    
    out_file = out_dir / f"results_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
