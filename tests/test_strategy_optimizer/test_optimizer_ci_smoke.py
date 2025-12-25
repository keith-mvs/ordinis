import asyncio
from datetime import datetime

import numpy as np
import pandas as pd

from scripts.strategy_optimizer import StrategyOptimizer, WalkForwardConfig


def _make_synthetic_df(n=800):
    idx = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='D')
    prices = np.linspace(100.0, 150.0, num=n)
    df = pd.DataFrame(
        {
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(100, 1000, size=n),
        },
        index=idx,
    )
    return df


def test_optimizer_smoke():
    """Lightweight smoke test: optimizer completes 1 cycle on synthetic data without using pytest fixtures."""
    # Prepare synthetic data for one symbol
    df = _make_synthetic_df(800)

    def fake_load(self):
        # Inject synthetic dataset and mark load_data as successful
        self.data_cache = {'FAKE': df}
        return True

    # Monkeypatch by assignment (no pytest fixture used)
    StrategyOptimizer.load_data = fake_load

    opt = StrategyOptimizer(
        strategy_name='fibonacci_adx',
        n_cycles=1,
        use_gpu=False,
        metric='cagr',
        n_workers=1,
        bootstrap_n=0,  # skip expensive bootstrap
        skip_bootstrap=True,
    )

    res = opt.run_optimization()

    assert hasattr(res, 'best_params')
    # Should return quickly and produce an OptimizationResult-like object
    assert isinstance(res.test_cagr, float)
