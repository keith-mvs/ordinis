import os
import pandas as pd
import numpy as np

from ordinis.tools.optimizer import optimize_from_config


def make_synthetic_price(n=300, seed=0):
    np.random.seed(seed)
    returns = np.random.normal(loc=0.0005, scale=0.01, size=n)
    price = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "open": price * (1 - 0.001),
        "high": price * (1 + 0.002),
        "low": price * (1 - 0.002),
        "close": price,
    }, index=pd.date_range("2020-01-01", periods=n, freq="D"))
    return df


def test_optimizer_runs_and_returns_results(tmp_path):
    df = make_synthetic_price(300, seed=123)
    cfg = {
        "trials": 3,
        "seed": 123,
        "metric": "total_return",
        "direction": "maximize",
    }

    res = optimize_from_config(df, cfg)

    assert "study_path" in res
    assert "best_params" in res
    assert "best_value" in res

    # study file must exist
    assert os.path.exists(res["study_path"])

    # new params should be present in best_params (may be from random search fallback)
    best = res["best_params"]
    assert "atr_scale" in best
    assert "require_volume_confirmation" in best
    assert "enforce_regime_gate" in best