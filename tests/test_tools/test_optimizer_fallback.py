import json
import os
import pandas as pd
import numpy as np

from ordinis.tools import optimizer as opt_mod
from ordinis.tools.optimizer import BacktestOptimizer, optimize_from_config, run_best_backtest_from_study


def make_synthetic_price(n=200, seed=0):
    np.random.seed(seed)
    returns = np.random.normal(loc=0.0005, scale=0.01, size=n)
    price = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "open": price * (1 - 0.001),
        "high": price * (1 + 0.002),
        "low": price * (1 - 0.002),
        "close": price,
        "volume": np.random.randint(100, 1000, size=n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="D"))
    return df


def test_random_search_fallback_returns_study_and_saves(tmp_path):
    df = make_synthetic_price(200, seed=42)

    # point artifacts to tmp_path to avoid polluting repo
    opt_mod.ARTIFACTS_DIR = tmp_path / "backtest_optimizations"
    opt_mod.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = {
        "trials": 5,
        "seed": 123,
        "metric": "total_return",
        "direction": "maximize",
    }

    # Run optimization via BacktestOptimizer.run (fallback path should be taken when optuna is missing)
    opt = BacktestOptimizer(df, opt_mod.OptimizerConfig(**cfg))
    study = opt.run(n_trials=cfg["trials"])

    assert hasattr(study, "best_params")
    assert hasattr(study, "best_value")
    assert len(study.trials) == cfg["trials"]

    best = study.best_params
    # categorical keys must be present and boolean
    assert "require_volume_confirmation" in best
    assert isinstance(best["require_volume_confirmation"], (bool,))
    assert "enforce_regime_gate" in best
    assert isinstance(best["enforce_regime_gate"], (bool,))

    # Save the study and ensure file exists and contains expected fields
    out = opt.save(study)
    assert os.path.exists(out)

    with open(out, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["study_name"] == study.study_name
    assert "best_params" in data

    # Run backtest from saved study JSON and verify output structure
    res = run_best_backtest_from_study(str(out), df)
    assert "summary_path" in res
    assert "backtest" in res
    assert "total_return" in res["backtest"]
