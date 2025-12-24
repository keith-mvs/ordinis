import os
from pathlib import Path

import numpy as np
import pandas as pd

from ordinis.tools.optimizer import optimize_from_config, run_best_backtest_from_study, BacktestOptimizer, OptimizerConfig
from ordinis.engines.signalcore.models.atr_optimized_rsi import backtest as atr_backtest


def make_price_series(n=200, seed=1, drift=0.0005, vol=0.01):
    np.random.seed(seed)
    returns = np.random.normal(loc=drift, scale=vol, size=n)
    price = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame({
        "open": price * (1 - 0.001),
        "high": price * (1 + 0.002),
        "low": price * (1 - 0.002),
        "close": price,
        "volume": np.random.randint(100, 1000, size=n),
    }, index=pd.date_range("2020-01-01", periods=n, freq="D"))
    return df


def test_optimize_from_config_and_run_best_backtest(tmp_path):
    df = make_price_series()

    cfg = {"trials": 3, "seed": 42, "metric": "total_return", "direction": "maximize"}

    res = optimize_from_config(df, cfg)

    assert "study_path" in res and Path(res["study_path"]).exists()
    assert "best_params" in res and isinstance(res["best_params"], dict)

    # run best backtest using generated study
    summary = run_best_backtest_from_study(res["study_path"], df)
    assert "backtest" in summary and "summary_path" in summary
    assert "total_return" in summary["backtest"]


def test_backtest_returns_expected_keys():
    df = make_price_series(seed=7)
    res = atr_backtest(df)
    assert isinstance(res, dict)
    assert "total_return" in res and "trades" in res and isinstance(res["trades"], list)


def test_fallback_run_returns_simple_study(monkeypatch):
    # Force fallback by monkeypatching OPTUNA_AVAILABLE
    import ordinis.tools.optimizer as opt_mod

    monkeypatch.setattr(opt_mod, "OPTUNA_AVAILABLE", False)

    df = make_price_series()
    cfg = {"trials": 2, "seed": 0}
    opt = BacktestOptimizer(df, OptimizerConfig(**cfg))

    study = opt.run(n_trials=2)

    # SimpleStudy replacement should have attributes study_name, trials and best_params
    assert hasattr(study, "study_name")
    assert hasattr(study, "trials") and len(study.trials) == 2
    assert hasattr(study, "best_params")

    # saving should produce a file
    p = opt.save(study)
    assert p.exists() and p.suffix == ".json"
    # cleanup
    p.unlink()
