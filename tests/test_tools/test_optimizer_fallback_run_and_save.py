import random
import pandas as pd
import numpy as np
import pytest
import importlib
import os
from datetime import datetime

import ordinis.tools.optimizer as opt_mod
from ordinis.tools.optimizer import BacktestOptimizer, OptimizerConfig, run_best_backtest_from_study
from ordinis.engines.signalcore.core.model import ModelConfig


def make_price_series(n=80, seed=7):
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


def test_fallback_run_and_save(tmp_path, monkeypatch):
    # Force fallback path (simulate Optuna not available)
    monkeypatch.setattr(opt_mod, "OPTUNA_AVAILABLE", False)
    monkeypatch.setattr(opt_mod, "optuna", None)

    df = make_price_series(n=60)
    cfg = OptimizerConfig(trials=3, seed=42, metric="total_return", direction="maximize")

    opt = BacktestOptimizer(df, cfg)
    study = opt.run(n_trials=3)

    assert hasattr(study, "study_name")
    assert hasattr(study, "trials")
    assert study.best_params is not None

    out = opt.save(study)
    assert out.exists()

    # Run backtest from the saved study file
    res = run_best_backtest_from_study(str(out), df)
    assert "summary_path" in res and "backtest" in res
    assert os.path.exists(res["summary_path"])