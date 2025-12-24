"""Optimizer utility for running hyperparameter optimization over backtests.

Uses Optuna to run iterative search over strategy parameters. Intended for quick
MVP integration with model-level backtests (e.g., ATR-Optimized-RSI).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:  # pragma: no cover - optuna may not be installed in CI
    optuna = None
    OPTUNA_AVAILABLE = False

import pandas as pd

from ordinis.engines.signalcore.models.atr_optimized_rsi import backtest as atr_backtest


ARTIFACTS_DIR = Path("artifacts/backtest_optimizations")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class OptimizerConfig:
    trials: int = 40
    seed: int | None = None
    metric: str = "total_return"  # total_return | profit_factor | win_rate | sharpe
    direction: str = "maximize"  # or 'minimize'


class BacktestOptimizer:
    def __init__(self, df: pd.DataFrame, config: OptimizerConfig | None = None):
        self.df = df
        self.config = config or OptimizerConfig()
        self.study: Optional[optuna.Study] = None

    def _objective(self, trial: optuna.Trial) -> float:
        # Suggest parameters
        rsi_period = trial.suggest_int("rsi_period", 5, 40)
        rsi_os = trial.suggest_int("rsi_oversold", 10, 50)
        rsi_exit = trial.suggest_int("rsi_exit", 30, 70)
        atr_period = trial.suggest_int("atr_period", 5, 60)
        atr_stop_mult = trial.suggest_float("atr_stop_mult", 0.5, 10.0, step=0.1)
        atr_tp_mult = trial.suggest_float("atr_tp_mult", 0.5, 20.0, step=0.1)
        # New tunables
        atr_scale = trial.suggest_float("atr_scale", 1.0, 50.0, step=0.1)
        require_volume = trial.suggest_categorical("require_volume_confirmation", [False, True])
        enforce_regime = trial.suggest_categorical("enforce_regime_gate", [False, True])

        res = atr_backtest(
            self.df,
            rsi_os=rsi_os,
            rsi_exit=rsi_exit,
            atr_stop_mult=atr_stop_mult,
            atr_tp_mult=atr_tp_mult,
            rsi_period=rsi_period,
            atr_period=atr_period,
            atr_scale=atr_scale,
            require_volume_confirmation=require_volume,
            enforce_regime_gate=enforce_regime,
        )

        # compute metrics
        if self.config.metric == "total_return":
            value = float(res.get("total_return", 0.0))
        elif self.config.metric == "profit_factor":
            value = float(res.get("profit_factor", 0.0))
        elif self.config.metric == "win_rate":
            value = float(res.get("win_rate", 0.0))
        elif self.config.metric == "sharpe":
            # compute from pnls
            pnls = [t["pnl"] for t in res.get("trades", [])]
            if len(pnls) < 2:
                value = 0.0
            else:
                import numpy as _np

                mean = _np.mean(pnls)
                std = _np.std(pnls, ddof=1)
                value = float(0.0 if std == 0 else mean / std)
        else:
            value = float(res.get("total_return", 0.0))

        # set user attrs for debugging
        trial.set_user_attr("trades", len(res.get("trades", [])))
        trial.set_user_attr("backtest_summary", {k: res.get(k) for k in ["total_return", "profit_factor", "win_rate"]})

        # Optuna maximizes by default; if direction is minimize, return negative
        return value if self.config.direction == "maximize" else -value

    def run(self, n_trials: Optional[int] = None, sampler: Optional[optuna.samplers.BaseSampler] = None) -> optuna.Study:
        n_trials = n_trials or self.config.trials
        storage = None
        study_name = f"optimizer_{int(time.time())}"

        if OPTUNA_AVAILABLE and optuna is not None:
            sampler = sampler or optuna.samplers.TPESampler(seed=self.config.seed)
            self.study = optuna.create_study(direction=("maximize" if self.config.direction == "maximize" else "minimize"), sampler=sampler, study_name=study_name, storage=storage)
            self.study.optimize(self._objective, n_trials=n_trials)

            return self.study
        else:
            # Fallback simple random search if optuna is not installed
            import random

            trials = []
            best_value = None
            best_params = None

            for i in range(n_trials):
                params = {
                    "rsi_period": random.randint(5, 40),
                    "rsi_oversold": random.randint(10, 50),
                    "rsi_exit": random.randint(30, 70),
                    "atr_period": random.randint(5, 60),
                    "atr_stop_mult": round(random.uniform(0.5, 10.0), 2),
                    "atr_tp_mult": round(random.uniform(0.5, 20.0), 2),
                    "atr_scale": round(random.uniform(1.0, 50.0), 2),
                    "require_volume_confirmation": random.choice([False, True]),
                    "enforce_regime_gate": random.choice([False, True]),
                }

                # emulate trial by injecting into _objective via a dummy object
                class DummyTrial:
                    def __init__(self, params):
                        self.params = params
                        self.user_attrs = {}

                    def suggest_int(self, name, a, b):
                        return self.params[name]

                    def suggest_float(self, name, a, b, step=None):
                        return self.params[name]

                    def suggest_categorical(self, name, choices):
                        return self.params[name]

                    def set_user_attr(self, k, v):
                        self.user_attrs[k] = v

                trial = DummyTrial(params)
                value = self._objective(trial)

                trials.append({"number": i, "value": value, "params": params, "state": "COMPLETE", "user_attrs": trial.user_attrs})

                if best_value is None or (value > best_value and self.config.direction == "maximize") or (
                    value < best_value and self.config.direction == "minimize"
                ):
                    best_value = value
                    best_params = params

            # create a simple study-like object
            class SimpleTrial:
                def __init__(self, t):
                    self.number = t["number"]
                    self.value = t["value"]
                    self.params = t["params"]
                    self.state = t["state"]
                    self.user_attrs = t["user_attrs"]

            class SimpleStudy:
                def __init__(self, name, trials, best_value, best_params):
                    self.study_name = name
                    self.trials = [SimpleTrial(t) for t in trials]
                    self.best_value = best_value
                    self.best_params = best_params

            self.study = SimpleStudy(study_name, trials, best_value, best_params)
            return self.study

    def save(self, study: optuna.Study | None = None) -> Path:
        study = study or self.study
        if study is None:
            raise RuntimeError("No study to save")

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out = ARTIFACTS_DIR / f"study_{study.study_name}_{ts}.json"

        data = {
            "study_name": study.study_name,
            "best_value": study.best_value,
            "best_params": dict(study.best_params),
            "n_trials": len(study.trials),
            "trials": [],
        }

        for t in study.trials:
            data["trials"].append({
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
                "user_attrs": t.user_attrs,
            })

        with open(out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return out


def optimize_from_config(df: pd.DataFrame, cfg: dict[str, Any]) -> Dict[str, Any]:
    conf = OptimizerConfig(**cfg)
    opt = BacktestOptimizer(df, conf)
    study = opt.run(n_trials=conf.trials)
    path = opt.save(study)
    return {"study_path": str(path), "best_params": study.best_params, "best_value": study.best_value}


def run_best_backtest_from_study(study_path: str, df: pd.DataFrame) -> dict:
    """Load a study JSON saved by `save` and run a backtest with the best params."""
    import json

    with open(study_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    best = data.get("best_params", {})
    # Map keys expected by atr_backtest
    res = atr_backtest(
        df,
        rsi_os=best.get("rsi_oversold", 35),
        rsi_exit=best.get("rsi_exit", 50),
        atr_stop_mult=best.get("atr_stop_mult", 1.5),
        atr_tp_mult=best.get("atr_tp_mult", 2.0),
        rsi_period=best.get("rsi_period", 14),
        atr_period=best.get("atr_period", 14),
    )

    out_path = ARTIFACTS_DIR / (Path(study_path).stem + "_backtest_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"study": data, "backtest": res}, f, indent=2)
    return {"summary_path": str(out_path), "backtest": res}


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Run optimizer against ATR-Optimized RSI backtest")
    parser.add_argument("--data-csv", required=True, help="CSV file with OHLC data")
    parser.add_argument("--config", default="configs/optimizer.yaml", help="Optimizer configuration YAML")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(args.data_csv, parse_dates=True, index_col=0)

    res = optimize_from_config(df, cfg)
    print("Optimization complete:", res)