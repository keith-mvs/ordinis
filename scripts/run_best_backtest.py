"""Run the saved study's best params through a backtest and write summary."""
import pandas as pd
import numpy as np
from ordinis.tools.optimizer import run_best_backtest_from_study


def make_synthetic_price(n=500, seed=42):
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


def main():
    df = make_synthetic_price()
    study_path = "artifacts/backtest_optimizations/study_optimizer_1766519265_20251223T194745Z.json"
    res = run_best_backtest_from_study(study_path, df)
    print("Backtest summary written:", res["summary_path"])
    print("Backtest result:", res["backtest"])

if __name__ == "__main__":
    main()
