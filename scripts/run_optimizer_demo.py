"""Demo runner for the backtest optimizer (quick run)."""
import pandas as pd
import numpy as np
from ordinis.tools.optimizer import optimize_from_config


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
    cfg = {"trials": 10, "seed": 42, "metric": "total_return", "direction": "maximize"}
    res = optimize_from_config(df, cfg)
    print("Optimization result:", res)

if __name__ == "__main__":
    main()
