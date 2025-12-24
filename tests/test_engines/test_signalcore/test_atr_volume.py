import numpy as np
import pandas as pd

from ordinis.engines.signalcore.models.atr_optimized_rsi import backtest


def make_synthetic_price_with_volume(n=300, seed=1):
    np.random.seed(seed)
    returns = np.random.normal(loc=0.0005, scale=0.01, size=n)
    price = 100 * np.exp(np.cumsum(returns))
    volume = np.random.randint(100, 10000, size=n)
    df = pd.DataFrame({
        "open": price * (1 - 0.001),
        "high": price * (1 + 0.002),
        "low": price * (1 - 0.002),
        "close": price,
        "volume": volume,
    }, index=pd.date_range("2020-01-01", periods=n, freq="D"))
    return df


def test_backtest_accepts_volume_confirmation():
    df = make_synthetic_price_with_volume(400)

    # Run backtest with volume confirmation required - should run without error
    res = backtest(
        df,
        rsi_os=35,
        rsi_exit=50,
        atr_stop_mult=1.5,
        atr_tp_mult=2.0,
        rsi_period=14,
        atr_period=14,
        atr_scale=1.0,
        require_volume_confirmation=True,
        volume_mean_period=10,
        enforce_regime_gate=False,
    )

    assert isinstance(res, dict)
    assert "trades" in res