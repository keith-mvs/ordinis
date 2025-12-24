import asyncio
import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models.atr_optimized_rsi import ATROptimizedRSIModel, OptimizedConfig, OPTIMIZED_CONFIGS, backtest


def make_price_series(n=250, seed=0, drift=0.0005, vol=0.01):
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


@pytest.mark.asyncio
async def test_volume_confirmation_blocks_entry_when_volume_high(monkeypatch):
    df = make_price_series(seed=42)

    # find current RSI and set rsi_oversold above it to force the entry condition
    model_cfg = ModelConfig(model_id="test", model_type="technical", parameters={
        "use_optimized": False,
        "require_volume_confirmation": True,
        "volume_mean_period": 5,
    })

    # force RSI to a low fixed value so entry conditions are deterministic
    monkeypatch.setattr(
        "ordinis.engines.signalcore.features.technical.TechnicalIndicators.rsi",
        lambda series, period: pd.Series(np.full(len(series), 10), index=series.index),
    )

    model = ATROptimizedRSIModel(model_cfg)

    # ensure last volume is high compared to rolling mean to trigger blocking
    df.at[df.index[-1], "volume"] = int(df["volume"].rolling(5).mean().iloc[-1] + 1000)

    # compute current RSI to set rsi_oversold threshold safely above
    from ordinis.engines.signalcore.features.technical import TechnicalIndicators

    rsi = TechnicalIndicators.rsi(df["close"], model.rsi_period)
    model.config.parameters["rsi_oversold"] = int(rsi.iloc[-1] + 10)

    sig = await model.generate("TEST", df, datetime.utcnow())

    assert sig is None, "Expected entry to be blocked by high volume when volume confirmation is required"

    # disable volume confirmation -> should produce a trade signal
    model.config.parameters["require_volume_confirmation"] = False
    model = ATROptimizedRSIModel(model.config)

    # recompute RSI threshold to ensure entry condition remains true
    from ordinis.engines.signalcore.features.technical import TechnicalIndicators
    rsi = TechnicalIndicators.rsi(df["close"], model.rsi_period)
    model.config.parameters["rsi_oversold"] = int(rsi.iloc[-1] + 5)

    sig = await model.generate("TEST", df, datetime.utcnow())
    assert sig is not None and sig.signal_type.name == "ENTRY"


@pytest.mark.asyncio
async def test_regime_gate_blocks_when_price_below_sma():
    # create a dataset where price is below its SMA
    df = make_price_series(seed=7)
    # make last price artificially low to be below SMA
    df.at[df.index[-1], "close"] = df["close"].rolling(10).mean().iloc[-1] - 5
    df.at[df.index[-1], "open"] = df.at[df.index[-1], "close"] * 0.99
    df.at[df.index[-1], "high"] = df.at[df.index[-1], "close"] * 1.01
    df.at[df.index[-1], "low"] = df.at[df.index[-1], "close"] * 0.995

    model_cfg = ModelConfig(model_id="test", model_type="technical", parameters={
        "use_optimized": False,
        "enforce_regime_gate": True,
        "regime_sma_period": 5,
    })
    model = ATROptimizedRSIModel(model_cfg)

    # set rsi_oversold high to allow entry if regime gate not blocking
    from ordinis.engines.signalcore.features.technical import TechnicalIndicators
    rsi = TechnicalIndicators.rsi(df["close"], model.rsi_period)
    model.config.parameters["rsi_oversold"] = int(rsi.iloc[-1] + 10)

    sig = await model.generate("TEST", df, datetime.utcnow())
    assert sig is None, "Expected regime gate to block entry when price below SMA"

    # make price above SMA then expect signal
    df.at[df.index[-1], "close"] = df["close"].rolling(5).mean().iloc[-1] + 10
    df.at[df.index[-1], "open"] = df.at[df.index[-1], "close"] * 0.99

    # recompute RSI threshold after changing price so the entry condition still holds
    from ordinis.engines.signalcore.features.technical import TechnicalIndicators
    rsi = TechnicalIndicators.rsi(df["close"], model.rsi_period)
    model.config.parameters["rsi_oversold"] = int(rsi.iloc[-1] + 5)

    model = ATROptimizedRSIModel(model.config)

    # Sanity checks - ensure SMA and RSI conditions are met before generating
    sma = df["close"].rolling(model.regime_sma_period).mean()
    assert len(sma.dropna()) > 0 and df["close"].iloc[-1] > sma.iloc[-1], "Sanity: price should be above SMA"

    from ordinis.engines.signalcore.features.technical import TechnicalIndicators
    rsi = TechnicalIndicators.rsi(df["close"], model.rsi_period)
    assert rsi.iloc[-1] < model.config.parameters["rsi_oversold"], "Sanity: RSI should be below oversold threshold"

    is_valid, msg = model.validate(df)
    assert is_valid, f"Data should validate before generating signals: {msg}"

    sig = await model.generate("TEST", df, datetime.utcnow())
    assert sig is not None and sig.signal_type.name == "ENTRY"


@pytest.mark.asyncio
async def test_atr_scale_from_config_reflected_in_metadata(monkeypatch):
    df = make_price_series(seed=21)

    # Create a temporary optimized config for symbol FOO with custom atr_scale
    monkeypatch.setitem(OPTIMIZED_CONFIGS, "FOO", OptimizedConfig(rsi_oversold=100, atr_scale=7.0))

    model_cfg = ModelConfig(model_id="test", model_type="technical", parameters={
        "use_optimized": True,
    })
    model = ATROptimizedRSIModel(model_cfg)

    sig = await model.generate("FOO", df, datetime.utcnow())
    assert sig is not None and sig.metadata.get("atr_scale") == 7.0


def test_backtest_atr_scale_changes_results():
    df = make_price_series(seed=99)

    res1 = backtest(df, atr_scale=1.0)
    res2 = backtest(df, atr_scale=10.0)

    # With a much larger ATR scale, stops/targets will be wider and trade outcomes should differ
    assert res1 != res2
    assert "total_return" in res1 and "total_return" in res2
