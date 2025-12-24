import numpy as np
import pandas as pd
import pytest

from datetime import datetime

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.models.atr_optimized_rsi import ATROptimizedRSIModel


def make_price_series(n=60, seed=3):
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


def test_describe_contains_parameters_and_optimized_symbols():
    cfg = ModelConfig(model_id="test", model_type="technical", parameters={})
    model = ATROptimizedRSIModel(cfg)
    d = model.describe()
    assert isinstance(d, dict)
    assert "parameters" in d and "optimized_symbols" in d


def test_validate_fails_on_short_data():
    cfg = ModelConfig(model_id="test", model_type="technical", parameters={})
    model = ATROptimizedRSIModel(cfg)
    # Intentionally short dataframe
    df = make_price_series(n=5)
    valid, msg = model.validate(df)
    assert not valid


@pytest.mark.asyncio
async def test_generate_returns_none_on_invalid_data():
    cfg = ModelConfig(model_id="test", model_type="technical", parameters={})
    model = ATROptimizedRSIModel(cfg)
    df = make_price_series(n=5)
    sig = await model.generate("TEST", df, datetime.utcnow())
    assert sig is None
