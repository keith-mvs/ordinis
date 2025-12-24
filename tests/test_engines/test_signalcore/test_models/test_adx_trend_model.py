import pandas as pd
import pytest

from ordinis.engines.signalcore.core.model import ModelConfig
from ordinis.engines.signalcore.core.signal import Direction, SignalType
from ordinis.engines.signalcore.models.adx_trend import ADXTrendModel


def _make_ohlcv(
    n: int,
    *,
    start: float = 100.0,
    step: float = 1.0,
    flat_tail: int = 0,
    symbol: str | None = "AAPL",
) -> pd.DataFrame:
    """Create synthetic OHLCV with an optional flat tail to weaken ADX."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")

    # Trending segment
    base = pd.Series([start + i * step for i in range(n)], index=idx, dtype=float)

    if flat_tail > 0:
        base.iloc[-flat_tail:] = base.iloc[-flat_tail - 1]

    close = base
    open_ = close.shift(1).fillna(close)

    # Keep a small, consistent range so DM dominates direction.
    high = close + 0.5
    low = close - 0.5
    volume = pd.Series(1_000, index=idx, dtype=float)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )

    if symbol is not None:
        df["symbol"] = symbol

    return df


@pytest.mark.asyncio
async def test_uptrend_emits_entry_long() -> None:
    cfg = ModelConfig(model_id="adx", model_type="technical", parameters={"adx_period": 14})
    model = ADXTrendModel(cfg)

    data = _make_ohlcv(100, step=1.0, symbol="SPY")
    ts = data.index[-1].to_pydatetime()

    signal = await model.generate(data, ts)

    assert signal.signal_type == SignalType.ENTRY
    assert signal.direction == Direction.LONG
    assert signal.score > 0
    assert 0.55 <= signal.probability <= 0.75
    assert signal.symbol == "SPY"
    assert signal.metadata["adx_threshold"] == model.adx_threshold


@pytest.mark.asyncio
async def test_downtrend_emits_entry_short() -> None:
    cfg = ModelConfig(model_id="adx", model_type="technical", parameters={"adx_period": 14})
    model = ADXTrendModel(cfg)

    data = _make_ohlcv(100, step=-1.0, symbol="SPY")
    ts = data.index[-1].to_pydatetime()

    signal = await model.generate(data, ts)

    assert signal.signal_type == SignalType.ENTRY
    assert signal.direction == Direction.SHORT
    assert signal.score < 0
    assert 0.55 <= signal.probability <= 0.75


@pytest.mark.asyncio
async def test_flat_prices_emit_hold() -> None:
    cfg = ModelConfig(model_id="adx", model_type="technical", parameters={"adx_period": 14})
    model = ADXTrendModel(cfg)

    data = _make_ohlcv(100, step=0.0, symbol="SPY")
    ts = data.index[-1].to_pydatetime()

    signal = await model.generate(data, ts)

    assert signal.signal_type == SignalType.HOLD
    assert signal.direction == Direction.NEUTRAL
    assert signal.score == 0.0
    assert signal.probability == 0.5


@pytest.mark.asyncio
async def test_trend_weakening_triggers_exit() -> None:
    # Use adx_period=1 so ADX equals DX, allowing an abrupt drop on a flat last candle.
    cfg = ModelConfig(
        model_id="adx",
        model_type="technical",
        parameters={"adx_period": 1, "adx_threshold": 25, "di_threshold": 5},
    )
    model = ADXTrendModel(cfg)

    # Make the last bar flat so DX (and thus ADX) drops below the threshold,
    # while the previous bar remains in a strong trend.
    data = _make_ohlcv(60, step=1.0, flat_tail=1, symbol="SPY")
    ts = data.index[-1].to_pydatetime()

    signal = await model.generate(data, ts)

    assert signal.signal_type == SignalType.EXIT
    assert signal.direction == Direction.NEUTRAL
    assert signal.score < 0
    assert signal.probability >= 0.60


@pytest.mark.asyncio
async def test_validation_missing_columns_raises() -> None:
    cfg = ModelConfig(model_id="adx", model_type="technical")
    model = ADXTrendModel(cfg)

    data = _make_ohlcv(100)
    data = data.drop(columns=["volume"])

    with pytest.raises(ValueError, match=r"Missing columns"):
        await model.generate(data, data.index[-1].to_pydatetime())


@pytest.mark.asyncio
async def test_validation_nulls_raises() -> None:
    cfg = ModelConfig(model_id="adx", model_type="technical")
    model = ADXTrendModel(cfg)

    data = _make_ohlcv(100)
    data.loc[data.index[-1], "close"] = float("nan")

    with pytest.raises(ValueError, match=r"null"):
        await model.generate(data, data.index[-1].to_pydatetime())
