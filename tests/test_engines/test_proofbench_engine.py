import pytest
import pandas as pd
from datetime import datetime, timezone

from ordinis.engines.proofbench.core.engine import ProofBenchEngine
from ordinis.engines.proofbench.core.simulator import SimulationResults


def make_price_series(n=10):
    import numpy as np

    np.random.seed(1)
    returns = np.random.normal(0.0005, 0.01, size=n)
    price = 100 * np.exp(returns.cumsum())
    df = pd.DataFrame({
        "open": price * 0.999,
        "high": price * 1.002,
        "low": price * 0.998,
        "close": price,
        "volume": (1000 * (1 + np.abs(np.random.randn(n)))).astype(int),
    }, index=pd.date_range("2020-01-01", periods=n, freq="D"))
    return df


@pytest.mark.asyncio
async def test_initialize_and_load_data_health():
    engine = ProofBenchEngine()

    # Using methods before initialize should raise
    with pytest.raises(RuntimeError):
        engine.load_data("FOO", make_price_series(3))

    await engine.initialize()

    health = await engine._do_health_check()
    assert "No data loaded" in health.message

    df = make_price_series(5)
    engine.load_data("FOO", df)
    assert "FOO" in engine.config.loaded_symbols

    health2 = await engine._do_health_check()
    assert health2.level.name == "HEALTHY"


@pytest.mark.asyncio
async def test_run_backtest_and_metrics():
    engine = ProofBenchEngine()
    await engine.initialize()

    df = make_price_series(6)
    engine.load_data("BAR", df)

    # trivial strategy that does nothing
    def strategy(engine_inst, symbol, bar):
        return None

    engine.set_strategy(strategy)

    res = await engine.run_backtest()
    assert isinstance(res, SimulationResults)
    assert engine.backtests_run >= 1

    # record trades and compute metrics
    await engine.record([
        {"symbol": "BAR", "side": "buy", "quantity": 1, "price": 100.0, "timestamp": datetime.now(timezone.utc), "order_id": "1", "pnl": 10.0},
        {"symbol": "BAR", "side": "sell", "quantity": 1, "price": 110.0, "timestamp": datetime.now(timezone.utc), "order_id": "2", "pnl": -5.0},
    ])

    assert len(engine.trade_history) >= 2
    metrics = engine.calculate_metrics()
    assert metrics["trade_count"] >= 2
    assert "sharpe_ratio" in metrics
