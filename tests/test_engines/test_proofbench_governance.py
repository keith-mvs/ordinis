import pytest
import pandas as pd
from datetime import datetime
from dataclasses import dataclass

from ordinis.engines.proofbench.core.engine import ProofBenchEngine
from ordinis.engines.base.hooks import PreflightResult, Decision


class DenyHook:
    """Simple denying governance hook for tests."""

    def __init__(self, reason: str = "policy"):  # not a real GovernanceHook class but sufficient for tests
        self.reason = reason
        self.audit_called = False

    async def preflight(self, context):
        return PreflightResult(decision=Decision.DENY, reason=self.reason)

    async def audit(self, record):
        self.audit_called = True

    async def on_error(self, error):
        pass


@pytest.mark.asyncio
async def test_run_backtest_blocked_by_governance():
    engine = ProofBenchEngine(governance_hook=DenyHook())

    await engine.initialize()
    df = pd.DataFrame({
        "open": [1, 2, 3],
        "high": [1, 2, 3],
        "low": [1, 2, 3],
        "close": [1, 2, 3],
        "volume": [100, 200, 300],
    }, index=pd.date_range("2020-01-01", periods=3, freq="D"))

    engine.load_data("FOO", df)

    with pytest.raises(PermissionError):
        await engine.run_backtest()


@pytest.mark.asyncio
async def test_run_backtest_allows_when_governance_allows():
    class AllowHook:
        async def preflight(self, context):
            return PreflightResult(decision=Decision.ALLOW, reason="ok")

        async def audit(self, record):
            pass

        async def on_error(self, error):
            pass

    engine = ProofBenchEngine(governance_hook=AllowHook())
    await engine.initialize()

    df = pd.DataFrame({
        "open": [1, 2, 3],
        "high": [1, 2, 3],
        "low": [1, 2, 3],
        "close": [1, 2, 3],
        "volume": [100, 200, 300],
    }, index=pd.date_range("2020-01-01", periods=3, freq="D"))

    engine.load_data("FOO", df)

    # set a trivial strategy to allow simulator to run
    def strategy(engine_inst, symbol, bar):
        return None

    engine.set_strategy(strategy)

    res = await engine.run_backtest()
    assert engine.backtests_run >= 1
