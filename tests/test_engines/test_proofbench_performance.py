import pytest
import pandas as pd
from datetime import datetime

from ordinis.engines.proofbench.analytics.performance import PerformanceAnalyzer


class DummyTrade:
    def __init__(self, pnl: float, duration_seconds: float):
        self.pnl = pnl
        self.duration = duration_seconds

    @property
    def is_winner(self):
        return self.pnl > 0

    @property
    def is_loser(self):
        return self.pnl <= 0


def make_equity_curve(values, start_ts=None):
    if start_ts is None:
        start_ts = datetime(2020, 1, 1)
    idx = pd.date_range(start_ts, periods=len(values), freq="D")
    return list(zip(idx, values))


def test_empty_metrics():
    pa = PerformanceAnalyzer()
    metrics = pa.analyze([], [], 100000.0)
    assert metrics.total_return == 0.0
    assert metrics.num_trades == 0


def test_basic_analyze():
    pa = PerformanceAnalyzer()

    equity_curve = make_equity_curve([100_000, 100_100, 100_200])
    trades = [DummyTrade(10.0, 3600), DummyTrade(-5.0, 7200)]

    m = pa.analyze(equity_curve, trades, 100_000)

    assert m.total_return >= 0.0
    assert m.num_trades == 2
    assert hasattr(m, "sharpe_ratio")
    assert isinstance(m.sharpe_ratio, float)
    assert m.equity_final == 100_200
