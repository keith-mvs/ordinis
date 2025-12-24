import pandas as pd
import numpy as np
import pytest

from ordinis.engines.proofbench.analytics.performance import (
    PerformanceAnalyzer,
    compare_to_benchmark,
)


def test_performance_analyzer_empty():
    pa = PerformanceAnalyzer()
    metrics = pa.analyze([], [], initial_capital=1000.0)
    assert metrics.total_return == 0.0
    assert metrics.num_trades == 0


def test_annualized_return_and_volatility_and_drawdown():
    pa = PerformanceAnalyzer()
    # Create simple equity curve over 2 days
    now = pd.Timestamp("2025-01-01T00:00:00Z")
    eq = [
        (now, 1000.0),
        (now + pd.Timedelta(days=1), 1100.0),
        (now + pd.Timedelta(days=2), 1050.0),
    ]
    metrics = pa.analyze(eq, [], initial_capital=1000.0)
    assert metrics.equity_final == 1050.0
    # volatility should be non-negative
    assert metrics.volatility >= 0.0
    assert metrics.max_drawdown <= 0.0


def test_trade_statistics():
    pa = PerformanceAnalyzer()

    class T:
        def __init__(self, pnl, duration):
            self.pnl = pnl
            self.duration = duration

        @property
        def is_winner(self):
            return self.pnl > 0

        @property
        def is_loser(self):
            return self.pnl < 0

    trades = [T(100.0, 3600.0), T(-50.0, 7200.0), T(200.0, 1800.0)]
    stats = pa._trade_statistics(trades)
    assert stats["num_trades"] == 3
    assert stats["win_rate"] == pytest.approx((2 / 3) * 100)
    assert stats["largest_win"] == 200.0
    assert stats["largest_loss"] == -50.0


def test_compare_to_benchmark_basic_and_errors():
    # Create non-degenerate returns series
    s = pd.Series([0.01, 0.02, -0.005], index=pd.date_range("2025-01-01", periods=3))
    b = pd.Series([0.009, 0.018, -0.004], index=s.index)
    metrics = compare_to_benchmark(s, b)
    assert not np.isnan(metrics.beta)
    assert not np.isnan(metrics.correlation)

    # Disjoint indices should raise
    s2 = pd.Series([0.01], index=[pd.Timestamp("2020-01-01")])
    with pytest.raises(ValueError):
        compare_to_benchmark(s2, b)

    # Degenerate identical constant series should produce NaN metrics
    c = pd.Series([0.0, 0.0, 0.0], index=pd.date_range("2025-01-01", periods=3))
    res = compare_to_benchmark(c, c)
    assert np.isnan(res.beta) or np.isnan(res.correlation)
