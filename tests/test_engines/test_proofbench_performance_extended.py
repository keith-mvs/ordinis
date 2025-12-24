from types import SimpleNamespace
from math import isclose, isnan

import pandas as pd
import numpy as np

from ordinis.engines.proofbench.analytics.performance import (
    PerformanceAnalyzer,
    compare_to_benchmark,
)


def test_empty_metrics():
    pa = PerformanceAnalyzer()
    metrics = pa.analyze([], [], 1000.0)
    assert metrics.total_return == 0.0
    assert metrics.num_trades == 0
    assert metrics.equity_final == 0.0


def test_basic_analyze_no_trades():
    pa = PerformanceAnalyzer()
    # 3 daily points
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    equity = [(dates[0], 1000.0), (dates[1], 1100.0), (dates[2], 1210.0)]
    metrics = pa.analyze(equity, [], 1000.0)
    # total return should be 21% ((1210-1000)/1000*100)
    assert isclose(metrics.total_return, 21.0, rel_tol=1e-6)
    assert metrics.num_trades == 0
    assert metrics.equity_final == 1210.0


def test_drawdown_and_trade_statistics():
    pa = PerformanceAnalyzer()
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    equity = [(dates[0], 1000.0), (dates[1], 1200.0), (dates[2], 800.0), (dates[3], 900.0), (dates[4], 1000.0)]

    # Create simple trades
    # winner: pnl 10, duration 3600 (1 hour)
    # loser1: pnl -5, duration 7200 (2 hours)
    # loser2: pnl -3, duration 1800 (0.5 hour)
    trades = [
        SimpleNamespace(is_winner=True, is_loser=False, pnl=10.0, duration=3600),
        SimpleNamespace(is_winner=False, is_loser=True, pnl=-5.0, duration=7200),
        SimpleNamespace(is_winner=False, is_loser=True, pnl=-3.0, duration=1800),
    ]

    metrics = pa.analyze(equity, trades, 1000.0)
    assert metrics.max_drawdown < 0.0
    assert metrics.num_trades == 3

    # Check trade statistics roughly match expectations
    # win_rate = 1/3 * 100
    assert isclose(metrics.win_rate, (1 / 3) * 100, rel_tol=1e-6)
    # profit_factor = gross_profit / gross_loss = 10 / 8 = 1.25
    assert isclose(metrics.profit_factor, 10.0 / 8.0, rel_tol=1e-6)


def test_compare_to_benchmark_errors_and_basic():
    # No overlapping indices -> should raise
    s = pd.Series([0.01], index=[pd.Timestamp("2020-01-01")])
    b = pd.Series([0.02], index=[pd.Timestamp("2020-02-01")])
    try:
        compare_to_benchmark(s, b)
        assert False, "Expected ValueError for no overlapping returns"
    except ValueError:
        pass

    # Basic identical series -> beta ~ 1, corr ~ 1, r_squared ~ 1
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    s2 = pd.Series(0.01, index=idx)
    b2 = pd.Series(0.01, index=idx)
    bm = compare_to_benchmark(s2, b2)
    # Degenerate constant series -> variance zero -> beta/corr/r_squared are NaN
    assert isnan(bm.beta)
    assert isnan(bm.correlation)
    assert isnan(bm.r_squared)
    assert isnan(bm.information_ratio)
    assert isnan(bm.treynor_ratio)
