import pandas as pd

from ordinis.engines.proofbench.analytics.monte_carlo import MonteCarloAnalyzer
from ordinis.engines.proofbench.analytics.performance import BenchmarkMetrics, compare_to_benchmark
from ordinis.engines.proofbench.analytics.walk_forward import WalkForwardAnalyzer


def test_walk_forward_basic():
    returns = pd.Series([0.01] * 120)  # constant 1% per period
    wf = WalkForwardAnalyzer(train_size=60, test_size=30)
    res = wf.analyze(returns)
    assert res.num_windows > 0
    assert res.robustness_ratio == 1.0  # identical in/out sample mean


def test_monte_carlo_bootstrap_and_shuffle():
    returns = pd.Series([0.02, -0.01, 0.03, 0.0])
    mc = MonteCarloAnalyzer(simulations=200, seed=42)

    boot = mc.return_bootstrap(returns)
    shuf = mc.trade_shuffle(returns)

    for res in (boot, shuf):
        assert 0 <= res.prob_loss <= 1
        assert res.simulations == 200


def test_compare_to_benchmark():
    strategy = pd.Series([0.02, 0.01, 0.015, -0.005])
    bench = pd.Series([0.01, -0.005, 0.02, 0.0])
    metrics = compare_to_benchmark(strategy, bench, risk_free_rate=0.0)
    assert isinstance(metrics, BenchmarkMetrics)
    assert metrics.beta > 0
    assert metrics.correlation > 0
