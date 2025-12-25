import os
from scripts.strategy_optimizer import StrategyOptimizer


def test_api_fetch_no_keys():
    opt = StrategyOptimizer(strategy_name='fibonacci_adx', n_cycles=1, use_gpu=False, skip_bootstrap=True)
    # Ensure env has no API keys
    os.environ.pop('POLYGON_API_KEY', None)
    os.environ.pop('FINNHUB_API_KEY', None)

    res = opt._fetch_symbol_via_api('FAKE', years=1)
    assert res is None
