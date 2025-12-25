import json
from pathlib import Path

import pytest

import scripts.optimizer_postprocess as post


class FakeOptimizer:
    def __init__(self, *args, **kwargs):
        pass

    def load_data(self):
        return True

    def _run_holdout_test(self, params):
        return {"cagr": 12.34, "sharpe": 1.78}

    def _compute_bootstrap_ci(self, params, n_bootstrap=100, confidence=0.95):
        return {"cagr": (10.0, 14.0), "sharpe": (1.2, 2.1)}


def test_postprocess_updates_artifact(tmp_path, monkeypatch):
    # Create artifact
    out = tmp_path / 'artifacts' / 'optimization'
    out.mkdir(parents=True)
    data = {"strategy_name": "fibonacci_adx", "best_params": {"adx_period": 13}}
    json_path = out / 'fibonacci_adx_optimization.json'
    json_path.write_text(json.dumps(data))

    # Patch StrategyOptimizer in module
    monkeypatch.setattr(post, 'StrategyOptimizer', FakeOptimizer)

    # Run main with args
    monkeypatch.chdir(tmp_path)
    post_args = ['--strategy', 'fibonacci_adx', '--bootstrap-n', '10']

    # Invoke main via run
    import sys
    orig_argv = sys.argv
    sys.argv = ['optimizer_postprocess.py'] + post_args
    try:
        post.main()
    finally:
        sys.argv = orig_argv

    # Validate artifact updated
    out_json = json.loads(json_path.read_text())
    assert 'test_metrics' in out_json
    assert out_json['test_metrics']['cagr'] == 12.34
    assert 'bootstrap_ci' in out_json
    assert out_json['bootstrap_ci']['cagr'] == [10.0, 14.0] if isinstance(out_json['bootstrap_ci']['cagr'], list) else out_json['bootstrap_ci']['cagr']
