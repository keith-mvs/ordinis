import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import optimizer_apply


def run_script_in_tmp(tmp_path, args):
    # Run the script by invoking it directly, but ensure PYTHONPATH contains repo root
    repo_root = Path(__file__).parents[2]
    cmd = [sys.executable, str(repo_root / 'scripts' / 'optimizer_apply.py')] + args
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo_root) + os.pathsep + env.get('PYTHONPATH', '')
    res = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True)
    return res


def test_dry_run_does_not_write_config(tmp_path, monkeypatch):
    # Prepare fake result
    result = {
        "strategy_name": "fibonacci_adx",
        "best_params": {
            "adx_period": 13,
            "adx_threshold": 30,
            "swing_lookback": 60,
            "tolerance": 0.01,
            "fib_382_weight": 0.3,
            "fib_500_weight": 0.5,
            "fib_618_weight": 0.7,
            "take_profit_1272_mult": 2.0,
            "take_profit_1618_mult": 3.0,
            "require_volume_confirmation": 1,
            "max_pyramids": 1,
        },
    }

    res_file = tmp_path / 'result.json'
    res_file.write_text(json.dumps(result))

    # Run dry-run
    res = run_script_in_tmp(tmp_path, ['--result', str(res_file), '--dry-run'])
    assert res.returncode == 0
    assert 'Dry-run mode' in res.stdout or 'Dry-run' in res.stdout

    # No configs directory should exist
    assert not (tmp_path / 'configs' / 'strategies').exists()


def test_apply_writes_config_and_audit(tmp_path, monkeypatch):
    result = {
        "strategy_name": "fibonacci_adx",
        "best_params": {
            "adx_period": 13,
            "adx_threshold": 30,
            "swing_lookback": 60,
            "tolerance": 0.01,
            "fib_382_weight": 0.3,
            "fib_500_weight": 0.5,
            "fib_618_weight": 0.7,
            "take_profit_1272_mult": 2.0,
            "take_profit_1618_mult": 3.0,
            "require_volume_confirmation": 1,
            "max_pyramids": 1,
        },
        "edge_zone_params": [],
    }

    res_file = tmp_path / 'result.json'
    res_file.write_text(json.dumps(result))

    # Run with --yes (auto-approve)
    res = run_script_in_tmp(tmp_path, ['--result', str(res_file), '--yes'])
    assert res.returncode == 0, res.stderr

    cfg_path = tmp_path / 'configs' / 'strategies' / 'fibonacci_adx.yaml'
    assert cfg_path.exists()

    text = cfg_path.read_text()
    assert 'global_params' in text
    assert 'adx_period' in text

    # Audit log written
    audit_log = tmp_path / 'artifacts' / 'optimization' / 'apply-log.json'
    assert audit_log.exists()
    lines = audit_log.read_text().strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert entry['strategy'] == 'fibonacci_adx'
    assert 'changes' in entry


def test_validation_rejects_out_of_bounds(tmp_path):
    # adx_threshold upper bound is 40 per ParameterSpec
    bad_result = {
        "strategy_name": "fibonacci_adx",
        "best_params": {
            "adx_period": 13,
            "adx_threshold": 400.0,
        },
    }
    res_file = tmp_path / 'bad.json'
    res_file.write_text(json.dumps(bad_result))

    res = run_script_in_tmp(tmp_path, ['--result', str(res_file), '--yes'])
    # Our script raises RuntimeError which manifests as non-zero exit code
    assert res.returncode != 0
    assert 'Parameter validation failed' in res.stderr or 'out of bounds' in res.stderr
