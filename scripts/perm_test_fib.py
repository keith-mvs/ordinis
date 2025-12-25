#!/usr/bin/env python3
"""
Permissive test for Fibonacci+ADX: reduce ADX threshold and increase tolerance to see whether the strategy can generate signals.
Writes per-symbol summary to `artifacts/optimization/perm_test_fib.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.strategy_optimizer import StrategyOptimizer
from ordinis.application.strategies.fibonacci_adx import FibonacciADXStrategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=int, default=5)
    parser.add_argument('--n-symbols', type=int, default=25)
    parser.add_argument('--sample-days', type=int, default=250)
    args = parser.parse_args()

    opt = StrategyOptimizer('fibonacci_adx', use_gpu=False, years=args.years, n_symbols=args.n_symbols)
    ok = opt.load_data()
    print('loaded', ok, 'symbols', len(opt.data_cache))

    results = {}
    for symbol, df in list(opt.data_cache.items())[: args.n_symbols]:
        # permissive params
        params = dict(adx_period=14, adx_threshold=10, swing_lookback=50, tolerance=0.05)
        strat = FibonacciADXStrategy('perm_test', **params)
        strat.configure()

        min_required = strat.required_bars
        if len(df) < min_required:
            results[symbol] = {'rows': len(df), 'min_required': min_required, 'entries': 0}
            continue

        entries = 0
        # iterate over last sample_days
        start_idx = max(min_required, len(df) - args.sample_days)
        for i in range(start_idx, len(df)):
            lookback = df.iloc[: i + 1]
            timestamp = lookback.index[-1]
            signal = None
            try:
                import asyncio
                signal = asyncio.get_event_loop().run_until_complete(strat.generate_signal(lookback, timestamp))
            except Exception:
                signal = None
            if signal is not None:
                entries += 1
        results[symbol] = {'rows': len(df), 'min_required': min_required, 'entries': entries}

    outdir = Path('artifacts/optimization')
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / 'perm_test_fib.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print('Wrote', outpath)


if __name__ == '__main__':
    main()
