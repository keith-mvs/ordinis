#!/usr/bin/env python3
"""
Run evaluate_params for a single symbol and paramset to inspect why no trades were generated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.strategy_optimizer import StrategyOptimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default=None)
    parser.add_argument('--artifact', type=str, default='artifacts/optimization/fibonacci_adx_optimization.json')
    parser.add_argument('--years', type=int, default=5)
    args = parser.parse_args()

    # Load params
    art = Path(args.artifact)
    if not art.exists():
        print('artifact not found', art)
        return
    j = json.loads(art.read_text())
    params = j.get('best_params', {})
    print('best_params:', params)

    opt = StrategyOptimizer(strategy_name='fibonacci_adx', use_gpu=False, years=args.years, n_symbols=10)
    ok = opt.load_data()
    print('load_data ok', ok, 'symbols:', list(opt.data_cache.keys())[:20])

    symbols = list(opt.data_cache.keys())
    if not symbols:
        print('No symbols available; aborting')
        return

    symbol = args.symbol or symbols[0]
    print('Testing on symbol', symbol)

    score, metrics = opt.evaluate_params(params, symbols=[symbol])
    print('score', score)
    print('metrics', json.dumps(metrics, indent=2))

    # Save to artifacts for reliable inspection
    outdir = Path('artifacts/optimization')
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f'single_symbol_eval_{symbol}.json'
    with open(outpath, 'w') as f:
        json.dump({'symbol': symbol, 'score': score, 'metrics': metrics}, f, indent=2)
    print(f'Wrote results to {outpath}')

if __name__ == '__main__':
    main()
