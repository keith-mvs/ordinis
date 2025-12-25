#!/usr/bin/env python3
"""
Debug tool: inspect ADX and Fibonacci model outputs for sample symbols.

Usage:
  PYTHONPATH=$(pwd) conda activate ordinis-gpu && python scripts/debug_fib_adx.py --n-symbols 5 --max-days 250

This script loads the optimizer artifact best params and runs through recent timestamps,
printing ADX and Fibonacci signals and reasons why the combined strategy returns None.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from ordinis.application.strategies.fibonacci_adx import FibonacciADXStrategy
from ordinis.engines.signalcore.core.model import ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("debug_fib_adx")


def load_best_params(artifact_path: Path) -> dict[str, Any]:
    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)
    with open(artifact_path) as f:
        j = json.load(f)
    return j.get("best_params", {})


from scripts.strategy_optimizer import StrategyOptimizer


def load_symbol_data_via_optimizer(n_symbols: int | None = None, years: int | None = 5) -> dict[str, pd.DataFrame]:
    """Use StrategyOptimizer.load_data which supports CSVs and API fallbacks."""
    opt = StrategyOptimizer(strategy_name="fibonacci_adx", use_gpu=False, years=years, n_symbols=n_symbols, n_cycles=1)
    ok = opt.load_data()
    if not ok:
        logger.warning("StrategyOptimizer failed to load data (no CSVs or API fetch).")
        return {}
    return opt.data_cache


async def inspect_symbol(symbol: str, df: pd.DataFrame, params: dict[str, Any], sample_days: int = 200):
    print(f"\n--- SYMBOL: {symbol} (rows={len(df)}) ---")

    # Build strategy instance with params
    s = FibonacciADXStrategy(name="debug_fib", **params)
    s.configure()

    # Report min required
    min_required = s.required_bars
    print(f"min_required: {min_required}")

    if len(df) < min_required:
        print("Insufficient data for this symbol (len < min_required). Skipping.")
        return

    # limit to last sample_days
    start_idx = max(min_required, len(df) - sample_days)
    indices = list(range(start_idx, len(df)))

    found_any = 0
    for i in indices:
        lookback = df.iloc[: i + 1]
        timestamp = lookback.index[-1]
        try:
            adx_sig = await s.adx_model.generate(lookback, timestamp)
        except Exception as e:
            print(f"{timestamp} ADX generate exception: {e}")
            adx_sig = None
        if adx_sig is None:
            # ADX did not produce a signal; print a short summary of ADX value if possible
            # some models expose last calculated adx via features
            last_adx = None
            try:
                # attempt to compute adx feature by calling feature functions if available
                # fallback: look for 'adx' in last row
                last_adx = lookback.get("adx") if "adx" in lookback.columns else None
            except Exception:
                last_adx = None
            print(f"{timestamp} ADX None (last_adx={last_adx})")
            continue

        # If adx sig exists, print its adx value and direction
        print(f"{timestamp} ADX OK: adx={adx_sig.metadata.get('adx')} direction={adx_sig.direction} score={getattr(adx_sig,'score',None)}")

        # Check threshold test
        if adx_sig.metadata.get("adx", 0) < s.adx_threshold:
            print(f"  ADX below threshold {s.adx_threshold}: {adx_sig.metadata.get('adx')}")
            continue

        # Next check fib
        try:
            fib_sig = await s.fib_model.generate(lookback, timestamp)
        except Exception as e:
            print(f"{timestamp} Fib generate exception: {e}")
            fib_sig = None

        if fib_sig is None:
            print(f"  Fib None (no level match)")
            continue

        print(f"  Fib OK: direction={fib_sig.direction} type={fib_sig.signal_type} score={getattr(fib_sig,'score',None)} metadata={dict(list(fib_sig.metadata.items())[:5])}")

        # Combined check
        if adx_sig.direction != fib_sig.direction:
            print("  Direction mismatch: ADX vs Fib")
            continue

        if fib_sig.signal_type != fib_sig.signal_type.ENTRY:
            print("  Fib not ENTRY")
            continue

        print(f"  -> Combined ENTRY at {timestamp} (prob={getattr(fib_sig,'probability',None)})")
        found_any += 1

    if found_any == 0:
        print("No combined entries found in sample window.")
    else:
        print(f"Found {found_any} combined entries in sample window.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-symbols", type=int, default=5)
    parser.add_argument("--max-days", type=int, default=250)
    parser.add_argument("--artifact", type=str, default="artifacts/optimization/fibonacci_adx_optimization.json")
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args()

    best_params = load_best_params(Path(args.artifact))
    print("Loaded best_params:", best_params)

    symmap = load_symbol_data_via_optimizer(n_symbols=args.n_symbols, years=args.years)
    if not symmap:
        print("No data available via CSVs or API for the requested symbols/horizon.")
        return

    # Run inspection per symbol
    loop = asyncio.get_event_loop()
    for symbol, df in list(symmap.items())[: args.n_symbols]:
        loop.run_until_complete(inspect_symbol(symbol, df, best_params, sample_days=args.max_days))


if __name__ == "__main__":
    main()
