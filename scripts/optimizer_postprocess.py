#!/usr/bin/env python3
"""
Post-process optimization artifacts: run holdout and bootstrap CI on best parameters.

Usage:
  python scripts/optimizer_postprocess.py --strategy fibonacci_adx --bootstrap-n 100

Behavior:
- Loads artifacts/optimization/{strategy}_optimization.json and pickle
- If best_params exist, runs final holdout test and bootstrap CI and updates the JSON/pickle
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

from scripts.strategy_optimizer import StrategyOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optimizer_postprocess")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--bootstrap-n", type=int, default=100)
    parser.add_argument("--years", type=int, default=5)
    args = parser.parse_args()

    strategy = args.strategy
    outdir = Path("artifacts/optimization")
    json_path = outdir / f"{strategy}_optimization.json"
    pkl_path = outdir / f"{strategy}_optimization.pkl"

    if not json_path.exists():
        logger.error(f"JSON result not found: {json_path}")
        return

    with open(json_path) as f:
        data = json.load(f)

    best_params = data.get("best_params") or {}
    if not best_params:
        logger.error("No best_params found in artifact; aborting post-process")
        return

    logger.info("Starting holdout test and bootstrap CI")

    opt = StrategyOptimizer(strategy_name=strategy, n_cycles=1, use_gpu=True, years=args.years, bootstrap_n=args.bootstrap_n, skip_bootstrap=False)
    ok = opt.load_data()
    if not ok:
        logger.error("Failed to load data for postprocessing")
        return

    test_metrics = opt._run_holdout_test(best_params)
    bootstrap_ci = opt._compute_bootstrap_ci(best_params, n_bootstrap=args.bootstrap_n, confidence=0.95)

    data["test_metrics"] = test_metrics
    data["bootstrap_ci"] = bootstrap_ci

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    # Update pickle if present
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)
            # attach computed metrics
            obj.test_cagr = test_metrics.get("cagr", 0)
            obj.test_sharpe = test_metrics.get("sharpe", 0)
            obj.bootstrap_cagr_ci = bootstrap_ci.get("cagr", (0, 0))
            obj.bootstrap_sharpe_ci = bootstrap_ci.get("sharpe", (0, 0))
            with open(pkl_path, "wb") as f:
                pickle.dump(obj, f)
            logger.info(f"Updated pickle at {pkl_path}")
        except Exception as e:
            logger.warning(f"Failed to update pickle: {e}")

    logger.info("Post-processing completed and artifacts updated.")


if __name__ == "__main__":
    main()
