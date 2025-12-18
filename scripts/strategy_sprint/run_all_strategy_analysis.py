"""
Strategy Sprint: Execute all strategy-specific next steps.

This script runs analysis for all 8 new strategies without interfering
with live paper trading.

Usage:
    python scripts/strategy_sprint/run_all_strategy_analysis.py
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("artifacts/logs/strategy_sprint.log"),
    ],
)
logger = logging.getLogger(__name__)


async def main():
    """Run all strategy analyses."""
    logger.info("=" * 60)
    logger.info("STRATEGY SPRINT - Starting all analyses")
    logger.info("=" * 60)

    # Import each analysis module using relative imports
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))

    import evt_overlay
    import garch_backtest
    import hmm_regime_train
    import kalman_optimize
    import mi_analysis
    import mtf_ranker
    import network_build
    import ou_pairs_discovery

    results = {}

    # 1. GARCH Backtest
    logger.info("\n[1/8] GARCH Backtest on volatile symbols...")
    results["garch"] = await garch_backtest.run()

    # 2. EVT Overlay
    logger.info("\n[2/8] EVT Risk Gate overlay wiring...")
    results["evt"] = await evt_overlay.run()

    # 3. MTF Ranker
    logger.info("\n[3/8] MTF Momentum universe ranker...")
    results["mtf"] = await mtf_ranker.run()

    # 4. Kalman Optimization
    logger.info("\n[4/8] Kalman Q/R parameter optimization...")
    results["kalman"] = await kalman_optimize.run()

    # 5. OU Pairs
    logger.info("\n[5/8] OU Pairs cointegration discovery...")
    results["ou_pairs"] = await ou_pairs_discovery.run()

    # 6. MI Analysis
    logger.info("\n[6/8] MI signal predictive analysis...")
    results["mi"] = await mi_analysis.run()

    # 7. HMM Regime
    logger.info("\n[7/8] HMM SPY regime training...")
    results["hmm"] = await hmm_regime_train.run()

    # 8. Network Build
    logger.info("\n[8/8] Network correlation construction...")
    results["network"] = await network_build.run()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("STRATEGY SPRINT - Complete")
    logger.info("=" * 60)

    for name, result in results.items():
        status = "✅" if result.get("success", False) else "❌"
        logger.info(f"{status} {name}: {result.get('summary', 'No summary')}")

    return results


if __name__ == "__main__":
    asyncio.run(main())
