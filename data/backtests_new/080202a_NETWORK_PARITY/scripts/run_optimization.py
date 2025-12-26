#!/usr/bin/env python3
"""
Network Parity Portfolio Optimization - Round 1

Runs the full ML optimization pipeline across multiple market regimes
using NVIDIA Nemo for intelligent parameter suggestions.

Target: >30% average profit across all periods and symbols.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
import aiohttp

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from config import (
    OUTPUT_DIR,
    NEMOTRON_MODEL,
    NEMOTRON_ENDPOINT,
    NVIDIA_API_KEY_ENV,
    NetworkParityParams,
    OptimizationConfig,
    OptimizationHyperparams,
    PARAMETER_BOUNDS,
)
from data_pipeline import HistoricalDataPipeline, DataPipelineResult
from backtesting import BacktestEngine, BacktestResult
from reporting import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
    ]
)
logger = logging.getLogger(__name__)

# NVIDIA API Key
NVIDIA_API_KEY = os.environ.get(NVIDIA_API_KEY_ENV, "")


@dataclass
class PeriodResult:
    """Results for a single period."""
    period_name: str
    n_symbols: int
    total_return: float
    avg_return_per_symbol: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    n_trades: int


@dataclass
class OptimizationResult:
    """Complete optimization result."""
    iteration: int
    params: dict
    period_results: list[PeriodResult]
    avg_return: float
    avg_sharpe: float
    avg_sortino: float
    avg_win_rate: float
    composite_score: float
    timestamp: str


async def call_nemo_api(
    session: aiohttp.ClientSession,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> str | None:
    """Call NVIDIA Nemo API for parameter suggestions."""
    if not NVIDIA_API_KEY:
        logger.warning("NVIDIA_API_KEY not set, using random perturbation")
        return None

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": NEMOTRON_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are a quantitative trading optimization expert.
Your task is to suggest parameter adjustments for a Network Parity portfolio strategy.
The strategy uses correlation networks to weight assets and z-score signals for mean reversion.
Always respond with valid JSON containing parameter suggestions."""
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        async with session.post(NEMOTRON_ENDPOINT, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data["choices"][0]["message"]["content"]
            else:
                text = await response.text()
                logger.error(f"Nemo API error {response.status}: {text[:200]}")
                return None
    except Exception as e:
        logger.error(f"Nemo API exception: {e}")
        return None


def generate_perturbation(
    current_params: dict[str, Any],
    scale: float = 0.15,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate random parameter perturbation."""
    rng = np.random.default_rng(seed)
    new_params = current_params.copy()

    integer_params = {"corr_lookback", "recalc_frequency", "momentum_lookback", "max_positions"}

    for param, (min_val, max_val) in PARAMETER_BOUNDS.items():
        if param in current_params:
            current = current_params[param]
            range_val = max_val - min_val
            delta = rng.normal(0, scale * range_val)
            new_value = max(min_val, min(max_val, current + delta))

            if param in integer_params:
                new_value = int(round(new_value))

            new_params[param] = new_value

    return new_params


async def parse_nemo_suggestions(response: str, current_params: dict) -> dict[str, Any]:
    """Parse Nemo response and extract parameter suggestions."""
    try:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            suggestions = json.loads(json_match.group())
            new_params = current_params.copy()

            for key, value in suggestions.items():
                if key in PARAMETER_BOUNDS:
                    min_val, max_val = PARAMETER_BOUNDS[key]
                    new_params[key] = max(min_val, min(max_val, float(value)))

            return new_params
    except Exception as e:
        logger.warning(f"Failed to parse Nemo response: {e}")

    # Fall back to random perturbation
    return generate_perturbation(current_params)


def run_backtest_for_period(
    period_data: DataPipelineResult,
    params: NetworkParityParams,
) -> PeriodResult | None:
    """Run backtest for a single period."""
    if period_data.n_symbols < 5:
        return None

    try:
        engine = BacktestEngine(params)
        returns_matrix = period_data.get_returns_matrix(timeframe="1D")

        if returns_matrix.empty or len(returns_matrix) < 5:
            return None

        # Simple backtest simulation
        n_days = len(returns_matrix)
        n_symbols = len(returns_matrix.columns)

        # Calculate correlation network weights
        corr_matrix = returns_matrix.corr()

        # Centrality-based weights (simplified eigenvector centrality)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix.values)
            centrality = np.abs(eigenvectors[:, -1])
            weights = centrality / centrality.sum()
        except Exception:
            weights = np.ones(n_symbols) / n_symbols

        # Apply weight constraints
        weights = np.clip(weights, params.min_weight, params.max_weight)
        weights = weights / weights.sum()

        # Calculate portfolio returns
        portfolio_returns = (returns_matrix.values * weights).sum(axis=1)

        # Calculate metrics
        total_return = (1 + portfolio_returns).prod() - 1
        avg_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()

        sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0

        downside = portfolio_returns[portfolio_returns < 0]
        downside_std = downside.std() if len(downside) > 0 else std_return
        sortino = (avg_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Win rate
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)

        return PeriodResult(
            period_name="",  # Will be set by caller
            n_symbols=n_symbols,
            total_return=float(total_return),
            avg_return_per_symbol=float(total_return / n_symbols),
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            max_drawdown=float(max_drawdown),
            win_rate=float(win_rate),
            n_trades=n_days,
        )

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        return None


def compute_composite_score(results: list[PeriodResult]) -> float:
    """Compute composite optimization score."""
    if not results:
        return 0.0

    avg_return = np.mean([r.total_return for r in results])
    avg_sharpe = np.mean([r.sharpe_ratio for r in results])
    avg_sortino = np.mean([r.sortino_ratio for r in results])
    avg_win_rate = np.mean([r.win_rate for r in results])
    avg_drawdown = np.mean([r.max_drawdown for r in results])

    # Weighted score favoring returns (target: >30%)
    score = (
        0.45 * avg_return +       # Heavy weight on returns
        0.20 * avg_sharpe / 3 +   # Normalized Sharpe
        0.15 * avg_sortino / 4 +  # Normalized Sortino
        0.10 * avg_win_rate +     # Win rate
        0.10 * (1 - avg_drawdown) # Drawdown penalty
    )

    return float(score)


async def run_optimization(
    max_iterations: int = 50,
    use_nemo: bool = True,
    target_return: float = 0.30,
) -> list[OptimizationResult]:
    """Run the full optimization loop."""

    logger.info("=" * 70)
    logger.info("NETWORK PARITY OPTIMIZATION - ROUND 1")
    logger.info(f"Target: >{target_return*100:.0f}% average return")
    logger.info("=" * 70)

    # Load historical data
    pipeline = HistoricalDataPipeline()
    periods = pipeline.get_available_periods()
    logger.info(f"Loading {len(periods)} historical periods...")

    period_data = pipeline.load_all_periods()

    # Filter out periods with insufficient data
    valid_periods = {k: v for k, v in period_data.items() if v.n_symbols >= 5}
    logger.info(f"Valid periods: {list(valid_periods.keys())}")

    # Initialize parameters
    current_params = asdict(NetworkParityParams())
    best_params = current_params.copy()
    best_score = float("-inf")
    best_return = float("-inf")

    results_history: list[OptimizationResult] = []

    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        for iteration in range(max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration + 1}/{max_iterations}")
            logger.info(f"{'='*60}")

            # Run backtests for all periods
            period_results = []
            for period_name, data in valid_periods.items():
                params = NetworkParityParams(**current_params)
                result = run_backtest_for_period(data, params)
                if result:
                    result.period_name = period_name
                    period_results.append(result)
                    logger.info(
                        f"  {period_name}: return={result.total_return*100:.2f}%, "
                        f"sharpe={result.sharpe_ratio:.2f}, win_rate={result.win_rate*100:.1f}%"
                    )

            if not period_results:
                logger.error("No valid results for this iteration")
                continue

            # Calculate aggregate metrics
            avg_return = np.mean([r.total_return for r in period_results])
            avg_sharpe = np.mean([r.sharpe_ratio for r in period_results])
            avg_sortino = np.mean([r.sortino_ratio for r in period_results])
            avg_win_rate = np.mean([r.win_rate for r in period_results])
            composite_score = compute_composite_score(period_results)

            logger.info(f"\n  AGGREGATE RESULTS:")
            logger.info(f"    Average Return: {avg_return*100:.2f}%")
            logger.info(f"    Average Sharpe: {avg_sharpe:.2f}")
            logger.info(f"    Average Sortino: {avg_sortino:.2f}")
            logger.info(f"    Average Win Rate: {avg_win_rate*100:.1f}%")
            logger.info(f"    Composite Score: {composite_score:.4f}")

            # Track best
            if composite_score > best_score:
                best_score = composite_score
                best_params = current_params.copy()
                best_return = avg_return
                logger.info(f"  *** NEW BEST SCORE: {best_score:.4f} ***")

            # Save result
            opt_result = OptimizationResult(
                iteration=iteration + 1,
                params=current_params.copy(),
                period_results=period_results,
                avg_return=avg_return,
                avg_sharpe=avg_sharpe,
                avg_sortino=avg_sortino,
                avg_win_rate=avg_win_rate,
                composite_score=composite_score,
                timestamp=datetime.now().isoformat(),
            )
            results_history.append(opt_result)

            # Save iteration results
            iteration_dir = OUTPUT_DIR / "iterations" / f"Iteration_{iteration + 1:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)

            with open(iteration_dir / "results.json", "w") as f:
                json.dump({
                    "iteration": iteration + 1,
                    "params": current_params,
                    "avg_return": avg_return,
                    "avg_sharpe": avg_sharpe,
                    "composite_score": composite_score,
                    "period_results": [asdict(r) for r in period_results],
                }, f, indent=2)

            # Check if target reached
            if avg_return >= target_return:
                logger.info(f"\n*** TARGET ACHIEVED: {avg_return*100:.2f}% >= {target_return*100:.0f}% ***")
                break

            # Generate new parameters using Nemo or random perturbation
            if use_nemo and NVIDIA_API_KEY:
                # Build prompt for Nemo
                prompt = f"""Current optimization iteration {iteration + 1} results:
- Average Return: {avg_return*100:.2f}%
- Target Return: {target_return*100:.0f}%
- Sharpe Ratio: {avg_sharpe:.2f}
- Win Rate: {avg_win_rate*100:.1f}%

Current parameters:
{json.dumps(current_params, indent=2)}

Parameter bounds:
{json.dumps({k: list(v) for k, v in PARAMETER_BOUNDS.items()}, indent=2)}

Best results so far came from periods with higher volatility stocks.
The strategy uses correlation-based network centrality for asset weighting.

Please suggest parameter adjustments to achieve the {target_return*100:.0f}% return target.
Focus on parameters that improve performance in high-volatility regimes.
Return a JSON object with the suggested parameter values."""

                logger.info("Requesting NVIDIA Nemo suggestions...")
                nemo_response = await call_nemo_api(session, prompt)

                if nemo_response:
                    current_params = await parse_nemo_suggestions(nemo_response, current_params)
                    logger.info("Applied Nemo-suggested parameters")
                else:
                    current_params = generate_perturbation(current_params, scale=0.15)
                    logger.info("Applied random perturbation (Nemo unavailable)")
            else:
                # Adaptive perturbation scale based on progress
                progress = (best_return - 0) / (target_return - 0) if target_return > 0 else 0
                scale = 0.20 * (1 - min(progress, 0.8))  # Reduce scale as we approach target
                current_params = generate_perturbation(current_params, scale=scale)
                logger.info(f"Applied random perturbation (scale={scale:.3f})")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Best Score: {best_score:.4f}")
    logger.info(f"Best Return: {best_return*100:.2f}%")
    logger.info(f"Target Return: {target_return*100:.0f}%")
    logger.info(f"Target Achieved: {'YES' if best_return >= target_return else 'NO'}")
    logger.info(f"\nBest Parameters:")
    for key, value in best_params.items():
        logger.info(f"  {key}: {value}")

    # Save final summary
    summary = {
        "optimization_id": f"R1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "target_return": target_return,
        "best_return": best_return,
        "best_score": best_score,
        "target_achieved": best_return >= target_return,
        "total_iterations": len(results_history),
        "best_params": best_params,
        "final_timestamp": datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / "summary" / f"optimization_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results_history


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Network Parity Optimization")
    parser.add_argument("--max-iterations", type=int, default=50, help="Maximum iterations")
    parser.add_argument("--no-nemo", action="store_true", help="Disable NVIDIA Nemo")
    parser.add_argument("--target-return", type=float, default=0.30, help="Target return (default: 0.30 = 30%)")
    args = parser.parse_args()

    results = await run_optimization(
        max_iterations=args.max_iterations,
        use_nemo=not args.no_nemo,
        target_return=args.target_return,
    )

    logger.info(f"\nCompleted {len(results)} iterations")


if __name__ == "__main__":
    asyncio.run(main())
