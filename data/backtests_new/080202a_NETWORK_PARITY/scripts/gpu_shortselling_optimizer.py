#!/usr/bin/env python3
"""
GPU-Accelerated Short-Selling Network Parity Optimizer

Implements a long/short strategy that can profit in both bull and bear markets.

Key improvements over the long-only version:
1. Short-selling capability during bear regimes
2. Regime detection for market direction
3. Per-symbol tracking and reporting
4. Leverage controls for amplified returns

Target: >30% average profit across all market regimes
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np
import pandas as pd

# GPU acceleration
try:
    import torch
    import torch.nn.functional as F
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
except ImportError:
    CUDA_AVAILABLE = False
    DEVICE = None
    torch = None

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

from config import OUTPUT_DIR, HISTORICAL_DATA_DIR, HISTORICAL_DATA_DIR_V2
from data_pipeline import HistoricalDataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ShortSellingParams:
    """Parameters for long/short strategy."""
    # Momentum
    momentum_lookback: int = 5
    momentum_threshold: float = 0.02

    # Mean reversion
    zscore_lookback: int = 10
    zscore_entry: float = 1.5
    zscore_exit: float = 0.5

    # Position sizing
    concentration_factor: float = 3.0
    max_position_pct: float = 0.25
    min_position_pct: float = 0.02

    # Short-selling specific
    short_leverage: float = 1.0  # Max 2x short leverage
    long_leverage: float = 1.0   # Max 2x long leverage
    market_direction_lookback: int = 10
    bear_threshold: float = -0.02  # Market drop threshold for bear regime
    bull_threshold: float = 0.01   # Market rise threshold for bull regime

    # Risk management
    stop_loss_pct: float = 0.10
    take_profit_pct: float = 0.20
    max_short_pct: float = 0.50  # Max portfolio in shorts

    # Regime detection
    vol_threshold: float = 0.02
    trend_threshold: float = 0.01


def gpu_compute_returns(prices: np.ndarray) -> torch.Tensor:
    """GPU-accelerated returns calculation."""
    if not CUDA_AVAILABLE:
        returns = np.diff(prices, axis=0) / prices[:-1]
        return torch.from_numpy(returns).float()

    prices_t = torch.from_numpy(prices).float().to(DEVICE)
    returns = (prices_t[1:] - prices_t[:-1]) / prices_t[:-1]
    return returns


def gpu_compute_momentum(returns: torch.Tensor, lookback: int) -> torch.Tensor:
    """GPU-accelerated momentum calculation."""
    if returns.shape[0] < lookback:
        return torch.zeros(returns.shape[1], device=returns.device if CUDA_AVAILABLE else None)

    if CUDA_AVAILABLE:
        returns = returns.to(DEVICE)

    window_returns = returns[-lookback:]
    momentum = (1 + window_returns).prod(dim=0) - 1
    return momentum


def gpu_compute_zscore(prices: torch.Tensor, lookback: int) -> torch.Tensor:
    """GPU-accelerated z-score calculation."""
    if prices.shape[0] < lookback:
        return torch.zeros(prices.shape[1], device=prices.device if CUDA_AVAILABLE else None)

    if CUDA_AVAILABLE:
        prices = prices.to(DEVICE)

    window = prices[-lookback:]
    mean = window.mean(dim=0)
    std = window.std(dim=0)
    std[std == 0] = 1.0
    zscore = (prices[-1] - mean) / std
    return zscore


def detect_market_regime(
    returns: torch.Tensor,
    lookback: int,
    bear_thresh: float,
    bull_thresh: float,
    vol_thresh: float = 0.03,
) -> str:
    """
    Detect overall market regime: bull, bear, neutral, or high_vol.

    Returns:
        'bull': Market trending up - prefer longs
        'bear': Market trending down - prefer shorts
        'neutral': Sideways market - mixed positions
        'high_vol': High volatility/choppy - reduce exposure
    """
    if returns.shape[0] < lookback:
        return 'neutral'

    # Use equal-weighted market return as proxy
    window = returns[-lookback:]
    market_return = window.mean(dim=1).sum()
    market_vol = window.mean(dim=1).std()

    if CUDA_AVAILABLE:
        market_return = market_return.item()
        market_vol = market_vol.item()

    # Check for high volatility first (choppy markets like 2015, 2023)
    if market_vol > vol_thresh:
        return 'high_vol'

    if market_return < bear_thresh:
        return 'bear'
    elif market_return > bull_thresh:
        return 'bull'
    else:
        return 'neutral'


class GPUShortSellingStrategy:
    """
    Long/Short strategy with GPU acceleration.

    Strategy Logic:
    1. Detect market regime (bull/bear/neutral)
    2. In bull regime: Go long winners, small short hedges on losers
    3. In bear regime: Go short losers, small long hedges on winners
    4. In neutral: Mean reversion with balanced long/short
    """

    def __init__(self, params: ShortSellingParams):
        self.params = params
        self.device = DEVICE

    def run_backtest(
        self,
        returns_df: pd.DataFrame,
        prices_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Run backtest with long/short positions."""

        if prices_df is None:
            prices = (1 + returns_df).cumprod()
            prices_df = prices

        returns = returns_df.values
        prices = prices_df.values
        symbols = returns_df.columns.tolist()
        n_days, n_assets = returns.shape

        if n_days < 10:
            return {
                "total_return": 0, "sharpe": 0, "sortino": 0,
                "max_dd": 0, "win_rate": 0, "symbol_returns": {}
            }

        # Convert to tensors
        returns_t = torch.from_numpy(returns).float()
        prices_t = torch.from_numpy(prices).float()

        if CUDA_AVAILABLE:
            returns_t = returns_t.to(DEVICE)
            prices_t = prices_t.to(DEVICE)

        # Track portfolio and per-symbol performance
        portfolio_returns = []
        symbol_pnl = {sym: [] for sym in symbols}
        positions_history = []

        for t in range(self.params.momentum_lookback, n_days):
            # Detect market regime (including volatility check)
            regime = detect_market_regime(
                returns_t[:t],
                self.params.market_direction_lookback,
                self.params.bear_threshold,
                self.params.bull_threshold,
                self.params.vol_threshold,
            )

            # Calculate signals
            momentum = gpu_compute_momentum(
                returns_t[:t],
                self.params.momentum_lookback
            )

            zscore = gpu_compute_zscore(
                prices_t[:t],
                self.params.zscore_lookback
            )

            # Calculate volatility for position sizing
            recent_vol = returns_t[max(0, t-5):t].std(dim=0)
            vol_adj = 1.0 / (recent_vol + 0.01)  # Inverse vol for position sizing

            # Generate long/short signals based on regime
            if regime == 'bull':
                # Bull market: Long winners, small short hedges
                long_signals = momentum.clone()
                long_signals = torch.where(
                    momentum > self.params.momentum_threshold,
                    long_signals * self.params.long_leverage,
                    torch.zeros_like(long_signals)
                )

                # Short the worst performers as hedge
                short_signals = -momentum.clone()
                short_signals = torch.where(
                    momentum < -self.params.momentum_threshold,
                    short_signals * self.params.short_leverage * 0.3,  # Smaller shorts in bull
                    torch.zeros_like(short_signals)
                )

            elif regime == 'bear':
                # Bear market: Short losers aggressively, small long hedges
                short_signals = -momentum.clone()
                short_signals = torch.where(
                    momentum < -self.params.momentum_threshold,
                    short_signals * self.params.short_leverage,
                    torch.zeros_like(short_signals)
                )

                # Small longs on relative strength
                long_signals = momentum.clone()
                long_signals = torch.where(
                    momentum > self.params.momentum_threshold * 2,  # Higher bar for longs in bear
                    long_signals * self.params.long_leverage * 0.3,
                    torch.zeros_like(long_signals)
                )

            elif regime == 'high_vol':
                # High volatility: Go flat or minimal exposure (protect capital)
                # Only trade extreme mean reversions with reduced size
                long_signals = -zscore.clone()
                long_signals = torch.where(
                    zscore < -self.params.zscore_entry * 1.5,  # Higher bar
                    long_signals * self.params.long_leverage * 0.2,  # 20% size
                    torch.zeros_like(long_signals)
                )

                short_signals = zscore.clone()
                short_signals = torch.where(
                    zscore > self.params.zscore_entry * 1.5,  # Higher bar
                    short_signals * self.params.short_leverage * 0.2,  # 20% size
                    torch.zeros_like(short_signals)
                )

            else:  # neutral
                # Mean reversion strategy
                # Buy oversold (negative zscore)
                long_signals = -zscore.clone()
                long_signals = torch.where(
                    zscore < -self.params.zscore_entry,
                    long_signals * self.params.long_leverage,
                    torch.zeros_like(long_signals)
                )

                # Short overbought (positive zscore)
                short_signals = zscore.clone()
                short_signals = torch.where(
                    zscore > self.params.zscore_entry,
                    short_signals * self.params.short_leverage,
                    torch.zeros_like(short_signals)
                )

            # Combine long and short positions
            # Positive = long, Negative = short
            positions = long_signals - short_signals

            # Normalize to control leverage
            total_exposure = positions.abs().sum()
            if total_exposure > 0:
                # Scale to target net exposure (1.0 = 100% invested)
                max_exposure = 1.0 + self.params.short_leverage
                if total_exposure > max_exposure:
                    positions = positions * (max_exposure / total_exposure)

            # Apply concentration (softmax on absolute signals, preserve sign)
            abs_positions = positions.abs()
            if abs_positions.sum() > 0:
                weights = F.softmax(abs_positions * self.params.concentration_factor, dim=0)
                positions = weights * positions.sign() * positions.abs().sum()

            # Ensure short positions don't exceed max short percentage
            short_mask = positions < 0
            short_exposure = positions[short_mask].abs().sum()
            if short_exposure > self.params.max_short_pct:
                positions[short_mask] = positions[short_mask] * (self.params.max_short_pct / short_exposure)

            positions_history.append(positions.cpu().numpy() if CUDA_AVAILABLE else positions.numpy())

            # Calculate return for this day
            # For longs: positive position * positive return = profit
            # For shorts: negative position * negative return = profit (double negative)
            day_returns = returns_t[t]
            position_returns = positions * day_returns
            portfolio_return = position_returns.sum()

            portfolio_returns.append(portfolio_return.item() if CUDA_AVAILABLE else float(portfolio_return))

            # Track per-symbol P&L
            for i, sym in enumerate(symbols):
                sym_ret = position_returns[i].item() if CUDA_AVAILABLE else float(position_returns[i])
                symbol_pnl[sym].append(sym_ret)

        if not portfolio_returns:
            return {
                "total_return": 0, "sharpe": 0, "sortino": 0,
                "max_dd": 0, "win_rate": 0, "symbol_returns": {}
            }

        portfolio_returns = np.array(portfolio_returns)

        # Calculate metrics
        total_return = float((1 + portfolio_returns).prod() - 1)
        avg_return = portfolio_returns.mean()
        std_return = portfolio_returns.std() if len(portfolio_returns) > 1 else 0.001

        sharpe = float(avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0

        downside = portfolio_returns[portfolio_returns < 0]
        downside_std = downside.std() if len(downside) > 1 else std_return
        sortino = float(avg_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(abs(drawdowns.min()))

        win_rate = float((portfolio_returns > 0).sum() / len(portfolio_returns))

        # Calculate per-symbol returns
        symbol_returns = {}
        for sym in symbols:
            if symbol_pnl[sym]:
                sym_rets = np.array(symbol_pnl[sym])
                symbol_returns[sym] = {
                    "total_return": float((1 + sym_rets).prod() - 1),
                    "avg_daily": float(sym_rets.mean()),
                    "win_rate": float((sym_rets > 0).sum() / len(sym_rets)) if len(sym_rets) > 0 else 0,
                    "n_days": len(sym_rets),
                }

        return {
            "total_return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "n_days": len(portfolio_returns),
            "symbol_returns": symbol_returns,
        }


def optimize_parameters(
    pipeline: HistoricalDataPipeline,
    n_iterations: int = 100,
    population_size: int = 30,
) -> tuple[ShortSellingParams, dict, list]:
    """
    GPU-accelerated parameter optimization for long/short strategy.
    """

    logger.info(f"GPU Short-Selling Optimization - Device: {DEVICE}, CUDA: {CUDA_AVAILABLE}")

    # Load all period data
    all_periods = pipeline.load_all_periods()
    valid_periods = {k: v for k, v in all_periods.items() if v.n_symbols >= 5}

    logger.info(f"Optimizing across {len(valid_periods)} periods")

    # Parameter bounds - EDGE values for aggressive small-cap trading
    param_bounds = {
        "momentum_lookback": (2, 8),        # Faster signals
        "momentum_threshold": (0.0, 0.02),  # Lower threshold
        "zscore_lookback": (3, 10),         # Shorter lookback
        "zscore_entry": (0.5, 2.0),         # Lower entry
        "zscore_exit": (0.0, 0.8),          # Tighter exit
        "concentration_factor": (3.0, 8.0), # Higher concentration
        "max_position_pct": (0.25, 0.60),   # Larger positions
        "min_position_pct": (0.02, 0.10),   # Higher floor
        "short_leverage": (1.0, 2.5),       # More short leverage
        "long_leverage": (1.0, 2.5),        # More long leverage
        "market_direction_lookback": (3, 10),
        "bear_threshold": (-0.03, -0.005),  # Quicker bear detection
        "bull_threshold": (0.003, 0.02),    # Quicker bull detection
        "stop_loss_pct": (0.08, 0.25),      # Wider stops for volatility
        "take_profit_pct": (0.15, 0.50),    # Bigger targets
        "max_short_pct": (0.40, 0.80),      # More shorts allowed
        "vol_threshold": (0.02, 0.06),      # Higher vol tolerance
        "trend_threshold": (0.01, 0.03),
    }

    int_params = {"momentum_lookback", "zscore_lookback", "market_direction_lookback"}

    # Initialize population
    rng = np.random.default_rng(42)
    population = []

    for _ in range(population_size):
        params = {}
        for key, (low, high) in param_bounds.items():
            val = rng.uniform(low, high)
            if key in int_params:
                val = int(round(val))
            params[key] = val
        population.append(params)

    best_params = None
    best_score = float("-inf")
    best_results = {}
    history = []

    for iteration in range(n_iterations):
        scores = []

        for params in population:
            strategy = GPUShortSellingStrategy(ShortSellingParams(**params))

            period_scores = []
            for period_name, data in valid_periods.items():
                returns_matrix = data.get_returns_matrix(timeframe="1D")
                if returns_matrix.empty or len(returns_matrix) < 10:
                    continue

                result = strategy.run_backtest(returns_matrix)

                # Multi-metric scoring with Calmar + Omega ratios
                # Calmar Ratio: Return / Max Drawdown (penalizes large DDs)
                calmar = result["total_return"] / max(result["max_dd"], 0.01)

                # Omega Ratio proxy: (Win Rate * Avg Win) / (Loss Rate * Avg Loss)
                # Approximate using win_rate and return distribution
                win_rate = result["win_rate"]
                omega_approx = (win_rate / max(1 - win_rate, 0.1)) if win_rate > 0 else 0

                # Burke Ratio: Return / sqrt(sum of squared DDs) - penalizes multiple DDs
                burke = result["total_return"] / max(np.sqrt(result["max_dd"] ** 2), 0.01)

                # Combined score: Calmar-dominant with Burke backup
                score = (
                    0.35 * calmar +                    # Calmar ratio (return/DD)
                    0.25 * burke +                     # Burke ratio (multiple DDs)
                    0.20 * (result["sortino"] / 10) +  # Sortino (downside risk)
                    0.10 * omega_approx +              # Omega (win/loss ratio)
                    0.10 * result["total_return"]      # Raw return
                )
                period_scores.append(score)

            avg_score = np.mean(period_scores) if period_scores else 0
            scores.append(avg_score)

            if avg_score > best_score:
                best_score = avg_score
                best_params = params.copy()

                # Calculate detailed results
                best_results = {}
                for period_name, data in valid_periods.items():
                    returns_matrix = data.get_returns_matrix(timeframe="1D")
                    if returns_matrix.empty:
                        continue
                    result = strategy.run_backtest(returns_matrix)
                    best_results[period_name] = result

        # Log progress
        if iteration % 10 == 0 or iteration == n_iterations - 1:
            avg_return = np.mean([r["total_return"] for r in best_results.values()]) * 100 if best_results else 0
            logger.info(
                f"Iteration {iteration + 1}/{n_iterations}: "
                f"Best Score={best_score:.4f}, Avg Return={avg_return:.2f}%"
            )

        history.append({
            "iteration": iteration + 1,
            "best_score": best_score,
            "avg_score": np.mean(scores),
        })

        # Evolutionary update
        sorted_indices = np.argsort(scores)[::-1]
        n_elite = max(2, population_size // 5)

        new_population = []

        # Keep elite
        for i in sorted_indices[:n_elite]:
            new_population.append(population[i].copy())

        # Create offspring by mutation
        while len(new_population) < population_size:
            parent_idx = sorted_indices[rng.integers(0, n_elite)]
            parent = population[parent_idx]
            child = parent.copy()

            n_mutations = rng.integers(1, 4)
            mutate_keys = rng.choice(list(param_bounds.keys()), size=n_mutations, replace=False)

            for key in mutate_keys:
                low, high = param_bounds[key]
                scale = 0.2 * (high - low)
                child[key] = np.clip(
                    child[key] + rng.normal(0, scale),
                    low,
                    high
                )
                if key in int_params:
                    child[key] = int(round(child[key]))

            new_population.append(child)

        population = new_population

    return ShortSellingParams(**best_params), best_results, history


async def main(data_dir: Path | None = None, batch_name: str = "v1", n_iterations: int = 100):
    """Main entry point.

    Args:
        data_dir: Directory containing historical data files. Defaults to HISTORICAL_DATA_DIR.
        batch_name: Name for this batch (e.g., "v1", "v2_alternative")
        n_iterations: Number of optimization iterations
    """

    logger.info("=" * 70)
    logger.info("GPU SHORT-SELLING NETWORK PARITY OPTIMIZER")
    logger.info(f"Batch: {batch_name}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"CUDA Available: {CUDA_AVAILABLE}")
    logger.info("=" * 70)

    if CUDA_AVAILABLE:
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    if data_dir is None:
        data_dir = HISTORICAL_DATA_DIR

    logger.info(f"Data directory: {data_dir}")
    pipeline = HistoricalDataPipeline(data_dir=data_dir)

    logger.info(f"\nRunning GPU-accelerated short-selling optimization ({n_iterations} iterations)...")
    best_params, best_results, history = optimize_parameters(
        pipeline,
        n_iterations=n_iterations,
        population_size=20,  # Reduced for speed
    )

    # Calculate aggregate metrics
    avg_return = np.mean([r["total_return"] for r in best_results.values()])
    avg_sharpe = np.mean([r["sharpe"] for r in best_results.values()])
    avg_sortino = np.mean([r["sortino"] for r in best_results.values()])
    avg_win_rate = np.mean([r["win_rate"] for r in best_results.values()])
    avg_max_dd = np.mean([r["max_dd"] for r in best_results.values()])

    logger.info("\n" + "=" * 70)
    logger.info("SHORT-SELLING OPTIMIZATION COMPLETE")
    logger.info("=" * 70)

    logger.info("\nPer-Period Results:")
    for period, result in best_results.items():
        logger.info(
            f"  {period}: return={result['total_return']*100:.2f}%, "
            f"sharpe={result['sharpe']:.2f}, win_rate={result['win_rate']*100:.1f}%, "
            f"max_dd={result['max_dd']*100:.1f}%"
        )

    logger.info(f"\nAggregate Results:")
    logger.info(f"  Average Return: {avg_return*100:.2f}%")
    logger.info(f"  Average Sharpe: {avg_sharpe:.2f}")
    logger.info(f"  Average Sortino: {avg_sortino:.2f}")
    logger.info(f"  Average Win Rate: {avg_win_rate*100:.1f}%")
    logger.info(f"  Average Max Drawdown: {avg_max_dd*100:.1f}%")

    target_return = 0.30
    target_achieved = avg_return >= target_return

    logger.info(f"\n  Target Return: {target_return*100:.0f}%")
    logger.info(f"  Target Achieved: {'YES' if target_achieved else 'NO'}")

    # Save per-symbol reports
    logger.info("\nGenerating per-symbol reports...")
    reports_dir = OUTPUT_DIR / "iterations" / f"shortselling_{batch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for period_name, result in best_results.items():
        period_dir = reports_dir / period_name
        period_dir.mkdir(exist_ok=True)

        # Save period summary
        period_summary = {
            "period": period_name,
            "total_return": result["total_return"],
            "sharpe": result["sharpe"],
            "sortino": result["sortino"],
            "max_dd": result["max_dd"],
            "win_rate": result["win_rate"],
            "n_days": result.get("n_days", 0),
        }
        with open(period_dir / "period_summary.json", "w") as f:
            json.dump(period_summary, f, indent=2)

        # Save per-symbol reports
        symbol_returns = result.get("symbol_returns", {})
        for sym, sym_data in symbol_returns.items():
            sym_report = {
                "symbol": sym,
                "period": period_name,
                **sym_data
            }
            with open(period_dir / f"{sym}.json", "w") as f:
                json.dump(sym_report, f, indent=2)

        logger.info(f"  Saved {len(symbol_returns)} symbol reports for {period_name}")

    # Save main summary
    summary = {
        "optimization_id": f"SHORT_{batch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "batch_name": batch_name,
        "data_directory": str(data_dir),
        "strategy_type": "long_short",
        "device": str(DEVICE),
        "cuda_available": CUDA_AVAILABLE,
        "target_return": target_return,
        "actual_return": float(avg_return),
        "target_achieved": bool(target_achieved),
        "avg_sharpe": float(avg_sharpe),
        "avg_sortino": float(avg_sortino),
        "avg_win_rate": float(avg_win_rate),
        "avg_max_dd": float(avg_max_dd),
        "best_params": asdict(best_params),
        "period_results": {
            k: {kk: (float(vv) if not isinstance(vv, dict) else vv) for kk, vv in v.items() if kk != "symbol_returns"}
            for k, v in best_results.items()
        },
        "reports_directory": str(reports_dir),
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "summary" / f"shortselling_{batch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nMain summary saved to: {summary_path}")
    logger.info(f"Symbol reports saved to: {reports_dir}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU Short-Selling Optimizer")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to historical data directory"
    )
    parser.add_argument(
        "--batch",
        type=str,
        default="v1",
        help="Batch name (e.g., v1, v2_alternative)"
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use v2 alternative periods (2006, 2012, 2015, 2019, 2022, 2023)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of optimization iterations (default: 50)"
    )

    args = parser.parse_args()

    # Determine data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.v2:
        data_dir = HISTORICAL_DATA_DIR_V2
    else:
        data_dir = HISTORICAL_DATA_DIR

    # Set batch name
    batch_name = args.batch
    if args.v2 and batch_name == "v1":
        batch_name = "v2_alternative"

    asyncio.run(main(data_dir=data_dir, batch_name=batch_name, n_iterations=args.iterations))
