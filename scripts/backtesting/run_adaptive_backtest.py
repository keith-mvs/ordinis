"""
Comprehensive Backtest of Adaptive Strategy System.

Tests the regime-adaptive strategy manager across diverse
market conditions using the multi-timeframe training data.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.data.training_data_generator import (
    DataChunk,
    MarketRegime,
    TrainingConfig,
    TrainingDataGenerator,
)
from src.engines.proofbench.core.simulator import SimulationConfig, SimulationEngine
from src.strategies.regime_adaptive import (
    AdaptiveConfig,
    AdaptiveStrategyManager,
    create_strategy_callback,
)


@dataclass
class BacktestResult:
    """Result from a single backtest run."""

    chunk_id: int
    regime: MarketRegime
    duration_months: int
    start_date: str
    end_date: str
    strategy_return: float
    benchmark_return: float
    alpha: float
    sharpe_ratio: float
    max_drawdown: float
    num_trades: int
    detected_regime: str
    regime_confidence: float


def run_adaptive_backtest(
    chunk: DataChunk,
    chunk_id: int,
    initial_capital: float = 100000.0,
) -> BacktestResult:
    """Run backtest with adaptive strategy on a single chunk."""
    # Create fresh manager for each test
    config = AdaptiveConfig(
        use_ensemble=True,
        confidence_scaling=True,
        volatility_scaling=True,
        base_position_size=0.8,
    )
    manager = AdaptiveStrategyManager(config)

    # Create simulation
    sim_config = SimulationConfig(
        initial_capital=initial_capital,
        bar_frequency="1d",
        enable_logging=False,
    )

    # Ensure data has DatetimeIndex without timezone
    data = chunk.data.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    # Remove timezone info if present
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Also ensure all datetime columns in the data are timezone-naive
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            if hasattr(data[col].dt, "tz") and data[col].dt.tz is not None:
                data[col] = data[col].dt.tz_localize(None)

    engine = SimulationEngine(sim_config)
    engine.load_data(chunk.symbol, data)

    # Create callback
    callback = create_strategy_callback(manager)
    engine.set_strategy(callback)

    # Run simulation
    results = engine.run()
    metrics = results.metrics

    # Get detected regime info
    regime_info = manager.get_regime_info()

    # Calculate benchmark
    benchmark_return = chunk.metrics["total_return"] * 100
    alpha = metrics.total_return - benchmark_return

    return BacktestResult(
        chunk_id=chunk_id,
        regime=chunk.regime,
        duration_months=chunk.duration_months,
        start_date=chunk.start_date.strftime("%Y-%m-%d"),
        end_date=chunk.end_date.strftime("%Y-%m-%d"),
        strategy_return=metrics.total_return,
        benchmark_return=benchmark_return,
        alpha=alpha,
        sharpe_ratio=metrics.sharpe_ratio,
        max_drawdown=metrics.max_drawdown,
        num_trades=metrics.num_trades,
        detected_regime=regime_info["regime"],
        regime_confidence=regime_info["confidence"],
    )


def print_regime_summary(results: list[BacktestResult], regime: MarketRegime):
    """Print summary for a specific regime."""
    regime_results = [r for r in results if r.regime == regime]
    if not regime_results:
        return

    alphas = [r.alpha for r in regime_results]
    returns = [r.strategy_return for r in regime_results]
    sharpes = [r.sharpe_ratio for r in regime_results]
    win_rate = sum(1 for r in regime_results if r.alpha > 0) / len(regime_results) * 100

    print(f"\n{regime.value.upper()} REGIME ({len(regime_results)} tests)")
    print("-" * 50)
    print(f"  Avg Return:     {np.mean(returns):+.2f}%")
    print(f"  Avg Alpha:      {np.mean(alphas):+.2f}%")
    print(f"  Avg Sharpe:     {np.mean(sharpes):.2f}")
    print(f"  Beat B&H Rate:  {win_rate:.1f}%")
    print(f"  Best Alpha:     {max(alphas):+.2f}%")
    print(f"  Worst Alpha:    {min(alphas):+.2f}%")


def main():
    print("=" * 80)
    print("ADAPTIVE STRATEGY SYSTEM BACKTEST")
    print("=" * 80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Generate training data
    print("[1/3] Generating training data chunks...")
    config = TrainingConfig(
        symbols=["SPY"],
        chunk_sizes_months=[2, 3, 4, 6, 8, 10, 12],
        lookback_years=[5, 10, 15, 20],
        random_seed=42,
    )

    generator = TrainingDataGenerator(config)
    chunks = generator.generate_chunks("SPY", num_chunks=100, balance_regimes=True)

    print(f"Generated {len(chunks)} chunks")
    print("\nRegime Distribution:")
    for regime, count in generator.get_regime_distribution(chunks).items():
        pct = count / len(chunks) * 100 if chunks else 0
        print(f"  {regime.value:<12}: {count:>4} ({pct:.1f}%)")

    # Run backtests
    print("\n[2/3] Running adaptive strategy backtests...")
    results = []

    for i, chunk in enumerate(chunks):
        try:
            result = run_adaptive_backtest(chunk, i)
            results.append(result)

            # Progress indicator
            if (i + 1) % 20 == 0:
                completed = i + 1
                win_rate = sum(1 for r in results if r.alpha > 0) / len(results) * 100
                print(f"  Completed {completed}/{len(chunks)} - Running win rate: {win_rate:.1f}%")

        except Exception as e:
            print(f"  [WARN] Chunk {i} failed: {e}")

    # Print results
    print("\n[3/3] Analyzing results...")
    print("\n" + "=" * 80)
    print("RESULTS BY MARKET REGIME")
    print("=" * 80)

    for regime in MarketRegime:
        print_regime_summary(results, regime)

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE")
    print("=" * 80)

    all_alphas = [r.alpha for r in results]
    all_returns = [r.strategy_return for r in results]
    all_sharpes = [r.sharpe_ratio for r in results]
    overall_win = sum(1 for a in all_alphas if a > 0) / len(all_alphas) * 100

    print(f"\nTotal Tests:        {len(results)}")
    print(f"Avg Return:         {np.mean(all_returns):+.2f}%")
    print(f"Avg Alpha vs B&H:   {np.mean(all_alphas):+.2f}%")
    print(f"Avg Sharpe:         {np.mean(all_sharpes):.2f}")
    print(f"Beat B&H Rate:      {overall_win:.1f}%")
    print(f"Std Dev of Alpha:   {np.std(all_alphas):.2f}%")

    # Regime detection accuracy
    print("\n" + "=" * 80)
    print("REGIME DETECTION ANALYSIS")
    print("=" * 80)

    correct_detections = 0
    for result in results:
        actual = result.regime.value
        detected = result.detected_regime
        if actual == detected:
            correct_detections += 1

    detection_rate = correct_detections / len(results) * 100 if results else 0
    print(f"\nRegime Detection Accuracy: {detection_rate:.1f}%")
    print(f"Avg Detection Confidence:  {np.mean([r.regime_confidence for r in results]):.1%}")

    # Comparison with old strategies
    print("\n" + "=" * 80)
    print("COMPARISON WITH NON-ADAPTIVE STRATEGIES")
    print("=" * 80)

    old_results = {
        "MA Crossover": {"alpha": -3.99, "win_rate": 31.6},
        "RSI": {"alpha": -4.88, "win_rate": 34.7},
        "Momentum": {"alpha": -5.85, "win_rate": 35.7},
        "Bollinger": {"alpha": -3.96, "win_rate": 34.7},
        "MACD": {"alpha": -7.12, "win_rate": 28.6},
    }

    print(f"\n{'Strategy':<25} {'Avg Alpha':>12} {'Beat B&H':>12}")
    print("-" * 50)
    for name, metrics in old_results.items():
        print(f"{name:<25} {metrics['alpha']:>+11.2f}% {metrics['win_rate']:>11.1f}%")
    print("-" * 50)
    print(f"{'ADAPTIVE SYSTEM':<25} {np.mean(all_alphas):>+11.2f}% {overall_win:>11.1f}%")

    # Improvement calculation
    best_old_alpha = max(m["alpha"] for m in old_results.values())
    best_old_win = max(m["win_rate"] for m in old_results.values())

    alpha_improvement = np.mean(all_alphas) - best_old_alpha
    win_improvement = overall_win - best_old_win

    print("\nImprovement vs Best Old Strategy:")
    print(f"  Alpha: {alpha_improvement:+.2f}%")
    print(f"  Win Rate: {win_improvement:+.1f}%")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    if overall_win >= 55:
        verdict = "SUCCESS - Adaptive system outperforms passive investing"
    elif overall_win >= 50:
        verdict = "MARGINAL - Slight edge over buy-and-hold"
    elif overall_win > best_old_win:
        verdict = "IMPROVED - Better than fixed strategies but still needs work"
    else:
        verdict = "NEEDS WORK - Adaptive system requires further refinement"

    print(f"\n{verdict}")

    if overall_win >= 50:
        print("\nStrengths:")
        for regime in MarketRegime:
            regime_results = [r for r in results if r.regime == regime]
            if regime_results:
                win_rate = sum(1 for r in regime_results if r.alpha > 0) / len(regime_results) * 100
                if win_rate >= 50:
                    print(f"  - {regime.value}: {win_rate:.1f}% beat B&H")

    print("\nAreas for Improvement:")
    for regime in MarketRegime:
        regime_results = [r for r in results if r.regime == regime]
        if regime_results:
            win_rate = sum(1 for r in regime_results if r.alpha > 0) / len(regime_results) * 100
            if win_rate < 40:
                print(f"  - {regime.value}: Only {win_rate:.1f}% beat B&H")


if __name__ == "__main__":
    main()
