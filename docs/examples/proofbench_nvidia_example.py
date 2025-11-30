"""
ProofBench with NVIDIA AI Integration Example.

Demonstrates:
- LLM-powered performance narration
- Strategy comparison and optimization
- Trade pattern analysis
- Metric explanations

Get API key: https://build.nvidia.com/
"""

from datetime import datetime
import os

import pandas as pd

from engines.proofbench import (
    LLMPerformanceNarrator,
    PerformanceMetrics,
    SimulationConfig,
    SimulationResults,
)
from engines.proofbench.core.portfolio import Portfolio

# ==================== Configuration ====================

# Option 1: Set API key via environment variable
# os.environ["NVIDIA_API_KEY"] = "nvapi-..."

# Option 2: Pass API key directly (for demonstration)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")  # None for rule-based fallback


# ==================== Example 1: Basic Performance Narration ====================


def example_basic_narration():
    """Basic performance narration without NVIDIA (rule-based)."""
    print("=" * 60)
    print("Example 1: Basic Performance Narration (Rule-Based)")
    print("=" * 60)

    # Create narrator
    narrator = LLMPerformanceNarrator(nvidia_api_key=None)

    # Create mock backtest results
    metrics = PerformanceMetrics(
        total_return=0.28,
        annualized_return=0.21,
        volatility=0.16,
        downside_deviation=0.11,
        sharpe_ratio=1.3,
        sortino_ratio=1.9,
        calmar_ratio=1.75,
        max_drawdown=-0.12,
        avg_drawdown=-0.05,
        max_drawdown_duration=25.0,
        num_trades=45,
        win_rate=0.58,
        profit_factor=1.9,
        avg_win=550.0,
        avg_loss=-320.0,
        largest_win=2200.0,
        largest_loss=-750.0,
        avg_trade_duration=4.5,
        expectancy=125.0,
        recovery_factor=2.3,
        equity_final=128000.0,
    )

    config = SimulationConfig(initial_capital=100000.0)
    portfolio = Portfolio(100000.0)

    # Create equity curve
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    equity_curve = pd.DataFrame(
        {
            "timestamp": dates,
            "equity": [100000.0 + i * 76.7 for i in range(len(dates))],
        }
    )

    trades = pd.DataFrame({"pnl": [100.0] * 45})

    results = SimulationResults(
        config=config,
        metrics=metrics,
        portfolio=portfolio,
        equity_curve=equity_curve,
        trades=trades,
        orders=[],
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 12, 31),
    )

    # Narrate results
    narration = narrator.narrate_results(results)

    print(f"\nBacktest Period: {results.start_time.date()} to {results.end_time.date()}")
    print(f"Model Used: {narration['llm_model']}")
    print(f"\nNarration:\n{narration['narration']}")
    print("\nKey Metrics:")
    print(f"  Total Return: {narration['metrics_summary']['total_return']:.2%}")
    print(f"  Sharpe Ratio: {narration['metrics_summary']['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {narration['metrics_summary']['max_drawdown']:.2%}")
    print(f"  Win Rate: {narration['metrics_summary']['win_rate']:.1%}")


# ==================== Example 2: ProofBench with NVIDIA AI ====================


def example_nvidia_narration():
    """Performance narration with NVIDIA AI integration."""
    print("\n" + "=" * 60)
    print("Example 2: ProofBench with NVIDIA AI")
    print("=" * 60)

    # Check if API key is available
    if not NVIDIA_API_KEY:
        print("\nNote: NVIDIA_API_KEY not set. Will use rule-based fallback.")
        print("To enable NVIDIA: export NVIDIA_API_KEY='nvapi-...'")
        print("Get your key: https://build.nvidia.com/\n")

    # Create narrator with NVIDIA integration
    narrator = LLMPerformanceNarrator(nvidia_api_key=NVIDIA_API_KEY)

    # Create a more realistic backtest scenario
    metrics = PerformanceMetrics(
        total_return=0.45,
        annualized_return=0.32,
        volatility=0.18,
        downside_deviation=0.12,
        sharpe_ratio=1.8,
        sortino_ratio=2.7,
        calmar_ratio=2.5,
        max_drawdown=-0.13,
        avg_drawdown=-0.04,
        max_drawdown_duration=18.0,
        num_trades=72,
        win_rate=0.62,
        profit_factor=2.3,
        avg_win=720.0,
        avg_loss=-310.0,
        largest_win=3500.0,
        largest_loss=-650.0,
        avg_trade_duration=3.8,
        expectancy=255.0,
        recovery_factor=3.5,
        equity_final=145000.0,
    )

    config = SimulationConfig(initial_capital=100000.0, risk_free_rate=0.02)
    portfolio = Portfolio(100000.0)

    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    equity_curve = pd.DataFrame(
        {
            "timestamp": dates,
            "equity": [100000.0 + i * 123.3 for i in range(len(dates))],
        }
    )

    trades = pd.DataFrame({"pnl": [100.0] * 72})

    results = SimulationResults(
        config=config,
        metrics=metrics,
        portfolio=portfolio,
        equity_curve=equity_curve,
        trades=trades,
        orders=[],
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 12, 31),
    )

    # Narrate results with NVIDIA
    print("\n[Narration] Generating AI-powered performance analysis...")
    narration = narrator.narrate_results(results)

    print(f"\nModel Used: {narration['llm_model']}")
    print(f"\nPerformance Narration:\n{narration['narration']}")

    # Analyze trade patterns
    print("\n[Trade Patterns] Analyzing trade execution patterns...")
    pattern_analysis = narrator.analyze_trade_patterns(results)

    print(f"\nTrade Pattern Analysis:\n{pattern_analysis['analysis']}")

    # Suggest optimizations
    print("\n[Optimizations] Generating improvement suggestions...")
    suggestions = narrator.suggest_optimizations(results, focus="returns")

    print("\nOptimization Suggestions (Focus: Returns):")
    for i, suggestion in enumerate(suggestions[:5], 1):
        print(f"{i}. {suggestion}")


# ==================== Example 3: Strategy Comparison ====================


def example_strategy_comparison():
    """Compare multiple backtest strategies."""
    print("\n" + "=" * 60)
    print("Example 3: Strategy Comparison")
    print("=" * 60)

    narrator = LLMPerformanceNarrator(nvidia_api_key=NVIDIA_API_KEY)

    # Create three different strategy results
    strategies = []

    # Strategy A: Aggressive (high returns, high risk)
    metrics_a = PerformanceMetrics(
        total_return=0.55,
        annualized_return=0.38,
        volatility=0.25,
        downside_deviation=0.18,
        sharpe_ratio=1.5,
        sortino_ratio=2.1,
        calmar_ratio=1.9,
        max_drawdown=-0.20,
        avg_drawdown=-0.08,
        max_drawdown_duration=45.0,
        num_trades=120,
        win_rate=0.52,
        profit_factor=1.7,
        avg_win=900.0,
        avg_loss=-550.0,
        largest_win=4500.0,
        largest_loss=-1800.0,
        avg_trade_duration=2.5,
        expectancy=182.0,
        recovery_factor=2.75,
        equity_final=155000.0,
    )

    # Strategy B: Balanced (moderate returns, moderate risk)
    metrics_b = PerformanceMetrics(
        total_return=0.32,
        annualized_return=0.24,
        volatility=0.14,
        downside_deviation=0.09,
        sharpe_ratio=1.7,
        sortino_ratio=2.7,
        calmar_ratio=2.4,
        max_drawdown=-0.10,
        avg_drawdown=-0.04,
        max_drawdown_duration=22.0,
        num_trades=65,
        win_rate=0.60,
        profit_factor=2.1,
        avg_win=680.0,
        avg_loss=-320.0,
        largest_win=2800.0,
        largest_loss=-700.0,
        avg_trade_duration=4.2,
        expectancy=216.0,
        recovery_factor=3.2,
        equity_final=132000.0,
    )

    # Strategy C: Conservative (low returns, low risk)
    metrics_c = PerformanceMetrics(
        total_return=0.18,
        annualized_return=0.14,
        volatility=0.08,
        downside_deviation=0.05,
        sharpe_ratio=1.75,
        sortino_ratio=2.8,
        calmar_ratio=2.8,
        max_drawdown=-0.05,
        avg_drawdown=-0.02,
        max_drawdown_duration=12.0,
        num_trades=40,
        win_rate=0.68,
        profit_factor=2.5,
        avg_win=520.0,
        avg_loss=-210.0,
        largest_win=1500.0,
        largest_loss=-400.0,
        avg_trade_duration=6.5,
        expectancy=210.0,
        recovery_factor=3.6,
        equity_final=118000.0,
    )

    config = SimulationConfig(initial_capital=100000.0)
    portfolio = Portfolio(100000.0)

    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")

    for name, metrics in [
        ("Aggressive Strategy", metrics_a),
        ("Balanced Strategy", metrics_b),
        ("Conservative Strategy", metrics_c),
    ]:
        equity_curve = pd.DataFrame(
            {
                "timestamp": dates,
                "equity": [100000.0] * len(dates),
            }
        )
        trades = pd.DataFrame({"pnl": [0.0]})

        results = SimulationResults(
            config=config,
            metrics=metrics,
            portfolio=portfolio,
            equity_curve=equity_curve,
            trades=trades,
            orders=[],
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 12, 31),
        )

        strategies.append((name, results))

    # Compare strategies
    print(f"\n[Comparison] Analyzing {len(strategies)} strategies...")

    comparison = narrator.compare_results(strategies)

    print(f"\nModel Used: {comparison['llm_model']}")
    print(f"Strategies Compared: {comparison['strategies_compared']}")
    print(f"\nComparative Analysis:\n{comparison['comparison']}")


# ==================== Example 4: Metric Explanations ====================


def example_metric_explanations():
    """Explain key performance metrics."""
    print("\n" + "=" * 60)
    print("Example 4: Metric Explanations")
    print("=" * 60)

    narrator = LLMPerformanceNarrator(nvidia_api_key=NVIDIA_API_KEY)

    # Explain key metrics
    metrics_to_explain = [
        ("Sharpe Ratio", 1.8),
        ("Sortino Ratio", 2.5),
        ("Calmar Ratio", 2.3),
        ("Profit Factor", 2.1),
    ]

    print("\n[Explanations] Generating plain-language metric explanations...")

    for metric_name, metric_value in metrics_to_explain:
        print(f"\n{metric_name}: {metric_value:.2f}")
        print("-" * 40)

        explanation = narrator.explain_metric(metric_name, metric_value)
        print(explanation)


# ==================== Example 5: Optimization Focus Areas ====================


def example_optimization_focus():
    """Test different optimization focus areas."""
    print("\n" + "=" * 60)
    print("Example 5: Optimization by Focus Area")
    print("=" * 60)

    narrator = LLMPerformanceNarrator(nvidia_api_key=NVIDIA_API_KEY)

    # Create sample results
    metrics = PerformanceMetrics(
        total_return=0.22,
        annualized_return=0.17,
        volatility=0.19,
        downside_deviation=0.14,
        sharpe_ratio=0.9,
        sortino_ratio=1.2,
        calmar_ratio=1.1,
        max_drawdown=-0.15,
        avg_drawdown=-0.07,
        max_drawdown_duration=35.0,
        num_trades=55,
        win_rate=0.50,
        profit_factor=1.4,
        avg_win=600.0,
        avg_loss=-430.0,
        largest_win=2100.0,
        largest_loss=-1100.0,
        avg_trade_duration=5.8,
        expectancy=93.0,
        recovery_factor=1.5,
        equity_final=122000.0,
    )

    config = SimulationConfig(initial_capital=100000.0)
    portfolio = Portfolio(100000.0)

    dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    equity_curve = pd.DataFrame({"timestamp": dates, "equity": [100000.0] * len(dates)})
    trades = pd.DataFrame({"pnl": [0.0]})

    results = SimulationResults(
        config=config,
        metrics=metrics,
        portfolio=portfolio,
        equity_curve=equity_curve,
        trades=trades,
        orders=[],
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 12, 31),
    )

    # Get suggestions for different focus areas
    focus_areas = ["returns", "risk", "consistency", "general"]

    for focus in focus_areas:
        print(f"\n[Optimization Focus: {focus.upper()}]")
        print("-" * 40)

        suggestions = narrator.suggest_optimizations(results, focus=focus)

        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"{i}. {suggestion}")


# ==================== Main ====================

if __name__ == "__main__":
    print("\nProofBench with NVIDIA AI Integration\n")

    # Run examples
    example_basic_narration()
    example_nvidia_narration()
    example_strategy_comparison()
    example_metric_explanations()
    example_optimization_focus()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Get NVIDIA API key: https://build.nvidia.com/")
    print("2. Set environment variable: export NVIDIA_API_KEY='nvapi-...'")
    print("3. Run real backtests with your strategies")
    print("4. Use AI insights to optimize performance")
    print("=" * 60)
