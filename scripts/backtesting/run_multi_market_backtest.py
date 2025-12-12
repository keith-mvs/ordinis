"""
Multi-Market Backtest - Test strategies across different market conditions.

Tests the same strategies on:
1. Trending UP market (SPY)
2. Volatile market (QQQ)
3. Sideways/Range-bound market (XYZ)
4. Trending DOWN market (ABC)
"""

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datetime import datetime  # noqa: E402

# Import from the main demo script
from run_backtest_demo import (  # noqa: E402
    create_bollinger_strategy,
    create_ma_crossover_strategy,
    create_macd_strategy,
    create_momentum_strategy,
    create_rsi_strategy,
    load_sample_data,
    run_backtest,
)


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def main():
    """Run multi-market backtest."""
    print_header("MULTI-MARKET BACKTEST ANALYSIS")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Define datasets
    datasets = [
        ("sample_spy_trending_up.csv", "TRENDING UP", "Strong bull market"),
        ("sample_qqq_volatile.csv", "VOLATILE", "High volatility"),
        ("sample_xyz_sideways.csv", "SIDEWAYS", "Range-bound market"),
        ("sample_abc_trending_down.csv", "TRENDING DOWN", "Bear market"),
    ]

    # Define strategies
    strategies = [
        ("MA Crossover", create_ma_crossover_strategy(20, 50)),
        ("RSI", create_rsi_strategy(14, 30, 70)),
        ("Momentum", create_momentum_strategy(20, 0.05)),
        ("Bollinger", create_bollinger_strategy(20, 2.0)),
        ("MACD", create_macd_strategy(12, 26, 9)),
    ]

    # Store all results
    all_results = {}

    for filename, market_type, description in datasets:
        print_header(f"MARKET: {market_type}")
        print(f"({description})")

        try:
            data = load_sample_data(filename)
            print(f"\nData: {len(data)} bars")
            print(
                f"Period: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
            )

            # Calculate buy & hold
            bh_return = (
                (data["close"].iloc[-1] - data["close"].iloc[0]) / data["close"].iloc[0] * 100
            )
            print(f"Buy & Hold: {bh_return:+.2f}%")

            print(f"\n{'Strategy':<15} {'Return':>10} {'Sharpe':>8} {'MaxDD':>8} {'Trades':>7}")
            print("-" * 55)

            market_results = []
            for name, callback in strategies:
                try:
                    result = run_backtest(data, callback, name)
                    market_results.append(result)
                    print(
                        f"{name:<15} "
                        f"{result.total_return:>+9.2f}% "
                        f"{result.sharpe_ratio:>8.2f} "
                        f"{result.max_drawdown:>7.2f}% "
                        f"{result.total_trades:>7}"
                    )
                except Exception as e:
                    print(f"{name:<15} [ERROR: {str(e)[:30]}]")

            all_results[market_type] = market_results

            # Find winner for this market
            if market_results:
                winner = max(market_results, key=lambda x: x.sharpe_ratio)
                print(f"\nBest Strategy: {winner.name} (Sharpe: {winner.sharpe_ratio:.2f})")

        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
            continue

    # Final summary
    print_header("STRATEGY RANKING ACROSS ALL MARKETS")

    # Calculate average performance for each strategy
    strategy_scores = {}
    for strat_name, _ in strategies:
        sharpes = []
        returns = []
        for market_results in all_results.values():
            for result in market_results:
                if result.name == strat_name:
                    sharpes.append(result.sharpe_ratio)
                    returns.append(result.total_return)
                    break

        if sharpes:
            strategy_scores[strat_name] = {
                "avg_sharpe": sum(sharpes) / len(sharpes),
                "avg_return": sum(returns) / len(returns),
                "markets_tested": len(sharpes),
            }

    # Sort by average Sharpe
    sorted_strategies = sorted(
        strategy_scores.items(), key=lambda x: x[1]["avg_sharpe"], reverse=True
    )

    print(f"\n{'Rank':<6} {'Strategy':<15} {'Avg Sharpe':>12} {'Avg Return':>12}")
    print("-" * 50)
    for rank, (name, scores) in enumerate(sorted_strategies, 1):
        print(
            f"{rank:<6} "
            f"{name:<15} "
            f"{scores['avg_sharpe']:>12.2f} "
            f"{scores['avg_return']:>+11.2f}%"
        )

    # Overall winner
    print_header("CONCLUSION")
    if sorted_strategies:
        best_name, best_scores = sorted_strategies[0]
        print(f"\n[OVERALL BEST] {best_name}")
        print(f"   Average Sharpe: {best_scores['avg_sharpe']:.2f}")
        print(f"   Average Return: {best_scores['avg_return']:+.2f}%")
        print(f"   Markets Tested: {best_scores['markets_tested']}")

    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nInsight: Different strategies excel in different market conditions.")
    print("Consider: Strategy selection based on market regime detection.\n")


if __name__ == "__main__":
    main()
