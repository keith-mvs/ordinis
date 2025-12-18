"""
Example: Testing and Optimizing Position Sizing with LearningEngine

This script demonstrates:
1. Testing different position sizing strategies
2. Recording outcomes in LearningEngine
3. Analyzing results to find optimal parameters
4. Generating optimized configuration

Run with:
    python examples/test_and_optimize_position_sizing.py
"""

import asyncio
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ordinis.engines.learning import (
    EventType,
    LearningEngine,
    LearningEngineConfig,
    LearningEvent,
)
from ordinis.engines.portfolio import (
    RiskParityRebalancer,
    SignalDrivenRebalancer,
    SignalInput,
    SignalMethod,
    TargetAllocation,
    TargetAllocationRebalancer,
)


def generate_synthetic_returns(symbols: list[str], days: int = 252, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic return data for testing."""
    np.random.seed(seed)

    returns = {}
    for symbol in symbols:
        # Different volatilities for each symbol
        vol = np.random.uniform(0.01, 0.03)
        drift = np.random.uniform(-0.0002, 0.0005)
        returns[symbol] = np.random.normal(drift, vol, days)

    dates = pd.date_range(end=datetime.now(), periods=days, freq="D")

    return pd.DataFrame(returns, index=dates)


async def test_target_allocation():
    """Test basic target allocation strategy."""
    print("\n" + "=" * 70)
    print("TEST 1: Target Allocation Strategy")
    print("=" * 70)

    # Setup
    targets = [
        TargetAllocation("AAPL", 0.40),
        TargetAllocation("MSFT", 0.30),
        TargetAllocation("GOOGL", 0.30),
    ]

    rebalancer = TargetAllocationRebalancer(targets, drift_threshold=0.05)

    # Current portfolio (unbalanced)
    positions = {"AAPL": 100, "MSFT": 200, "GOOGL": 50}
    prices = {"AAPL": 150.0, "MSFT": 100.0, "GOOGL": 120.0}

    # Calculate current state
    total_value = sum(positions[s] * prices[s] for s in positions)
    print(f"\nPortfolio Value: ${total_value:,.2f}")

    for symbol in positions:
        value = positions[symbol] * prices[symbol]
        weight = value / total_value
        target = targets[[t.symbol for t in targets].index(symbol)].target_weight
        print(f"  {symbol}: ${value:,.2f} ({weight:.1%}) - Target: {target:.1%}")

    # Generate rebalancing decisions
    decisions = rebalancer.generate_rebalance_orders(positions, prices)

    print(f"\nRebalancing Decisions:")
    for decision in decisions:
        action = "BUY" if decision.adjustment_shares > 0 else "SELL"
        print(f"  {action} {abs(decision.adjustment_shares):.0f} shares of {decision.symbol}")
        print(f"    Current: {decision.current_weight:.1%} → Target: {decision.target_weight:.1%}")

    return decisions


async def test_risk_parity():
    """Test risk parity strategy."""
    print("\n" + "=" * 70)
    print("TEST 2: Risk Parity Strategy")
    print("=" * 70)

    # Generate synthetic returns
    symbols = ["AAPL", "MSFT", "GOOGL"]
    returns = generate_synthetic_returns(symbols, days=252)

    print(f"\nHistorical Volatilities (annualized):")
    for symbol in symbols:
        vol = returns[symbol].std() * np.sqrt(252)
        print(f"  {symbol}: {vol:.2%}")

    # Calculate risk parity weights
    rebalancer = RiskParityRebalancer(
        lookback_days=252,
        min_weight=0.10,
        max_weight=0.50,
    )

    weights = rebalancer.calculate_weights(returns)

    print(f"\nRisk Parity Weights:")
    for symbol, weight in weights.weights.items():
        vol = weights.volatilities[symbol]
        risk = weights.risk_contributions[symbol]
        print(f"  {symbol}: {weight:.1%} (vol: {vol:.2%}, risk contrib: {risk:.1%})")

    print(f"\nTotal: {sum(weights.weights.values()):.1%}")

    return weights


async def test_signal_driven():
    """Test signal-driven strategy."""
    print("\n" + "=" * 70)
    print("TEST 3: Signal-Driven Strategy")
    print("=" * 70)

    # Create trading signals
    signals = [
        SignalInput("AAPL", signal=0.8, confidence=0.9, source="RSI"),
        SignalInput("MSFT", signal=0.4, confidence=0.7, source="MACD"),
        SignalInput("GOOGL", signal=0.6, confidence=0.8, source="RSI"),
        SignalInput("NVDA", signal=-0.3, confidence=0.6, source="MACD"),  # Negative
    ]

    print("\nInput Signals:")
    for sig in signals:
        print(f"  {sig.symbol}: signal={sig.signal:+.2f}, confidence={sig.confidence:.2f}")

    # Test PROPORTIONAL method
    print("\nPROPORTIONAL Method:")
    rebalancer = SignalDrivenRebalancer(
        method=SignalMethod.PROPORTIONAL,
        min_weight=0.0,
        max_weight=0.50,
        cash_buffer=0.10,
    )

    weights = rebalancer.calculate_weights(signals)

    for symbol, weight in weights.weights.items():
        print(f"  {symbol}: {weight:.1%}")
    print(f"  CASH: {0.10:.1%}")
    print(f"  Total Invested: {sum(weights.weights.values()):.1%}")

    # Test BINARY method
    print("\nBINARY Method:")
    rebalancer_binary = SignalDrivenRebalancer(
        method=SignalMethod.BINARY,
        cash_buffer=0.10,
    )

    weights_binary = rebalancer_binary.calculate_weights(signals)

    for symbol, weight in weights_binary.weights.items():
        print(f"  {symbol}: {weight:.1%}")
    print(f"  CASH: {0.10:.1%}")

    return weights


async def optimize_with_learning_engine():
    """Optimize position sizing parameters using LearningEngine."""
    print("\n" + "=" * 70)
    print("TEST 4: Optimization with LearningEngine")
    print("=" * 70)

    # Setup learning engine
    learning_config = LearningEngineConfig(
        data_dir=Path("artifacts/position_sizing_optimization"),
        max_events_memory=10000,
    )
    learning_engine = LearningEngine(learning_config)
    await learning_engine.initialize()

    try:
        # Generate synthetic market data
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
        returns = generate_synthetic_returns(symbols, days=252)

        # Calculate cumulative returns for each symbol
        cumulative_returns = (1 + returns).cumprod()

        # Test different max position sizes
        position_sizes_to_test = [0.05, 0.10, 0.15, 0.20, 0.25]

        print("\nTesting different max position sizes...")

        results = []

        for max_pos_pct in position_sizes_to_test:
            print(f"\n  Testing max_position_pct = {max_pos_pct:.0%}")

            # Create equal-weight target allocation
            weight_per_symbol = 1.0 / len(symbols)
            targets = [TargetAllocation(sym, weight_per_symbol) for sym in symbols]

            # Simulate portfolio performance
            # Simple simulation: cap individual positions at max_pos_pct
            portfolio_weights = {}
            for sym in symbols:
                raw_weight = weight_per_symbol
                capped_weight = min(raw_weight, max_pos_pct)
                portfolio_weights[sym] = capped_weight

            # Renormalize
            total_weight = sum(portfolio_weights.values())
            portfolio_weights = {k: v / total_weight for k, v in portfolio_weights.items()}

            # Calculate portfolio returns
            portfolio_returns = sum(returns[sym] * portfolio_weights[sym] for sym in symbols)

            # Calculate metrics
            total_return = (1 + portfolio_returns).prod() - 1
            volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (portfolio_returns.mean() * 252) / volatility if volatility > 0 else 0

            # Calculate max drawdown
            cumulative = (1 + portfolio_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Record in learning engine
            learning_engine.record_event(
                LearningEvent(
                    event_type=EventType.METRIC_RECORDED,
                    source_engine="optimization",
                    symbol="PORTFOLIO",
                    payload={
                        "max_position_pct": max_pos_pct,
                        "total_return": total_return,
                        "volatility": volatility,
                        "sharpe_ratio": sharpe_ratio,
                        "max_drawdown": max_drawdown,
                    },
                    outcome=sharpe_ratio,  # Use Sharpe as optimization metric
                )
            )

            results.append(
                {
                    "max_position_pct": max_pos_pct,
                    "sharpe_ratio": sharpe_ratio,
                    "total_return": total_return,
                    "max_drawdown": max_drawdown,
                }
            )

            print(f"    Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"    Total Return: {total_return:.2%}")
            print(f"    Max Drawdown: {max_drawdown:.2%}")

        # Find best parameters
        performance_events = learning_engine.get_events(event_type=EventType.METRIC_RECORDED)

        best_event = max(
            performance_events, key=lambda e: e.outcome if e.outcome else -float("inf")
        )

        print("\n" + "=" * 70)
        print("OPTIMIZATION RESULTS")
        print("=" * 70)

        if best_event.payload:
            print(f"\nOptimal Parameters:")
            print(f"  Max Position Size: {best_event.payload['max_position_pct']:.0%}")
            print(f"\nPerformance Metrics:")
            print(f"  Sharpe Ratio: {best_event.payload['sharpe_ratio']:.2f}")
            print(f"  Total Return: {best_event.payload['total_return']:.2%}")
            print(f"  Max Drawdown: {best_event.payload['max_drawdown']:.2%}")
            print(f"  Volatility: {best_event.payload['volatility']:.2%}")

        # Save results
        results_df = pd.DataFrame(results)
        print(f"\nAll Results:")
        print(results_df.to_string(index=False))

        # Generate optimized config
        optimized_config = {
            "position_sizing": {
                "max_position_pct": best_event.payload["max_position_pct"],
                "optimization_date": datetime.now().isoformat(),
                "optimization_metric": "sharpe_ratio",
                "backtest_sharpe": best_event.payload["sharpe_ratio"],
                "backtest_return": best_event.payload["total_return"],
            }
        }

        # Save to file
        config_path = Path("artifacts/optimized_position_sizing_config.json")
        config_path.parent.mkdir(parents=True, exist_ok=True)

        import json

        with open(config_path, "w") as f:
            json.dump(optimized_config, f, indent=2)

        print(f"\n✓ Saved optimized config to: {config_path}")

        return optimized_config

    finally:
        await learning_engine.shutdown()


async def main():
    """Run all position sizing tests and optimization."""
    print("\n" + "=" * 70)
    print("POSITION SIZING TESTING AND OPTIMIZATION")
    print("=" * 70)

    # Run tests
    await test_target_allocation()
    await test_risk_parity()
    await test_signal_driven()

    # Run optimization
    optimized_config = await optimize_with_learning_engine()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)
    print(
        f"\nOptimized max_position_pct: {optimized_config['position_sizing']['max_position_pct']:.0%}"
    )
    print("\nNext steps:")
    print("  1. Review artifacts/optimized_position_sizing_config.json")
    print("  2. Update your config file with optimized parameters")
    print("  3. Run walk-forward validation on out-of-sample data")
    print("  4. Monitor performance in production")


if __name__ == "__main__":
    asyncio.run(main())
