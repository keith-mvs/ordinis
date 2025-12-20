# Testing Position Sizing Logic

This guide explains how to test position sizing logic in Ordinis and use the LearningEngine to optimize position sizing parameters.

## Overview

Position sizing can be tested at multiple levels:
1. **Unit Tests** - Test individual strategies in isolation
2. **Integration Tests** - Test strategy interaction with portfolio engine
3. **Backtests** - Test strategies with historical data
4. **Learning Engine Integration** - Optimize parameters based on outcomes

## Unit Testing Position Sizing Strategies

### Testing Target Allocation

```python
import pytest
from ordinis.engines.portfolio import (
    TargetAllocationRebalancer,
    TargetAllocation,
)

def test_target_allocation_basic():
    """Test basic target allocation calculation."""
    # Setup target allocations (must sum to 1.0)
    targets = [
        TargetAllocation("AAPL", 0.40),
        TargetAllocation("MSFT", 0.30),
        TargetAllocation("GOOGL", 0.30),
    ]

    rebalancer = TargetAllocationRebalancer(targets, drift_threshold=0.05)

    # Current portfolio
    positions = {"AAPL": 10, "MSFT": 5, "GOOGL": 3}
    prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 100.0}

    # Calculate drift
    drift = rebalancer.calculate_drift(positions, prices)

    # Total value: 10*150 + 5*300 + 3*100 = 1500 + 1500 + 300 = 3300
    # Current weights: AAPL=45.45%, MSFT=45.45%, GOOGL=9.09%
    # Expected drift: AAPL=+5.45%, MSFT=+15.45%, GOOGL=-20.91%

    assert abs(drift["AAPL"] - 0.0545) < 0.01
    assert abs(drift["MSFT"] - 0.1545) < 0.01
    assert abs(drift["GOOGL"] - (-0.2091)) < 0.01

def test_target_allocation_rebalancing():
    """Test rebalancing order generation."""
    targets = [
        TargetAllocation("AAPL", 0.40),
        TargetAllocation("MSFT", 0.30),
        TargetAllocation("GOOGL", 0.30),
    ]

    rebalancer = TargetAllocationRebalancer(targets, drift_threshold=0.05)

    positions = {"AAPL": 10, "MSFT": 5, "GOOGL": 3}
    prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 100.0}

    # Generate rebalancing decisions
    decisions = rebalancer.generate_rebalance_orders(positions, prices)

    assert len(decisions) == 3

    # AAPL should decrease (overweight)
    aapl_decision = next(d for d in decisions if d.symbol == "AAPL")
    assert aapl_decision.adjustment_shares < 0

    # GOOGL should increase significantly (underweight)
    googl_decision = next(d for d in decisions if d.symbol == "GOOGL")
    assert googl_decision.adjustment_shares > 0

def test_target_allocation_should_rebalance():
    """Test rebalancing trigger logic."""
    targets = [
        TargetAllocation("AAPL", 0.50),
        TargetAllocation("MSFT", 0.50),
    ]

    rebalancer = TargetAllocationRebalancer(targets, drift_threshold=0.10)

    # Small drift - should not rebalance
    positions = {"AAPL": 51, "MSFT": 49}
    prices = {"AAPL": 100.0, "MSFT": 100.0}
    assert not rebalancer.should_rebalance(positions, prices)

    # Large drift - should rebalance
    positions = {"AAPL": 70, "MSFT": 30}
    prices = {"AAPL": 100.0, "MSFT": 100.0}
    assert rebalancer.should_rebalance(positions, prices)
```

### Testing Risk Parity

```python
import pandas as pd
import numpy as np
from ordinis.engines.portfolio import RiskParityRebalancer

def test_risk_parity_weights():
    """Test risk parity weight calculation."""
    # Generate synthetic returns with different volatilities
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")

    returns = pd.DataFrame({
        "LOW_VOL": np.random.normal(0.0005, 0.01, 252),   # Low vol
        "MED_VOL": np.random.normal(0.0005, 0.015, 252),  # Medium vol
        "HIGH_VOL": np.random.normal(0.0005, 0.02, 252),  # High vol
    }, index=dates)

    rebalancer = RiskParityRebalancer(
        lookback_days=252,
        min_weight=0.10,
        max_weight=0.50,
    )

    weights = rebalancer.calculate_weights(returns)

    # Low vol should get highest weight
    assert weights.weights["LOW_VOL"] > weights.weights["MED_VOL"]
    assert weights.weights["MED_VOL"] > weights.weights["HIGH_VOL"]

    # Weights should sum to 1.0
    assert abs(sum(weights.weights.values()) - 1.0) < 0.01

    # All weights should respect bounds
    for w in weights.weights.values():
        assert 0.10 <= w <= 0.50

def test_risk_parity_inverse_volatility():
    """Test inverse volatility weighting property."""
    # Create returns with known volatilities
    np.random.seed(42)
    returns = pd.DataFrame({
        "A": np.random.normal(0, 0.01, 252),  # σ ≈ 0.01
        "B": np.random.normal(0, 0.02, 252),  # σ ≈ 0.02
    })

    rebalancer = RiskParityRebalancer(
        lookback_days=252,
        min_weight=0.0,
        max_weight=1.0,
    )

    weights = rebalancer.calculate_weights(returns)

    # Weight ratio should be inverse of volatility ratio
    # w_A / w_B ≈ σ_B / σ_A ≈ 2.0
    weight_ratio = weights.weights["A"] / weights.weights["B"]
    assert 1.8 < weight_ratio < 2.2  # Allow some tolerance
```

### Testing Signal-Driven Strategy

```python
from ordinis.engines.portfolio import (
    SignalDrivenRebalancer,
    SignalInput,
    SignalMethod,
)

def test_signal_driven_proportional():
    """Test proportional signal weighting."""
    signals = [
        SignalInput("AAPL", signal=0.8, confidence=0.9, source="RSI"),
        SignalInput("MSFT", signal=0.4, confidence=0.7, source="MACD"),
        SignalInput("GOOGL", signal=0.2, confidence=0.5, source="RSI"),
    ]

    rebalancer = SignalDrivenRebalancer(
        method=SignalMethod.PROPORTIONAL,
        min_weight=0.0,
        max_weight=0.50,
        cash_buffer=0.0,
    )

    weights = rebalancer.calculate_weights(signals)

    # AAPL has highest signal×confidence, should have highest weight
    assert weights.weights["AAPL"] > weights.weights["MSFT"]
    assert weights.weights["MSFT"] > weights.weights["GOOGL"]

    # Weights should sum to 1.0
    assert abs(sum(weights.weights.values()) - 1.0) < 0.01

def test_signal_driven_binary():
    """Test binary signal method."""
    signals = [
        SignalInput("AAPL", signal=0.8, confidence=0.9, source="RSI"),
        SignalInput("MSFT", signal=0.2, confidence=0.7, source="MACD"),
        SignalInput("GOOGL", signal=-0.5, confidence=0.8, source="RSI"),
    ]

    rebalancer = SignalDrivenRebalancer(
        method=SignalMethod.BINARY,
        min_weight=0.0,
        max_weight=0.60,
        cash_buffer=0.10,  # 10% cash
    )

    weights = rebalancer.calculate_weights(signals)

    # Only positive signals should be included
    assert "AAPL" in weights.weights
    assert "MSFT" in weights.weights
    assert "GOOGL" not in weights.weights

    # Equal weights for positive signals (90% invested due to cash buffer)
    assert abs(weights.weights["AAPL"] - 0.45) < 0.01
    assert abs(weights.weights["MSFT"] - 0.45) < 0.01

def test_signal_driven_cash_buffer():
    """Test cash buffer application."""
    signals = [
        SignalInput("AAPL", signal=1.0, confidence=1.0, source="Test"),
    ]

    rebalancer = SignalDrivenRebalancer(
        method=SignalMethod.BINARY,
        cash_buffer=0.20,  # 20% cash
    )

    weights = rebalancer.calculate_weights(signals)

    # Should only invest 80%
    assert abs(weights.weights["AAPL"] - 0.80) < 0.01
```

## Integration Testing with Portfolio Engine

```python
import asyncio
from ordinis.engines.portfolio import (
    PortfolioEngine,
    PortfolioEngineConfig,
    StrategyType,
)

async def test_portfolio_engine_integration():
    """Test position sizing through full portfolio engine."""
    # Setup configuration
    config = PortfolioEngineConfig(
        initial_capital=100000.0,
        max_position_pct=0.25,
        min_trade_value=100.0,
    )

    engine = PortfolioEngine(config)
    await engine.initialize()

    try:
        # Create target allocation strategy
        targets = [
            TargetAllocation("AAPL", 0.40),
            TargetAllocation("MSFT", 0.30),
            TargetAllocation("GOOGL", 0.30),
        ]

        strategy = TargetAllocationRebalancer(targets)
        engine.register_strategy(StrategyType.TARGET_ALLOCATION, strategy)

        # Current portfolio state
        positions = {"AAPL": 100, "MSFT": 50, "GOOGL": 30}
        prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 100.0}

        # Generate decisions
        decisions = await engine.generate_rebalancing_decisions(
            positions=positions,
            prices=prices,
            strategy_type=StrategyType.TARGET_ALLOCATION,
        )

        assert len(decisions) > 0

        # Validate decisions respect constraints
        for decision in decisions:
            # No position should exceed max_position_pct
            target_value = decision.target_weight * sum(
                positions.get(s, 0) * prices.get(s, 0)
                for s in ["AAPL", "MSFT", "GOOGL"]
            )
            assert decision.target_weight <= config.max_position_pct

    finally:
        await engine.shutdown()
```

## Backtesting Position Sizing

### Basic Backtest

```python
from ordinis.backtesting import BacktestRunner, BacktestConfig
from ordinis.engines.proofbench import ProofBenchEngine

async def test_position_sizing_backtest():
    """Backtest position sizing strategy."""
    # Configure backtest
    config = BacktestConfig(
        initial_capital=100000.0,
        max_position_size=0.10,  # 10% max per position
        max_portfolio_exposure=1.0,
        commission_pct=0.001,
        slippage_bps=5,
    )

    # Setup backtest engine
    runner = BacktestRunner(config, output_dir="backtest_results")

    # Run backtest with target allocation strategy
    results = await runner.run_backtest(
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date="2023-01-01",
        end_date="2023-12-31",
        strategy_type="target_allocation",
        strategy_params={
            "targets": {
                "AAPL": 0.40,
                "MSFT": 0.30,
                "GOOGL": 0.30,
            }
        }
    )

    # Validate results
    assert results.total_return is not None
    assert results.sharpe_ratio is not None
    assert results.max_drawdown is not None

    print(f"Total Return: {results.total_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")

    return results
```

### Parameter Sweep for Optimization

```python
import itertools
import pandas as pd

async def parameter_sweep_position_sizing():
    """Test multiple position sizing parameters to find optimal."""
    # Parameter grid
    max_position_sizes = [0.05, 0.10, 0.15, 0.20, 0.25]
    drift_thresholds = [0.02, 0.05, 0.08, 0.10]

    results = []

    for max_pos, drift in itertools.product(max_position_sizes, drift_thresholds):
        print(f"Testing max_position={max_pos}, drift={drift}")

        config = BacktestConfig(
            initial_capital=100000.0,
            max_position_size=max_pos,
        )

        runner = BacktestRunner(config)
        result = await runner.run_backtest(
            symbols=["AAPL", "MSFT", "GOOGL"],
            start_date="2023-01-01",
            end_date="2023-12-31",
            strategy_type="target_allocation",
            strategy_params={
                "drift_threshold": drift,
                "targets": {"AAPL": 0.40, "MSFT": 0.30, "GOOGL": 0.30}
            }
        )

        results.append({
            "max_position_size": max_pos,
            "drift_threshold": drift,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "win_rate": result.win_rate,
        })

    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)

    # Find optimal parameters (highest Sharpe ratio)
    best = df.loc[df["sharpe_ratio"].idxmax()]
    print("\nBest Parameters:")
    print(f"  max_position_size: {best['max_position_size']}")
    print(f"  drift_threshold: {best['drift_threshold']}")
    print(f"  Sharpe Ratio: {best['sharpe_ratio']:.2f}")

    return df
```

## Learning Engine Integration

### Recording Position Sizing Events

```python
from ordinis.engines.learning import (
    LearningEngine,
    LearningEngineConfig,
    LearningEvent,
    EventType,
)

async def test_position_sizing_with_learning():
    """Test position sizing with learning engine feedback."""
    # Setup learning engine
    learning_config = LearningEngineConfig(
        data_dir=Path("artifacts/learning"),
        max_events_memory=10000,
    )
    learning_engine = LearningEngine(learning_config)
    await learning_engine.initialize()

    try:
        # Setup portfolio engine
        portfolio_config = PortfolioEngineConfig(
            initial_capital=100000.0,
            max_position_pct=0.15,
        )
        portfolio_engine = PortfolioEngine(portfolio_config)
        await portfolio_engine.initialize()

        # Record position sizing decision
        learning_engine.record_event(
            LearningEvent(
                event_type=EventType.REBALANCE_TRIGGERED,
                source_engine="portfolio",
                symbol="AAPL",
                payload={
                    "current_weight": 0.25,
                    "target_weight": 0.40,
                    "adjustment_shares": 50,
                    "max_position_pct": 0.15,
                },
            )
        )

        # Simulate position execution
        learning_engine.record_event(
            LearningEvent(
                event_type=EventType.POSITION_OPENED,
                source_engine="execution",
                symbol="AAPL",
                payload={
                    "shares": 50,
                    "price": 150.0,
                    "position_value": 7500.0,
                    "portfolio_pct": 0.075,
                },
            )
        )

        # Record outcome after 30 days
        learning_engine.record_event(
            LearningEvent(
                event_type=EventType.POSITION_CLOSED,
                source_engine="execution",
                symbol="AAPL",
                payload={
                    "entry_price": 150.0,
                    "exit_price": 165.0,
                    "return": 0.10,  # 10% gain
                    "holding_period_days": 30,
                },
                outcome=0.10,  # Track outcome for learning
            )
        )

        # Query learning data
        position_events = learning_engine.get_events(
            event_type=EventType.POSITION_CLOSED
        )
        print(f"Recorded {len(position_events)} position outcomes")

        # Analyze outcomes by position size
        outcomes_by_size = {}
        for event in position_events:
            if event.payload and "position_value" in event.payload:
                size = event.payload["position_value"]
                size_bucket = round(size / 1000) * 1000  # Bucket by $1000
                if size_bucket not in outcomes_by_size:
                    outcomes_by_size[size_bucket] = []
                if event.outcome is not None:
                    outcomes_by_size[size_bucket].append(event.outcome)

        # Calculate average return by position size
        for size, outcomes in sorted(outcomes_by_size.items()):
            avg_return = sum(outcomes) / len(outcomes) if outcomes else 0
            print(f"Position size ${size}: Avg return = {avg_return:.2%}")

    finally:
        await learning_engine.shutdown()
        await portfolio_engine.shutdown()
```

### Optimizing Position Sizing with Learning Engine

```python
async def optimize_position_sizing_with_learning():
    """Use learning engine to optimize position sizing parameters."""
    learning_engine = LearningEngine(LearningEngineConfig(
        data_dir=Path("artifacts/learning_optimization")
    ))
    await learning_engine.initialize()

    try:
        # Test different position sizing strategies
        strategies_to_test = [
            {
                "name": "conservative",
                "max_position_pct": 0.05,
                "min_position_pct": 0.02,
            },
            {
                "name": "balanced",
                "max_position_pct": 0.10,
                "min_position_pct": 0.03,
            },
            {
                "name": "aggressive",
                "max_position_pct": 0.20,
                "min_position_pct": 0.05,
            },
        ]

        results = {}

        for strategy in strategies_to_test:
            print(f"\nTesting strategy: {strategy['name']}")

            # Run backtest with this strategy
            config = BacktestConfig(
                initial_capital=100000.0,
                max_position_size=strategy["max_position_pct"],
            )

            runner = BacktestRunner(config)
            result = await runner.run_backtest(
                symbols=["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"],
                start_date="2022-01-01",
                end_date="2023-12-31",
                strategy_type="target_allocation",
            )

            # Record results in learning engine
            learning_engine.record_event(
                LearningEvent(
                    event_type=EventType.STRATEGY_PERFORMANCE,
                    source_engine="backtest",
                    symbol="PORTFOLIO",
                    payload={
                        "strategy_name": strategy["name"],
                        "max_position_pct": strategy["max_position_pct"],
                        "total_return": result.total_return,
                        "sharpe_ratio": result.sharpe_ratio,
                        "max_drawdown": result.max_drawdown,
                        "win_rate": result.win_rate,
                        "num_trades": result.num_trades,
                    },
                    outcome=result.sharpe_ratio,  # Use Sharpe as optimization metric
                )
            )

            results[strategy["name"]] = {
                "sharpe_ratio": result.sharpe_ratio,
                "total_return": result.total_return,
                "max_drawdown": result.max_drawdown,
            }

        # Analyze results
        performance_events = learning_engine.get_events(
            event_type=EventType.STRATEGY_PERFORMANCE
        )

        best_strategy = max(
            performance_events,
            key=lambda e: e.outcome if e.outcome else -float("inf")
        )

        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)

        if best_strategy.payload:
            print(f"Best Strategy: {best_strategy.payload['strategy_name']}")
            print(f"  Max Position %: {best_strategy.payload['max_position_pct']:.1%}")
            print(f"  Sharpe Ratio: {best_strategy.payload['sharpe_ratio']:.2f}")
            print(f"  Total Return: {best_strategy.payload['total_return']:.2%}")
            print(f"  Max Drawdown: {best_strategy.payload['max_drawdown']:.2%}")

        # Save optimized parameters
        optimized_config = {
            "max_position_pct": best_strategy.payload["max_position_pct"],
            "strategy": best_strategy.payload["strategy_name"],
            "backtest_sharpe": best_strategy.payload["sharpe_ratio"],
        }

        return optimized_config

    finally:
        await learning_engine.shutdown()
```

## Running the Tests

### Run Unit Tests

```bash
# Run all position sizing tests
pytest tests/ -k "position" -v

# Run specific strategy tests
pytest tests/ -k "target_allocation" -v
pytest tests/ -k "risk_parity" -v
pytest tests/ -k "signal_driven" -v
```

### Run Integration Tests

```bash
# Run portfolio engine integration tests
pytest tests/ -k "portfolio_engine" -v

# Run with coverage
pytest tests/ -k "position" --cov=src/ordinis/engines/portfolio --cov-report=html
```

### Run Backtests

```bash
# Run single backtest
python examples/backtest_position_sizing.py

# Run parameter sweep
python examples/optimize_position_sizing.py

# Run with learning engine
python examples/position_sizing_with_learning.py
```

## Best Practices

1. **Start Simple**: Test basic strategies before complex optimizations
2. **Use Synthetic Data**: Generate known patterns for unit tests
3. **Test Edge Cases**: Empty portfolios, single assets, extreme drifts
4. **Validate Constraints**: Ensure weights sum to 1.0, respect bounds
5. **Track Outcomes**: Use LearningEngine to record all decisions and results
6. **Compare Strategies**: Run multiple strategies on same data
7. **Walk-Forward Testing**: Test on out-of-sample periods
8. **Monitor Drift**: Track parameter performance over time

## Common Issues and Solutions

### Issue: Weights Don't Sum to 1.0

```python
# Check floating point precision
assert abs(sum(weights.values()) - 1.0) < 1e-6

# Renormalize if needed
total = sum(weights.values())
weights = {k: v/total for k, v in weights.items()}
```

### Issue: Insufficient Historical Data

```python
# Ensure minimum lookback period
if len(returns) < rebalancer.lookback_days:
    raise ValueError(f"Need at least {rebalancer.lookback_days} days of data")
```

### Issue: No Rebalancing Triggered

```python
# Lower drift threshold for more frequent rebalancing
rebalancer = TargetAllocationRebalancer(
    targets,
    drift_threshold=0.02  # 2% instead of default 5%
)
```

## Next Steps

- Review [POSITION_SIZING_LOGIC.md](../POSITION_SIZING_LOGIC.md) for detailed strategy documentation
- Check [examples/](../../examples/) for complete working examples
- See [scripts/ai/test_learning_engine_integration.py](../../scripts/ai/test_learning_engine_integration.py) for learning engine patterns
