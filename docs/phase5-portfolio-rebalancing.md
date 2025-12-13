# Phase 5: Portfolio Rebalancing Engine

**Status**: Complete
**Tests**: 137 passing
**Commit**: 8eef150b

## Overview

Phase 5 implements a comprehensive portfolio rebalancing system with five distinct strategies and a unified orchestration engine. The system provides sophisticated rebalancing logic with history tracking, execution management, and flexible strategy switching.

## Architecture

### Core Components

```
portfolio/
├── target_allocation.py    # Target allocation rebalancing
├── risk_parity.py          # Risk parity rebalancing
├── signal_driven.py        # Signal-based rebalancing
├── threshold_based.py      # Corridor/threshold rebalancing
└── engine.py              # Unified orchestration engine
```

### Strategy Types

The system supports five rebalancing strategies via the `StrategyType` enum:

1. **TARGET_ALLOCATION** - Fixed target weights
2. **RISK_PARITY** - Equal risk contribution
3. **SIGNAL_DRIVEN** - Signal-based weights
4. **THRESHOLD_BASED** - Corridor/band rebalancing

## Rebalancing Strategies

### 1. Target Allocation Rebalancing

Maintains fixed target weights for each symbol in the portfolio.

**Key Classes:**
- `TargetAllocation` - Target weight configuration
- `TargetAllocationRebalancer` - Rebalancing logic
- `RebalanceDecision` - Rebalancing decision output

**Example:**
```python
from ordinis.engines.portfolio import (
    TargetAllocationRebalancer,
    TargetAllocation,
)

# Define target allocations
targets = [
    TargetAllocation("AAPL", 0.40),
    TargetAllocation("MSFT", 0.30),
    TargetAllocation("GOOGL", 0.30),
]

# Create rebalancer with 5% drift threshold
rebalancer = TargetAllocationRebalancer(
    target_allocations=targets,
    drift_threshold=0.05,
)

# Check if rebalancing is needed
positions = {"AAPL": 10, "MSFT": 5, "GOOGL": 5}
prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}

if rebalancer.should_rebalance(positions, prices):
    decisions = rebalancer.generate_rebalance_orders(
        positions, prices, cash=5000.0
    )
    for decision in decisions:
        print(f"{decision.symbol}: {decision.adjustment_shares:+.2f} shares")
```

**Features:**
- Drift threshold to avoid excessive trading
- Cash injection/withdrawal support
- Current vs. target weight tracking
- Dollar and share adjustment calculations

---

### 2. Risk Parity Rebalancing

Allocates capital such that each asset contributes equally to portfolio risk.

**Key Classes:**
- `RiskParityRebalancer` - Risk parity logic
- `RiskParityWeights` - Weight calculation results
- `RiskParityDecision` - Rebalancing decision output

**Example:**
```python
from ordinis.engines.portfolio import RiskParityRebalancer
import pandas as pd

# Historical returns for volatility estimation
returns = pd.DataFrame({
    "AAPL": [0.01, -0.02, 0.03, ...],
    "MSFT": [0.02, 0.01, -0.01, ...],
    "GOOGL": [0.00, 0.02, -0.02, ...],
})

rebalancer = RiskParityRebalancer(
    symbols=["AAPL", "MSFT", "GOOGL"],
    lookback_period=252,
    rebalance_threshold=0.05,
)

# Calculate risk parity weights
weights = rebalancer.calculate_risk_parity_weights(returns)
print(f"Risk parity weights: {weights.weights}")
print(f"Risk contributions: {weights.risk_contributions}")

# Generate rebalancing decisions
positions = {"AAPL": 10, "MSFT": 5, "GOOGL": 5}
prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}

decisions = rebalancer.generate_rebalance_orders(
    positions, prices, returns, cash=5000.0
)
```

**Features:**
- Volatility-based weight calculation
- Equal risk contribution across assets
- Correlation-aware rebalancing
- Configurable lookback period

---

### 3. Signal-Driven Rebalancing

Adjusts allocations based on external signals (momentum, value, sentiment, etc.).

**Key Classes:**
- `SignalDrivenRebalancer` - Signal-based logic
- `SignalInput` - Signal data structure
- `SignalMethod` - Signal aggregation method enum
- `SignalDrivenWeights` - Weight calculation results
- `SignalDrivenDecision` - Rebalancing decision output

**Example:**
```python
from ordinis.engines.portfolio import (
    SignalDrivenRebalancer,
    SignalInput,
    SignalMethod,
)

# Define signals
signals = [
    SignalInput("AAPL", momentum=0.8, value=0.5, sentiment=0.7),
    SignalInput("MSFT", momentum=0.6, value=0.8, sentiment=0.6),
    SignalInput("GOOGL", momentum=0.4, value=0.6, sentiment=0.5),
]

rebalancer = SignalDrivenRebalancer(
    symbols=["AAPL", "MSFT", "GOOGL"],
    signal_method=SignalMethod.WEIGHTED_AVERAGE,
    signal_weights={"momentum": 0.5, "value": 0.3, "sentiment": 0.2},
    rebalance_threshold=0.05,
)

# Calculate signal-based weights
weights = rebalancer.calculate_signal_weights(signals)
print(f"Signal-based weights: {weights.weights}")
print(f"Composite signals: {weights.composite_signals}")

# Generate rebalancing decisions
positions = {"AAPL": 10, "MSFT": 5, "GOOGL": 5}
prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}

decisions = rebalancer.generate_rebalance_orders(
    positions, prices, signals, cash=5000.0
)
```

**Features:**
- Multiple signal aggregation methods (average, weighted, rank-based)
- Configurable signal weights
- Signal normalization and validation
- Composite signal calculation

---

### 4. Threshold-Based Rebalancing

Uses corridor/band thresholds to trigger rebalancing only when drift exceeds specified limits.

**Key Classes:**
- `ThresholdConfig` - Threshold configuration per symbol
- `ThresholdBasedRebalancer` - Threshold logic
- `ThresholdStatus` - Current threshold status
- `ThresholdDecision` - Rebalancing decision output

**Example:**
```python
from ordinis.engines.portfolio import (
    ThresholdBasedRebalancer,
    ThresholdConfig,
)
from datetime import datetime, UTC

# Define threshold configurations with asymmetric bands
configs = [
    ThresholdConfig("AAPL", target_weight=0.40, lower_band=-0.05, upper_band=0.05),
    ThresholdConfig("MSFT", target_weight=0.30, lower_band=-0.05, upper_band=0.05),
    ThresholdConfig("GOOGL", target_weight=0.30, lower_band=-0.05, upper_band=0.05),
]

rebalancer = ThresholdBasedRebalancer(
    threshold_configs=configs,
    min_days_between_rebalance=30,
    min_trade_value=100.0,
)

# Check threshold status
positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}
last_rebalance = datetime(2023, 11, 1, tzinfo=UTC)

statuses = rebalancer.check_thresholds(positions, prices, last_rebalance)
for status in statuses:
    if status.should_rebalance:
        print(f"{status.symbol}: Drift {status.drift:+.2%} breaches threshold")

# Generate rebalancing decisions
decisions = rebalancer.generate_rebalance_orders(
    positions, prices, last_rebalance, cash=5000.0
)
```

**Features:**
- Upper and lower drift bands per symbol
- Time-based constraints (minimum days between rebalances)
- Transaction cost filtering (minimum trade value)
- Asymmetric bands support
- Detailed trigger reason tracking

---

## Unified Rebalancing Engine

The `RebalancingEngine` provides a single interface for managing multiple strategies with history tracking and execution management.

### Key Features

1. **Multi-Strategy Support** - Register and switch between strategies at runtime
2. **History Tracking** - Audit trail of all rebalancing events
3. **Execution Management** - Callback-based execution with error handling
4. **Flexible API** - Strategy-agnostic interface

### Example Usage

```python
from ordinis.engines.portfolio import (
    RebalancingEngine,
    StrategyType,
    TargetAllocationRebalancer,
    TargetAllocation,
)

# Create engine with history tracking
engine = RebalancingEngine(
    default_strategy=StrategyType.TARGET_ALLOCATION,
    track_history=True,
)

# Register strategies
targets = [
    TargetAllocation("AAPL", 0.40),
    TargetAllocation("MSFT", 0.30),
    TargetAllocation("GOOGL", 0.30),
]
target_strategy = TargetAllocationRebalancer(targets, drift_threshold=0.05)
engine.register_strategy(StrategyType.TARGET_ALLOCATION, target_strategy)

# Generate rebalancing decisions
positions = {"AAPL": 30, "MSFT": 5, "GOOGL": 5}
prices = {"AAPL": 200.0, "MSFT": 300.0, "GOOGL": 500.0}

if engine.should_rebalance(positions, prices):
    decisions = engine.generate_rebalancing_decisions(
        positions, prices, cash=5000.0
    )

    # Execute with custom callback
    def execute_order(decision):
        # Your execution logic here
        try:
            # Place order via broker API
            return (True, None)  # Success
        except Exception as e:
            return (False, str(e))  # Failure

    result = engine.execute_rebalancing(decisions, execution_callback=execute_order)

    print(f"Executed: {result.decisions_executed}")
    print(f"Failed: {result.decisions_failed}")
    print(f"Total traded: ${result.total_value_traded:,.2f}")

    if result.errors:
        for error in result.errors:
            print(f"Error: {error}")

# View history
history_df = engine.get_history_summary()
print(history_df)
```

### History Tracking

The engine tracks:
- Timestamp of each rebalancing event
- Strategy type used
- Number of decisions generated
- Total adjustment value
- Execution status (planned, executed, partial)
- Strategy-specific metadata

```python
# Access raw history
for entry in engine.history:
    print(f"{entry.timestamp}: {entry.strategy_type.value}")
    print(f"  Decisions: {entry.decisions_count}")
    print(f"  Value: ${entry.total_adjustment_value:,.2f}")
    print(f"  Status: {entry.execution_status}")

# Get summary DataFrame
df = engine.get_history_summary(limit=10)
print(df)
```

### Execution Management

The engine supports three execution modes:

1. **Simulation** - No callback, simulates successful execution
2. **Custom Callback** - User-provided execution logic
3. **Error Handling** - Tracks failures with detailed error messages

```python
# Simulation mode (default)
result = engine.execute_rebalancing(decisions)

# Custom callback mode
def my_callback(decision):
    # Return (success, error_message)
    if decision.symbol == "RESTRICTED":
        return (False, "Symbol restricted")
    # Execute order...
    return (True, None)

result = engine.execute_rebalancing(decisions, execution_callback=my_callback)

# Check results
if result.success:
    print("All orders executed successfully")
else:
    print(f"Partial execution: {result.decisions_failed} failed")
    for error in result.errors:
        print(f"  - {error}")
```

## Integration Points

### With ProofBench (Backtesting)

```python
from ordinis.engines.proofbench import SimulationEngine
from ordinis.engines.portfolio import RebalancingEngine

# In your strategy's on_bar() method
def on_bar(self, bar):
    # Check if rebalancing is needed
    if self.rebalance_engine.should_rebalance(
        self.portfolio.positions,
        self.current_prices
    ):
        decisions = self.rebalance_engine.generate_rebalancing_decisions(
            self.portfolio.positions,
            self.current_prices,
            cash=self.portfolio.cash,
        )

        # Execute decisions
        for decision in decisions:
            if decision.adjustment_shares > 0:
                self.buy(decision.symbol, abs(decision.adjustment_shares))
            elif decision.adjustment_shares < 0:
                self.sell(decision.symbol, abs(decision.adjustment_shares))
```

### With FlowRoute (Live Trading)

```python
from ordinis.engines.flowroute import TradingEngine
from ordinis.engines.portfolio import RebalancingEngine

# Execution callback that places live orders
def execute_via_flowroute(decision):
    try:
        if decision.adjustment_shares > 0:
            order = trading_engine.market_order(
                decision.symbol,
                abs(decision.adjustment_shares),
                side="BUY"
            )
        else:
            order = trading_engine.market_order(
                decision.symbol,
                abs(decision.adjustment_shares),
                side="SELL"
            )
        return (True, None)
    except Exception as e:
        return (False, str(e))

result = engine.execute_rebalancing(decisions, execution_callback=execute_via_flowroute)
```

### With RiskGuard (Risk Management)

```python
from ordinis.engines.riskguard import RiskEngine
from ordinis.engines.portfolio import RebalancingEngine

# Pre-validate rebalancing decisions
def risk_aware_callback(decision):
    # Check risk limits before execution
    if not risk_engine.validate_order(decision.symbol, decision.adjustment_shares):
        return (False, "Risk limit exceeded")

    # Execute if risk check passes
    return execute_order(decision)

result = engine.execute_rebalancing(decisions, execution_callback=risk_aware_callback)
```

## Testing

Phase 5 includes comprehensive test coverage:

- **137 total tests** across 5 modules
- **Target Allocation**: 22 tests
- **Risk Parity**: 22 tests
- **Signal-Driven**: 32 tests
- **Threshold-Based**: 32 tests
- **Unified Engine**: 29 tests

Run tests:
```bash
# All portfolio tests
pytest tests/test_engines/test_portfolio/ -v

# Specific strategy
pytest tests/test_engines/test_portfolio/test_target_allocation.py -v
pytest tests/test_engines/test_portfolio/test_risk_parity.py -v
pytest tests/test_engines/test_portfolio/test_signal_driven.py -v
pytest tests/test_engines/test_portfolio/test_threshold_based.py -v

# Unified engine
pytest tests/test_engines/test_portfolio/test_engine.py -v
```

## Performance Considerations

### Transaction Cost Optimization

1. **Threshold-Based Strategy** - Only rebalance when drift exceeds bands
2. **Time Constraints** - Minimum days between rebalances
3. **Trade Value Filtering** - Skip small trades below minimum value
4. **Drift Thresholds** - Prevent excessive rebalancing

### Computational Efficiency

1. **Risk Parity** - O(n²) for correlation matrix, configurable lookback
2. **Signal Aggregation** - O(n) for weighted average, O(n log n) for rank-based
3. **Decision Generation** - O(n) for all strategies
4. **History Tracking** - Optional, can be disabled for production

## Best Practices

### 1. Choosing a Strategy

- **Target Allocation** - Simple, predictable allocations
- **Risk Parity** - Equal risk contribution, volatility-aware
- **Signal-Driven** - Factor-based, adaptive allocations
- **Threshold-Based** - Cost-aware, minimizes trading

### 2. Configuration Guidelines

**Drift Thresholds:**
- Conservative: 1-2% (more frequent rebalancing)
- Moderate: 3-5% (balanced approach)
- Aggressive: 7-10% (minimize trading costs)

**Time Constraints:**
- Daily: 1-7 days (high-frequency strategies)
- Weekly: 7-14 days (tactical rebalancing)
- Monthly: 30-60 days (strategic rebalancing)
- Quarterly: 90-120 days (long-term portfolios)

**Transaction Cost Filtering:**
- Small accounts: $50-100 minimum trade
- Medium accounts: $100-500 minimum trade
- Large accounts: $500-1000+ minimum trade

### 3. Error Handling

Always handle execution errors gracefully:

```python
result = engine.execute_rebalancing(decisions, execution_callback=my_callback)

if not result.success:
    # Log errors
    for error in result.errors:
        logger.error(f"Rebalancing error: {error}")

    # Notify operators
    if result.decisions_failed > 0:
        alert_operations(f"{result.decisions_failed} orders failed")

    # Retry failed orders (optional)
    if result.decisions_failed > 0 and result.decisions_failed < 3:
        retry_failed_orders(result.errors)
```

### 4. History Analysis

Use history to monitor rebalancing effectiveness:

```python
# Get recent history
df = engine.get_history_summary(limit=30)

# Analyze rebalancing frequency
rebalances_per_month = df.groupby(df['timestamp'].dt.to_period('M')).size()

# Track total adjustment value
total_traded = df['total_adjustment_value'].sum()

# Monitor execution success rate
execution_rate = (df['execution_status'] == 'executed').mean()
```

## Troubleshooting

### Issue: Rebalancing not triggered

**Symptoms:** `should_rebalance()` returns False when expected

**Possible causes:**
1. Drift below threshold
2. Time constraint not met (threshold-based)
3. Insufficient price data (risk parity)
4. Missing signals (signal-driven)

**Solution:**
```python
# Check current drift
from ordinis.engines.portfolio import TargetAllocationRebalancer

rebalancer = TargetAllocationRebalancer(targets, drift_threshold=0.05)
current_weights = rebalancer.calculate_current_weights(positions, prices)

for symbol, weight in current_weights.items():
    target = next(t for t in targets if t.symbol == symbol)
    drift = weight - target.target_weight
    print(f"{symbol}: {drift:+.2%} drift (threshold: ±{0.05:.1%})")
```

### Issue: Execution callback errors

**Symptoms:** All executions fail, errors not descriptive

**Solution:**
```python
def robust_callback(decision):
    try:
        # Your execution logic
        order = place_order(decision.symbol, decision.adjustment_shares)
        return (True, None)
    except BrokerError as e:
        # Return descriptive error
        return (False, f"Broker error: {e.message}")
    except Exception as e:
        # Catch unexpected errors
        logger.exception("Unexpected execution error")
        return (False, f"Unexpected error: {e!s}")
```

### Issue: Risk parity weights unstable

**Symptoms:** Weights change drastically between rebalances

**Possible causes:**
1. Insufficient lookback period
2. High market volatility
3. Missing return data

**Solution:**
```python
# Increase lookback period
rebalancer = RiskParityRebalancer(
    symbols=symbols,
    lookback_period=504,  # 2 years instead of 1
    rebalance_threshold=0.05,
)

# Add minimum weight constraints (manual clipping)
weights = rebalancer.calculate_risk_parity_weights(returns)
for symbol in weights.weights:
    if weights.weights[symbol] < 0.05:  # Minimum 5%
        weights.weights[symbol] = 0.05
    if weights.weights[symbol] > 0.50:  # Maximum 50%
        weights.weights[symbol] = 0.50

# Renormalize
total = sum(weights.weights.values())
weights.weights = {s: w/total for s, w in weights.weights.items()}
```

## Future Enhancements

Potential Phase 6 features:

1. **Smart Order Routing** - VWAP, TWAP, iceberg orders
2. **Tax-Loss Harvesting** - Minimize capital gains
3. **Multi-Asset Class** - Bonds, commodities, alternatives
4. **Mean-Variance Optimization** - Markowitz efficient frontier
5. **Black-Litterman** - Combine market equilibrium with views
6. **Hierarchical Risk Parity** - Cluster-based allocation
7. **Kelly Criterion** - Optimal position sizing
8. **Conditional Rebalancing** - Market regime awareness

## References

### Academic Papers

1. **Risk Parity:**
   - Maillard, S., Roncalli, T., & Teiletche, J. (2010). "The Properties of Equally Weighted Risk Contribution Portfolios"

2. **Threshold-Based Rebalancing:**
   - Donohue, C., & Yip, K. (2003). "Optimal Portfolio Rebalancing with Transaction Costs"

3. **Signal-Driven:**
   - Fama, E., & French, K. (1993). "Common risk factors in the returns on stocks and bonds"

### Industry Standards

- CFA Institute: "Portfolio Management in Practice"
- GARP: "Market Risk Handbook"
- SEC: "Investment Adviser Regulations"

## Changelog

### Phase 5.5 (2025-12-13)
- Added unified rebalancing engine
- Implemented history tracking
- Added execution management with callbacks

### Phase 5.4 (2025-12-13)
- Added threshold-based rebalancing
- Implemented corridor/band logic
- Added time and transaction cost constraints

### Phase 5.3 (2025-12-12)
- Added signal-driven rebalancing
- Implemented multiple signal aggregation methods
- Added composite signal calculation

### Phase 5.2 (2025-12-12)
- Added risk parity rebalancing
- Implemented volatility-based weight calculation
- Added equal risk contribution logic

### Phase 5.1 (2025-12-12)
- Initial release: Target allocation rebalancing
- Basic drift threshold logic
- Cash injection/withdrawal support
