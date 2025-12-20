# Position Sizing Quick Reference

> For detailed documentation, see [POSITION_SIZING_LOGIC.md](../POSITION_SIZING_LOGIC.md)

## Overview

Position sizing in Ordinis is handled through:
- **Portfolio Engine**: Weight-based rebalancing strategies
- **PortfolioOpt Engine**: GPU-accelerated optimization with Mean-CVaR

## Quick Start Examples

### 1. Target Allocation (Fixed Weights)

```python
from ordinis.engines.portfolio import (
    PortfolioEngine,
    TargetAllocationRebalancer,
    TargetAllocation,
    StrategyType
)

# Define target allocations (must sum to 1.0)
targets = [
    TargetAllocation("AAPL", 0.40),   # 40% of portfolio
    TargetAllocation("MSFT", 0.30),   # 30% of portfolio
    TargetAllocation("GOOGL", 0.30),  # 30% of portfolio
]

# Create and configure engine
engine = PortfolioEngine()
strategy = TargetAllocationRebalancer(targets, drift_threshold=0.05)
engine.register_strategy(StrategyType.TARGET_ALLOCATION, strategy)

# Generate rebalancing decisions
positions = {"AAPL": 10, "MSFT": 5, "GOOGL": 3}
prices = {"AAPL": 150.0, "MSFT": 300.0, "GOOGL": 100.0}

decisions = await engine.generate_rebalancing_decisions(
    positions=positions,
    prices=prices
)
```

### 2. Risk Parity (Equal Risk Contribution)

```python
from ordinis.engines.portfolio import RiskParityRebalancer
import pandas as pd

# Historical returns (columns = symbols, rows = dates)
returns = pd.DataFrame({
    "AAPL": [0.01, -0.02, 0.015, ...],
    "MSFT": [0.005, 0.01, -0.005, ...],
    "GOOGL": [0.02, -0.01, 0.025, ...],
})

# Create strategy
strategy = RiskParityRebalancer(
    lookback_days=252,    # 1 year lookback
    min_weight=0.05,      # 5% minimum per asset
    max_weight=0.50,      # 50% maximum per asset
)

# Calculate risk parity weights
weights = strategy.calculate_weights(returns)
print(weights.weights)  # {"AAPL": 0.30, "MSFT": 0.35, "GOOGL": 0.35}
```

### 3. Signal-Driven (Based on Trading Signals)

```python
from ordinis.engines.portfolio import (
    SignalDrivenRebalancer,
    SignalInput,
    SignalMethod
)

# Trading signals from your strategy
signals = [
    SignalInput("AAPL", signal=0.8, confidence=0.9, source="RSI"),
    SignalInput("MSFT", signal=0.3, confidence=0.7, source="MACD"),
    SignalInput("GOOGL", signal=-0.2, confidence=0.6, source="RSI"),  # Negative = skip
]

# Create strategy
strategy = SignalDrivenRebalancer(
    method=SignalMethod.PROPORTIONAL,  # Weight ∝ signal strength
    min_weight=0.0,
    max_weight=0.50,
    cash_buffer=0.10,  # Keep 10% in cash
)

# Calculate weights
weights = strategy.calculate_weights(signals)
# Only positive signals are included, weighted by signal × confidence
```

### 4. Portfolio Optimization (Mean-CVaR)

```python
from ordinis.engines.portfolioopt import PortfolioOptEngine, PortfolioOptEngineConfig

# Configure optimization
config = PortfolioOptEngineConfig(
    default_api="cvxpy",        # or "cuopt" for GPU
    target_return=0.001,        # 0.1% daily target return
    max_weight=0.20,            # 20% max per asset
    risk_aversion=0.5,          # Balance risk/return
    max_concentration=0.25,     # 25% max single asset
    min_diversification=5,      # At least 5 positions
)

# Create and initialize engine
engine = PortfolioOptEngine(config)
await engine.initialize()

# Run optimization
result = await engine.optimize(returns_df)

print(result.weights)          # Optimal weights
print(result.expected_return)  # Expected return
print(result.cvar)             # Conditional Value at Risk
```

## Common Configuration Patterns

### Conservative Portfolio

```python
# Portfolio Engine
PortfolioEngineConfig(
    max_position_pct=0.05,      # 5% max per position
    initial_capital=100000,
)

# Portfolio Optimization
PortfolioOptEngineConfig(
    max_weight=0.10,            # 10% max weight
    max_concentration=0.15,     # 15% max single asset
    min_diversification=10,     # At least 10 positions
    risk_aversion=0.8,          # High risk aversion
    max_cvar=0.05,              # 5% CVaR limit
)
```

### Balanced Portfolio

```python
# Portfolio Engine
PortfolioEngineConfig(
    max_position_pct=0.10,      # 10% max per position
    initial_capital=100000,
)

# Portfolio Optimization
PortfolioOptEngineConfig(
    max_weight=0.20,            # 20% max weight
    max_concentration=0.25,     # 25% max single asset
    min_diversification=5,      # At least 5 positions
    risk_aversion=0.5,          # Moderate risk aversion
    max_cvar=0.10,              # 10% CVaR limit
)
```

### Aggressive Portfolio

```python
# Portfolio Engine
PortfolioEngineConfig(
    max_position_pct=0.20,      # 20% max per position
    initial_capital=100000,
)

# Portfolio Optimization
PortfolioOptEngineConfig(
    max_weight=0.30,            # 30% max weight
    max_concentration=0.40,     # 40% max single asset
    min_diversification=3,      # At least 3 positions
    risk_aversion=0.2,          # Low risk aversion
    max_cvar=0.15,              # 15% CVaR limit
)
```

## Key Parameters Reference

### Portfolio Engine

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `max_position_pct` | Maximum % per position | 0.25 | 0.0-1.0 |
| `min_trade_value` | Minimum trade size ($) | 10.0 | >0 |
| `initial_capital` | Starting capital ($) | 100000 | >0 |

### Target Allocation

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `target_weight` | Desired allocation | - | 0.0-1.0 |
| `drift_threshold` | Max drift before rebalance | 0.05 | 0.0-1.0 |

### Risk Parity

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `lookback_days` | Historical window | 252 | ≥20 |
| `min_weight` | Min weight per asset | 0.01 | 0.0-1.0 |
| `max_weight` | Max weight per asset | 0.50 | 0.0-1.0 |
| `drift_threshold` | Rebalance trigger | 0.05 | 0.0-1.0 |

### Signal-Driven

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `method` | Signal→weight conversion | PROPORTIONAL | PROPORTIONAL, BINARY, RANKED |
| `min_weight` | Min weight per asset | 0.0 | 0.0-1.0 |
| `max_weight` | Max weight per asset | 0.50 | 0.0-1.0 |
| `cash_buffer` | Cash % to reserve | 0.0 | 0.0-1.0 |

### PortfolioOpt

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `target_return` | Daily target return | 0.001 | ≥0 |
| `max_weight` | Max weight per asset | 0.20 | 0.0-1.0 |
| `risk_aversion` | CVaR penalty (λ) | 0.5 | 0.0-1.0 |
| `max_concentration` | Max single asset | 0.25 | 0.0-1.0 |
| `min_diversification` | Min # of positions | 5 | ≥1 |
| `max_cvar` | CVaR limit | 0.10 | >0 |

## Formulas

### Weight Calculation

```python
# Current weight
current_weight = (shares × price) / total_portfolio_value

# Target value
target_value = total_portfolio_value × target_weight

# Shares adjustment
adjustment_shares = (target_value - current_value) / price
```

### Risk Parity

```python
# Inverse volatility weighting
weight_i = (1 / volatility_i) / Σ(1 / volatility_j)

# Risk contribution
risk_contribution_i = weight_i × volatility_i / portfolio_volatility
```

### Confidence-Adjusted Sizing

```python
# Base size
base_size = portfolio.equity × position_size_pct  # e.g., 5%

# Confidence adjustment
confidence = signal.probability × abs(signal.score)
position_value = base_size × confidence

# Cap at maximum
max_value = portfolio.equity × max_position_size_pct  # e.g., 15%
final_value = min(position_value, max_value)

# Convert to shares
quantity = int(final_value / price)
```

### Mean-CVaR Optimization

```
minimize: (1 - λ) × (-μᵀw) + λ × CVaR(w)

subject to:
  Σw_i = 1
  0 ≤ w_i ≤ max_weight
  μᵀw ≥ target_return
```

Where:
- `w`: Portfolio weights
- `μ`: Expected returns
- `λ`: Risk aversion (0=return focus, 1=risk focus)
- `CVaR`: Conditional Value at Risk

## Risk Management Layers

Position sizes are validated through multiple layers:

1. **Engine Configuration**
   - `max_weight`, `max_concentration`
   - Configured in engine initialization

2. **RiskGuard Rules**
   - `max_position_pct`, `min_position_size`
   - Applied pre-trade

3. **Governance Policies**
   - `daily_trade_limit`, `max_drawdown`
   - System-wide limits

## Integration Patterns

### Hybrid: Optimization + Rebalancing

```python
# Step 1: Generate optimal weights
opt_engine = PortfolioOptEngine(config)
result = await opt_engine.optimize(returns_df)

# Step 2: Convert to target allocations
targets = [
    TargetAllocation(symbol, weight)
    for symbol, weight in result.weights.items()
]

# Step 3: Execute via Portfolio Engine
portfolio_engine = PortfolioEngine()
strategy = TargetAllocationRebalancer(targets)
portfolio_engine.register_strategy(StrategyType.TARGET_ALLOCATION, strategy)

decisions = await portfolio_engine.generate_rebalancing_decisions(
    positions=current_positions,
    prices=current_prices
)
```

### Regime-Adaptive Sizing

```python
from ordinis.application.strategies.regime_adaptive import AdaptiveManager

# Position sizes adapt to market regime
manager = AdaptiveManager(config)
signal = await manager.generate_signal(data)

# Signal includes regime-adjusted position size
print(signal.position_size)  # e.g., 0.8 in trending, 0.5 in volatile
```

## Common Pitfalls

❌ **Don't**: Set weights that don't sum to 1.0
```python
# WRONG
targets = [
    TargetAllocation("AAPL", 0.40),
    TargetAllocation("MSFT", 0.40),  # Sums to 0.80, not 1.0!
]
```

✅ **Do**: Ensure weights sum to 1.0
```python
# CORRECT
targets = [
    TargetAllocation("AAPL", 0.40),
    TargetAllocation("MSFT", 0.30),
    TargetAllocation("GOOGL", 0.30),  # Sums to 1.0 ✓
]
```

❌ **Don't**: Ignore min_weight and max_weight constraints
```python
# Can lead to extreme allocations
strategy = RiskParityRebalancer(
    min_weight=0.0,  # Allows zero allocation
    max_weight=1.0,  # Allows 100% allocation
)
```

✅ **Do**: Set reasonable bounds
```python
# Better: enforce diversification
strategy = RiskParityRebalancer(
    min_weight=0.05,  # At least 5% per asset
    max_weight=0.30,  # At most 30% per asset
)
```

❌ **Don't**: Use insufficient historical data
```python
# Only 10 days of data for volatility calc
returns = returns_df.tail(10)
strategy.calculate_weights(returns)  # Unreliable!
```

✅ **Do**: Use adequate history
```python
# At least 60 days, ideally 252 (1 year)
returns = returns_df.tail(252)
strategy.calculate_weights(returns)
```

## Troubleshooting

### Issue: Rebalancing not triggered
**Check**: `drift_threshold` might be too high
```python
# Lower the threshold
strategy = TargetAllocationRebalancer(targets, drift_threshold=0.02)  # 2% instead of 5%
```

### Issue: Optimization fails with "infeasible"
**Check**: Constraints might be too restrictive
```python
# Relax constraints
config = PortfolioOptEngineConfig(
    target_return=0.0005,    # Lower target
    max_weight=0.30,         # Increase max weight
    min_diversification=3,   # Reduce minimum positions
)
```

### Issue: Positions too small
**Check**: `min_trade_value` or position sizing percentages
```python
# Increase minimum sizes
config = PortfolioEngineConfig(
    min_trade_value=100.0,   # Increase minimum
)

# Or increase base position size
position_size_pct = 0.10     # 10% instead of 5%
```

## Next Steps

- Read full documentation: [POSITION_SIZING_LOGIC.md](../POSITION_SIZING_LOGIC.md)
- View flow diagrams: [diagrams/position_sizing_flow.md](../diagrams/position_sizing_flow.md)
- Check examples: `examples/portfolio_rebalancing.py`
- Review tests: `tests/test_engines/test_portfolio/`
