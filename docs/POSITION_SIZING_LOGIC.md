# Position Sizing Logic in Ordinis

This document explains how position sizing works within the Portfolio Engine and Portfolio Optimization (PortfolioOpt) components of the Ordinis trading system.

## Overview

Position sizing in Ordinis is handled through two primary components:

1. **Portfolio Engine** (`engines/portfolio`) - Manages rebalancing strategies using weight-based allocation
2. **PortfolioOpt Engine** (`engines/portfolioopt`) - GPU-accelerated Mean-CVaR optimization with constraints

## Portfolio Engine Position Sizing

### Architecture

The Portfolio Engine supports multiple rebalancing strategies, each with its own approach to position sizing:

- **Target Allocation** - Fixed percentage weights per symbol
- **Risk Parity** - Inverse volatility weighting for equal risk contribution
- **Signal-Driven** - Signal-based dynamic weighting
- **Threshold-Based** - Drift-triggered rebalancing

### Core Position Sizing Logic

#### 1. Weight-Based Allocation

All strategies convert position sizes to portfolio weights (0.0 to 1.0):

```python
# Calculate current weights
total_value = sum(positions[sym] * prices[sym] for sym in symbols)
current_weight = (positions[sym] * prices[sym]) / total_value
```

#### 2. Target Allocation Strategy

**File**: `src/ordinis/engines/portfolio/target_allocation.py`

Maintains fixed percentage allocations:

```python
# Configuration
targets = {
    "AAPL": 0.40,  # 40% of portfolio
    "MSFT": 0.30,  # 30% of portfolio
    "GOOGL": 0.30  # 30% of portfolio
}

# Calculate drift
drift = current_weight - target_weight

# Generate rebalancing order
target_value = total_value * target_weight
current_value = positions[sym] * prices[sym]
adjustment_value = target_value - current_value
adjustment_shares = adjustment_value / prices[sym]
```

**Key Parameters**:
- `target_weight`: Desired allocation (0.0 to 1.0)
- `drift_threshold`: Max acceptable drift before rebalancing (default: 0.05 = 5%)

#### 3. Risk Parity Strategy

**File**: `src/ordinis/engines/portfolio/risk_parity.py`

Allocates based on inverse volatility for equal risk contribution:

```python
# Calculate annualized volatility
volatilities = returns.std() * np.sqrt(252)

# Inverse volatility weighting
inverse_vols = 1.0 / volatilities
raw_weights = inverse_vols / inverse_vols.sum()

# Apply min/max constraints
weights = raw_weights.clip(lower=min_weight, upper=max_weight)
# Renormalize
weights = weights / weights.sum()
```

**Key Parameters**:
- `lookback_days`: Historical period for volatility calculation (default: 252)
- `min_weight`: Minimum weight per asset (default: 0.01 = 1%)
- `max_weight`: Maximum weight per asset (default: 0.50 = 50%)
- `drift_threshold`: Rebalancing trigger (default: 0.05 = 5%)

**Formula**: Each asset contributes equally to portfolio risk by weighting inversely to its volatility.

#### 4. Signal-Driven Strategy

**File**: `src/ordinis/engines/portfolio/signal_driven.py`

Converts trading signals into portfolio weights:

```python
# Signal methods available:
# 1. PROPORTIONAL: Weight proportional to signal strength
if method == "PROPORTIONAL":
    positive_signals = [s for s in signals if s.signal > threshold]
    total_signal = sum(abs(s.signal * s.confidence) for s in positive_signals)
    weights = {
        s.symbol: (abs(s.signal * s.confidence) / total_signal) * (1 - cash_buffer)
        for s in positive_signals
    }

# 2. BINARY: Full weight to positive signals, zero to negative
elif method == "BINARY":
    positive_signals = [s for s in signals if s.signal > 0]
    equal_weight = (1 - cash_buffer) / len(positive_signals)
    weights = {s.symbol: equal_weight for s in positive_signals}

# 3. RANKED: Weight based on signal ranking
elif method == "RANKED":
    sorted_signals = sorted(signals, key=lambda s: s.signal * s.confidence, reverse=True)
    # Top-ranked get higher weights
```

**Key Parameters**:
- `method`: Signal conversion method (PROPORTIONAL, BINARY, RANKED)
- `min_weight`: Minimum weight per asset (default: 0.0)
- `max_weight`: Maximum weight per asset (default: 0.50 = 50%)
- `min_signal_threshold`: Minimum signal value to consider (default: 0.0)
- `cash_buffer`: Percentage to keep as cash (default: 0.0)

### Configuration Parameters

#### Portfolio Engine Config

**File**: `src/ordinis/engines/portfolio/core/config.py`

```python
@dataclass
class PortfolioEngineConfig:
    initial_capital: float = 100000.0      # Starting capital
    min_trade_value: float = 10.0          # Minimum dollar value for execution
    max_position_pct: float = 0.25         # Maximum single position (25%)
```

### Position Sizing in Execution

**File**: `src/ordinis/orchestration/pipeline.py`

When executing signals through the orchestration pipeline:

```python
# Base position sizing
base_position_size = portfolio.equity * position_size_pct  # Default: 5%

# Adjust based on signal confidence
confidence_adjustment = signal.probability * abs(signal.score)
position_value = base_position_size * confidence_adjustment

# Cap at maximum position size
max_position_value = portfolio.equity * max_position_size_pct  # Default: 15%
position_value = min(position_value, max_position_value)

# Convert to quantity
quantity = int(position_value / current_price)
```

**Key Parameters**:
- `position_size_pct`: Default percentage of equity per position (default: 0.05 = 5%)
- `max_position_size_pct`: Maximum percentage per symbol (default: 0.15 = 15%)
- `max_portfolio_exposure_pct`: Maximum total exposure (default: 1.0 = 100%)

## PortfolioOpt Engine Position Sizing

### Architecture

The PortfolioOpt Engine uses GPU-accelerated Mean-CVaR optimization via NVIDIA's Quantitative Portfolio Optimization (QPO) blueprint.

**File**: `src/ordinis/engines/portfolioopt/core/engine.py`

### Optimization-Based Position Sizing

#### Mean-CVaR Optimization

The engine solves:

```
minimize: (1 - λ) * (-μᵀw) + λ * CVaR(w)

subject to:
  - sum(w) = 1                    # Fully invested
  - 0 ≤ w_i ≤ max_weight          # Per-asset weight cap
  - μᵀw ≥ target_return           # Minimum return target
```

Where:
- `w`: Portfolio weights vector
- `μ`: Expected returns vector
- `λ`: Risk aversion parameter (higher = more risk-averse)
- `CVaR`: Conditional Value at Risk

#### Configuration Parameters

**File**: `src/ordinis/engines/portfolioopt/core/config.py`

```python
@dataclass
class PortfolioOptEngineConfig:
    # Optimization targets
    target_return: float = 0.001           # Daily target return (0.1%)
    max_weight: float = 0.20               # Per-asset weight cap (20%)
    risk_aversion: float = 0.5             # CVaR penalty weight

    # Constraints
    min_weight: float = 0.0                # Minimum weight (allows zero)
    max_concentration: float = 0.25        # Max single-asset concentration (25%)
    min_diversification: int = 5           # Minimum number of non-zero positions

    # Risk limits
    max_cvar: float = 0.10                 # Maximum acceptable CVaR (10%)
    max_volatility: float = 0.25           # Maximum portfolio volatility (25%)
```

#### Optimization Flow

```python
async def optimize(
    self,
    returns: pd.DataFrame,
    target_return: float = 0.001,
    max_weight: float = 0.20,
    risk_aversion: float = 0.5
) -> OptimizationResult:
    """
    Run Mean-CVaR portfolio optimization.

    Returns:
        OptimizationResult with:
          - weights: dict[str, float]  # Optimized position sizes
          - expected_return: float
          - cvar: float
          - constraints_satisfied: bool
    """
```

### Constraint Validation

After optimization, the engine validates:

```python
# 1. Concentration limit
max_actual_weight = max(weights.values())
if max_actual_weight > max_concentration:
    warnings.append("Concentration limit exceeded")

# 2. Diversification requirement
non_zero_count = sum(1 for w in weights.values() if w > 0.001)
if non_zero_count < min_diversification:
    warnings.append("Diversification below minimum")

# 3. CVaR limit
if cvar > max_cvar:
    warnings.append("CVaR exceeds limit")
```

### Solver Options

The engine supports two solvers:

1. **CVXPY (CPU)**: Fallback solver using SciPy
2. **cuOpt (GPU)**: GPU-accelerated solver for large portfolios

```python
# Configuration
config = PortfolioOptEngineConfig(
    default_api="cvxpy",  # or "cuopt" for GPU
)
```

## Integration: Portfolio Engine + PortfolioOpt

### Hybrid Approach

You can use PortfolioOpt to generate optimal weights, then feed them to Portfolio Engine for execution:

```python
# Step 1: Generate optimal weights using PortfolioOpt
opt_engine = PortfolioOptEngine(config)
result = await opt_engine.optimize(returns_df)
optimal_weights = result.weights  # {"AAPL": 0.35, "MSFT": 0.40, "GOOGL": 0.25}

# Step 2: Create target allocations for Portfolio Engine
target_allocations = [
    TargetAllocation(symbol, weight)
    for symbol, weight in optimal_weights.items()
]

# Step 3: Execute rebalancing
portfolio_engine = PortfolioEngine()
strategy = TargetAllocationRebalancer(target_allocations)
portfolio_engine.register_strategy(StrategyType.TARGET_ALLOCATION, strategy)

decisions = await portfolio_engine.generate_rebalancing_decisions(
    positions=current_positions,
    prices=current_prices
)
```

## Regime-Adaptive Position Sizing

**File**: `src/ordinis/application/strategies/regime_adaptive/adaptive_manager.py`

Position sizing is adapted based on market regime:

```python
def _apply_position_sizing(self, signal: TradingSignal, regime: RegimeSignal) -> TradingSignal:
    # Start with base size
    size = base_position_size  # e.g., 0.8 (80% of capital)

    # Scale by regime confidence
    if confidence_scaling:
        size *= regime.confidence

    # Adjust for volatility
    if volatility_factor:
        vol_factor = 1.0 / (1.0 + current_volatility)
        size *= vol_factor

    # Ensure minimum
    size = max(size, min_position_size)  # e.g., 0.3 (30% minimum)

    # Apply signal's position size recommendation
    size *= signal.position_size

    return TradingSignal(..., position_size=size)
```

**Regime-Specific Multipliers** (from `config/optimizer.py`):

```python
"regime_specific_params": {
    "trending": {
        "position_size_multiplier": 1.2,    # Increase size in trends
    },
    "ranging": {
        "position_size_multiplier": 0.8,    # Reduce size in ranges
    },
    "volatile": {
        "position_size_multiplier": 0.5,    # Reduce size in volatility
    },
}
```

## Risk Management Integration

### RiskGuard Integration

**File**: `src/ordinis/engines/riskguard/rules/standard.py`

Position sizing is constrained by risk rules:

```python
STANDARD_RULES = {
    "max_position_pct": RiskRule(
        rule_id="RT001",
        threshold=0.10,  # 10% max per position
        action_on_breach="resize",
    ),
    "min_position_size": RiskRule(
        rule_id="RT003",
        threshold=1000.0,  # $1,000 minimum
        action_on_breach="reject",
    ),
}

# Risk profiles
CONSERVATIVE_PROFILE = {
    "max_position_pct": 0.05,      # 5% max
    "min_position_size": 2000.0,   # $2,000 min
}

AGGRESSIVE_PROFILE = {
    "max_position_pct": 0.20,      # 20% max
    "min_position_size": 500.0,    # $500 min
}
```

### Governance Integration

**File**: `src/ordinis/engines/governance/core/config.py`

Governance enforces additional limits:

```python
@dataclass
class GovernanceEngineConfig:
    max_position_size_pct: float = 0.10    # 10% default
    daily_trade_limit: int = 100
    max_drawdown_pct: float = -0.10
```

## Configuration Summary

### Recommended Settings by Strategy Type

#### Conservative Portfolio

```yaml
portfolio_engine:
  max_position_pct: 0.05           # 5% max per position
  initial_capital: 100000

portfolio_opt:
  max_weight: 0.10                 # 10% max weight
  max_concentration: 0.15          # 15% max single asset
  min_diversification: 10          # At least 10 positions
  risk_aversion: 0.8               # High risk aversion
  max_cvar: 0.05                   # 5% CVaR limit
```

#### Balanced Portfolio

```yaml
portfolio_engine:
  max_position_pct: 0.10           # 10% max per position
  initial_capital: 100000

portfolio_opt:
  max_weight: 0.20                 # 20% max weight
  max_concentration: 0.25          # 25% max single asset
  min_diversification: 5           # At least 5 positions
  risk_aversion: 0.5               # Moderate risk aversion
  max_cvar: 0.10                   # 10% CVaR limit
```

#### Aggressive Portfolio

```yaml
portfolio_engine:
  max_position_pct: 0.20           # 20% max per position
  initial_capital: 100000

portfolio_opt:
  max_weight: 0.30                 # 30% max weight
  max_concentration: 0.40          # 40% max single asset
  min_diversification: 3           # At least 3 positions
  risk_aversion: 0.2               # Low risk aversion
  max_cvar: 0.15                   # 15% CVaR limit
```

## Best Practices

### 1. Position Sizing for Backtesting

**File**: `src/ordinis/backtesting/runner.py`

```python
@dataclass
class BacktestConfig:
    max_position_size: float = 0.1         # 10% of equity
    max_portfolio_exposure: float = 1.0     # 100% deployed
    stop_loss_pct: float = 0.08            # 8% stop loss
```

### 2. Signal-Based Position Sizing

```python
# Confidence-adjusted sizing
base_size = portfolio.equity * 0.05
confidence_adjustment = signal.probability * abs(signal.score)
position_value = base_size * confidence_adjustment

# Cap at limits
max_value = portfolio.equity * 0.15
position_value = min(position_value, max_value)
```

### 3. Volatility-Adjusted Sizing

```python
# Risk parity approach
vol_target = 0.15  # 15% annualized volatility target
asset_vol = calculate_volatility(returns)
vol_scalar = vol_target / asset_vol
position_weight = base_weight * vol_scalar
```

## Key Takeaways

1. **Portfolio Engine** focuses on **weight-based allocation** with multiple rebalancing strategies
   - Uses portfolio weights (0.0 to 1.0) as the core abstraction
   - Supports fixed allocations, risk parity, signal-driven, and threshold-based approaches

2. **PortfolioOpt Engine** uses **optimization-based position sizing**
   - Solves Mean-CVaR optimization problem for optimal weights
   - Enforces constraints on concentration, diversification, and risk
   - GPU-accelerated for large portfolios

3. **Integration** is seamless: use PortfolioOpt for weight generation, Portfolio Engine for execution

4. **Risk Management** is enforced through multiple layers:
   - Engine configuration (max_weight, max_concentration)
   - RiskGuard rules (position limits, risk checks)
   - Governance policies (trading limits, approval thresholds)

5. **Regime Adaptation** dynamically adjusts position sizing based on market conditions

## References

- Portfolio Engine: `src/ordinis/engines/portfolio/`
- PortfolioOpt Engine: `src/ordinis/engines/portfolioopt/`
- Risk Management: `src/ordinis/engines/riskguard/`
- Orchestration: `src/ordinis/orchestration/pipeline.py`
- Configuration: `src/ordinis/config/optimizer.py`
