# Phase 6: Advanced Portfolio Optimization

**Status**: Planning
**Start Date**: 2025-12-13
**Target Completion**: TBD
**Dependencies**: Phase 5 Complete (167 tests passing)

## Overview

Phase 6 extends the portfolio rebalancing engine with advanced optimization techniques, moving beyond simple weight allocation to mathematically optimal portfolio construction.

## Objectives

1. **Mean-Variance Optimization** - Markowitz efficient frontier
2. **Black-Litterman Model** - Combine market equilibrium with investor views
3. **Hierarchical Risk Parity** - Cluster-based allocation using machine learning
4. **Kelly Criterion** - Optimal position sizing based on edge and odds
5. **Tax-Loss Harvesting** - Minimize capital gains through strategic rebalancing
6. **Multi-Asset Class Support** - Extend beyond equities to bonds, commodities, alternatives

## Architecture

### New Modules

```
portfolio/
├── optimization/
│   ├── __init__.py
│   ├── mean_variance.py        # Markowitz optimization
│   ├── black_litterman.py      # Views-based optimization
│   ├── hierarchical_rp.py      # HRP clustering
│   └── constraints.py          # Portfolio constraints (leverage, turnover, etc.)
├── sizing/
│   ├── __init__.py
│   ├── kelly.py                # Kelly criterion
│   ├── risk_budgeting.py       # Risk-based sizing
│   └── fractional_kelly.py     # Conservative Kelly variants
├── tax/
│   ├── __init__.py
│   ├── loss_harvesting.py      # Tax-loss harvesting logic
│   ├── lot_selection.py        # FIFO, LIFO, specific lot
│   └── wash_sale.py            # Wash sale rule compliance
└── multi_asset/
    ├── __init__.py
    ├── asset_classes.py        # Asset class definitions
    ├── correlation.py          # Cross-asset correlations
    └── constraints.py          # Asset class constraints
```

## Detailed Design

### 6.1: Mean-Variance Optimization (Week 1)

**Goal**: Implement Markowitz efficient frontier optimization

**Components**:
- `MeanVarianceOptimizer` class
- Expected returns estimation (historical, CAPM, factor models)
- Covariance matrix estimation (sample, shrinkage, factor models)
- Efficient frontier calculation
- Tangency portfolio (max Sharpe ratio)
- Minimum variance portfolio
- Target return/risk portfolios

**Mathematical Foundation**:
```
minimize: w^T Σ w  (portfolio variance)
subject to: w^T μ = r_target  (target return)
            w^T 1 = 1         (fully invested)
            w_i >= 0          (long-only, optional)
```

**Features**:
- Multiple risk models (sample, shrinkage, factor-based)
- Constraints (box constraints, sector limits, turnover)
- Regularization (L1, L2 for stability)
- Robust optimization (uncertainty sets)

**Tests**: 25 tests
- Basic optimization
- Efficient frontier generation
- Constraint handling
- Edge cases (singular matrices, negative weights)

---

### 6.2: Black-Litterman Model (Week 1-2)

**Goal**: Incorporate investor views into market equilibrium

**Components**:
- `BlackLittermanOptimizer` class
- Market equilibrium (reverse optimization from cap weights)
- Views specification (absolute, relative)
- Confidence matrix (view uncertainty)
- Posterior distribution calculation
- Integration with mean-variance optimizer

**Mathematical Foundation**:
```
Posterior Returns:
E[R] = [(τΣ)^-1 + P^T Ω^-1 P]^-1 [(τΣ)^-1 π + P^T Ω^-1 Q]

Where:
π = market equilibrium (implied returns)
P = pick matrix (views structure)
Q = views vector (expected returns)
Ω = view confidence matrix
τ = uncertainty scalar
```

**Features**:
- Absolute views ("AAPL will return 15%")
- Relative views ("AAPL will outperform MSFT by 5%")
- Confidence levels per view
- Market cap weighted equilibrium
- Custom equilibrium specifications

**Tests**: 20 tests
- View specification
- Equilibrium calculation
- Posterior distribution
- Integration with optimization

---

### 6.3: Hierarchical Risk Parity (Week 2)

**Goal**: Cluster-based allocation using machine learning

**Components**:
- `HierarchicalRPOptimizer` class
- Distance metrics (correlation, covariance-based)
- Hierarchical clustering (single, complete, average linkage)
- Recursive bisection algorithm
- Quasi-diagonalization

**Algorithm**:
1. Calculate distance matrix from correlations
2. Hierarchical clustering (dendrogram)
3. Quasi-diagonalize covariance matrix
4. Recursive bisection to allocate weights

**Features**:
- Multiple distance metrics
- Configurable linkage methods
- Visualization of dendrograms
- Robustness to estimation error

**Tests**: 18 tests
- Clustering logic
- Weight allocation
- Edge cases (perfect correlation, uncorrelated)

---

### 6.4: Kelly Criterion (Week 3)

**Goal**: Optimal position sizing based on edge and odds

**Components**:
- `KellyOptimizer` class
- Full Kelly calculation
- Fractional Kelly (Kelly fraction)
- Simultaneous Kelly (multi-asset)
- Continuous Kelly (for log-normal returns)

**Mathematical Foundation**:
```
Full Kelly:
f* = (p(b+1) - 1) / b

Where:
f* = optimal fraction
p = win probability
b = odds (payoff ratio)

Continuous Kelly:
f* = μ / σ^2  (for log-normal)
```

**Features**:
- Edge/odds input
- Return distribution input
- Fractional Kelly (25%, 50%, 75%)
- Drawdown-aware Kelly
- Multi-asset simultaneous Kelly

**Tests**: 15 tests
- Single asset Kelly
- Fractional Kelly
- Multi-asset Kelly
- Edge cases (negative edge, zero variance)

---

### 6.5: Tax-Loss Harvesting (Week 3-4)

**Goal**: Minimize capital gains through strategic rebalancing

**Components**:
- `TaxLossHarvester` class
- Lot tracking (FIFO, LIFO, specific lot)
- Unrealized gains/losses calculation
- Harvest opportunity identification
- Wash sale rule compliance (30-day window)
- Replacement security selection

**Features**:
- Multiple lot selection methods
- Tax bracket awareness
- Short-term vs long-term gains
- Wash sale prevention
- Correlation-based replacements

**Tests**: 22 tests
- Lot tracking
- Harvest identification
- Wash sale detection
- Replacement selection

---

### 6.6: Multi-Asset Class Support (Week 4)

**Goal**: Extend beyond equities to bonds, commodities, alternatives

**Components**:
- `AssetClass` enum (EQUITY, FIXED_INCOME, COMMODITY, REAL_ESTATE, ALTERNATIVE)
- Cross-asset correlation modeling
- Asset class constraints
- Rebalancing frequency per asset class
- Liquidity constraints

**Features**:
- Asset class hierarchies
- Cross-asset optimization
- Class-specific constraints
- Liquidity tiers
- Rebalancing schedules

**Tests**: 12 tests
- Asset class definitions
- Cross-asset optimization
- Constraint enforcement

---

## Implementation Timeline

### Week 1 (Days 1-7)
- Design and implement Mean-Variance Optimization
- Design and start Black-Litterman
- **Deliverables**:
  - `mean_variance.py` (complete)
  - `black_litterman.py` (in progress)
  - 25 MVO tests passing

### Week 2 (Days 8-14)
- Complete Black-Litterman
- Implement Hierarchical Risk Parity
- **Deliverables**:
  - `black_litterman.py` (complete)
  - `hierarchical_rp.py` (complete)
  - 45 tests passing (25 MVO + 20 BL)

### Week 3 (Days 15-21)
- Implement Kelly Criterion
- Start Tax-Loss Harvesting
- **Deliverables**:
  - `kelly.py` (complete)
  - `loss_harvesting.py` (in progress)
  - 63 tests passing

### Week 4 (Days 22-28)
- Complete Tax-Loss Harvesting
- Implement Multi-Asset Class Support
- Integration and documentation
- **Deliverables**:
  - All Phase 6 modules complete
  - 112 Phase 6 tests passing
  - Comprehensive documentation

## Testing Strategy

### Unit Tests
- Each optimizer independently tested
- Edge cases and boundary conditions
- Numerical stability tests
- Performance benchmarks

### Integration Tests
- Integration with Phase 5 rebalancing engine
- Event hooks for optimization events
- End-to-end optimization workflows
- Backtesting integration

### Performance Tests
- Large portfolio optimization (100+ assets)
- Covariance matrix computation
- Optimization solver performance
- Memory profiling

## Dependencies

### Python Libraries
```python
# Optimization
scipy>=1.10.0       # Optimization solvers
cvxpy>=1.3.0        # Convex optimization
quadprog>=0.1.11    # Quadratic programming

# Linear algebra
numpy>=1.24.0
scipy.linalg        # Matrix operations

# Machine learning
scikit-learn>=1.2.0  # Hierarchical clustering
scipy.cluster        # Dendrogram utilities

# Visualization (optional)
matplotlib>=3.7.0
seaborn>=0.12.0
```

## Integration Points

### With Phase 5 Rebalancing Engine

```python
from ordinis.engines.portfolio import RebalancingEngine, StrategyType
from ordinis.engines.portfolio.optimization import MeanVarianceOptimizer

# Create optimizer
mvo = MeanVarianceOptimizer(
    expected_returns=returns_forecast,
    covariance=cov_matrix,
    risk_aversion=1.0,
)

# Register as strategy
engine = RebalancingEngine()
engine.register_strategy(StrategyType.MEAN_VARIANCE, mvo)

# Generate optimal weights
decisions = engine.generate_rebalancing_decisions(
    positions, prices, strategy_type=StrategyType.MEAN_VARIANCE
)
```

### With ProofBench (Backtesting)

```python
from ordinis.engines.proofbench import SimulationEngine
from ordinis.engines.portfolio.optimization import BlackLittermanOptimizer

# In strategy on_bar() method
def on_bar(self, engine, symbol, bar):
    # Update views periodically
    if should_rebalance():
        bl = BlackLittermanOptimizer(market_caps, views, confidences)
        weights = bl.optimize()

        # Rebalance to optimal weights
        for sym, target_weight in weights.items():
            current_weight = get_current_weight(sym)
            trade_shares = calculate_rebalance(sym, current_weight, target_weight)
            engine.submit_order(Order(sym, trade_shares))
```

### With Tax Module

```python
from ordinis.engines.portfolio.tax import TaxLossHarvester

harvester = TaxLossHarvester(
    tax_rate_st=0.37,  # Short-term cap gains
    tax_rate_lt=0.20,  # Long-term cap gains
    lot_method="HIFO", # Highest in, first out
)

# Check for harvest opportunities
opportunities = harvester.identify_opportunities(
    positions, prices, cost_basis_lots
)

# Execute harvests while avoiding wash sales
decisions = harvester.generate_harvest_orders(
    opportunities, replacement_candidates
)
```

## Success Metrics

### Code Metrics
- **112 new tests** (100% passing)
- **>95% code coverage** on optimization modules
- **<100ms** for 50-asset optimization
- **<500ms** for 200-asset optimization

### Functionality Metrics
- Efficient frontier generation (<1s for 100 assets)
- Black-Litterman convergence in <10 iterations
- HRP clustering stable across random seeds
- Kelly sizing within 1% of analytical solution
- Tax harvesting identifies >90% of opportunities

### Quality Metrics
- All mathematical formulas validated against academic papers
- Numerical stability tested (condition numbers, ill-conditioned matrices)
- Integration tests with Phase 5 engine
- Documentation with mathematical derivations

## Risk Mitigation

### Numerical Stability
**Risk**: Ill-conditioned covariance matrices cause optimization failures
**Mitigation**:
- Shrinkage estimators
- Regularization (ridge penalty)
- Condition number checks
- Fallback to equal-weight

### Optimization Performance
**Risk**: Large portfolios (200+ assets) take too long
**Mitigation**:
- Sparse matrix representations
- Warm starts for sequential optimizations
- Parallel processing for monte carlo
- Caching of covariance matrices

### Tax Complexity
**Risk**: Tax rules vary by jurisdiction and change over time
**Mitigation**:
- Configurable tax parameters
- Plugin architecture for different tax regimes
- Warning messages about jurisdiction
- Annual tax rule reviews

## Documentation

### API Documentation
- Docstrings for all public methods
- Mathematical formulations in docstrings
- Usage examples for each optimizer
- Integration examples

### Conceptual Documentation
- White paper on portfolio optimization theory
- Comparison of optimization methods
- When to use each technique
- Limitations and assumptions

### User Guide
- Getting started with optimization
- Configuring optimizers
- Interpreting results
- Troubleshooting common issues

## Future Enhancements (Phase 7+)

1. **Robust Optimization** - Uncertainty sets, worst-case scenarios
2. **Multi-Period Optimization** - Dynamic programming, stochastic control
3. **Transaction Cost Models** - Price impact, market impact
4. **Factor Models** - Fama-French, Carhart, custom factors
5. **Alternative Risk Measures** - CVaR, drawdown, semi-variance
6. **Machine Learning Integration** - RL for dynamic allocation
7. **ESG Constraints** - Carbon footprint, ESG scores
8. **Currency Hedging** - Multi-currency portfolios

## References

### Academic Papers
1. Markowitz, H. (1952). "Portfolio Selection"
2. Black, F., & Litterman, R. (1992). "Global Portfolio Optimization"
3. Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform Out of Sample"
4. Kelly, J. (1956). "A New Interpretation of Information Rate"
5. Arnott, R., et al. (2001). "The Surprising Alpha from Malkiel's Monkey and Upside-Down Strategies"

### Books
- "Portfolio Optimization and Performance Analysis" - Guerard & Schwartz
- "Quantitative Equity Portfolio Management" - Chincarini & Kim
- "Advanced Portfolio Management" - Paleologo
- "Tax-Aware Investment Management" - Horan & Robinson

---

**Next Steps**:
1. Review and approve Phase 6 plan
2. Set up development environment (scipy, cvxpy)
3. Begin Week 1 implementation (Mean-Variance Optimization)
4. Create Phase 6 test suite structure
