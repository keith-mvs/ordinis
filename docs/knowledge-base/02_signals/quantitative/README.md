# Quantitative Analysis - Knowledge Base

## Overview

Quantitative analysis applies mathematical models, statistical methods, and computational techniques to develop systematic trading strategies. This section focuses on **practical trading applications** of quantitative methods.

**Related Sections:**
- [10_mathematical_foundations](../10_mathematical_foundations/) - Underlying mathematical theory
- [07_risk_management](../07_risk_management/) - Position sizing and risk control
- [08_strategy_design](../08_strategy_design/) - Backtesting methodology

---

## Directory Structure

```
12_quantitative_analysis/
├── README.md                      # This file
├── statistical_arbitrage/         # Mean reversion & pairs trading
│   ├── README.md
│   ├── pairs_trading.md           # Cointegration-based pairs
│   ├── mean_reversion.md          # Statistical mean reversion
│   └── spread_trading.md          # ETF arbitrage, index arb
├── factor_investing/              # Factor-based strategies
│   ├── README.md
│   ├── fama_french.md             # Classic factor models
│   ├── momentum_factor.md         # Cross-sectional momentum
│   ├── value_factor.md            # Value investing factors
│   └── quality_factor.md          # Quality and profitability
├── ml_strategies/                 # Machine learning approaches
│   ├── README.md
│   ├── signal_classification.md   # ML for buy/sell signals
│   ├── return_prediction.md       # Cross-sectional return prediction
│   ├── regime_classification.md   # ML regime detection
│   └── feature_engineering.md     # Financial feature creation
├── execution_algorithms/          # Trade execution
│   ├── README.md
│   ├── twap_vwap.md               # Time/volume weighted execution
│   ├── optimal_execution.md       # Almgren-Chriss framework
│   └── market_impact.md           # Impact modeling
└── portfolio_construction/        # Portfolio optimization
    ├── README.md
    ├── mean_variance.md           # Markowitz optimization
    ├── risk_parity.md             # Equal risk contribution
    └── hrp.md                     # Hierarchical Risk Parity
```

---

## Strategy Categories

### 1. Statistical Arbitrage
Exploits statistical relationships between securities.

| Strategy | Edge | Risk |
|----------|------|------|
| Pairs Trading | Mean reversion of spread | Relationship breakdown |
| Index Arbitrage | ETF vs basket mispricing | Execution risk |
| Sector Arbitrage | Relative value in sector | Factor exposure |

### 2. Factor Investing
Systematic exposure to return-predictive factors.

| Factor | Academic Support | Implementation |
|--------|------------------|----------------|
| Value | Fama-French (1992) | P/B, P/E screens |
| Momentum | Jegadeesh-Titman (1993) | 12-1 month returns |
| Quality | Novy-Marx (2013) | Profitability, accruals |
| Size | Fama-French (1992) | Market cap weighting |
| Low Volatility | Ang et al. (2006) | Volatility sorting |

### 3. Machine Learning Strategies
Data-driven signal generation.

| Approach | Use Case | Pitfalls |
|----------|----------|----------|
| Classification | Signal direction | Overfitting |
| Regression | Return prediction | Non-stationarity |
| Clustering | Regime detection | Unstable clusters |
| Reinforcement | Dynamic allocation | Sample complexity |

### 4. Execution Algorithms
Minimize market impact and execution costs.

| Algorithm | Use Case | Benchmark |
|-----------|----------|-----------|
| TWAP | Uniform execution | Time-average price |
| VWAP | Follow volume curve | Volume-weighted price |
| Implementation Shortfall | Minimize slippage | Arrival price |
| Optimal Execution | Risk-adjusted | Almgren-Chriss |

### 5. Portfolio Construction
Optimal allocation across strategies/assets.

| Method | Objective | Requirement |
|--------|-----------|-------------|
| Mean-Variance | Max Sharpe | Return estimates |
| Risk Parity | Equal risk contribution | Covariance only |
| Black-Litterman | Bayesian allocation | Views + equilibrium |
| HRP | Hierarchical clustering | Correlation matrix |

---

## Key Principles

### 1. Statistical Significance
```python
# Always test for significance
def is_significant(returns, benchmark, alpha=0.05):
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(returns, benchmark)
    return p_value < alpha
```

### 2. Out-of-Sample Validation
```python
# Never trust in-sample results alone
VALIDATION_RULES = {
    'train_test_split': 0.7,
    'walk_forward': True,
    'min_out_of_sample': 252,  # 1 year minimum
    'multiple_testing_correction': 'bonferroni'
}
```

### 3. Transaction Cost Awareness
```python
# Gross alpha ≠ Net alpha
def net_sharpe(gross_sharpe, turnover, cost_per_turnover):
    cost_drag = turnover * cost_per_turnover * 252
    return gross_sharpe - cost_drag / volatility
```

### 4. Capacity Constraints
```python
# Strategies have limited capacity
def estimate_capacity(strategy_aum, market_impact_coefficient, target_impact):
    # At what AUM does impact exceed target?
    return target_impact / market_impact_coefficient
```

---

## Implementation in Ordinis

### Current Implementation
- **Regime Detection**: `src/strategies/regime_adaptive/regime_detector.py`
- **Factor Signals**: `src/engines/signalcore/` (planned)
- **Backtesting**: `src/engines/proofbench/`

### Planned Enhancements
- Statistical arbitrage scanner
- Factor exposure analysis
- ML signal generation framework
- Optimal execution engine

---

## Common Pitfalls

| Pitfall | Description | Mitigation |
|---------|-------------|------------|
| Overfitting | Fitting noise, not signal | Cross-validation, simplicity |
| Look-ahead Bias | Using future data | Point-in-time data |
| Survivorship Bias | Ignoring delisted stocks | Survivorship-free data |
| Transaction Costs | Ignoring trading costs | Realistic cost modeling |
| Regime Changes | Parameters drift | Adaptive methods |
| Data Mining | Multiple testing inflation | Deflated Sharpe |

---

## Academic References

### Foundational Papers

| Paper | Author(s) | Year | Topic |
|-------|-----------|------|-------|
| "Common Risk Factors in Returns" | Fama & French | 1993 | Factor models |
| "Returns to Buying Winners" | Jegadeesh & Titman | 1993 | Momentum |
| "Does the Stock Market Overreact?" | De Bondt & Thaler | 1985 | Mean reversion |
| "Optimal Execution of Portfolio Transactions" | Almgren & Chriss | 2001 | Execution |

### Essential Books

1. **De Prado**: "Advances in Financial Machine Learning" (2018)
2. **Chan**: "Quantitative Trading" (2008)
3. **Narang**: "Inside the Black Box" (2013)
4. **Grinold & Kahn**: "Active Portfolio Management" (1999)

---

## Quick Links

- [Pairs Trading](statistical_arbitrage/pairs_trading.md)
- [Fama-French Factors](factor_investing/fama_french.md)
- [ML Signal Classification](ml_strategies/signal_classification.md)
- [TWAP/VWAP](execution_algorithms/twap_vwap.md)
- [Risk Parity](portfolio_construction/risk_parity.md)
