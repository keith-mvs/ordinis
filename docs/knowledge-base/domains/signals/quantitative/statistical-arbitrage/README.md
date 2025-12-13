# Statistical Arbitrage

## Overview

Statistical arbitrage exploits temporary mispricings identified through statistical relationships between securities. Unlike pure arbitrage, stat arb strategies carry risk as relationships may not revert.

**Mathematical Foundation**: See [10_mathematical_foundations - Cointegration](../../10_mathematical_foundations/README.md#34-cointegration)

---

## Strategy Types

| Strategy | Relationship | Holding Period |
|----------|--------------|----------------|
| [Pairs Trading](pairs_trading.md) | Two cointegrated stocks | Days to weeks |
| [Mean Reversion](mean_reversion.md) | Asset to its mean | Hours to days |
| [Spread Trading](spread_trading.md) | ETF vs components | Intraday to days |

---

## Core Concept: Cointegration

Two series X and Y are cointegrated if:
1. Both are non-stationary (I(1))
2. A linear combination is stationary (I(0))

```python
# Test for cointegration
from statsmodels.tsa.stattools import coint

def test_cointegration(series_x, series_y, significance=0.05):
    stat, p_value, crit_values = coint(series_x, series_y)
    return {
        'cointegrated': p_value < significance,
        'p_value': p_value,
        'test_statistic': stat
    }
```

---

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Half-life | Time to 50% mean reversion | < 20 days |
| Hurst Exponent | Mean-reverting tendency | < 0.5 |
| Spread Stationarity | ADF test p-value | < 0.05 |
| Hedge Ratio Stability | Rolling beta variance | Low |

---

## Risk Factors

1. **Relationship Breakdown**: Fundamentals change
2. **Execution Risk**: Can't get fills at expected prices
3. **Funding Risk**: Margin calls during divergence
4. **Model Risk**: Wrong hedge ratio estimation
5. **Convergence Time**: May take longer than expected

---

## Best Practices

1. **Test for cointegration** before trading any pair
2. **Use rolling hedge ratios** to adapt to changing relationships
3. **Set max divergence limits** to cut losses on broken relationships
4. **Diversify across pairs** to reduce single-pair risk
5. **Monitor half-life** - faster reversion = better
