# Factor Investing

## Overview

Factor investing systematically tilts portfolios toward characteristics (factors) that have historically explained differences in returns across securities. Unlike stock picking, factor investing captures broad, persistent return premiums.

---

## Academic Foundation

### Fama-French Three-Factor Model (1993)
```
R_i - R_f = α + β_mkt(R_mkt - R_f) + β_smb(SMB) + β_hml(HML) + ε

Where:
- R_i = Return of asset i
- R_f = Risk-free rate
- R_mkt = Market return
- SMB = Small Minus Big (size factor)
- HML = High Minus Low (value factor)
```

### Five-Factor Model (2015)
Adds:
- **RMW**: Robust Minus Weak (profitability)
- **CMA**: Conservative Minus Aggressive (investment)

### Carhart Four-Factor Model (1997)
Adds:
- **MOM**: Momentum factor (12-1 month returns)

---

## Factor Types

| File | Factor | Description |
|------|--------|-------------|
| [fama_french.md](fama_french.md) | Market, Size, Value | Classic FF factors |
| [momentum_factor.md](momentum_factor.md) | Momentum | Cross-sectional momentum |
| [value_factor.md](value_factor.md) | Value | Book-to-market, earnings |
| [quality_factor.md](quality_factor.md) | Quality | Profitability, stability |

---

## Core Factors

### 1. Market (MKT)
- **Premium**: ~6-8% annually (equity risk premium)
- **Source**: Compensation for systematic risk
- **Implementation**: Broad market exposure (SPY, VTI)

### 2. Size (SMB)
- **Premium**: ~2-3% historically (declining)
- **Source**: Small caps are riskier, less liquid
- **Implementation**: Long small cap, short large cap

### 3. Value (HML)
- **Premium**: ~3-5% historically
- **Source**: Distress risk, behavioral mispricing
- **Implementation**: Long high B/M, short low B/M

### 4. Momentum (MOM)
- **Premium**: ~6-8% historically
- **Source**: Underreaction, herding behavior
- **Implementation**: Long past winners, short past losers

### 5. Quality (QMJ)
- **Premium**: ~3-4% historically
- **Source**: Market underprices quality
- **Implementation**: Long profitable, stable companies

### 6. Low Volatility
- **Premium**: ~2-3% historically
- **Source**: Leverage aversion, lottery preference
- **Implementation**: Long low vol, short high vol

---

## Factor Construction

### Generic Factor Portfolio
```python
def construct_factor_portfolio(
    data: pd.DataFrame,
    signal_column: str,
    n_quantiles: int = 5,
    long_quantile: int = 5,
    short_quantile: int = 1
) -> pd.DataFrame:
    """
    Construct long-short factor portfolio.
    """
    # Rank stocks by signal
    data['quantile'] = pd.qcut(
        data[signal_column],
        q=n_quantiles,
        labels=range(1, n_quantiles + 1)
    )

    # Long top quantile, short bottom quantile
    long_stocks = data[data['quantile'] == long_quantile]
    short_stocks = data[data['quantile'] == short_quantile]

    # Equal-weight within each leg
    long_weight = 1 / len(long_stocks)
    short_weight = -1 / len(short_stocks)

    portfolio = pd.concat([
        long_stocks.assign(weight=long_weight),
        short_stocks.assign(weight=short_weight)
    ])

    return portfolio
```

### Factor Return Calculation
```python
def calculate_factor_return(
    returns: pd.DataFrame,
    factor_weights: pd.DataFrame
) -> pd.Series:
    """
    Calculate factor return series.
    """
    # Align dates
    common_dates = returns.index.intersection(factor_weights.index)

    factor_returns = []
    for date in common_dates:
        weights = factor_weights.loc[date]
        ret = returns.loc[date]

        # Handle missing stocks
        common_stocks = weights.index.intersection(ret.index)
        weights = weights[common_stocks]
        ret = ret[common_stocks]

        # Normalize weights
        weights = weights / weights.abs().sum()

        factor_ret = (weights * ret).sum()
        factor_returns.append(factor_ret)

    return pd.Series(factor_returns, index=common_dates)
```

---

## Factor Timing

### Regime-Based Factor Allocation
```python
def factor_allocation_by_regime(regime: str) -> dict:
    """
    Adjust factor exposures based on market regime.
    """
    allocations = {
        'BULL': {
            'momentum': 0.30,
            'value': 0.20,
            'quality': 0.20,
            'size': 0.20,
            'low_vol': 0.10
        },
        'BEAR': {
            'momentum': 0.10,  # Momentum crashes in reversals
            'value': 0.20,
            'quality': 0.35,  # Quality outperforms in downturns
            'size': 0.05,     # Avoid small caps
            'low_vol': 0.30   # Defensive
        },
        'VOLATILE': {
            'momentum': 0.05,
            'value': 0.20,
            'quality': 0.30,
            'size': 0.05,
            'low_vol': 0.40   # Maximum defensive
        },
        'RECOVERY': {
            'momentum': 0.25,
            'value': 0.30,  # Value leads recoveries
            'quality': 0.15,
            'size': 0.20,   # Small caps bounce hard
            'low_vol': 0.10
        }
    }
    return allocations.get(regime, allocations['BULL'])
```

### Factor Momentum
```python
def factor_momentum_signal(factor_returns: pd.DataFrame, lookback: int = 12) -> pd.Series:
    """
    Overweight factors with recent positive performance.
    """
    cumulative_returns = (1 + factor_returns).rolling(lookback).apply(
        lambda x: x.prod() - 1
    )

    # Rank factors by recent performance
    ranks = cumulative_returns.rank(axis=1, ascending=False)

    return ranks
```

---

## Multi-Factor Portfolios

### Equal-Weighted Multi-Factor
```python
def multi_factor_score(
    data: pd.DataFrame,
    factors: list,
    weights: dict = None
) -> pd.Series:
    """
    Combine multiple factor scores.
    """
    if weights is None:
        weights = {f: 1/len(factors) for f in factors}

    # Z-score each factor
    z_scores = pd.DataFrame()
    for factor in factors:
        z_scores[factor] = (data[factor] - data[factor].mean()) / data[factor].std()

    # Weighted combination
    combined = sum(z_scores[f] * weights[f] for f in factors)

    return combined
```

### Factor Diversification
```python
def factor_correlation_matrix(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Correlation matrix for factor timing and diversification.
    """
    return factor_returns.corr()

# Target low correlation between factors for diversification
# Momentum and Value are typically negatively correlated
```

---

## Risk Decomposition

```python
def factor_exposure_analysis(
    returns: pd.Series,
    factor_returns: pd.DataFrame
) -> dict:
    """
    Decompose returns into factor exposures.
    """
    from sklearn.linear_model import LinearRegression

    # Align data
    aligned = pd.concat([returns, factor_returns], axis=1).dropna()
    y = aligned.iloc[:, 0]
    X = aligned.iloc[:, 1:]

    model = LinearRegression()
    model.fit(X, y)

    # Factor betas (exposures)
    betas = dict(zip(factor_returns.columns, model.coef_))

    # Alpha (unexplained return)
    alpha = model.intercept_ * 252  # Annualized

    # R-squared (explained variance)
    r_squared = model.score(X, y)

    return {
        'betas': betas,
        'alpha': alpha,
        'r_squared': r_squared,
        'residual_risk': np.std(y - model.predict(X)) * np.sqrt(252)
    }
```

---

## Factor Data Sources

| Source | Factors Available | Cost |
|--------|-------------------|------|
| Kenneth French Data Library | FF 3/5 factors, momentum | Free |
| AQR Data Library | Quality, BAB, momentum | Free |
| MSCI Barra | Full factor suite | $$$ |
| Bloomberg | Various | $$$ |

---

## Implementation Considerations

### Transaction Costs
```python
def factor_turnover(weights_t: pd.Series, weights_t1: pd.Series) -> float:
    """
    Calculate portfolio turnover from rebalancing.
    """
    return (weights_t - weights_t1).abs().sum() / 2

# High turnover factors (momentum) have higher costs
# Low turnover factors (value) are cheaper to implement
```

### Rebalancing Frequency
| Factor | Recommended Frequency | Rationale |
|--------|----------------------|-----------|
| Momentum | Monthly | Signal decays quickly |
| Value | Quarterly | Fundamentals change slowly |
| Quality | Quarterly | Earnings-based |
| Low Vol | Monthly | Volatility changes |

---

## Academic References

- Fama & French (1992, 1993): Size and value factors
- Jegadeesh & Titman (1993): Momentum
- Novy-Marx (2013): Profitability factor
- Frazzini & Pedersen (2014): Betting against beta
- Asness et al. (2019): Quality minus junk
