### 1.2 Moments and Tail Risk

**Standard Moments**:

```python
# Moment calculations
def moments(returns: np.array) -> dict:
    return {
        'mean': np.mean(returns),                    # 1st moment: E[X]
        'variance': np.var(returns),                 # 2nd central moment
        'skewness': scipy.stats.skew(returns),       # 3rd standardized moment
        'kurtosis': scipy.stats.kurtosis(returns)    # 4th standardized moment (excess)
    }

# Interpretation for trading
SKEWNESS_INTERPRETATION = {
    'negative': 'Left tail risk (crash risk)',
    'zero': 'Symmetric distribution',
    'positive': 'Right tail (windfall potential)'
}

KURTOSIS_INTERPRETATION = {
    'leptokurtic': 'kurtosis > 0: Fat tails, more extreme events',
    'mesokurtic': 'kurtosis ≈ 0: Normal-like tails',
    'platykurtic': 'kurtosis < 0: Thin tails'
}
```

**Tail Risk Measures**:

```python
def value_at_risk(returns: np.array, confidence: float = 0.95) -> float:
    """
    VaR: Maximum loss at given confidence level.
    VaR_α = -inf{x : P(X ≤ x) ≥ 1 - α}
    """
    return -np.percentile(returns, (1 - confidence) * 100)

def expected_shortfall(returns: np.array, confidence: float = 0.95) -> float:
    """
    ES (CVaR): Expected loss beyond VaR.
    ES_α = E[X | X ≤ -VaR_α]
    """
    var = value_at_risk(returns, confidence)
    return -returns[returns <= -var].mean()

def tail_index_hill(returns: np.array, k: int) -> float:
    """
    Hill estimator for tail index (power law exponent).
    Used for extreme value analysis.
    """
    sorted_returns = np.sort(np.abs(returns))[::-1]
    return k / np.sum(np.log(sorted_returns[:k] / sorted_returns[k]))
```

---
