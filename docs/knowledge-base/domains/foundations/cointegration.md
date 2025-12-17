### 3.4 Cointegration

**Engle-Granger Two-Step Procedure**:

```python
from statsmodels.tsa.stattools import coint

def test_cointegration(y1: np.array, y2: np.array) -> dict:
    """
    Test for cointegration between two time series.

    Two series are cointegrated if:
    1. Both are I(1) (integrated of order 1)
    2. A linear combination is I(0) (stationary)
    """
    # Engle-Granger test
    coint_stat, p_value, crit_values = coint(y1, y2)

    # Estimate hedge ratio via OLS
    hedge_ratio = np.polyfit(y2, y1, 1)[0]

    # Spread
    spread = y1 - hedge_ratio * y2

    # Test spread for stationarity
    spread_test = stationarity_tests(spread)

    return {
        'coint_statistic': coint_stat,
        'p_value': p_value,
        'critical_values': crit_values,
        'hedge_ratio': hedge_ratio,
        'spread': spread,
        'spread_stationary': spread_test['adf']['is_stationary'],
        'is_cointegrated': p_value < 0.05
    }

def johansen_cointegration(data: np.array, det_order: int = 0) -> dict:
    """
    Johansen cointegration test for multiple time series.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    result = coint_johansen(data, det_order, 1)

    return {
        'eigenvalues': result.eig,
        'trace_statistic': result.lr1,
        'max_eigen_statistic': result.lr2,
        'critical_values_trace': result.cvt,
        'critical_values_max_eigen': result.cvm,
        'cointegrating_vectors': result.evec
    }
```

---
