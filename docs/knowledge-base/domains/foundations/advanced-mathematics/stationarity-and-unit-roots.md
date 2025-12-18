### 3.1 Stationarity and Unit Roots

**Stationarity Conditions**:
- Strict stationarity: Joint distribution invariant under time shifts
- Weak stationarity: Constant mean, variance, and autocovariance

```python
from statsmodels.tsa.stattools import adfuller, kpss

def stationarity_tests(series: np.array) -> dict:
    """
    Test for stationarity using ADF and KPSS tests.

    ADF: H0 = unit root (non-stationary)
    KPSS: H0 = stationary

    Ideal: Reject ADF, fail to reject KPSS
    """
    # Augmented Dickey-Fuller test
    adf_result = adfuller(series, autolag='AIC')

    # KPSS test
    kpss_result = kpss(series, regression='c')

    return {
        'adf': {
            'statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_stationary': adf_result[1] < 0.05
        },
        'kpss': {
            'statistic': kpss_result[0],
            'p_value': kpss_result[1],
            'critical_values': kpss_result[3],
            'is_stationary': kpss_result[1] > 0.05
        }
    }

def make_stationary(series: pd.Series) -> Tuple[pd.Series, int]:
    """
    Difference series until stationary.
    Returns (stationary_series, order_of_differencing).
    """
    d = 0
    diff_series = series.copy()

    while not stationarity_tests(diff_series.dropna())['adf']['is_stationary']:
        diff_series = diff_series.diff()
        d += 1
        if d > 2:
            break

    return diff_series.dropna(), d
```

---
