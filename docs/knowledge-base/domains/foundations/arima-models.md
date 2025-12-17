### 3.2 ARIMA Models

**ARIMA(p, d, q)**: AutoRegressive Integrated Moving Average

```
(1 - Σφ_i L^i)(1-L)^d X_t = (1 + Σθ_j L^j)ε_t
```

Where L is the lag operator.

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

def identify_arima_order(series: np.array, max_order: int = 5) -> dict:
    """
    Identify ARIMA order using ACF/PACF analysis.

    AR(p): PACF cuts off after lag p, ACF decays
    MA(q): ACF cuts off after lag q, PACF decays
    """
    acf_values = acf(series, nlags=max_order)
    pacf_values = pacf(series, nlags=max_order)

    # Significance threshold (approximate)
    threshold = 1.96 / np.sqrt(len(series))

    # Find where ACF/PACF become insignificant
    significant_acf = np.where(np.abs(acf_values[1:]) > threshold)[0]
    significant_pacf = np.where(np.abs(pacf_values[1:]) > threshold)[0]

    return {
        'acf': acf_values,
        'pacf': pacf_values,
        'suggested_p': len(significant_pacf) if len(significant_pacf) > 0 else 0,
        'suggested_q': len(significant_acf) if len(significant_acf) > 0 else 0,
        'threshold': threshold
    }

def fit_arima(series: np.array, order: Tuple[int, int, int]) -> dict:
    """
    Fit ARIMA model and return diagnostics.
    """
    model = ARIMA(series, order=order)
    fitted = model.fit()

    return {
        'params': fitted.params,
        'aic': fitted.aic,
        'bic': fitted.bic,
        'residuals': fitted.resid,
        'forecast': fitted.forecast,
        'summary': fitted.summary()
    }

def auto_arima(series: np.array, max_p: int = 5, max_q: int = 5) -> dict:
    """
    Automatic ARIMA order selection using AIC.
    """
    best_aic = np.inf
    best_order = (0, 0, 0)

    _, d = make_stationary(pd.Series(series))

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                model = ARIMA(series, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except:
                continue

    return {
        'best_order': best_order,
        'best_aic': best_aic
    }
```

---
