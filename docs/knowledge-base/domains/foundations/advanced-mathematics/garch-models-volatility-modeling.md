### 3.3 GARCH Models (Volatility Modeling)

**GARCH(p, q)**: Generalized Autoregressive Conditional Heteroskedasticity

```
r_t = μ + ε_t,  ε_t = σ_t × z_t,  z_t ~ N(0,1)
σ²_t = ω + Σα_i ε²_{t-i} + Σβ_j σ²_{t-j}
```

```python
from arch import arch_model

def fit_garch(returns: np.array, p: int = 1, q: int = 1) -> dict:
    """
    Fit GARCH(p,q) model for volatility forecasting.
    """
    model = arch_model(returns, vol='Garch', p=p, q=q, rescale=False)
    fitted = model.fit(disp='off')

    return {
        'params': {
            'omega': fitted.params['omega'],
            'alpha': [fitted.params[f'alpha[{i}]'] for i in range(1, p+1)],
            'beta': [fitted.params[f'beta[{i}]'] for i in range(1, q+1)]
        },
        'conditional_volatility': fitted.conditional_volatility,
        'standardized_residuals': fitted.std_resid,
        'forecast': fitted.forecast,
        'aic': fitted.aic,
        'bic': fitted.bic
    }

def forecast_volatility(
    returns: np.array,
    horizon: int = 5,
    model_type: str = 'GARCH'
) -> np.array:
    """
    Forecast volatility using GARCH-family models.
    """
    if model_type == 'GARCH':
        model = arch_model(returns, vol='Garch', p=1, q=1)
    elif model_type == 'EGARCH':
        model = arch_model(returns, vol='EGARCH', p=1, q=1)
    elif model_type == 'GJR-GARCH':
        model = arch_model(returns, vol='Garch', p=1, o=1, q=1)

    fitted = model.fit(disp='off')
    forecast = fitted.forecast(horizon=horizon)

    return np.sqrt(forecast.variance.values[-1])
```

**GARCH Variants**:

| Model | Feature | Use Case |
|-------|---------|----------|
| GARCH | Symmetric shocks | General volatility |
| EGARCH | Asymmetric, no positivity constraint | Leverage effect |
| GJR-GARCH | Asymmetric threshold | Equity volatility |
| TGARCH | Threshold model | Regime-dependent vol |

---
