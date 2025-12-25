### 5.1 Factor Models

**Fama-French Type Models**:

```python
def estimate_factor_model(
    returns: np.array,        # Asset returns (T × N)
    factors: np.array,        # Factor returns (T × K)
    fit_intercept: bool = True
) -> dict:
    """
    Estimate multi-factor model: R = α + β × F + ε
    """
    from sklearn.linear_model import LinearRegression

    n_assets = returns.shape[1]
    results = {
        'alphas': [],
        'betas': [],
        'r_squared': [],
        'residuals': []
    }

    for i in range(n_assets):
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(factors, returns[:, i])

        results['alphas'].append(model.intercept_ if fit_intercept else 0)
        results['betas'].append(model.coef_)
        results['r_squared'].append(model.score(factors, returns[:, i]))
        results['residuals'].append(returns[:, i] - model.predict(factors))

    return {
        'alphas': np.array(results['alphas']),
        'betas': np.array(results['betas']),
        'r_squared': np.array(results['r_squared']),
        'residual_matrix': np.array(results['residuals']).T
    }

def pca_factor_extraction(
    returns: np.array,
    n_factors: int = 5
) -> dict:
    """
    Extract statistical factors using PCA.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(returns)

    return {
        'factors': factors,
        'loadings': pca.components_.T,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
    }
```

---
