### 4.3 Convex Optimization

**Risk Parity Portfolio**:

```python
def risk_parity_portfolio(cov_matrix: np.array) -> np.array:
    """
    Equal risk contribution portfolio.

    Each asset contributes equally to total portfolio risk.
    """
    n_assets = cov_matrix.shape[0]

    def risk_contribution(weights):
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol
        return risk_contrib

    def objective(weights):
        rc = risk_contribution(weights)
        # Minimize difference from equal contribution
        return np.sum((rc - rc.mean())**2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 1) for _ in range(n_assets)]
    w0 = np.ones(n_assets) / n_assets

    result = minimize(objective, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)

    return result.x
```

**Robust Optimization**:

```python
def robust_portfolio_optimization(
    expected_returns: np.array,
    cov_matrix: np.array,
    uncertainty_set: float = 0.1  # ε for ||μ - μ̂|| ≤ ε
) -> dict:
    """
    Robust portfolio optimization accounting for estimation error.

    Uses ellipsoidal uncertainty set around expected returns.
    """
    n_assets = len(expected_returns)

    def worst_case_return(weights):
        # Worst-case return within uncertainty set
        nominal_return = weights @ expected_returns
        uncertainty_penalty = uncertainty_set * np.sqrt(weights @ cov_matrix @ weights)
        return nominal_return - uncertainty_penalty

    def neg_worst_case_return(weights):
        return -worst_case_return(weights)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n_assets)]
    w0 = np.ones(n_assets) / n_assets

    result = minimize(neg_worst_case_return, w0, method='SLSQP',
                     bounds=bounds, constraints=constraints)

    return {
        'weights': result.x,
        'worst_case_return': -result.fun
    }
```

---
