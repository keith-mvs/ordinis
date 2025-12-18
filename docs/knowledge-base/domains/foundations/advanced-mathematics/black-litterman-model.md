### 4.2 Black-Litterman Model

**Combines market equilibrium with investor views**:

```python
def black_litterman(
    cov_matrix: np.array,
    market_caps: np.array,
    views: np.array,          # P × μ = Q + ε
    view_matrix: np.array,    # P (picking matrix)
    view_confidence: np.array, # Ω (uncertainty)
    risk_aversion: float = 2.5,
    tau: float = 0.05
) -> dict:
    """
    Black-Litterman model for portfolio optimization.
    """
    n_assets = len(market_caps)

    # Market-implied equilibrium returns
    market_weights = market_caps / market_caps.sum()
    pi = risk_aversion * cov_matrix @ market_weights

    # View precision (inverse of view covariance)
    omega = np.diag(view_confidence)
    omega_inv = np.linalg.inv(omega)

    # Posterior expected returns
    tau_sigma = tau * cov_matrix
    tau_sigma_inv = np.linalg.inv(tau_sigma)

    # Combined estimate
    M = np.linalg.inv(tau_sigma_inv + view_matrix.T @ omega_inv @ view_matrix)
    posterior_mean = M @ (tau_sigma_inv @ pi + view_matrix.T @ omega_inv @ views)

    # Posterior covariance
    posterior_cov = M + cov_matrix

    return {
        'equilibrium_returns': pi,
        'posterior_returns': posterior_mean,
        'posterior_covariance': posterior_cov,
        'market_weights': market_weights
    }
```

---
