### 4.1 Mean-Variance Optimization (Markowitz)

**Classic Formulation**:

```
min  w'Σw           (minimize variance)
s.t. w'μ ≥ r_target (return constraint)
     w'1 = 1        (weights sum to 1)
     w ≥ 0          (no short selling, optional)
```

```python
from scipy.optimize import minimize

def mean_variance_optimization(
    expected_returns: np.array,
    cov_matrix: np.array,
    target_return: Optional[float] = None,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Mean-variance portfolio optimization.
    """
    n_assets = len(expected_returns)

    def portfolio_variance(weights):
        return weights @ cov_matrix @ weights

    def portfolio_return(weights):
        return weights @ expected_returns

    # Constraints
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    if target_return is not None:
        constraints.append({
            'type': 'eq',
            'fun': lambda w: portfolio_return(w) - target_return
        })

    # Bounds (long-only)
    bounds = [(0, 1) for _ in range(n_assets)]

    # Initial guess
    w0 = np.ones(n_assets) / n_assets

    result = minimize(
        portfolio_variance,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    optimal_return = portfolio_return(optimal_weights)
    optimal_volatility = np.sqrt(portfolio_variance(optimal_weights))
    sharpe_ratio = (optimal_return - risk_free_rate) / optimal_volatility

    return {
        'weights': optimal_weights,
        'expected_return': optimal_return,
        'volatility': optimal_volatility,
        'sharpe_ratio': sharpe_ratio
    }

def efficient_frontier(
    expected_returns: np.array,
    cov_matrix: np.array,
    n_points: int = 50
) -> Tuple[np.array, np.array, np.array]:
    """
    Compute the efficient frontier.
    """
    min_ret = expected_returns.min()
    max_ret = expected_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_volatilities = []
    frontier_weights = []

    for target in target_returns:
        try:
            result = mean_variance_optimization(
                expected_returns, cov_matrix, target_return=target
            )
            frontier_volatilities.append(result['volatility'])
            frontier_weights.append(result['weights'])
        except:
            frontier_volatilities.append(np.nan)
            frontier_weights.append(None)

    return target_returns, np.array(frontier_volatilities), frontier_weights
```

---
