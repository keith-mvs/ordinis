# Portfolio Construction

## Overview

Portfolio construction determines optimal allocation across assets or strategies. The goal is to maximize risk-adjusted returns while controlling drawdowns and maintaining diversification.

---

## Methods

| File | Method | Requirement |
|------|--------|-------------|
| [mean_variance.md](mean_variance.md) | Markowitz | Return estimates |
| [risk_parity.md](risk_parity.md) | Risk Parity | Covariance only |
| [hrp.md](hrp.md) | Hierarchical Risk Parity | Correlation matrix |

---

## Mean-Variance Optimization

### Classic Markowitz
```python
from scipy.optimize import minimize

def mean_variance_portfolio(
    expected_returns: np.array,
    cov_matrix: np.array,
    target_return: float = None,
    risk_free: float = 0.0
) -> dict:
    """
    Classic mean-variance optimization.

    min w'Σw  (minimize variance)
    s.t. w'μ >= target_return
         w'1 = 1
         w >= 0
    """
    n = len(expected_returns)

    def portfolio_variance(w):
        return w @ cov_matrix @ w

    def portfolio_return(w):
        return w @ expected_returns

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
    ]

    if target_return is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda w: portfolio_return(w) - target_return
        })

    # Bounds (long only)
    bounds = [(0, 1) for _ in range(n)]

    # Initial guess
    w0 = np.ones(n) / n

    result = minimize(
        portfolio_variance,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    weights = result.x
    ret = portfolio_return(weights)
    vol = np.sqrt(portfolio_variance(weights))
    sharpe = (ret - risk_free) / vol

    return {
        'weights': weights,
        'return': ret,
        'volatility': vol,
        'sharpe': sharpe
    }
```

### Maximum Sharpe Portfolio
```python
def max_sharpe_portfolio(
    expected_returns: np.array,
    cov_matrix: np.array,
    risk_free: float = 0.0
) -> dict:
    """
    Find portfolio with maximum Sharpe ratio.
    """
    n = len(expected_returns)

    def neg_sharpe(w):
        ret = w @ expected_returns
        vol = np.sqrt(w @ cov_matrix @ w)
        return -(ret - risk_free) / vol

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n

    result = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    return {
        'weights': result.x,
        'sharpe': -result.fun
    }
```

### Minimum Variance Portfolio
```python
def min_variance_portfolio(cov_matrix: np.array) -> dict:
    """
    Minimum variance portfolio (no return estimates needed).
    """
    n = cov_matrix.shape[0]

    def portfolio_variance(w):
        return w @ cov_matrix @ w

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, 1) for _ in range(n)]
    w0 = np.ones(n) / n

    result = minimize(portfolio_variance, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    return {
        'weights': result.x,
        'variance': result.fun
    }
```

---

## Risk Parity

### Equal Risk Contribution
```python
def risk_parity_portfolio(cov_matrix: np.array) -> dict:
    """
    Each asset contributes equally to portfolio risk.
    No return estimates needed.
    """
    n = cov_matrix.shape[0]

    def risk_contribution(w):
        """Calculate each asset's contribution to total risk."""
        port_vol = np.sqrt(w @ cov_matrix @ w)
        marginal_contrib = cov_matrix @ w
        risk_contrib = w * marginal_contrib / port_vol
        return risk_contrib

    def objective(w):
        """Minimize deviation from equal risk contribution."""
        rc = risk_contribution(w)
        target_rc = np.ones(n) / n * np.sqrt(w @ cov_matrix @ w)
        return np.sum((rc - rc.mean())**2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.01, 1) for _ in range(n)]  # Minimum 1% per asset
    w0 = np.ones(n) / n

    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    weights = result.x
    rc = risk_contribution(weights)

    return {
        'weights': weights,
        'risk_contributions': rc,
        'risk_contribution_pct': rc / rc.sum()
    }
```

### Inverse Volatility
```python
def inverse_volatility_weights(cov_matrix: np.array) -> np.array:
    """
    Simple approximation to risk parity.
    Weight inversely proportional to volatility.
    """
    volatilities = np.sqrt(np.diag(cov_matrix))
    inv_vol = 1 / volatilities
    weights = inv_vol / inv_vol.sum()
    return weights
```

---

## Hierarchical Risk Parity (HRP)

```python
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def hierarchical_risk_parity(cov_matrix: np.array) -> dict:
    """
    Graph-theoretic portfolio construction.
    More stable than mean-variance.
    """
    n = cov_matrix.shape[0]
    corr = cov_to_corr(cov_matrix)

    # Step 1: Tree clustering
    dist = np.sqrt((1 - corr) / 2)  # Correlation distance
    link = linkage(squareform(dist), method='single')
    sort_ix = leaves_list(link)

    # Step 2: Quasi-diagonalization
    sorted_cov = cov_matrix[sort_ix, :][:, sort_ix]

    # Step 3: Recursive bisection
    weights = np.ones(n)
    clusters = [list(range(n))]

    while len(clusters) > 0:
        clusters = [
            c[start:end]
            for c in clusters
            for start, end in ((0, len(c)//2), (len(c)//2, len(c)))
            if len(c) > 1
        ] if clusters[0] else []

        for i in range(0, len(clusters), 2):
            if i + 1 < len(clusters):
                c1, c2 = clusters[i], clusters[i+1]
                var1 = get_cluster_variance(sorted_cov, c1)
                var2 = get_cluster_variance(sorted_cov, c2)

                # Allocate inversely to variance
                alpha = var2 / (var1 + var2)
                weights[c1] *= alpha
                weights[c2] *= (1 - alpha)

    # Unsort weights
    final_weights = np.zeros(n)
    final_weights[sort_ix] = weights

    return {
        'weights': final_weights / final_weights.sum(),
        'sort_order': sort_ix
    }

def get_cluster_variance(cov: np.array, indices: list) -> float:
    """Inverse-variance allocation within cluster."""
    sub_cov = cov[np.ix_(indices, indices)]
    ivp = inverse_volatility_weights(sub_cov)
    return ivp @ sub_cov @ ivp

def cov_to_corr(cov: np.array) -> np.array:
    """Convert covariance to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)
```

---

## Black-Litterman Model

```python
def black_litterman(
    cov_matrix: np.array,
    market_caps: np.array,
    views: np.array,
    view_matrix: np.array,
    view_confidence: np.array,
    risk_aversion: float = 2.5,
    tau: float = 0.05
) -> dict:
    """
    Bayesian combination of market equilibrium and investor views.

    Parameters:
    - cov_matrix: Asset covariance matrix
    - market_caps: Market capitalization weights
    - views: Expected returns from views (Q)
    - view_matrix: Picking matrix (P)
    - view_confidence: Diagonal of Omega (uncertainty)
    - risk_aversion: Lambda
    - tau: Scalar for equilibrium uncertainty
    """
    n = len(market_caps)

    # Market-implied equilibrium returns
    mkt_weights = market_caps / market_caps.sum()
    pi = risk_aversion * cov_matrix @ mkt_weights

    # View uncertainty
    omega = np.diag(view_confidence)
    omega_inv = np.linalg.inv(omega)

    # Posterior calculation
    tau_cov = tau * cov_matrix
    tau_cov_inv = np.linalg.inv(tau_cov)

    # Combined posterior mean
    M = np.linalg.inv(tau_cov_inv + view_matrix.T @ omega_inv @ view_matrix)
    posterior_mean = M @ (tau_cov_inv @ pi + view_matrix.T @ omega_inv @ views)

    # Posterior covariance
    posterior_cov = M + cov_matrix

    return {
        'equilibrium_returns': pi,
        'posterior_returns': posterior_mean,
        'posterior_cov': posterior_cov,
        'market_weights': mkt_weights
    }
```

---

## Constraints

```python
def constrained_optimization(
    expected_returns: np.array,
    cov_matrix: np.array,
    constraints: dict
) -> dict:
    """
    Mean-variance with practical constraints.
    """
    n = len(expected_returns)

    def objective(w):
        return w @ cov_matrix @ w

    constraint_list = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    ]

    # Maximum position size
    if 'max_weight' in constraints:
        bounds = [(0, constraints['max_weight']) for _ in range(n)]
    else:
        bounds = [(0, 1) for _ in range(n)]

    # Minimum position size
    if 'min_weight' in constraints:
        bounds = [(constraints['min_weight'], b[1]) for b in bounds]

    # Sector constraints
    if 'sector_limits' in constraints:
        for sector, limit in constraints['sector_limits'].items():
            sector_mask = constraints['sector_mapping'] == sector
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda w, m=sector_mask, l=limit: l - np.sum(w[m])
            })

    # Turnover constraint
    if 'max_turnover' in constraints:
        current_weights = constraints['current_weights']
        constraint_list.append({
            'type': 'ineq',
            'fun': lambda w: constraints['max_turnover'] - np.sum(np.abs(w - current_weights))
        })

    w0 = np.ones(n) / n
    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraint_list)

    return {'weights': result.x}
```

---

## Rebalancing

```python
class RebalancingStrategy:
    """
    When and how to rebalance portfolio.
    """
    def __init__(
        self,
        target_weights: np.array,
        threshold: float = 0.05,  # 5% drift trigger
        method: str = 'threshold'
    ):
        self.target = target_weights
        self.threshold = threshold
        self.method = method

    def should_rebalance(self, current_weights: np.array) -> bool:
        """Check if rebalancing needed."""
        if self.method == 'threshold':
            max_drift = np.max(np.abs(current_weights - self.target))
            return max_drift > self.threshold

        elif self.method == 'calendar':
            # Rebalance monthly/quarterly regardless
            return True

        elif self.method == 'tolerance_band':
            # Each weight has its own band
            drifts = np.abs(current_weights - self.target) / self.target
            return np.any(drifts > self.threshold)

    def rebalance_trades(
        self,
        current_weights: np.array,
        portfolio_value: float
    ) -> pd.Series:
        """Calculate trades needed to rebalance."""
        trade_weights = self.target - current_weights
        trade_dollars = trade_weights * portfolio_value
        return pd.Series(trade_dollars)
```

---

## Performance Attribution

```python
def portfolio_attribution(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_weights: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    asset_returns: pd.DataFrame
) -> dict:
    """
    Brinson attribution: Allocation + Selection + Interaction.
    """
    # Allocation effect: Over/underweight in outperforming sectors
    allocation = (portfolio_weights - benchmark_weights) * (benchmark_returns - benchmark_returns.mean())

    # Selection effect: Stock picking within sectors
    selection = benchmark_weights * (portfolio_returns - benchmark_returns)

    # Interaction effect
    interaction = (portfolio_weights - benchmark_weights) * (portfolio_returns - benchmark_returns)

    return {
        'allocation': allocation.sum(),
        'selection': selection.sum(),
        'interaction': interaction.sum(),
        'total_active': allocation.sum() + selection.sum() + interaction.sum()
    }
```

---

## Best Practices

1. **Robust estimation**: Use shrinkage estimators for covariance
2. **Constraint appropriately**: Avoid corner solutions
3. **Control turnover**: Transaction costs matter
4. **Diversify**: Avoid concentrated bets
5. **Rebalance systematically**: Rules-based, not emotional
6. **Monitor**: Track drift and risk contributions

---

## Academic References

- Markowitz (1952): "Portfolio Selection"
- Black & Litterman (1992): Model for asset allocation
- Maillard, Roncalli, Teiletche (2010): Risk parity
- López de Prado (2016): Hierarchical Risk Parity
