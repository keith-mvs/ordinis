### 2.5 Mean-Reverting Processes

**Ornstein-Uhlenbeck Process**:

```
dX(t) = θ(μ - X(t))dt + σdW(t)
```

Where θ = speed of mean reversion, μ = long-term mean.

```python
def simulate_ornstein_uhlenbeck(
    X0: float,
    theta: float,  # Mean reversion speed
    mu: float,     # Long-term mean
    sigma: float,  # Volatility
    T: float,
    N: int,
    n_paths: int = 1
) -> np.array:
    """
    Simulate Ornstein-Uhlenbeck (OU) process.

    Used for:
    - Interest rate modeling
    - Pairs trading spread
    - Volatility modeling
    """
    dt = T / N
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = X0

    # Exact simulation (not Euler discretization)
    exp_theta = np.exp(-theta * dt)
    std = sigma * np.sqrt((1 - exp_theta**2) / (2 * theta))

    for i in range(N):
        paths[:, i+1] = (
            mu + (paths[:, i] - mu) * exp_theta +
            std * np.random.normal(0, 1, n_paths)
        )

    return paths

def estimate_ou_parameters(spread: np.array, dt: float = 1/252) -> dict:
    """
    Estimate OU parameters from time series (for pairs trading).

    Uses OLS regression: X(t+1) - X(t) = θ(μ - X(t))dt + noise
    """
    X = spread[:-1]
    dX = np.diff(spread)

    # Regression: dX = a + b*X
    b, a = np.polyfit(X, dX, 1)

    theta = -b / dt
    mu = a / (theta * dt)
    residuals = dX - (a + b * X)
    sigma = np.std(residuals) / np.sqrt(dt)

    # Half-life of mean reversion
    half_life = np.log(2) / theta

    return {
        'theta': theta,
        'mu': mu,
        'sigma': sigma,
        'half_life_days': half_life
    }
```

---
