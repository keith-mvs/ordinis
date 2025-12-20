### 2.1 Brownian Motion (Wiener Process)

**Definition**: A stochastic process W(t) is standard Brownian motion if:
1. W(0) = 0
2. W(t) has independent increments
3. W(t) - W(s) ~ N(0, t-s) for t > s
4. W(t) has continuous paths

**Geometric Brownian Motion (GBM)**:

The standard model for asset prices:

```
dS(t) = μS(t)dt + σS(t)dW(t)
```

Solution:
```
S(t) = S(0) × exp((μ - σ²/2)t + σW(t))
```

```python
def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    N: int,
    n_paths: int = 1
) -> np.array:
    """
    Simulate Geometric Brownian Motion paths.

    Parameters:
        S0: Initial price
        mu: Drift (expected return)
        sigma: Volatility
        T: Time horizon
        N: Number of time steps
        n_paths: Number of simulation paths

    Returns:
        Array of shape (n_paths, N+1) with simulated prices
    """
    dt = T / N
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0

    # Generate random increments
    dW = np.random.normal(0, np.sqrt(dt), (n_paths, N))

    # Simulate paths using exact solution
    for i in range(N):
        paths[:, i+1] = paths[:, i] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * dW[:, i]
        )

    return paths
```

---
