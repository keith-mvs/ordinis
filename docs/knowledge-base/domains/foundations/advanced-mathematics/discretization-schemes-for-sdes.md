### 6.3 Discretization Schemes for SDEs

**Euler-Maruyama vs Milstein**:

```python
def euler_maruyama(
    drift: Callable,
    diffusion: Callable,
    X0: float,
    T: float,
    N: int,
    n_paths: int = 1
) -> np.array:
    """
    Euler-Maruyama discretization.

    dX = μ(X,t)dt + σ(X,t)dW

    Weak order 1, strong order 0.5
    """
    dt = T / N
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = X0

    for i in range(N):
        t = i * dt
        X = paths[:, i]
        dW = np.random.normal(0, np.sqrt(dt), n_paths)

        paths[:, i+1] = X + drift(X, t) * dt + diffusion(X, t) * dW

    return paths

def milstein(
    drift: Callable,
    diffusion: Callable,
    diffusion_derivative: Callable,
    X0: float,
    T: float,
    N: int,
    n_paths: int = 1
) -> np.array:
    """
    Milstein scheme with higher-order correction.

    Strong order 1.0
    """
    dt = T / N
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = X0

    for i in range(N):
        t = i * dt
        X = paths[:, i]
        dW = np.random.normal(0, np.sqrt(dt), n_paths)

        # Milstein correction term
        correction = 0.5 * diffusion(X, t) * diffusion_derivative(X, t) * (dW**2 - dt)

        paths[:, i+1] = (
            X +
            drift(X, t) * dt +
            diffusion(X, t) * dW +
            correction
        )

    return paths
```

---
