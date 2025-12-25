### 2.3 Jump-Diffusion Processes

**Merton Jump-Diffusion Model**:

```
dS(t) = μS(t)dt + σS(t)dW(t) + S(t)dJ(t)
```

Where J(t) is a compound Poisson process with jump size Y.

```python
def simulate_merton_jump_diffusion(
    S0: float,
    mu: float,
    sigma: float,
    lambda_: float,  # Jump intensity
    mu_j: float,     # Mean jump size (log)
    sigma_j: float,  # Jump size volatility
    T: float,
    N: int,
    n_paths: int = 1
) -> np.array:
    """
    Simulate Merton jump-diffusion model.
    """
    dt = T / N
    paths = np.zeros((n_paths, N + 1))
    paths[:, 0] = S0

    for i in range(N):
        # Diffusion component
        dW = np.random.normal(0, np.sqrt(dt), n_paths)

        # Jump component (Poisson arrivals)
        n_jumps = np.random.poisson(lambda_ * dt, n_paths)
        jump_sizes = np.zeros(n_paths)
        for j in range(n_paths):
            if n_jumps[j] > 0:
                jumps = np.random.normal(mu_j, sigma_j, n_jumps[j])
                jump_sizes[j] = np.sum(np.exp(jumps) - 1)

        # Update prices
        paths[:, i+1] = paths[:, i] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * dW
        ) * (1 + jump_sizes)

    return paths
```

---
