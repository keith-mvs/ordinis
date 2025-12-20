### 6.2 Finite Difference Methods

**Solving Black-Scholes PDE**:

```python
def crank_nicolson_option(
    S_max: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    M: int = 100,  # Price steps
    N: int = 100,  # Time steps
    option_type: str = 'call'
) -> np.array:
    """
    Crank-Nicolson scheme for American/European options.
    """
    dS = S_max / M
    dt = T / N
    S = np.linspace(0, S_max, M + 1)

    # Initialize option values at maturity
    if option_type == 'call':
        V = np.maximum(S - K, 0)
    else:
        V = np.maximum(K - S, 0)

    # Coefficients
    alpha = 0.25 * dt * (sigma**2 * np.arange(M+1)**2 - r * np.arange(M+1))
    beta = -0.5 * dt * (sigma**2 * np.arange(M+1)**2 + r)
    gamma = 0.25 * dt * (sigma**2 * np.arange(M+1)**2 + r * np.arange(M+1))

    # Tridiagonal matrices
    A = np.diag(1 - beta[1:M]) + np.diag(-alpha[2:M], -1) + np.diag(-gamma[1:M-1], 1)
    B = np.diag(1 + beta[1:M]) + np.diag(alpha[2:M], -1) + np.diag(gamma[1:M-1], 1)

    # Time stepping
    for n in range(N):
        rhs = B @ V[1:M]

        # Boundary conditions
        rhs[0] += alpha[1] * (V[0] + V[0])  # At S=0
        rhs[-1] += gamma[M-1] * (V[M] + V[M])  # At S=S_max

        V[1:M] = np.linalg.solve(A, rhs)

    return S, V
```

---
