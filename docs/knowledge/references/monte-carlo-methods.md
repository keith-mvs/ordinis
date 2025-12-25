### 6.1 Monte Carlo Methods

**Variance Reduction Techniques**:

```python
def monte_carlo_option_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    n_simulations: int = 100000,
    antithetic: bool = True,
    control_variate: bool = True
) -> dict:
    """
    Monte Carlo option pricing with variance reduction.
    """
    dt = T
    discount = np.exp(-r * T)

    # Generate random numbers
    Z = np.random.normal(0, 1, n_simulations)

    # Antithetic variates
    if antithetic:
        Z = np.concatenate([Z, -Z])

    # Simulate terminal prices
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Option payoffs
    payoffs = np.maximum(S_T - K, 0)

    # Control variate using forward price
    if control_variate:
        forward = S0 * np.exp(r * T)
        # Covariance between payoff and control
        cov = np.cov(payoffs, S_T)[0, 1]
        var_control = np.var(S_T)
        beta = cov / var_control

        # Adjusted payoffs
        payoffs = payoffs - beta * (S_T - forward)

    # Price estimate
    price = discount * np.mean(payoffs)
    std_error = discount * np.std(payoffs) / np.sqrt(len(payoffs))

    return {
        'price': price,
        'std_error': std_error,
        '95_ci': (price - 1.96 * std_error, price + 1.96 * std_error)
    }

def importance_sampling_var(
    returns: np.array,
    confidence: float = 0.99,
    n_simulations: int = 100000
) -> dict:
    """
    Importance sampling for rare event (VaR) estimation.
    """
    mu = np.mean(returns)
    sigma = np.std(returns)

    # Shift distribution to sample more tail events
    shift = -2 * sigma  # Shift mean to focus on left tail

    # Generate samples from shifted distribution
    samples = np.random.normal(mu + shift, sigma, n_simulations)

    # Importance weights (likelihood ratio)
    weights = np.exp(
        -0.5 * ((samples - mu)**2 - (samples - (mu + shift))**2) / sigma**2
    )

    # Weighted quantile estimation
    sorted_indices = np.argsort(samples)
    sorted_samples = samples[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights) / np.sum(sorted_weights)

    var_index = np.searchsorted(cumulative_weights, 1 - confidence)
    var_estimate = -sorted_samples[var_index]

    return {
        'var': var_estimate,
        'effective_sample_size': np.sum(weights)**2 / np.sum(weights**2)
    }
```

---
