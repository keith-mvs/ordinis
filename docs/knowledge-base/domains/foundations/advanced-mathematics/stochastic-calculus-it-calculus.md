### 2.4 Stochastic Calculus (Itô Calculus)

**Itô's Lemma**: For f(S,t) where S follows dS = μdt + σdW:

```
df = (∂f/∂t + μ∂f/∂S + ½σ²∂²f/∂S²)dt + σ(∂f/∂S)dW
```

**Applications**:

```python
# Example: Deriving Black-Scholes using Itô's Lemma
"""
Let V(S,t) be option value. Under risk-neutral measure:

dV = (∂V/∂t + rS∂V/∂S + ½σ²S²∂²V/∂S²)dt + σS(∂V/∂S)dW

For replicating portfolio (Δ shares + bond):
dΠ = Δ×dS + r(V - ΔS)dt

Matching terms and eliminating dW gives Black-Scholes PDE:
∂V/∂t + rS∂V/∂S + ½σ²S²∂²V/∂S² - rV = 0
"""

def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Black-Scholes call option price.

    Derived from solving the BS PDE with boundary condition max(S-K, 0).
    """
    from scipy.stats import norm

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
```

---
