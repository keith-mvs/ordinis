### 2.2 Martingales

**Definition**: A process M(t) is a martingale if:
```
E[M(t) | ℱ(s)] = M(s)  for all s ≤ t
```

**Key Properties for Trading**:
- Under risk-neutral measure, discounted asset prices are martingales
- No free lunch with vanishing risk (NFLVR) theorem
- Optional stopping theorem (cannot profit from perfect timing alone)

```python
def is_martingale_test(prices: np.array, lags: int = 10) -> dict:
    """
    Statistical tests for martingale property.

    Uses variance ratio test (Lo-MacKinlay).
    Under martingale: Var(r_k) = k × Var(r_1)
    """
    returns = np.diff(np.log(prices))

    results = {}
    for k in range(2, lags + 1):
        # k-period returns
        returns_k = np.diff(np.log(prices[::k]))

        # Variance ratio
        vr = np.var(returns_k) / (k * np.var(returns))

        # Test statistic (under null, VR = 1)
        results[k] = {
            'variance_ratio': vr,
            'deviation_from_1': abs(vr - 1)
        }

    return results
```

---
