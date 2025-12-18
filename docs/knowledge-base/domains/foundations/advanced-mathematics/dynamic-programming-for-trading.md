### 4.4 Dynamic Programming for Trading

**Optimal Execution (Almgren-Chriss)**:

```python
def almgren_chriss_optimal_execution(
    total_shares: int,
    time_steps: int,
    volatility: float,
    temporary_impact: float,  # η
    permanent_impact: float,  # γ
    risk_aversion: float      # λ
) -> Tuple[np.array, float]:
    """
    Optimal trade schedule to minimize execution cost + risk.

    Objective: min E[cost] + λ × Var[cost]
    """
    # Optimal trading rate (closed-form solution)
    kappa = np.sqrt(risk_aversion * volatility**2 / temporary_impact)

    # Trade schedule
    trade_times = np.arange(time_steps + 1)
    remaining_shares = total_shares * np.sinh(kappa * (time_steps - trade_times)) / np.sinh(kappa * time_steps)

    # Trades at each step
    trades = -np.diff(remaining_shares)

    # Expected cost
    expected_cost = (
        permanent_impact * total_shares**2 / 2 +
        temporary_impact * np.sum(trades**2)
    )

    return trades, expected_cost
```

---
