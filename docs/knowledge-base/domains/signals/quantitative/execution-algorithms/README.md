# Execution Algorithms

## Overview

Execution algorithms minimize the cost of trading by optimizing order placement. The goal is to achieve the best possible price while minimizing market impact, timing risk, and explicit costs.

---

## Algorithm Types

| File | Algorithm | Objective |
|------|-----------|-----------|
| [twap_vwap.md](twap_vwap.md) | TWAP/VWAP | Track benchmark price |
| [optimal_execution.md](optimal_execution.md) | Almgren-Chriss | Minimize impact + risk |
| [market_impact.md](market_impact.md) | Impact Models | Estimate price impact |

---

## Execution Costs

### Cost Components
```python
EXECUTION_COSTS = {
    'commission': 0.001,        # Broker fee
    'spread': 0.0005,           # Half bid-ask spread
    'market_impact': 'variable', # Depends on size
    'timing_risk': 'variable',  # Price moves during execution
    'opportunity_cost': 'variable'  # Delay cost
}

def total_execution_cost(
    order_value: float,
    spread: float,
    impact: float,
    commission_rate: float
) -> float:
    """
    Total cost of executing an order.
    """
    spread_cost = order_value * spread / 2
    impact_cost = order_value * impact
    commission = order_value * commission_rate

    return spread_cost + impact_cost + commission
```

### Market Impact Models
```python
def square_root_impact(
    order_size: float,
    adv: float,  # Average daily volume
    volatility: float,
    impact_coefficient: float = 0.1
) -> float:
    """
    Square-root impact model (most common).
    Impact ~ sqrt(order_size / ADV) * volatility
    """
    participation = order_size / adv
    return impact_coefficient * volatility * np.sqrt(participation)

def linear_impact(
    order_size: float,
    adv: float,
    impact_coefficient: float = 0.1
) -> float:
    """
    Linear impact model (simpler).
    """
    return impact_coefficient * (order_size / adv)
```

---

## TWAP (Time-Weighted Average Price)

```python
class TWAPAlgorithm:
    """
    Execute order evenly over time.
    Benchmark: Time-weighted average price
    """
    def __init__(
        self,
        total_shares: int,
        duration_minutes: int,
        interval_minutes: int = 5
    ):
        self.total_shares = total_shares
        self.duration = duration_minutes
        self.interval = interval_minutes
        self.n_slices = duration_minutes // interval_minutes

    def generate_schedule(self) -> list:
        """
        Generate execution schedule.
        """
        shares_per_slice = self.total_shares / self.n_slices

        schedule = []
        for i in range(self.n_slices):
            schedule.append({
                'time': i * self.interval,
                'shares': int(shares_per_slice),
                'type': 'limit'  # or 'market' at end
            })

        return schedule

    def twap_benchmark(self, prices: pd.Series) -> float:
        """
        Calculate TWAP benchmark price.
        """
        return prices.mean()
```

---

## VWAP (Volume-Weighted Average Price)

```python
class VWAPAlgorithm:
    """
    Execute proportional to historical volume pattern.
    Benchmark: Volume-weighted average price
    """
    def __init__(
        self,
        total_shares: int,
        volume_profile: pd.Series  # Historical intraday volume
    ):
        self.total_shares = total_shares
        self.volume_profile = volume_profile

    def generate_schedule(self) -> list:
        """
        Distribute order according to volume pattern.
        """
        # Normalize volume profile
        volume_pct = self.volume_profile / self.volume_profile.sum()

        schedule = []
        for time, pct in volume_pct.items():
            shares = int(self.total_shares * pct)
            schedule.append({
                'time': time,
                'shares': shares,
                'pct_of_volume': pct
            })

        return schedule

    def vwap_benchmark(self, prices: pd.Series, volumes: pd.Series) -> float:
        """
        Calculate VWAP benchmark price.
        """
        return (prices * volumes).sum() / volumes.sum()

    @staticmethod
    def typical_intraday_volume_profile() -> pd.Series:
        """
        U-shaped volume pattern (high at open/close).
        """
        # Simplified hourly profile (9:30 AM to 4:00 PM)
        hours = pd.date_range('09:30', '16:00', freq='30T')
        volumes = [
            0.15,  # 9:30-10:00 (high - open)
            0.10,
            0.08,
            0.07,
            0.06,
            0.05,  # 12:00-12:30 (low - lunch)
            0.05,
            0.06,
            0.07,
            0.08,
            0.10,
            0.13,  # 3:30-4:00 (high - close)
        ]
        return pd.Series(volumes, index=hours[:len(volumes)])
```

---

## Optimal Execution (Almgren-Chriss)

```python
class AlmgrenChrissSolver:
    """
    Optimal trade schedule balancing impact vs timing risk.

    Objective: min E[Cost] + λ × Var[Cost]
    """
    def __init__(
        self,
        total_shares: int,
        time_horizon: float,
        volatility: float,
        permanent_impact: float,
        temporary_impact: float,
        risk_aversion: float
    ):
        self.X = total_shares
        self.T = time_horizon
        self.sigma = volatility
        self.gamma = permanent_impact  # Permanent impact coefficient
        self.eta = temporary_impact    # Temporary impact coefficient
        self.lambda_ = risk_aversion

    def solve(self, n_intervals: int = 20) -> dict:
        """
        Solve for optimal trading trajectory.
        """
        dt = self.T / n_intervals

        # Key parameter
        kappa = np.sqrt(self.lambda_ * self.sigma**2 / self.eta)

        # Optimal position trajectory
        t = np.linspace(0, self.T, n_intervals + 1)
        x_t = self.X * np.sinh(kappa * (self.T - t)) / np.sinh(kappa * self.T)

        # Trading rate
        n_t = -np.diff(x_t) / dt

        # Expected cost
        expected_cost = (
            0.5 * self.gamma * self.X**2 +
            self.eta * np.sum(n_t**2) * dt
        )

        # Variance of cost
        variance_cost = self.sigma**2 * np.sum(x_t[1:]**2) * dt

        return {
            'position_trajectory': x_t,
            'trading_rate': n_t,
            'expected_cost': expected_cost,
            'variance_cost': variance_cost,
            'total_cost': expected_cost + self.lambda_ * variance_cost
        }
```

---

## Implementation Shortfall

```python
class ImplementationShortfall:
    """
    Minimize slippage vs decision price.
    Benchmark: Price at time of decision (arrival price)
    """
    def __init__(
        self,
        decision_price: float,
        total_shares: int,
        urgency: str = 'medium'
    ):
        self.decision_price = decision_price
        self.total_shares = total_shares
        self.urgency = urgency

    def calculate_shortfall(
        self,
        execution_prices: list,
        execution_shares: list
    ) -> float:
        """
        Calculate implementation shortfall.
        """
        avg_exec_price = sum(
            p * s for p, s in zip(execution_prices, execution_shares)
        ) / sum(execution_shares)

        shortfall = (avg_exec_price - self.decision_price) / self.decision_price

        return shortfall

    def get_aggressiveness(self) -> dict:
        """
        Trading aggressiveness based on urgency.
        """
        profiles = {
            'low': {'participation_rate': 0.05, 'limit_offset': 0.002},
            'medium': {'participation_rate': 0.10, 'limit_offset': 0.001},
            'high': {'participation_rate': 0.20, 'limit_offset': 0.0005},
            'urgent': {'participation_rate': 0.30, 'limit_offset': 0.0}
        }
        return profiles.get(self.urgency, profiles['medium'])
```

---

## Participation Rate

```python
def calculate_participation_rate(
    order_shares: int,
    time_window: int,
    historical_volume: float
) -> float:
    """
    What fraction of market volume is our order?
    """
    expected_volume = historical_volume * (time_window / 390)  # 390 mins/day
    return order_shares / expected_volume

# Guidelines
PARTICIPATION_LIMITS = {
    'low_impact': 0.05,     # < 5% of volume
    'moderate_impact': 0.10, # 5-10%
    'high_impact': 0.20,    # 10-20%
    'significant_impact': 0.30  # > 20% - consider spreading over days
}
```

---

## Order Types for Execution

```python
class ExecutionOrderTypes:
    """
    Order types used in execution algorithms.
    """
    @staticmethod
    def limit_order(price: float, shares: int, side: str) -> dict:
        """Standard limit order."""
        return {
            'type': 'LIMIT',
            'price': price,
            'qty': shares,
            'side': side,
            'tif': 'DAY'
        }

    @staticmethod
    def pegged_order(offset: float, shares: int, side: str) -> dict:
        """Pegged to midpoint/bid/ask."""
        return {
            'type': 'PEG',
            'offset': offset,
            'qty': shares,
            'side': side,
            'peg_type': 'MIDPOINT'
        }

    @staticmethod
    def iceberg_order(
        total_shares: int,
        display_qty: int,
        price: float,
        side: str
    ) -> dict:
        """Hide order size."""
        return {
            'type': 'ICEBERG',
            'total_qty': total_shares,
            'display_qty': display_qty,
            'price': price,
            'side': side
        }
```

---

## Performance Metrics

```python
def execution_performance_metrics(
    fills: list,
    benchmark_type: str = 'vwap'
) -> dict:
    """
    Evaluate execution quality.
    """
    avg_fill_price = sum(f['price'] * f['qty'] for f in fills) / sum(f['qty'] for f in fills)

    if benchmark_type == 'vwap':
        benchmark = calculate_vwap(fills)
    elif benchmark_type == 'twap':
        benchmark = calculate_twap(fills)
    elif benchmark_type == 'arrival':
        benchmark = fills[0]['arrival_price']

    slippage = (avg_fill_price - benchmark) / benchmark

    return {
        'avg_fill_price': avg_fill_price,
        'benchmark_price': benchmark,
        'slippage_bps': slippage * 10000,
        'participation_rate': calculate_participation(fills),
        'fill_rate': sum(f['qty'] for f in fills) / fills[0]['target_qty']
    }
```

---

## Best Practices

1. **Know your benchmark**: Different algos optimize different objectives
2. **Consider urgency**: Faster execution = higher impact
3. **Monitor participation**: Stay under 10% of volume when possible
4. **Use limit orders**: Avoid market orders except in urgency
5. **Adapt to conditions**: Widen in volatile markets
6. **Measure performance**: Track slippage vs benchmark

---

## Academic References

- Almgren & Chriss (2001): "Optimal Execution of Portfolio Transactions"
- Bertsimas & Lo (1998): "Optimal Control of Execution Costs"
- Kissell & Glantz (2003): "Optimal Trading Strategies"
