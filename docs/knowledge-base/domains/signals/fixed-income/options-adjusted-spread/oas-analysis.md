# Option-Adjusted Spread (OAS) Analysis

**Section**: 06_options/advanced
**Last Updated**: 2025-12-12
**Source Skills**: [option-adjusted-spread](../../../../.claude/skills/option-adjusted-spread/SKILL.md)

---

## Overview

Advanced framework for valuing bonds with embedded options using Monte Carlo simulation and binomial tree methodologies.

---

## Embedded Option Valuation

### Callable Bond

```
Value_callable = Value_straight - Value_call_option
```

The issuer owns the call option, reducing bond value to investors.

### Puttable Bond

```
Value_puttable = Value_straight + Value_put_option
```

The investor owns the put option, increasing bond value.

---

## Spread Hierarchy

| Spread | Formula | Considers |
|--------|---------|-----------|
| Nominal | YTM_corp - YTM_treasury | Nothing |
| G-Spread | YTM_corp - Interpolated Treasury | Term structure |
| Z-Spread | Constant over spot curve | Full curve |
| OAS | Z-Spread - Option value | Credit only |

### Relationship

```python
# For callable bonds:
oas = z_spread - option_cost

# For puttable bonds:
oas = z_spread + option_value
```

---

## Binomial Tree Methodology

### Interest Rate Tree

```python
class BinomialRateTree:
    """Binomial interest rate tree for bond valuation."""

    def __init__(self, r0, u, d, q):
        """
        r0: Initial short rate
        u: Up multiplier
        d: Down multiplier
        q: Risk-neutral up probability
        """
        self.r0 = r0
        self.u = u
        self.d = d
        self.q = q

    def build_tree(self, periods):
        """Build rate tree structure."""
        tree = [[self.r0]]

        for t in range(1, periods + 1):
            level = []
            for i in range(t + 1):
                rate = self.r0 * (self.u ** (t - i)) * (self.d ** i)
                level.append(rate)
            tree.append(level)

        return tree
```

### Callable Bond Pricing

```python
def price_callable_bond(self, face_value, coupon_rate, periods,
                       call_schedule, frequency=2):
    """
    Price callable bond via backward induction.

    call_schedule: {period: call_price}
    """
    rate_tree = self.build_tree(periods)
    coupon = face_value * coupon_rate / frequency

    # Initialize at maturity
    value_tree = [[face_value + coupon] * (periods + 1)]

    # Backward induction
    for t in range(periods - 1, -1, -1):
        level = []
        for i in range(t + 1):
            r = rate_tree[t][i]

            # Expected value
            up_val = value_tree[0][i]
            down_val = value_tree[0][i + 1]
            expected = self.q * up_val + (1 - self.q) * down_val
            discounted = expected / (1 + r / frequency)

            value = discounted + coupon

            # Apply call constraint
            if t in call_schedule:
                value = min(value, call_schedule[t])

            level.append(value)

        value_tree.insert(0, level)

    return value_tree[0][0]
```

---

## Monte Carlo Simulation

### Vasicek Interest Rate Model

```python
import numpy as np

class VasicekModel:
    """Mean-reverting short rate model."""

    def __init__(self, r0, a, b, sigma):
        """
        r0: Initial rate
        a: Mean reversion speed
        b: Long-term mean
        sigma: Volatility
        """
        self.r0 = r0
        self.a = a
        self.b = b
        self.sigma = sigma

    def simulate_paths(self, T, num_paths, num_steps):
        """Simulate interest rate paths."""
        dt = T / num_steps
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.r0

        for t in range(1, num_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt), num_paths)
            dr = self.a * (self.b - paths[:, t-1]) * dt + self.sigma * dW
            paths[:, t] = paths[:, t-1] + dr

        return paths
```

### Monte Carlo OAS Calculation

```python
def monte_carlo_oas(bond, market_price, num_simulations=10000):
    """
    Calculate OAS via Monte Carlo.

    1. Simulate interest rate paths
    2. Price bond along each path
    3. Find spread that matches market price
    """
    model = VasicekModel(r0=0.03, a=0.1, b=0.05, sigma=0.01)
    paths = model.simulate_paths(T=bond.maturity, num_paths=num_simulations,
                                 num_steps=100)

    from scipy.optimize import brentq

    def price_error(spread):
        prices = []
        for path in paths:
            price = price_bond_on_path(bond, path, spread)
            prices.append(price)
        return np.mean(prices) - market_price

    oas = brentq(price_error, -0.05, 0.10)
    return oas
```

---

## OAS Interpretation

### Investment Signals

| OAS vs. Peers | Interpretation | Action |
|---------------|----------------|--------|
| Higher OAS | Cheap | Potential buy |
| Lower OAS | Rich | Avoid |
| OAS widening | Credit deterioration | Review fundamentals |
| OAS tightening | Credit improvement | Hold/add |

### OAS Decomposition

```
OAS = Credit Spread + Liquidity Premium
```

Both components reflect non-option risks.

---

## Volatility Sensitivity

### Vega Analysis

```python
def option_vega(bond_price_func, base_vol, vol_shock=0.01):
    """Sensitivity to volatility changes."""
    price_base = bond_price_func(base_vol)
    price_up = bond_price_func(base_vol + vol_shock)
    return (price_up - price_base) / vol_shock
```

### OAS Sensitivity Table

| Volatility | Callable OAS | Puttable OAS |
|------------|--------------|--------------|
| 10% | Higher | Lower |
| 15% | Baseline | Baseline |
| 20% | Lower | Higher |

Higher volatility increases option value, reducing callable OAS and increasing puttable OAS.

---

## Effective Duration with Options

```python
def effective_duration(price, price_up, price_down, yield_change):
    """Duration accounting for embedded options."""
    return (price_up - price_down) / (2 * price * yield_change)

def effective_convexity(price, price_up, price_down, yield_change):
    """Convexity accounting for embedded options."""
    return (price_up + price_down - 2*price) / (price * yield_change**2)
```

### Callable Bond Duration

- Effective duration < Modified duration (capped upside)
- Negative convexity possible near call strike

---

## Convertible Bond Analysis

### Value Components

```python
class ConvertibleBond:
    """Convertible bond pricing."""

    def __init__(self, face_value, coupon_rate, maturity,
                 conversion_ratio, stock_price):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.conversion_ratio = conversion_ratio
        self.stock_price = stock_price

    def conversion_value(self):
        """Value if converted now."""
        return self.conversion_ratio * self.stock_price

    def conversion_premium(self, market_price):
        """Premium over conversion value."""
        cv = self.conversion_value()
        return (market_price - cv) / cv
```

### Valuation Framework

```
Convertible Value = max(Straight Bond, Conversion Value) + Option Value
```

---

## Model Calibration

### Calibration to Market

```python
from scipy.optimize import minimize

def calibrate_tree(market_prices, bonds):
    """Calibrate tree parameters to match market."""
    def objective(params):
        r0, u, d, q = params
        tree = BinomialRateTree(r0, u, d, q)

        errors = []
        for price, bond in zip(market_prices, bonds):
            model_price = tree.price_callable_bond(
                bond['face'], bond['coupon'], bond['periods'], {})
            errors.append((model_price - price)**2)

        return sum(errors)

    result = minimize(objective, [0.03, 1.1, 0.9, 0.5],
                     bounds=[(0.01, 0.10), (1.0, 1.5), (0.5, 1.0), (0.3, 0.7)])
    return result.x
```

---

## Documentation Standards

For OAS analysis, document:
- Interest rate model and parameters
- Volatility assumptions
- Number of simulation paths or tree steps
- Calibration methodology
- Validation against market prices

---

## Cross-References

- [Fixed Income Risk](../../03_risk/fixed_income_risk.md)
- [Yield Analysis](../../02_signals/fundamental/yield_analysis.md)
- [Greeks Library](../greeks_library.md)

---

**Template**: KB Skills Integration v1.0
