# Yield Measures and Analysis

**Section**: 02_signals/fundamental
**Last Updated**: 2025-12-12
**Source Skills**: [yield-measures](../../../../.claude/skills/yield-measures/SKILL.md)

---

## Overview

Comprehensive yield metrics for fixed income analysis, including YTM, YTC, YTW, and yield curve interpretation.

---

## Primary Yield Measures

### Current Yield

```python
def current_yield(annual_coupon, market_price):
    """Simple income return measure."""
    return annual_coupon / market_price
```

**Limitations**: Ignores capital gains/losses and time value.

### Yield to Maturity (YTM)

```python
from scipy.optimize import newton

def ytm_calculate(price, face_value, coupon_rate, years, frequency=2):
    """
    Calculate YTM using Newton-Raphson method.

    Parameters:
        price: Current market price (clean)
        face_value: Par value
        coupon_rate: Annual coupon rate (decimal)
        years: Years to maturity
        frequency: Payments per year (2 = semi-annual)

    Returns:
        YTM as annual rate (decimal)
    """
    periods = int(years * frequency)
    coupon = (face_value * coupon_rate) / frequency

    def price_function(y):
        periodic = y / frequency
        pv_coupons = sum(coupon / (1 + periodic)**t
                        for t in range(1, periods + 1))
        pv_principal = face_value / (1 + periodic)**periods
        return pv_coupons + pv_principal - price

    return newton(price_function, coupon_rate, maxiter=100)
```

**Interpretation**: IRR of bond cash flows; assumes coupon reinvestment at YTM.

### Yield to Call (YTC)

```python
def ytc_calculate(price, call_price, coupon_rate, years_to_call, frequency=2):
    """Calculate YTC for callable bonds."""
    periods = int(years_to_call * frequency)
    coupon = (call_price * coupon_rate) / frequency

    def price_function(y):
        periodic = y / frequency
        pv_coupons = sum(coupon / (1 + periodic)**t
                        for t in range(1, periods + 1))
        pv_call = call_price / (1 + periodic)**periods
        return pv_coupons + pv_call - price

    return newton(price_function, coupon_rate, maxiter=100)
```

### Yield to Worst (YTW)

```python
def ytw_calculate(price, face_value, coupon_rate, years, call_schedule, frequency=2):
    """
    Calculate YTW: minimum yield across all scenarios.

    call_schedule: [(years_to_call, call_price), ...]
    """
    yields = []

    # YTM scenario
    ytm = ytm_calculate(price, face_value, coupon_rate, years, frequency)
    yields.append(ytm)

    # YTC scenarios
    for years_to_call, call_price in call_schedule:
        ytc = ytc_calculate(price, call_price, coupon_rate,
                           years_to_call, frequency)
        yields.append(ytc)

    return min(yields)
```

---

## Yield Measure Selection

| Scenario | Preferred Measure | Rationale |
|----------|-------------------|-----------|
| Non-callable bond | YTM | Only redemption scenario |
| Premium callable | YTC or YTW | Call likely |
| Discount callable | YTM | Call unlikely |
| Multiple call dates | YTW | Conservative estimate |

---

## Spot and Forward Rates

### Spot Rate Extraction

```python
def spot_rate_from_zero(price, face_value, maturity):
    """Extract spot rate from zero-coupon bond."""
    return (face_value / price)**(1/maturity) - 1
```

### Forward Rate Calculation

```python
import numpy as np

def forward_rates(spot_rates, maturities):
    """Calculate implied forward rates from spot curve."""
    forwards = []
    for i in range(len(spot_rates) - 1):
        t1, s1 = maturities[i], spot_rates[i]
        t2, s2 = maturities[i + 1], spot_rates[i + 1]

        forward = ((1 + s2)**t2 / (1 + s1)**t1)**(1/(t2-t1)) - 1
        forwards.append(forward)

    return np.array(forwards)
```

---

## Yield Curve Shapes

| Shape | Characteristic | Economic Signal |
|-------|----------------|-----------------|
| Normal | Upward sloping | Economic expansion |
| Inverted | Downward sloping | Recession indicator |
| Flat | Minimal slope | Transition/uncertainty |
| Humped | Peak at intermediate | Mixed expectations |

### Key Spreads

| Spread | Calculation | Significance |
|--------|-------------|--------------|
| 2s10s | 10Y - 2Y Treasury | Economic cycle indicator |
| 3m10Y | 10Y - 3M Treasury | Fed policy gauge |
| TED Spread | LIBOR - T-Bill | Banking stress indicator |

---

## Real vs. Nominal Yield

### Fisher Equation

```python
def real_yield(nominal_yield, inflation_rate):
    """Calculate real yield (exact)."""
    return (1 + nominal_yield) / (1 + inflation_rate) - 1

def breakeven_inflation(nominal_yield, tips_yield):
    """Calculate implied inflation expectation."""
    return nominal_yield - tips_yield
```

### TIPS Analysis

| Scenario | Implication |
|----------|-------------|
| Actual inflation > Breakeven | TIPS outperform nominals |
| Actual inflation < Breakeven | Nominals outperform TIPS |
| Breakeven rising | Inflation expectations increasing |

---

## Reinvestment Risk

### Realized Compound Yield

```python
def realized_compound_yield(coupon, periods, reinvest_rates, principal_gain):
    """Calculate actual yield accounting for reinvestment."""
    future_value = 0
    for i, rate in enumerate(reinvest_rates):
        compound_periods = periods - i - 1
        future_value += coupon * (1 + rate)**compound_periods

    future_value += principal_gain
    total_return = future_value / (coupon * periods)
    return total_return**(1/periods) - 1
```

### Reinvestment Impact

| Rate Environment | Effect on Realized Yield |
|------------------|--------------------------|
| Rising rates | Realized > YTM |
| Falling rates | Realized < YTM |
| Stable rates | Realized = YTM |

---

## Yield Curve Visualization

```python
import matplotlib.pyplot as plt

def plot_yield_curve(maturities, yields, title='Yield Curve'):
    """Visualize yield curve."""
    plt.figure(figsize=(10, 6))
    plt.plot(maturities, yields, 'b-o', linewidth=2)
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Yield (%)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    return plt
```

---

## Practice Examples

**Example 1**: Bond pricing
- Face: $1,000, Coupon: 6%, 10 years, Price: $950
- Current Yield: 6.32% (60/950)
- YTM: ~6.7% (premium over coupon rate)

**Example 2**: Callable bond
- Trading at $1,080, callable in 3 years at $1,050
- 7% coupon, 8 years to maturity
- YTM: ~5.8%, YTC: ~5.2%, YTW: 5.2%

---

## Cross-References

- [Fixed Income Analysis](fixed_income_analysis.md)
- [Fixed Income Risk](../../03_risk/fixed_income_risk.md)
- [Bond Pricing Skill](../../../../.claude/skills/bond-pricing/SKILL.md)

---

**Template**: KB Skills Integration v1.0
