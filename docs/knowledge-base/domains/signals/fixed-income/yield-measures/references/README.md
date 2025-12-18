# Yield Measures and Return Analysis

**Comprehensive yield metric calculation and interpretation for fixed-income risk/return assessment**

## Quick Reference

**Skill Level**: Intermediate
**Time Commitment**: 12-15 hours
**Prerequisites**: Bond Pricing and Valuation

## Core Capabilities

- Calculate Current Yield, YTM, YTC, YTW
- Derive spot and forward rates
- Interpret yield curve shapes (normal, inverted, flat)
- Adjust for inflation (real vs. nominal yields)
- Analyze TIPS and breakeven inflation

## Key Formulas

```
Current Yield = Annual Coupon / Market Price

YTM: Price = Σ[Coupon/(1+YTM)^t] + FV/(1+YTM)^n

Forward Rate: f_t = [(1+S_t+1)^(t+1) / (1+S_t)^t] - 1

Real Yield ≈ Nominal Yield - Inflation Rate
```

## Python Quick Start

```python
from scipy.optimize import newton

def ytm_calculate(price, face_value, coupon_rate, years, freq=2):
    periods = int(years * freq)
    coupon = (face_value * coupon_rate) / freq

    def price_func(y):
        py = y / freq
        return sum(coupon/(1+py)**t for t in range(1,periods+1)) \
               + face_value/(1+py)**periods - price

    return newton(price_func, coupon_rate)
```

## Deliverables

- `yield_measures_calculator.py`: Complete yield calculation module
- `yield_curves_visualization.md`: Curve analysis and interpretation

## Essential References

- Fabozzi: *Bond Markets* (Ch. 3, 5)
- Federal Reserve FRED: Treasury yield curves
- U.S. Treasury: Daily yield curve rates

## Validation Checkpoint

Can you calculate YTM for a bond at $950, 6% coupon, 10 years to maturity?
**Expected Answer**: ~6.75% (approximately)

---

**Full Documentation**: See `SKILL.md`
