# Bond Pricing and Valuation

**Master bond pricing fundamentals and valuation techniques for fixed-income securities**

## Quick Reference

**Skill Level**: Foundation
**Time Commitment**: 15-20 hours
**Prerequisites**: Basic financial mathematics

## Core Capabilities

- Price bonds using present value of cash flows
- Calculate Yield to Maturity (YTM) and Yield to Call (YTC)
- Price specialized bonds (zero-coupon, callable, puttable, floating-rate)
- Apply market conventions (clean vs. dirty price, day count)
- Implement Python-based valuation models

## Key Formula

```
Bond Price = Σ [Coupon / (1+y)^t] + [Face Value / (1+y)^n]
```

## Python Quick Start

```python
def bond_price(face_value, coupon_rate, yield_rate, periods):
    coupon = face_value * coupon_rate
    pv_coupons = sum(coupon / (1 + yield_rate)**t
                     for t in range(1, periods + 1))
    pv_principal = face_value / (1 + yield_rate)**periods
    return pv_coupons + pv_principal
```

## Deliverables

- `bond_pricing_model.ipynb`: Interactive pricing simulation
- `pricing_examples.md`: Case studies across bond types

## Essential References

- Fabozzi: *Bond Markets, Analysis, and Strategies* (Ch. 5-7)
- CFA Institute: Fixed Income Valuation readings
- QuantLib-Python documentation

## Validation Checkpoint

Can you price a 10-year, 5% coupon bond with 4.5% YTM?
**Expected Answer**: $1,038.40 (approximately)

---

**Full Documentation**: See `SKILL.md`
**Scripts**: See `scripts/` directory
**References**: See `references/` directory
