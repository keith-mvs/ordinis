# Duration and Convexity Analysis

**Quantify interest rate risk and manage portfolio duration for fixed-income risk management**

## Quick Reference

**Skill Level**: Advanced  
**Time Commitment**: 15-18 hours  
**Prerequisites**: Bond Pricing, Yield Measures

## Core Capabilities

- Calculate Macaulay and Modified Duration
- Compute convexity for non-linear price sensitivity
- Estimate price changes from yield shifts
- Manage portfolio duration through optimization
- Implement immunization strategies
- Use futures/swaps for duration adjustment

## Key Formulas

```
Macaulay Duration = Σ[t × PV(CF_t)] / Price

Modified Duration = Macaulay Duration / (1 + YTM/freq)

ΔP/P ≈ -ModDur × ΔY + ½ × Convexity × (ΔY)²
```

## Python Quick Start

```python
def modified_duration(face_value, coupon_rate, yield_rate, years, freq=2):
    periods = int(years * freq)
    coupon = (face_value * coupon_rate) / freq
    py = yield_rate / freq
    
    weighted_pv = sum(t * (coupon/(1+py)**t) 
                      for t in range(1, periods))
    weighted_pv += periods * (coupon + face_value) / (1+py)**periods
    
    price = sum(coupon/(1+py)**t for t in range(1, periods+1))
    price += face_value / (1+py)**periods
    
    mac_dur = (weighted_pv / price) / freq
    return mac_dur / (1 + yield_rate / freq)
```

## Deliverables

- `duration_convexity.ipynb`: Sensitivity analysis and scenario modeling
- `duration_vs_convexity_visuals.md`: Rate risk visualization

## Essential References

- Fabozzi: *Bond Markets* (Ch. 4)
- Tuckman & Serrat: *Fixed Income Securities* (Ch. 5)
- CFA: Fixed-Income Risk and Return

## Validation Checkpoint

For a 10-year, 5% coupon bond with 4% YTM:  
**Expected Modified Duration**: ~8.11 years

---

**Full Documentation**: See `SKILL.md`
