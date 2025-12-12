# Option-Adjusted Spread (OAS) and Embedded Option Valuation

**Analyze and adjust for embedded options to determine fair yield spreads and pricing efficiency**

## Quick Reference

**Skill Level**: Advanced  
**Time Commitment**: 20-25 hours  
**Prerequisites**: Bond Pricing, Yield Measures, Duration/Convexity

## Core Capabilities

- Price callable and puttable bonds with embedded options
- Compute Option-Adjusted Spread (OAS) using Monte Carlo simulation
- Implement binomial tree and lattice models for option valuation
- Compare OAS vs. Z-spread vs. nominal spread
- Perform sensitivity analysis to volatility and optionality assumptions

## Key Concepts

**OAS Definition**: Spread added to spot curve that equates model price to market price, adjusting for embedded option value.

```
P_market = P_model(OAS)
OAS = Spread that makes model price equal market price
```

**Option Value Decomposition**:
```
Value of Call Option = P_straight - P_callable
Value of Put Option = P_puttable - P_straight
```

## Python Quick Start

```python
def oas_binomial(market_price, face_value, coupon_rate, 
                 maturity, rate_vol, call_price=None):
    """
    Calculate OAS using binomial lattice for callable bond.
    
    Returns spread that equates model price to market price.
    """
    # Implementation uses iterative search over spread values
    # to find OAS where model price matches market price
    pass  # See full implementation in scripts/
```

## Deliverables

- `oas_model.ipynb`: Binomial lattice implementation for callable bonds
- `oas_analysis.md`: Case study comparing callable vs. non-callable structures

## Essential References

- Fabozzi: *Bond Markets, Analysis, and Strategies* (Ch. 17-18)
- CFA Institute: Valuation of Bonds with Embedded Options
- Bloomberg: OAS Analytics Guide
- Hull: *Options, Futures, and Other Derivatives* (Interest Rate Trees)

## Validation Checkpoint

Can you explain why OAS is typically lower than nominal spread for callable bonds?  
**Expected Answer**: Because the call option has value to the issuer, reducing the effective spread investor receives after adjusting for optionality.

---

**Full Documentation**: See `SKILL.md`  
**Scripts**: See `scripts/` directory  
**References**: See `references/` directory
