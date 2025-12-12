# Duration and Convexity - Reference Materials

## Primary Sources

### 1. CFA Institute
- **Reading**: Understanding Fixed-Income Risk and Return
- **Level**: I & II
- **Key Topics**:
  - Macaulay Duration vs. Modified Duration
  - Effective Duration for bonds with embedded options
  - Convexity and convexity adjustment
  - Duration-convexity approximation formula
  - Portfolio immunization strategies

### 2. Fabozzi - Bond Markets
- **Chapters**: 4, 7
- **Focus**: Price volatility characteristics, duration measures
- **Key Sections**:
  - Duration as weighted average time to cash flows
  - Modified duration as price sensitivity measure
  - Convexity as second-order price sensitivity

### 3. Academic References
- Bierwag, G.O. (1987). *Duration Analysis*
- Fisher, L., and Weil, R.L. (1971). "Coping with the Risk of Interest Rate Fluctuations"

## Key Formulas

### Macaulay Duration
```
D_Mac = Σ [t × PV(CFt)] / P

Where:
  t = Time period
  PV(CFt) = Present value of cash flow at time t
  P = Bond price
```

### Modified Duration
```
D_Mod = D_Mac / (1 + y/k)

Where:
  y = Yield to maturity
  k = Compounding frequency per year
```

### Price Change Approximation
```
ΔP/P ≈ -D_Mod × Δy + (1/2) × C × (Δy)²

Where:
  D_Mod = Modified duration
  C = Convexity
  Δy = Change in yield
```

### Convexity Formula
```
C = [1/P × (1+y)²] × Σ [t(t+1) × CFt / (1+y)^t]
```

## Portfolio Applications

### Dollar Duration
```
DD = D_Mod × P × 0.0001

Measures price change for 1 basis point yield change
```

### DV01 (Dollar Value of 01)
```
DV01 = D_Mod × P / 10,000

Price change per $100 face for 1bp yield change
```

## Python Implementation Notes

**Libraries**:
- `numpy`: Matrix calculations for portfolio duration
- `pandas`: Managing multi-bond portfolios
- `scipy.optimize`: Finding immunization targets

**Key Functions to Implement**:
- `macaulay_duration(cashflows, yields, times)`
- `modified_duration(macaulay_dur, yield_rate, freq)`
- `convexity(cashflows, yields, times, price)`
- `price_change_estimate(duration, convexity, yield_change)`

---

**Status**: Reference placeholder  
**Last Updated**: 2025-12-07
