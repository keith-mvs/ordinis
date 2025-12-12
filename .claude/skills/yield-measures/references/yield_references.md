# Yield Measures - Reference Materials

## Primary Sources

### 1. CFA Institute
- **Reading**: Yield Measures, Spot Rates, and Forward Rates
- **Level**: I & II
- **Key Topics**:
  - Current Yield vs. YTM vs. YTW
  - Spot rate curves and bootstrapping
  - Forward rates and expectations hypothesis
  - Real vs. nominal yields (TIPS)

### 2. Fabozzi - Bond Markets
- **Chapters**: 6-8
- **Focus**: Yield curve analysis, term structure theories
- **Applications**: Portfolio management using yield measures

### 3. Federal Reserve Economic Data (FRED)
- **URL**: https://fred.stlouisfed.org/
- **Data Series**:
  - `DGS10`: 10-Year Treasury Constant Maturity Rate
  - `DGS2`: 2-Year Treasury Constant Maturity Rate
  - `T10Y2Y`: 10-Year Treasury - 2-Year Treasury Spread
  - `DFII10`: 10-Year TIPS Constant Maturity Rate

## Key Formulas

### Spot Rate Extraction (Bootstrapping)
```
Given par yields, solve iteratively:
P = C/(1+s1) + C/(1+s2)² + ... + (C+FV)/(1+sn)ⁿ = Par

Where si are spot rates to be solved
```

### Forward Rate Calculation
```
(1 + sn)ⁿ = (1 + sm)ᵐ × (1 + fm,n)ⁿ⁻ᵐ

Where:
  fm,n = Forward rate from year m to year n
  sn, sm = Spot rates for year n and m
```

## Python Libraries

### yfinance
```bash
pip install yfinance
```
**Use**: Download Treasury yield data

### pandas-datareader
```bash
pip install pandas-datareader
```
**Use**: Access FRED data directly

---

**Status**: Reference placeholder  
**Last Updated**: 2025-12-07
