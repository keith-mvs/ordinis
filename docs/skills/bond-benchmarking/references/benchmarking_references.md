# Bond Benchmarking - Reference Materials

## Primary Sources

### 1. Bloomberg Indices
- **Bloomberg Barclays U.S. Aggregate Bond Index**: Broad U.S. investment-grade bonds
- **Bloomberg Barclays U.S. Corporate Index**: Investment-grade corporate bonds
- **Bloomberg Barclays High Yield Index**: Below investment-grade corporates

**Documentation**: https://www.bloomberg.com/professional/product/indices/

### 2. ICE BofA Indices
- **ICE BofA US Corporate Index**: Tracks USD corporate debt
- **ICE BofA US High Yield Index**: Sub-investment grade
- **Sector-Specific Indices**: Energy, Financials, Industrials, etc.

**Access**: Via Bloomberg Terminal or ICE Data Services

### 3. S&P Dow Jones Indices
- **S&P U.S. Aggregate Bond Index**
- **S&P/LSTA Leveraged Loan Index**

### 4. U.S. Treasury Benchmarks
- **FRED Data**: Treasury yield curves
- **Series**:
  - `DGS2`, `DGS5`, `DGS10`, `DGS30`: Constant maturity yields
  - `DFII10`: 10-Year TIPS (real yield benchmark)

## Key Benchmarking Metrics

### Spread to Benchmark
```
Nominal Spread = YTM_Bond - YTM_Benchmark

Z-Spread = Spread added to spot curve that equates PV to price

OAS = Option-Adjusted Spread (for embedded options)
```

### Relative Value Analysis
```
Relative Yield = (YTM_Bond / YTM_Benchmark) - 1

Excess Return = Return_Bond - Return_Benchmark
```

### Tracking Error
```
TE = σ(Return_Portfolio - Return_Benchmark)

Measures volatility of excess returns
```

### Information Ratio
```
IR = (Return_Portfolio - Return_Benchmark) / Tracking_Error

Risk-adjusted excess return
```

## Peer Group Construction

### Selection Criteria
1. **Sector**: Same industry classification (e.g., Energy, Finance)
2. **Rating**: Same credit rating tier (e.g., BBB, A)
3. **Maturity**: Similar duration bucket (e.g., 5-10 years)
4. **Size**: Comparable issue size for liquidity

### Percentile Analysis
```python
def percentile_ranking(bond_metric, peer_metrics):
    """
    Calculate percentile ranking within peer group.

    Returns:
        Percentile (0-100) where higher is better for yield
    """
    return (sum(bond_metric >= p for p in peer_metrics) / len(peer_metrics)) * 100
```

## Performance Attribution

### Decomposition Framework
```
Total Return = Income Return + Price Return
             = Coupon Yield + (Price_end - Price_start)/Price_start

Excess Return vs Benchmark:
  = Duration Effect + Curve Effect + Spread Effect + Residual
```

## Data Sources

### Real-Time
- **Bloomberg Terminal**: CBBT (Corporate Bond Ticker)
- **TRACE**: FINRA bond trade reporting
- **ICE Data Services**: Evaluated pricing

### Historical
- **FRED**: Treasury historical yields
- **Bloomberg API**: Index and constituent data
- **Morningstar**: Fund benchmarking data

## CFA Institute Framework

**Reading**: Relative-Value Methodologies for Global Credit Bond Portfolio Management

**Key Concepts**:
- Bottom-up vs. top-down approach
- Benchmark-aware vs. benchmark-agnostic strategies
- Active share and tracking error budgets

## Python Implementation

### Key Libraries
```python
import pandas as pd
import numpy as np
from scipy import stats

# For API access
import yfinance as yf
from fredapi import Fred
```

### Example: Calculate Spread to Treasury
```python
def spread_to_treasury(bond_ytm, maturity_years, treasury_curve):
    """
    Calculate spread to interpolated Treasury yield.

    Parameters:
        bond_ytm: Bond yield to maturity
        maturity_years: Bond maturity
        treasury_curve: DataFrame with maturities and yields

    Returns:
        Spread in basis points
    """
    # Interpolate Treasury yield at bond's maturity
    from scipy.interpolate import interp1d

    f = interp1d(treasury_curve['maturity'], treasury_curve['yield'])
    treasury_ytm = f(maturity_years)

    spread_bps = (bond_ytm - treasury_ytm) * 10000
    return spread_bps
```

---

**Status**: Reference placeholder
**Last Updated**: 2025-12-07
