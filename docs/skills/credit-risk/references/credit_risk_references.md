# Credit Risk Assessment - Reference Materials

## Primary Sources

### 1. Credit Rating Agencies

#### Moody's
- **Methodologies**: https://www.moodys.com/researchandratings/methodology
- **Rating Scale**: Aaa to C
- **Key Publications**:
  - "Rating Methodology: Corporate Bonds"
  - "Default and Recovery Rates Studies"
  - "Expected Default Frequency (EDF) Models"

#### S&P Global
- **Rating Criteria**: https://www.spglobal.com/ratings/
- **Rating Scale**: AAA to D
- **Key Publications**:
  - "Corporate Methodology"
  - "Annual Default Study"
  - "Rating Above the Sovereign"

#### Fitch Ratings
- **Rating Scale**: AAA to D
- **Criteria**: https://www.fitchratings.com/
- **Focus**: Financial institutions, structured finance

### 2. Academic Models

#### Merton Model (1974)
**Structural Approach**: Treats equity as call option on firm assets
```
Firm Value = Debt + Equity
Default occurs when V < D at maturity
```

#### Reduced-Form Models
- **Jarrow-Turnbull**: Intensity-based default modeling
- **Duffie-Singleton**: Credit spread decomposition

### 3. CFA Institute
- **Level I**: Introduction to Credit Analysis
- **Level II**: Advanced Credit Analysis and Modeling
- **Topics**:
  - Credit rating process
  - Credit spread analysis
  - Credit derivatives
  - Recovery rate estimation

## Key Metrics

### Probability of Default (PD)
```
PD = 1 - e^(-λt)

Where:
  λ = Default intensity (hazard rate)
  t = Time horizon
```

### Loss Given Default (LGD)
```
LGD = 1 - Recovery Rate
Expected Loss = PD × LGD × Exposure
```

### Credit Spread
```
Credit Spread = YTM_Corporate - YTM_Treasury

Alternative:
OAS = Option-Adjusted Spread (for embedded options)
```

### Z-Score (Altman)
```
Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5

Where:
  X1 = Working Capital / Total Assets
  X2 = Retained Earnings / Total Assets
  X3 = EBIT / Total Assets
  X4 = Market Value Equity / Book Value Debt
  X5 = Sales / Total Assets
```

## Data Sources

### Market Data
- **Bloomberg**: Credit Default Swap (CDS) spreads
- **FINRA TRACE**: Corporate bond trade data
- **ICE BofA Indices**: Credit spread indices

### Financial Statements
- **SEC EDGAR**: 10-K, 10-Q filings
- **Company Websites**: Investor relations sections
- **Bloomberg Terminal**: Fundamental data

### Default and Recovery Data
- **Moody's**: Annual Default Study
- **S&P**: Default & Recovery Database
- **CreditPro**: Historical default and recovery data

## Python Implementation

### Key Libraries
```python
# Financial statement analysis
import pandas as pd
import numpy as np

# Web scraping for EDGAR
from sec_edgar_downloader import Downloader

# Credit modeling
from scipy.stats import norm  # For Merton model
from lifelines import CoxPHFitter  # For survival analysis
```

### Merton Model Implementation
```python
from scipy.stats import norm

def merton_probability_default(V, D, sigma, r, T):
    """
    Calculate default probability using Merton model.

    Parameters:
        V: Firm value
        D: Debt face value
        sigma: Asset volatility
        r: Risk-free rate
        T: Time to maturity

    Returns:
        Probability of default
    """
    d2 = (np.log(V/D) + (r - 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    pd = norm.cdf(-d2)
    return pd
```

---

**Status**: Reference placeholder
**Last Updated**: 2025-12-07
