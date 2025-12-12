# Option-Adjusted Spread (OAS) - Reference Materials

## Primary Sources

### 1. Academic Foundations

#### Fabozzi - Bond Markets
- **Chapters**: 17-18 (Valuation of Bonds with Embedded Options)
- **Key Topics**:
  - Binomial interest rate trees
  - Callable bond valuation
  - Option cost calculation
  - OAS vs. Z-spread comparison

#### Hull - Options, Futures, and Other Derivatives
- **Chapter**: Interest Rate Models
- **Models Covered**:
  - Black-Derman-Toy (BDT) model
  - Ho-Lee model
  - Heath-Jarrow-Morton (HJM) framework

### 2. CFA Institute
- **Level II**: Valuation and Analysis of Bonds with Embedded Options
- **Learning Objectives**:
  - Build binomial interest rate trees
  - Value callable/puttable bonds
  - Calculate and interpret OAS
  - Perform effective duration/convexity for embedded options

### 3. Bloomberg Methodologies
- **Bloomberg Function**: `OAS1` screen
- **Documentation**: "OAS Analytics Guide"
- **Models**:
  - Monte Carlo simulation for MBS
  - Binomial trees for corporate callables
  - Trinomial trees for complex structures

## Key Concepts

### OAS Definition
```
OAS = Constant spread added to each node of interest rate tree
      such that model price equals market price

Price_Market = E[PV(Cash Flows discounted at Spot + OAS)]
```

### Z-Spread vs. OAS
```
Z-Spread = Static spread to spot curve (no optionality)
OAS = Z-Spread - Option Cost

For callable bonds: OAS < Z-Spread
For puttable bonds: OAS > Z-Spread
```

### Option Cost
```
Option Cost (callable) = Z-Spread - OAS
                       = Value of call option to issuer
                       
Price_Straight = Price_Callable + Call_Option_Value
```

## Binomial Interest Rate Tree

### Construction Steps
1. Calibrate to current term structure
2. Specify volatility assumptions
3. Generate up/down interest rate paths
4. Ensure tree is arbitrage-free (matches spot curve)

### Tree Notation
```
      r_uu
     /
  r_u
 /   \
r      r_ud
 \   /
  r_d
     \
      r_dd

Where:
  r_u = r × e^(2σ√Δt)
  r_d = r × e^(-2σ√Δt)
  σ = Interest rate volatility
```

## Monte Carlo Simulation

### Application
- Mortgage-Backed Securities (MBS)
- Complex path-dependent securities
- Multiple embedded options

### Algorithm
```
1. Generate N interest rate paths
2. For each path:
   a. Determine cash flows (considering prepayment/call)
   b. Discount at (spot + OAS)
   c. Calculate present value
3. Average across all paths
4. Iterate OAS until Price_Model = Price_Market
```

## Python Implementation

### Key Libraries
```python
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# For binomial trees
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
```

### Binomial Tree Template
```python
class BinomialTree:
    """
    Interest rate tree for option-embedded bond valuation.
    """
    def __init__(self, spot_curve, volatility, periods):
        self.spot_curve = spot_curve
        self.volatility = volatility
        self.periods = periods
        self.tree = self.build_tree()
    
    def build_tree(self):
        """Construct calibrated binomial tree."""
        # TODO: Implement tree calibration
        pass
    
    def value_bond(self, cashflows, call_schedule, oas=0):
        """
        Value callable bond using backward induction.
        
        Parameters:
            cashflows: Array of coupon payments
            call_schedule: Dict of {period: call_price}
            oas: Option-adjusted spread (bps)
        
        Returns:
            Bond value at time 0
        """
        # TODO: Implement backward induction
        pass
```

### OAS Calculator Template
```python
def calculate_oas(market_price, bond_params, rate_tree, call_schedule):
    """
    Calculate OAS using binary search.
    
    Finds OAS where model price equals market price.
    
    Parameters:
        market_price: Observed market price
        bond_params: Coupon, maturity, etc.
        rate_tree: Calibrated interest rate tree
        call_schedule: Call dates and prices
    
    Returns:
        OAS in basis points
    """
    def price_diff(oas):
        model_price = rate_tree.value_bond(
            bond_params.cashflows,
            call_schedule,
            oas
        )
        return model_price - market_price
    
    # Binary search for OAS
    oas = brentq(price_diff, -500, 500)  # Search -500 to +500 bps
    return oas
```

## Volatility Assumptions

### Historical Volatility
```python
def calculate_rate_volatility(yield_changes):
    """
    Calculate historical interest rate volatility.
    
    Parameters:
        yield_changes: Time series of yield changes
    
    Returns:
        Annualized volatility
    """
    daily_vol = np.std(yield_changes)
    annual_vol = daily_vol * np.sqrt(252)  # 252 trading days
    return annual_vol
```

### Implied Volatility
- Extract from swaption markets
- Use cap/floor prices
- Bloomberg: `VCUB` screen (Volatility Cube)

## Advanced Topics

### Effective Duration/Convexity
```
Effective Duration = (P- - P+) / (2 × P0 × Δy)
Effective Convexity = (P+ + P- - 2×P0) / (P0 × (Δy)²)

Where:
  P+, P- = Prices with yield shifts
  P0 = Initial price
  Δy = Yield change
```

### Key Rate Duration
- Sensitivity to specific points on yield curve
- More granular than effective duration
- Used for hedging callable bonds

## Validation Approaches

1. **Compare to Bloomberg OAS**: Validate implementation
2. **Z-Spread Cross-Check**: OAS should be less than Z-spread for callables
3. **Option Cost Reasonableness**: Typically 10-100 bps for investment-grade
4. **Monotonicity Check**: OAS should decrease as volatility increases (for callables)

## Data Sources

- **Bloomberg**: `OAS1`, `VCUB` (volatility)
- **FRED**: Treasury yields for spot curve
- **ICE**: Swaption volatility surfaces

---

**Status**: Reference placeholder  
**Last Updated**: 2025-12-07
