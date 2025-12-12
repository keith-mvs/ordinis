---
name: yield-measures
description: Calculate comprehensive bond yield measures including current yield, yield-to-maturity, yield-to-call, and yield-to-worst. Handles various compounding conventions and day-count methods. Requires numpy>=1.24.0, pandas>=2.0.0, scipy>=1.10.0. Use when comparing bond returns, evaluating callable bonds, or analyzing yield curve relationships.
---

# Yield Measures and Return Analysis Skill Development

## Objective

Develop comprehensive understanding of yield metrics and their application to risk/return assessment. Master the calculation and interpretation of multiple yield measures for accurate bond performance evaluation and portfolio-level return forecasting.

## Skill Classification

**Domain**: Fixed Income Analytics  
**Level**: Intermediate  
**Prerequisites**: Bond Pricing and Valuation  
**Estimated Time**: 12-15 hours

## Focus Areas

### 1. Current Yield

**Definition**: Annual coupon income as percentage of current market price.

**Formula**:
```
Current Yield = (Annual Coupon Payment / Current Market Price) × 100
```

**Characteristics**:
- Simple, quick assessment of income return
- Ignores time value of money
- Does not account for capital gains/losses
- Useful for income-focused investors

**Python Implementation**:
```python
def current_yield(coupon_payment, market_price):
    """
    Calculate current yield.
    
    Parameters:
    -----------
    coupon_payment : float
        Annual coupon payment
    market_price : float
        Current market price of bond
    
    Returns:
    --------
    float : Current yield as decimal
    """
    return coupon_payment / market_price
```

### 2. Yield to Maturity (YTM)

**Definition**: Total return anticipated if bond held until maturity, assuming all coupons reinvested at YTM.

**Characteristics**:
- Most comprehensive yield measure
- Internal rate of return (IRR) of bond
- Accounts for coupon income, capital gain/loss, time value
- Market standard for bond comparison

**Calculation Method** (Iterative):
```python
from scipy.optimize import newton

def ytm_calculate(price, face_value, coupon_rate, years_to_maturity, frequency=2):
    """
    Calculate Yield to Maturity using Newton-Raphson method.
    
    Parameters:
    -----------
    price : float
        Current market price (clean)
    face_value : float
        Par value
    coupon_rate : float
        Annual coupon rate (decimal)
    years_to_maturity : float
        Years until maturity
    frequency : int
        Coupon payments per year (default: 2 for semi-annual)
    
    Returns:
    --------
    float : YTM as annual rate (decimal)
    """
    periods = int(years_to_maturity * frequency)
    coupon = (face_value * coupon_rate) / frequency
    
    def price_function(y):
        periodic_yield = y / frequency
        pv_coupons = sum(coupon / (1 + periodic_yield)**t 
                        for t in range(1, periods + 1))
        pv_principal = face_value / (1 + periodic_yield)**periods
        return pv_coupons + pv_principal - price
    
    # Initial guess: coupon rate
    ytm = newton(price_function, coupon_rate, maxiter=100)
    return ytm
```

### 3. Yield to Call (YTC)

**Definition**: Total return if bond called at first call date.

**Key Considerations**:
- Relevant for callable bonds only
- Assumes issuer exercises call option
- Call price may differ from par value
- Important when bond trades above call price

**Calculation**:
```python
def ytc_calculate(price, call_price, coupon_rate, years_to_call, frequency=2):
    """
    Calculate Yield to Call.
    
    Parameters:
    -----------
    price : float
        Current market price
    call_price : float
        Price at which bond can be called
    coupon_rate : float
        Annual coupon rate (decimal)
    years_to_call : float
        Years until first call date
    frequency : int
        Coupon payments per year
    
    Returns:
    --------
    float : YTC as annual rate (decimal)
    """
    periods = int(years_to_call * frequency)
    coupon = (call_price * coupon_rate) / frequency
    
    def price_function(y):
        periodic_yield = y / frequency
        pv_coupons = sum(coupon / (1 + periodic_yield)**t 
                        for t in range(1, periods + 1))
        pv_call = call_price / (1 + periodic_yield)**periods
        return pv_coupons + pv_call - price
    
    ytc = newton(price_function, coupon_rate, maxiter=100)
    return ytc
```

### 4. Yield to Worst (YTW)

**Definition**: Lowest yield among all possible call dates and maturity.

**Methodology**:
```python
def ytw_calculate(price, face_value, coupon_rate, years_to_maturity, 
                  call_schedule, frequency=2):
    """
    Calculate Yield to Worst.
    
    Parameters:
    -----------
    price : float
        Current market price
    face_value : float
        Par value
    coupon_rate : float
        Annual coupon rate
    years_to_maturity : float
        Years to maturity
    call_schedule : list of tuples
        [(years_to_call, call_price), ...]
    frequency : int
        Coupon frequency
    
    Returns:
    --------
    float : YTW (minimum yield across all scenarios)
    """
    yields = []
    
    # Calculate YTM
    ytm = ytm_calculate(price, face_value, coupon_rate, 
                       years_to_maturity, frequency)
    yields.append(ytm)
    
    # Calculate YTC for each call date
    for years_to_call, call_price in call_schedule:
        ytc = ytc_calculate(price, call_price, coupon_rate, 
                           years_to_call, frequency)
        yields.append(ytc)
    
    return min(yields)
```

### 5. Spot Rates vs. Forward Rates

#### Spot Rates
Zero-coupon yield for specific maturity.

**Formula**:
```
S_t = [(FV / P_t)^(1/t)] - 1
```

#### Forward Rates
Implied future spot rates derived from current term structure.

**Formula** (1-year forward rate, t years from now):
```
f_t = [(1 + S_(t+1))^(t+1) / (1 + S_t)^t] - 1
```

**Python Implementation**:
```python
def calculate_forward_rates(spot_rates, maturities):
    """
    Calculate implied forward rates from spot curve.
    
    Parameters:
    -----------
    spot_rates : array-like
        Spot rates for each maturity
    maturities : array-like
        Corresponding maturities (years)
    
    Returns:
    --------
    array : Forward rates
    """
    forward_rates = []
    for i in range(len(spot_rates) - 1):
        t1, s1 = maturities[i], spot_rates[i]
        t2, s2 = maturities[i + 1], spot_rates[i + 1]
        
        forward = ((1 + s2)**(t2) / (1 + s1)**(t1))**(1/(t2-t1)) - 1
        forward_rates.append(forward)
    
    return np.array(forward_rates)
```

### 6. Coupon Reinvestment Assumptions

**Key Principle**: YTM assumes all coupons reinvested at the YTM rate.

**Realized Compound Yield**:
```python
def realized_compound_yield(coupon, periods, reinvestment_rates, 
                           principal_gain):
    """
    Calculate actual realized yield accounting for reinvestment.
    
    Parameters:
    -----------
    coupon : float
        Periodic coupon payment
    periods : int
        Number of periods held
    reinvestment_rates : array-like
        Actual reinvestment rates for each coupon
    principal_gain : float
        Capital gain/loss at sale or maturity
    
    Returns:
    --------
    float : Realized compound yield
    """
    future_value = 0
    for i, rate in enumerate(reinvestment_rates):
        periods_to_compound = periods - i - 1
        future_value += coupon * (1 + rate)**periods_to_compound
    
    future_value += principal_gain
    
    # Calculate annualized yield
    total_return = future_value / (coupon * periods)
    return (total_return)**(1/periods) - 1
```

### 7. Yield Curve Interpretation

#### Normal Yield Curve
- Upward sloping
- Long-term yields > short-term yields
- Indicates economic expansion expectations

#### Inverted Yield Curve
- Downward sloping
- Short-term yields > long-term yields
- Often precedes recession

#### Flat Yield Curve
- Minimal difference across maturities
- Transition period or uncertainty

**Visualization**:
```python
import matplotlib.pyplot as plt

def plot_yield_curve(maturities, yields, curve_date):
    """
    Visualize yield curve.
    
    Parameters:
    -----------
    maturities : array-like
        Bond maturities (years)
    yields : array-like
        Corresponding yields
    curve_date : str
        Date of yield curve
    """
    plt.figure(figsize=(10, 6))
    plt.plot(maturities, yields, 'b-o', linewidth=2)
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Yield (%)')
    plt.title(f'Yield Curve - {curve_date}')
    plt.grid(True, alpha=0.3)
    plt.show()
```

### 8. Real vs. Nominal Yield

#### Nominal Yield
Stated yield without inflation adjustment.

#### Real Yield
Inflation-adjusted return.

**Fisher Equation**:
```
(1 + nominal_rate) = (1 + real_rate) × (1 + inflation_rate)

Approximation: real_rate ≈ nominal_rate - inflation_rate
```

#### TIPS (Treasury Inflation-Protected Securities)
- Principal adjusts with CPI
- Real yield remains constant
- Provides inflation hedge

**Python Implementation**:
```python
def real_yield_calculator(nominal_yield, inflation_rate):
    """
    Calculate real yield from nominal yield and inflation.
    
    Parameters:
    -----------
    nominal_yield : float
        Nominal yield (decimal)
    inflation_rate : float
        Expected inflation rate (decimal)
    
    Returns:
    --------
    float : Real yield (exact calculation)
    """
    return (1 + nominal_yield) / (1 + inflation_rate) - 1

def tips_breakeven_inflation(nominal_yield, tips_yield):
    """
    Calculate breakeven inflation rate.
    
    Parameters:
    -----------
    nominal_yield : float
        Yield on nominal Treasury
    tips_yield : float
        Yield on comparable TIPS
    
    Returns:
    --------
    float : Implied inflation expectation
    """
    return nominal_yield - tips_yield
```

## Expected Competency

Upon completion, you will be able to:

1. **Calculate all major yield measures**: Current Yield, YTM, YTC, YTW
2. **Interpret yield curves** and understand economic implications
3. **Derive forward rates** from spot rate term structure
4. **Assess reinvestment risk** and calculate realized compound yields
5. **Adjust for inflation** using real yield and TIPS analysis
6. **Select appropriate yield measure** for specific investment scenarios
7. **Implement Python calculators** for automated yield analysis

## Deliverables

### 1. yield_measures_calculator.py
Production-ready Python module containing:
- Current yield, YTM, YTC, YTW calculators
- Spot/forward rate converters
- Real yield calculators
- Comprehensive error handling and validation
- Unit tests for all functions

### 2. yield_curves_visualization.md
Analysis documentation including:
- Historical yield curve comparison
- Normal vs. inverted curve interpretation
- Forward rate implications
- Economic scenario analysis
- Interactive visualizations using matplotlib/plotly

## Reference Materials

### Foundational Texts
1. Fabozzi, F. *Bond Markets, Analysis, and Strategies*
   - Chapter 3: Bond Yield Measures
   - Chapter 5: Factors Affecting Bond Yields

2. CFA Institute. *Fixed Income* (CFA Program Curriculum)
   - Yield and Yield Spread Measures
   - The Term Structure of Interest Rates

### Online Resources
1. **Federal Reserve FRED**: [Treasury Yield Curves](https://fred.stlouisfed.org/categories/115)
2. **U.S. Treasury**: [Daily Treasury Yield Curve Rates](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve)
3. **Investopedia**: [Yield Curve Analysis](https://www.investopedia.com/terms/y/yieldcurve.asp)

### Python Libraries
```bash
pip install numpy scipy pandas matplotlib plotly yfinance
```

## Validation Framework

### Self-Assessment Checklist

- [ ] Calculate YTM for bonds trading at premium, par, and discount
- [ ] Interpret difference between YTM and YTC for callable bonds
- [ ] Derive forward rates from spot curve
- [ ] Understand reinvestment risk impact on realized returns
- [ ] Analyze yield curve shapes and economic implications
- [ ] Calculate real yields and breakeven inflation
- [ ] Select appropriate yield measure for specific scenarios

### Practice Problems

1. **Problem 1**: Bond with $1,000 face value, 6% coupon, 10 years to maturity, trading at $950. Calculate current yield and YTM.

2. **Problem 2**: Callable bond trading at $1,080, callable in 3 years at $1,050, 7% coupon, 8 years to maturity. Calculate YTC and YTW.

3. **Problem 3**: Given spot rates (1yr: 2%, 2yr: 2.5%, 3yr: 3%), calculate 1-year forward rates.

4. **Problem 4**: Nominal Treasury yield 4%, TIPS yield 1.5%. What is breakeven inflation?

## Integration with Other Skills

- **Bond Pricing**: YTM serves as discount rate for valuation
- **Duration/Convexity**: Yield changes drive price sensitivity analysis
- **Credit Risk**: Yield spreads quantify credit risk premium
- **Benchmarking**: Yield comparisons against benchmarks
- **OAS Analysis**: Yield spread decomposition

## Standards and Compliance

- All yield calculations must specify:
  - Calculation date
  - Coupon payment frequency
  - Day count convention
  - Accrued interest treatment
  - Reinvestment assumptions

- Python implementations must include:
  - Input validation (price > 0, coupon_rate >= 0)
  - Convergence checks for iterative methods
  - Documentation of assumptions
  - Comparison with market conventions

## Version Control

**Version**: 1.0.0  
**Last Updated**: 2025-12-07  
**Author**: Ordinis-1 Bond Analysis Framework  
**Status**: Production Ready

## Notes

Yield measures are central to fixed-income analysis. Understanding the differences between yield metrics and their appropriate use cases is critical for accurate bond evaluation and portfolio management.

---

**Previous Skill**: `bond-pricing/`  
**Next Skill**: `duration-convexity/`
