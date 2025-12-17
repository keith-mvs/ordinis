---
name: duration-convexity
description: Calculate bond duration, modified duration, and convexity for interest rate risk management. Implements Macaulay duration, effective duration, and dollar duration metrics. Requires numpy>=1.24.0, pandas>=2.0.0. Use when hedging interest rate exposure, immunizing portfolios, or analyzing price sensitivity to rate changes.
---

# Duration and Convexity Analysis Skill Development

## Objective

Quantify bond price sensitivity to interest rate changes using duration and convexity measures. Master portfolio immunization strategies and interest rate risk management through systematic application of duration-based hedging techniques.

## Skill Classification

**Domain**: Fixed Income Risk Management
**Level**: Advanced
**Prerequisites**: Bond Pricing, Yield Measures
**Estimated Time**: 15-18 hours

## Focus Areas

### 1. Macaulay Duration

**Definition**: Weighted average time to receive bond's cash flows.

**Formula**:
```
Macaulay Duration = Σ[t × PV(CF_t)] / Bond Price

Where:
  t = Time period
  PV(CF_t) = Present value of cash flow at time t
```

**Python Implementation**:
```python
def macaulay_duration(face_value, coupon_rate, yield_rate, years, frequency=2):
    """
    Calculate Macaulay Duration.

    Parameters:
    -----------
    face_value : float
        Par value
    coupon_rate : float
        Annual coupon rate (decimal)
    yield_rate : float
        Yield to maturity (decimal)
    years : float
        Years to maturity
    frequency : int
        Coupon frequency per year

    Returns:
    --------
    float : Macaulay duration in years
    """
    periods = int(years * frequency)
    coupon = (face_value * coupon_rate) / frequency
    periodic_yield = yield_rate / frequency

    # Calculate weighted present values
    weighted_pv = 0
    bond_price = 0

    for t in range(1, periods + 1):
        cf = coupon if t < periods else coupon + face_value
        pv = cf / (1 + periodic_yield)**t
        weighted_pv += t * pv
        bond_price += pv

    # Convert to years
    mac_duration = (weighted_pv / bond_price) / frequency
    return mac_duration
```

**Key Properties**:
- Zero-coupon bond duration = maturity
- Lower coupon bonds have higher duration
- Higher yield → lower duration
- Longer maturity → higher duration (but not linear)

### 2. Modified Duration

**Definition**: Percentage price change for 1% yield change.

**Formula**:
```
Modified Duration = Macaulay Duration / (1 + YTM/frequency)
```

**Price Change Approximation**:
```
ΔP/P ≈ -Modified Duration × ΔY

Where:
  ΔP = Change in price
  P = Initial price
  ΔY = Change in yield
```

**Python Implementation**:
```python
def modified_duration(macaulay_dur, yield_rate, frequency=2):
    """
    Calculate Modified Duration.

    Parameters:
    -----------
    macaulay_dur : float
        Macaulay duration (years)
    yield_rate : float
        Yield to maturity (decimal)
    frequency : int
        Compounding frequency

    Returns:
    --------
    float : Modified duration
    """
    return macaulay_dur / (1 + yield_rate / frequency)

def price_change_estimate(modified_dur, yield_change):
    """
    Estimate percentage price change using modified duration.

    Parameters:
    -----------
    modified_dur : float
        Modified duration
    yield_change : float
        Change in yield (decimal)

    Returns:
    --------
    float : Estimated percentage price change
    """
    return -modified_dur * yield_change
```

### 3. Convexity

**Definition**: Curvature in price-yield relationship; measures duration estimation error.

**Formula**:
```
Convexity = Σ[t(t+1) × PV(CF_t)] / [P × (1+y)²]
```

**Price Change with Convexity**:
```
ΔP/P ≈ -ModDur × ΔY + ½ × Convexity × (ΔY)²
```

**Python Implementation**:
```python
def convexity_calculate(face_value, coupon_rate, yield_rate, years, frequency=2):
    """
    Calculate convexity.

    Parameters:
    -----------
    face_value : float
        Par value
    coupon_rate : float
        Annual coupon rate
    yield_rate : float
        YTM
    years : float
        Years to maturity
    frequency : int
        Coupon frequency

    Returns:
    --------
    float : Convexity
    """
    periods = int(years * frequency)
    coupon = (face_value * coupon_rate) / frequency
    periodic_yield = yield_rate / frequency

    # Calculate weighted second derivatives
    weighted_pv2 = 0
    bond_price = 0

    for t in range(1, periods + 1):
        cf = coupon if t < periods else coupon + face_value
        pv = cf / (1 + periodic_yield)**t
        weighted_pv2 += t * (t + 1) * pv
        bond_price += pv

    convexity = weighted_pv2 / (bond_price * (1 + periodic_yield)**2)
    # Adjust for annual basis
    return convexity / frequency**2

def price_change_with_convexity(modified_dur, convexity, yield_change):
    """
    Estimate price change including convexity adjustment.

    Returns:
    --------
    float : Estimated percentage price change
    """
    duration_effect = -modified_dur * yield_change
    convexity_effect = 0.5 * convexity * yield_change**2
    return duration_effect + convexity_effect
```

**Positive Convexity Benefits**:
- Price increases more when yields fall
- Price decreases less when yields rise
- Asymmetric gain/loss profile favors investor

### 4. Portfolio Duration Management

**Portfolio Duration**:
```
Portfolio Duration = Σ[w_i × Duration_i]

Where:
  w_i = Weight of bond i in portfolio
```

**Python Implementation**:
```python
def portfolio_duration(durations, weights):
    """
    Calculate portfolio duration.

    Parameters:
    -----------
    durations : array-like
        Modified durations of individual bonds
    weights : array-like
        Portfolio weights (must sum to 1)

    Returns:
    --------
    float : Portfolio duration
    """
    return np.dot(durations, weights)

def portfolio_convexity(convexities, weights):
    """
    Calculate portfolio convexity.
    """
    return np.dot(convexities, weights)
```

**Duration Targeting**:
```python
def duration_gap(portfolio_dur, target_dur):
    """
    Calculate duration gap for immunization.

    Returns:
    --------
    float : Duration gap (positive = extend, negative = shorten)
    """
    return target_dur - portfolio_dur

def rebalance_weights(current_weights, durations, target_dur):
    """
    Optimize portfolio weights to achieve target duration.

    Uses quadratic programming to minimize tracking error
    while matching target duration.
    """
    from scipy.optimize import minimize

    def objective(w):
        # Minimize deviation from current weights
        return np.sum((w - current_weights)**2)

    def duration_constraint(w):
        # Portfolio duration must equal target
        return np.dot(w, durations) - target_dur

    constraints = [
        {'type': 'eq', 'fun': duration_constraint},
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Sum to 1
    ]

    bounds = [(0, 1) for _ in current_weights]

    result = minimize(objective, current_weights, method='SLSQP',
                     bounds=bounds, constraints=constraints)

    return result.x
```

### 5. Immunization Strategies

**Classical Immunization**: Match asset duration to liability duration.

**Conditions for Immunization**:
1. PV(Assets) = PV(Liabilities)
2. Duration(Assets) = Duration(Liabilities)
3. Convexity(Assets) > Convexity(Liabilities)

**Python Framework**:
```python
class ImmunizationPortfolio:
    """
    Portfolio immunization framework.
    """

    def __init__(self, liability_value, liability_duration,
                 liability_convexity, horizon):
        self.liability_value = liability_value
        self.liability_duration = liability_duration
        self.liability_convexity = liability_convexity
        self.horizon = horizon

    def check_immunization(self, asset_value, asset_duration,
                          asset_convexity):
        """
        Verify immunization conditions.

        Returns:
        --------
        dict : Immunization status for each condition
        """
        conditions = {
            'value_matched': abs(asset_value - self.liability_value) < 0.01,
            'duration_matched': abs(asset_duration - self.liability_duration) < 0.1,
            'convexity_adequate': asset_convexity > self.liability_convexity
        }

        conditions['fully_immunized'] = all(conditions.values())
        return conditions

    def rebalancing_needed(self, current_duration, tolerance=0.25):
        """
        Determine if rebalancing required.

        Parameters:
        -----------
        tolerance : float
            Duration deviation tolerance (years)

        Returns:
        --------
        bool : True if rebalancing needed
        """
        return abs(current_duration - self.liability_duration) > tolerance
```

### 6. Duration Targeting with Futures/Swaps

**Futures-Based Duration Adjustment**:
```
Number of Contracts = (Target Duration - Portfolio Duration) × Portfolio Value
                      ─────────────────────────────────────────────────────────
                      Futures Duration × Futures Price × Contract Multiplier
```

**Python Implementation**:
```python
def futures_hedge_ratio(portfolio_value, portfolio_duration,
                       target_duration, futures_duration,
                       futures_price, contract_multiplier=1000):
    """
    Calculate number of futures contracts for duration adjustment.

    Parameters:
    -----------
    portfolio_value : float
        Current portfolio value
    portfolio_duration : float
        Current portfolio duration
    target_duration : float
        Desired portfolio duration
    futures_duration : float
        Duration of futures contract
    futures_price : float
        Current futures price
    contract_multiplier : float
        Contract size multiplier

    Returns:
    --------
    int : Number of contracts (positive = long, negative = short)
    """
    duration_gap = target_duration - portfolio_duration
    numerator = duration_gap * portfolio_value
    denominator = futures_duration * futures_price * contract_multiplier

    contracts = numerator / denominator
    return round(contracts)
```

**Interest Rate Swaps**:
```python
def swap_duration_adjustment(portfolio_value, portfolio_duration,
                             target_duration, swap_duration):
    """
    Calculate notional value of interest rate swap for duration targeting.

    Returns:
    --------
    float : Notional swap value
    """
    duration_gap = target_duration - portfolio_duration
    swap_notional = (duration_gap * portfolio_value) / swap_duration
    return swap_notional
```

## Expected Competency

Upon completion, you will be able to:

1. **Calculate Macaulay and Modified Duration** for individual bonds
2. **Compute Convexity** and understand its risk/return implications
3. **Estimate price changes** using duration and convexity
4. **Manage portfolio duration** through weight optimization
5. **Implement immunization strategies** for liability matching
6. **Use derivatives** (futures, swaps) for duration adjustment
7. **Assess interest rate risk** through scenario analysis

## Deliverables

### 1. duration_convexity.ipynb
Interactive Jupyter notebook containing:
- Duration/convexity calculators
- Price sensitivity simulation
- Portfolio duration optimization
- Immunization analysis
- Futures/swap hedging models
- Scenario analysis (parallel shifts, twists, butterfly)

### 2. duration_vs_convexity_visuals.md
Visualization documentation including:
- Price-yield relationship plots
- Duration estimation accuracy vs. convexity
- Portfolio immunization dashboards
- Hedging effectiveness analysis
- Interactive rate scenario modeling

## Reference Materials

### Foundational Texts
1. Fabozzi, F. *Bond Markets, Analysis, and Strategies*
   - Chapter 4: Bond Price Volatility

2. Tuckman, B. & Serrat, A. *Fixed Income Securities*
   - Chapter 5: Duration and Convexity

3. CFA Institute. *Fixed Income*
   - Understanding Fixed-Income Risk and Return

### Online Resources
1. **PIMCO**: [Duration and Convexity Primer](https://www.pimco.com/en-us/resources/education)
2. **Investopedia**: [Duration Guide](https://www.investopedia.com/terms/d/duration.asp)

### Python Libraries
```bash
pip install numpy scipy pandas matplotlib seaborn
```

## Validation Framework

### Self-Assessment Checklist

- [ ] Calculate Macaulay and Modified Duration accurately
- [ ] Compute convexity for any bond structure
- [ ] Estimate price changes for given yield shifts
- [ ] Understand convexity's impact on duration estimates
- [ ] Build immunized portfolios matching liabilities
- [ ] Use futures/swaps for duration management
- [ ] Perform sensitivity analysis across rate scenarios

### Practice Problems

1. Calculate Modified Duration for: 5% coupon, 10-year bond, 4% YTM
2. Estimate price change for +100 bps yield increase using duration only
3. Calculate convexity and improve price change estimate
4. Build 2-bond portfolio with target duration of 7 years
5. Determine futures contracts needed to extend duration from 5 to 6 years

## Integration with Other Skills

- **Bond Pricing**: Foundation for duration calculations
- **Yield Measures**: YTM used in duration formulas
- **Credit Risk**: Credit spreads affect duration
- **Benchmarking**: Duration-matched benchmarking
- **Portfolio Management**: Key risk metric for fixed-income portfolios

## Standards and Compliance

All duration/convexity calculations must include:
- Calculation methodology (Macaulay vs. Modified vs. Effective)
- Yield curve assumptions
- Frequency conventions
- Validation against market-standard tools (Bloomberg, FactSet)

## Version Control

**Version**: 1.0.0 | **Last Updated**: 2025-12-07 | **Status**: Production Ready

**Previous Skill**: `yield-measures/` | **Next Skill**: `credit-risk/`
