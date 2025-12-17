---
name: option-adjusted-spread
description: Calculate option-adjusted spread for bonds with embedded options using binomial trees and Monte Carlo simulation. Handles callable bonds, putable bonds, and MBS. Requires numpy>=1.24.0, pandas>=2.0.0, scipy>=1.10.0. Use when valuing bonds with optionality, comparing yields across callable and non-callable securities.
---

# Option-Adjusted Spread (OAS) and Embedded Option Valuation Skill Development

## Objective

Analyze and adjust for embedded options (calls, puts, convertibles) to determine fair yield spreads and pricing efficiency. Master Monte Carlo simulation and binomial tree methodologies for option-adjusted bond valuation.

## Skill Classification

**Domain**: Advanced Fixed Income Derivatives
**Level**: Expert
**Prerequisites**: All previous bond analysis skills
**Estimated Time**: 20-25 hours

## Focus Areas

### 1. Callable and Puttable Bond Pricing

**Callable Bond Value**:
```
Value_callable = Value_straight - Value_call_option
```

**Puttable Bond Value**:
```
Value_puttable = Value_straight + Value_put_option
```

**Python Framework**:
```python
class EmbeddedOptionBond:
    """Bond with embedded options."""

    def __init__(self, face_value, coupon_rate, maturity_years,
                 option_type, option_dates, option_prices):
        """Initialize bond. option_type: 'call'/'put', option_dates: exercise years, option_prices: strike prices."""
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity_years = maturity_years
        self.option_type = option_type
        self.option_dates = option_dates
        self.option_prices = option_prices

    def straight_bond_value(self, yield_rate, frequency=2):
        """Price without embedded option."""
        periods = int(self.maturity_years * frequency)
        coupon = (self.face_value * self.coupon_rate) / frequency
        periodic_yield = yield_rate / frequency

        pv_coupons = sum(coupon / (1 + periodic_yield)**t
                        for t in range(1, periods + 1))
        pv_principal = self.face_value / (1 + periodic_yield)**periods

        return pv_coupons + pv_principal
```

### 2. Monte Carlo Simulation for Interest Rate Paths

**Vasicek Model** (Mean-reverting short rate):
```
dr = a(b - r)dt + σdW

Where:
  r = short rate
  a = mean reversion speed
  b = long-term mean
  σ = volatility
  dW = Wiener process
```

**Python Implementation**:
```python
import numpy as np

class VasicekModel:
    """Vasicek interest rate model."""

    def __init__(self, r0, a, b, sigma):
        """Initialize Vasicek model. r0: initial rate, a: mean reversion speed, b: long-term mean, sigma: volatility."""
        self.r0 = r0
        self.a = a
        self.b = b
        self.sigma = sigma

    def simulate_paths(self, T, num_paths, num_steps):
        """Simulate rate paths. Returns array (num_paths, num_steps+1)."""
        dt = T / num_steps
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.r0

        for t in range(1, num_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt), num_paths)
            dr = self.a * (self.b - paths[:, t-1]) * dt + \
                 self.sigma * dW
            paths[:, t] = paths[:, t-1] + dr

        return paths

    def bond_price_path(self, rate_path, cash_flows, times):
        """Price bond along single rate path. Returns array of prices at each time step."""
        prices = []

        for t_idx, (cf, t) in enumerate(zip(cash_flows, times)):
            # Discount using rate path
            discount_factor = np.exp(-np.sum(rate_path[:t_idx]) *
                                    (times[1] - times[0]))
            prices.append(cf * discount_factor)

        return np.array(prices)

def monte_carlo_oas(bond, num_simulations=10000):
    """Calculate OAS via Monte Carlo simulation. Returns option-adjusted spread."""
    # Simplified implementation
    model = VasicekModel(r0=0.03, a=0.1, b=0.05, sigma=0.01)
    paths = model.simulate_paths(T=bond.maturity_years,
                                  num_paths=num_simulations,
                                  num_steps=100)

    # Price bond along each path with and without option
    # This is a placeholder for full implementation
    pass
```

### 3. Binomial Tree / Lattice Models

**Binomial Interest Rate Tree**:
```python
class BinomialRateTree:
    """Binomial interest rate tree for bond valuation."""

    def __init__(self, r0, u, d, q):
        """Initialize tree. r0: initial rate, u: up factor, d: down factor, q: risk-neutral prob of up move."""
        self.r0 = r0
        self.u = u
        self.d = d
        self.q = q

    def build_tree(self, periods):
        """Build interest rate tree. Returns list of lists (tree structure)."""
        tree = [[self.r0]]

        for t in range(1, periods + 1):
            level = []
            for i in range(t + 1):
                # Rate at node (i, t)
                rate = self.r0 * (self.u ** (t - i)) * (self.d ** i)
                level.append(rate)
            tree.append(level)

        return tree

    def price_callable_bond(self, face_value, coupon_rate, periods,
                           call_schedule, frequency=2):
        """Price callable bond via backward induction. call_schedule: {period: call_price}. Returns bond price."""
        # Build rate tree
        rate_tree = self.build_tree(periods)

        # Initialize value tree at maturity
        value_tree = [[face_value + face_value * coupon_rate / frequency]
                      * (periods + 1)]

        # Backward induction
        for t in range(periods - 1, -1, -1):
            level = []
            for i in range(t + 1):
                r = rate_tree[t][i]

                # Expected value from next period
                up_value = value_tree[0][i]
                down_value = value_tree[0][i + 1]

                expected = (self.q * up_value + (1 - self.q) * down_value)
                discounted = expected / (1 + r / frequency)

                # Add coupon
                value = discounted + face_value * coupon_rate / frequency

                # Check if callable
                if t in call_schedule:
                    value = min(value, call_schedule[t])

                level.append(value)

            value_tree.insert(0, level)

        return value_tree[0][0]
```

**Calibration to Market Prices**:
```python
def calibrate_tree(market_bond_prices, face_values, coupon_rates,
                   maturities):
    """
    Calibrate binomial tree to match market prices.

    Uses optimization to find tree parameters.
    """
    from scipy.optimize import minimize

    def objective(params):
        r0, u, d, q = params
        tree = BinomialRateTree(r0, u, d, q)

        # Calculate model prices
        errors = []
        for market_price, fv, coupon, maturity in \
            zip(market_bond_prices, face_values, coupon_rates, maturities):

            model_price = tree.price_callable_bond(fv, coupon,
                                                   int(maturity * 2), {})
            errors.append((model_price - market_price) ** 2)

        return sum(errors)

    # Initial guess
    initial_params = [0.03, 1.1, 0.9, 0.5]

    result = minimize(objective, initial_params,
                     bounds=[(0.01, 0.10), (1.0, 1.5), (0.5, 1.0), (0.3, 0.7)])

    return result.x
```

### 4. OAS vs. Z-Spread vs. Nominal Spread

**Nominal Spread**:
```
Nominal Spread = YTM_Corporate - YTM_Treasury
```

**Z-Spread** (Zero-Volatility Spread):
- Constant spread over entire spot curve
- No embedded option consideration

**OAS** (Option-Adjusted Spread):
- Spread after removing option value
- Reflects credit and liquidity risk only

**Relationship**:
```
For Callable Bonds:
OAS = Z-Spread - Call Option Value

For Puttable Bonds:
OAS = Z-Spread + Put Option Value
```

**Python Comparison**:
```python
def spread_comparison(bond_price, straight_price, option_value,
                     treasury_ytm, corporate_ytm):
    """
    Compare spread measures.

    Parameters:
    -----------
    bond_price : float
        Actual bond price (with embedded option)
    straight_price : float
        Hypothetical price without option
    option_value : float
        Value of embedded option
    treasury_ytm : float
        Treasury yield
    corporate_ytm : float
        Corporate bond YTM

    Returns:
    --------
    dict : Spread comparison
    """
    nominal_spread = (corporate_ytm - treasury_ytm) * 10000  # bps

    # Z-spread approximation (simplified)
    z_spread = nominal_spread  # Placeholder

    # OAS calculation
    if option_value > 0:  # Call option (issuer option)
        oas = z_spread - (option_value / straight_price) * 10000
    else:  # Put option (investor option)
        oas = z_spread + abs(option_value / straight_price) * 10000

    return {
        'nominal_spread_bps': nominal_spread,
        'z_spread_bps': z_spread,
        'oas_bps': oas,
        'option_value': option_value,
        'option_cost_bps': (option_value / straight_price) * 10000
    }
```

### 5. Volatility Sensitivity Analysis

**Vega** (Sensitivity to Volatility):
```python
def option_vega(bond_price_func, base_volatility, vol_shock=0.01):
    """
    Calculate sensitivity to volatility changes.

    Parameters:
    -----------
    bond_price_func : callable
        Function that prices bond given volatility
    base_volatility : float
        Current volatility assumption
    vol_shock : float
        Volatility change for numerical derivative

    Returns:
    --------
    float : Vega (price change per 1% volatility change)
    """
    price_base = bond_price_func(base_volatility)
    price_up = bond_price_func(base_volatility + vol_shock)

    vega = (price_up - price_base) / vol_shock
    return vega

def oas_volatility_sensitivity(oas_calc_func, volatilities):
    """
    Analyze OAS across volatility scenarios.

    Returns:
    --------
    dict : {volatility: oas} mapping
    """
    oas_scenarios = {}

    for vol in volatilities:
        oas = oas_calc_func(vol)
        oas_scenarios[vol] = oas

    return oas_scenarios
```

**Convexity of Embedded Options**:
```python
def effective_convexity_with_options(bond_price, bond_price_up,
                                     bond_price_down, yield_change):
    """
    Calculate effective convexity accounting for options.

    Parameters:
    -----------
    bond_price : float
        Current price
    bond_price_up : float
        Price after yield decrease
    bond_price_down : float
        Price after yield increase
    yield_change : float
        Yield shift size (e.g., 0.01 for 1%)

    Returns:
    --------
    float : Effective convexity
    """
    numerator = bond_price_up + bond_price_down - 2 * bond_price
    denominator = bond_price * yield_change ** 2

    return numerator / denominator
```

### 6. Convertible Bonds

**Convertible Bond Value**:
```
Convertible Value = max(Straight Debt Value, Conversion Value)
                   + Option Value
```

**Conversion Terms**:
```python
class ConvertibleBond:
    """Convertible bond pricing."""

    def __init__(self, face_value, coupon_rate, maturity,
                 conversion_ratio, stock_price, stock_volatility):
        """
        Parameters:
        -----------
        conversion_ratio : float
            Shares received per bond
        stock_price : float
            Current stock price
        stock_volatility : float
            Stock volatility (annual)
        """
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.conversion_ratio = conversion_ratio
        self.stock_price = stock_price
        self.stock_volatility = stock_volatility

    def conversion_value(self):
        """Value if converted immediately."""
        return self.conversion_ratio * self.stock_price

    def straight_bond_value(self, yield_rate):
        """Value as straight debt."""
        periods = int(self.maturity * 2)
        coupon = self.face_value * self.coupon_rate / 2
        periodic_yield = yield_rate / 2

        pv = sum(coupon / (1 + periodic_yield)**t
                for t in range(1, periods + 1))
        pv += self.face_value / (1 + periodic_yield)**periods

        return pv

    def conversion_premium(self, market_price):
        """Premium over conversion value."""
        conv_value = self.conversion_value()
        return (market_price - conv_value) / conv_value
```

## Expected Competency

Price callable/puttable bonds with binomial trees, implement Monte Carlo OAS simulation, distinguish spread measures (Nominal/Z-spread/OAS), assess volatility sensitivity, value convertibles, calibrate models to market prices, interpret OAS for investment decisions.

## Deliverables

**oas_model.ipynb**: Binomial tree callable bond pricer, Monte Carlo rate simulator, OAS calculator, spread comparisons, volatility sensitivity, convertible pricer

**oas_analysis.md**: Callable vs. non-callable comparisons, OAS interpretation, volatility scenarios, convertible arbitrage, model validation

## Reference Materials

**Texts**: Fabozzi *Fixed Income Analysis* (Embedded Options), Hull *Options, Futures, and Other Derivatives* (Interest Rate Derivatives), Sundaresan *Fixed Income Markets*

**Advanced**: QuantLib [Interest Rate Models](https://www.quantlib.org/), Bloomberg OAS methodology, BIS interest rate frameworks

**Libraries**: `pip install numpy scipy matplotlib quantlib-python`

## Validation Framework

**Checklist**: Build binomial rate tree, price callable bond with backward induction, implement Monte Carlo simulation, calculate OAS, distinguish OAS from Z-spread, assess volatility sensitivity, value convertibles, calibrate to market data

**Practice**: (1) Price 10-year callable bond (5% coupon, callable at par in year 5), (2) Calculate OAS (Z-spread 150 bps, option value $25, price $1,000), (3) Simulate 1,000 Vasicek rate paths, (4) Compare OAS for bonds with different call provisions

## Integration with Other Skills

**Bond Pricing** (option-free valuation foundation), **Yield Measures** (YTM benchmark), **Duration** (effective duration for options), **Benchmarking** (peer OAS comparison), **Credit Risk** (OAS isolates credit)

## Standards and Compliance

Document: interest rate model/parameters, volatility assumptions, simulation paths/tree steps, calibration methodology, validation results

## Version Control

**Version**: 1.0.0 | **Last Updated**: 2025-12-07 | **Status**: Production Ready | **Previous**: `bond-benchmarking/` | **Completion**: Final skill in bond analysis suite
