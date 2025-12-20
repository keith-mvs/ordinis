# QuantLib-Python Reference for Bond Pricing

## Overview

**QuantLib** is a comprehensive free/open-source library for quantitative finance, providing tools for pricing, trading, and risk management of financial instruments.

**QuantLib-Python**: Python bindings for QuantLib C++ library

## Installation

```bash
# Using pip
pip install QuantLib-Python

# Using conda
conda install -c conda-forge quantlib-python

# Verify installation
python -c "import QuantLib as ql; print(ql.__version__)"
```

## Core Bond Pricing Classes

### Date and Calendar Objects

```python
import QuantLib as ql

# Create date
settlement_date = ql.Date(15, 6, 2025)
ql.Settings.instance().evaluationDate = settlement_date

# Calendar (for business day conventions)
calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)  # US Treasury calendar
calendar_nyse = ql.UnitedStates(ql.UnitedStates.NYSE)      # NYSE calendar

# Business day convention
business_convention = ql.Following  # Adjust to next business day
```

### Interest Rate and Day Count

```python
# Day count conventions
day_count_30_360 = ql.Thirty360(ql.Thirty360.BondBasis)
day_count_actual_actual = ql.ActualActual(ql.ActualActual.ISDA)
day_count_actual_360 = ql.Actual360()

# Interest rate
rate = 0.05  # 5%
annual_rate = ql.InterestRate(rate, day_count_actual_actual,
                              ql.Compounded, ql.Annual)
```

### Fixed-Rate Bond Construction

```python
# Bond parameters
settlement_date = ql.Date(15, 6, 2025)
maturity_date = ql.Date(15, 6, 2035)
face_value = 100.0
coupon_rate = 0.05
frequency = ql.Semiannual

# Create schedule
schedule = ql.Schedule(
    settlement_date,
    maturity_date,
    ql.Period(frequency),
    calendar,
    business_convention,
    business_convention,
    ql.DateGeneration.Backward,  # Generate from maturity backwards
    False  # End of month
)

# Create fixed-rate bond
bond = ql.FixedRateBond(
    2,  # Settlement days
    face_value,
    schedule,
    [coupon_rate],  # Coupon rates (can be vector for stepped coupons)
    day_count_30_360
)
```

### Bond Pricing Engine

```python
# Flat yield curve
flat_rate = ql.SimpleQuote(0.045)  # 4.5% flat curve
rate_handle = ql.QuoteHandle(flat_rate)
flat_curve = ql.FlatForward(settlement_date, rate_handle, day_count_actual_actual)
curve_handle = ql.YieldTermStructureHandle(flat_curve)

# Pricing engine
pricing_engine = ql.DiscountingBondEngine(curve_handle)
bond.setPricingEngine(pricing_engine)

# Get price
clean_price = bond.cleanPrice()
dirty_price = bond.dirtyPrice()
accrued = bond.accruedAmount()

print(f"Clean Price: ${clean_price:.2f}")
print(f"Dirty Price: ${dirty_price:.2f}")
print(f"Accrued Interest: ${accrued:.2f}")
```

### Yield Calculations

```python
# Calculate YTM from price
market_price = 103.5
ytm = bond.bondYield(
    market_price,
    day_count_actual_actual,
    ql.Compounded,
    ql.Semiannual
)
print(f"YTM: {ytm:.4%}")

# Calculate price from YTM
ytm_given = 0.04
price_from_ytm = bond.cleanPrice(ytm_given, day_count_actual_actual,
                                  ql.Compounded, ql.Semiannual)
print(f"Price at {ytm_given:.2%} YTM: ${price_from_ytm:.2f}")
```

## Advanced Features

### Zero-Coupon Bond

```python
# Zero-coupon bond
zero_bond = ql.ZeroCouponBond(
    2,  # Settlement days
    calendar,
    face_value,
    maturity_date
)

zero_bond.setPricingEngine(pricing_engine)
zero_price = zero_bond.cleanPrice()
```

### Callable Bond

```python
# Call schedule
call_dates = [ql.Date(15, 6, 2030)]
call_prices = [102.0]  # Call at 102% of par

call_schedule = ql.CallabilitySchedule()
for call_date, call_price in zip(call_dates, call_prices):
    callability = ql.Callability(
        ql.BondPrice(call_price, ql.BondPrice.Clean),
        ql.Callability.Call,
        call_date
    )
    call_schedule.append(callability)

# Create callable bond
callable_bond = ql.CallableFixedRateBond(
    2,  # Settlement days
    face_value,
    schedule,
    [coupon_rate],
    day_count_30_360,
    business_convention,
    face_value,  # Redemption
    settlement_date,
    call_schedule
)
```

### Term Structure Construction

```python
# Build curve from market instruments
helpers = []

# Add deposit rates (short end)
deposit_rates = [(1, 0.025), (3, 0.028), (6, 0.030)]
for months, rate in deposit_rates:
    helper = ql.DepositRateHelper(
        ql.QuoteHandle(ql.SimpleQuote(rate)),
        ql.Period(months, ql.Months),
        2,  # Settlement days
        calendar,
        business_convention,
        False,  # End of month
        day_count_actual_360
    )
    helpers.append(helper)

# Add bond rates (long end)
bond_maturities = [2, 5, 10, 30]
bond_yields = [0.032, 0.038, 0.042, 0.045]
for years, ytm in zip(bond_maturities, bond_yields):
    maturity = settlement_date + ql.Period(years, ql.Years)
    helper = ql.FixedRateBondHelper(
        ql.QuoteHandle(ql.SimpleQuote(100.0)),  # Price quote
        2,
        face_value,
        ql.Schedule(settlement_date, maturity, ql.Period(ql.Semiannual),
                   calendar, business_convention, business_convention,
                   ql.DateGeneration.Backward, False),
        [ytm],
        day_count_30_360
    )
    helpers.append(helper)

# Build yield curve
curve = ql.PiecewiseLogCubicDiscount(settlement_date, helpers, day_count_actual_actual)
curve_handle = ql.YieldTermStructureHandle(curve)
```

## Best Practices

### Memory Management
```python
# QuantLib uses reference counting
# Keep references to quotes and handles
quotes = []  # Store quotes to prevent garbage collection
handles = []  # Store handles
```

### Date Arithmetic
```python
# Add periods to dates
future_date = settlement_date + ql.Period(6, ql.Months)

# Calculate days between dates
days = day_count_actual_actual.dayCount(settlement_date, future_date)

# Year fraction
year_frac = day_count_actual_actual.yearFraction(settlement_date, future_date)
```

### Error Handling
```python
try:
    price = bond.cleanPrice()
except RuntimeError as e:
    print(f"QuantLib Error: {e}")
    # Handle pricing errors (negative rates, invalid dates, etc.)
```

## Integration with Pandas

```python
import pandas as pd

def bond_cashflows_to_df(bond):
    """Extract bond cashflows to pandas DataFrame."""
    cashflows = []
    for cf in bond.cashflows():
        date = cf.date()
        amount = cf.amount()
        cashflows.append({'Date': date, 'Amount': amount})

    return pd.DataFrame(cashflows)

# Use it
cf_df = bond_cashflows_to_df(bond)
print(cf_df)
```

## Performance Optimization

```python
# Disable today's date updates if doing batch pricing
ql.Settings.instance().evaluationDate = settlement_date

# Use cached pricing engines when pricing multiple bonds
# Create engine once, reuse for many bonds
```

## Resources

**Official Documentation**:
- QuantLib Website: https://www.quantlib.org/
- Python Cookbook: https://quantlib-python-docs.readthedocs.io/

**Tutorials**:
- Goutham Balaraman's Blog: http://gouthamanbalaraman.com/blog/quantlib-python-tutorials-with-examples.html
- Luigi Ballabio's QuantLib YouTube Channel

**GitHub**:
- QuantLib C++: https://github.com/lballabio/QuantLib
- QuantLib-Python: https://github.com/lballabio/QuantLib-SWIG

---

**Status**: Reference complete
**Last Updated**: 2025-12-07
**QuantLib Version**: 1.31+
