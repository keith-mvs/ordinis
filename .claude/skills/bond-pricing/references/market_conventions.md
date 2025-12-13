# Bond Market Conventions and Standards

## Day Count Conventions

### 30/360 (Bond Basis)
**Usage**: U.S. corporate bonds, municipal bonds
**Convention**: Assumes 30 days per month, 360 days per year

**Formula**:
```
Days = (Y2 - Y1) × 360 + (M2 - M1) × 30 + (D2 - D1)

Where:
  Y1, Y2 = Start and end year
  M1, M2 = Start and end month
  D1, D2 = Start and end day (capped at 30)
```

**Python Implementation**:
```python
def days_30_360(start_date, end_date):
    """Calculate days using 30/360 convention."""
    d1, m1, y1 = start_date.day, start_date.month, start_date.year
    d2, m2, y2 = end_date.day, end_date.month, end_date.year

    # Adjust days if 31st
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 >= 30:
        d2 = 30

    return (y2 - y1) * 360 + (m2 - m1) * 30 + (d2 - d1)
```

### Actual/Actual (ISDA)
**Usage**: U.S. Treasury bonds, Treasury notes
**Convention**: Actual days in period, actual days in year

**Characteristics**:
- Most accurate reflection of time
- No day/month assumptions
- Handles leap years correctly

**Python Implementation**:
```python
from datetime import datetime

def days_actual_actual(start_date, end_date):
    """Calculate days using Actual/Actual convention."""
    return (end_date - start_date).days

def year_fraction_actual_actual(start_date, end_date):
    """Calculate year fraction for Actual/Actual."""
    days = (end_date - start_date).days
    year = start_date.year
    days_in_year = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365
    return days / days_in_year
```

### Actual/360
**Usage**: Money market instruments, floating-rate notes
**Convention**: Actual days in period, 360-day year

**Formula**:
```
Year Fraction = Actual Days / 360
```

### 30/360 European
**Usage**: Eurobonds
**Convention**: Similar to 30/360 but with different end-of-month rules

## Settlement Conventions

### T+1 Settlement
**Markets**: U.S. Treasury securities
**Rule**: Settlement occurs one business day after trade date

### T+2 Settlement
**Markets**: U.S. corporate bonds
**Rule**: Settlement occurs two business days after trade date

### T+3 Settlement
**Markets**: Municipal bonds (some)
**Rule**: Settlement occurs three business days after trade date

## Price Quotation Standards

### U.S. Treasuries
**Quote Basis**: 32nds of a point
**Example**: 99-16 = 99 + 16/32 = 99.50% of par

**Python Conversion**:
```python
def quote_32nds_to_decimal(whole, thirty_seconds):
    """Convert Treasury quote to decimal price."""
    return whole + (thirty_seconds / 32)

# Example: 99-16
price = quote_32nds_to_decimal(99, 16)  # Returns 99.50
```

### Corporate Bonds
**Quote Basis**: Decimal percent of par
**Example**: 99.50 = 99.50% of par

## Clean vs. Dirty Price

### Clean Price (Flat Price)
**Definition**: Quoted price excluding accrued interest
**Usage**: Market quotation standard
**Formula**: `Clean Price = Dirty Price - Accrued Interest`

### Dirty Price (Full Price)
**Definition**: Actual settlement price including accrued interest
**Usage**: Settlement calculations
**Formula**: `Dirty Price = Clean Price + Accrued Interest`

**Accrued Interest Calculation**:
```python
def accrued_interest(face_value, coupon_rate, last_coupon_date,
                     settlement_date, next_coupon_date, convention='30/360'):
    """
    Calculate accrued interest.

    Parameters:
    -----------
    face_value : float
    coupon_rate : float (annual)
    last_coupon_date : datetime
    settlement_date : datetime
    next_coupon_date : datetime
    convention : str

    Returns:
    --------
    float : Accrued interest amount
    """
    annual_coupon = face_value * coupon_rate

    if convention == '30/360':
        days_accrued = days_30_360(last_coupon_date, settlement_date)
        days_in_period = days_30_360(last_coupon_date, next_coupon_date)
    else:  # Actual/Actual
        days_accrued = (settlement_date - last_coupon_date).days
        days_in_period = (next_coupon_date - last_coupon_date).days

    return (annual_coupon / 2) * (days_accrued / days_in_period)
```

## Coupon Payment Frequencies

### Semi-Annual
**Usage**: U.S. Treasuries, most U.S. corporate bonds
**Payment**: Every 6 months
**Period Rate**: `r_period = r_annual / 2`

### Annual
**Usage**: Many Eurobonds
**Payment**: Once per year
**Period Rate**: `r_period = r_annual`

### Quarterly
**Usage**: Some corporate bonds, floating-rate notes
**Payment**: Every 3 months
**Period Rate**: `r_period = r_annual / 4`

## Ex-Dividend Trading

**Ex-Dividend Date**: Typically 1 business day before record date
**Impact**: Buyer does not receive next coupon if purchasing on or after ex-dividend date

## Market Data Sources

### Real-Time Pricing
- **Bloomberg Terminal**: CBBT (Corporate Bond Ticker)
- **TRACE**: Trade Reporting and Compliance Engine (FINRA)
- **ICE Data Services**: Evaluated pricing

### Historical Data
- **FRED (Federal Reserve)**: Treasury rates, yields
- **FINRA TRACE**: Historical corporate bond trades
- **Bloomberg API**: Historical bond data

## Regulatory Framework

**SEC Rule 15c2-12**: Municipal securities disclosure
**FINRA Rule 5130**: Restricted Persons (new issues)
**MSRB Rules**: Municipal Securities Rulemaking Board standards

---

**Status**: Reference complete
**Last Updated**: 2025-12-07
**Sources**: FINRA, SIFMA, Bloomberg Market Conventions Guide
