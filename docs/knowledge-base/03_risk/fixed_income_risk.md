# Fixed Income Risk Management

**Section**: 03_risk
**Last Updated**: 2025-12-12
**Source Skills**: [bond-pricing](../../../../.claude/skills/bond-pricing/SKILL.md), [duration-convexity](../../../../.claude/skills/duration-convexity/SKILL.md), [credit-risk](../../../../.claude/skills/credit-risk/SKILL.md)

---

## 1. Bond Pricing Fundamentals

### Price-Yield Relationship

```python
def bond_price(face_value, coupon_rate, yield_rate, periods):
    """PV of all cash flows."""
    coupon = face_value * coupon_rate
    pv_coupons = sum(coupon / (1 + yield_rate)**t for t in range(1, periods + 1))
    pv_principal = face_value / (1 + yield_rate)**periods
    return pv_coupons + pv_principal
```

### Market Conventions

| Convention | Description | Usage |
|------------|-------------|-------|
| Clean Price | Excludes accrued interest | Quoted |
| Dirty Price | Includes accrued interest | Settlement |
| 30/360 | 30 days/month, 360 days/year | Corporates |
| Actual/Actual | Actual days | Treasuries |

---

## 2. Duration and Convexity

### Macaulay Duration

Weighted average time to receive cash flows.

```python
def macaulay_duration(face_value, coupon_rate, yield_rate, years, frequency=2):
    periods = int(years * frequency)
    coupon = (face_value * coupon_rate) / frequency
    periodic_yield = yield_rate / frequency

    weighted_pv = 0
    bond_price = 0

    for t in range(1, periods + 1):
        cf = coupon if t < periods else coupon + face_value
        pv = cf / (1 + periodic_yield)**t
        weighted_pv += t * pv
        bond_price += pv

    return (weighted_pv / bond_price) / frequency
```

### Modified Duration

```python
def modified_duration(macaulay_dur, yield_rate, frequency=2):
    return macaulay_dur / (1 + yield_rate / frequency)

# Price change approximation
price_change_pct = -modified_duration * yield_change
```

### Convexity

```python
def convexity(face_value, coupon_rate, yield_rate, years, frequency=2):
    # Second derivative of price-yield function
    # Corrects duration estimate for large yield changes
    pass

# Full price change estimate
price_change = -mod_dur * delta_y + 0.5 * convexity * delta_y**2
```

### Duration Properties

| Factor | Effect on Duration |
|--------|-------------------|
| Longer maturity | Higher |
| Lower coupon | Higher |
| Higher yield | Lower |

---

## 3. Credit Risk

### Credit Spread Components

```
Credit Spread = Expected Loss + Risk Premium + Liquidity Premium
```

### Implied Default Probability

```python
def implied_default_prob(credit_spread, recovery_rate):
    """Market-implied annual PD."""
    lgd = 1 - recovery_rate
    return credit_spread / lgd
```

### Historical Default Rates

| Rating | 10-Year Cumulative |
|--------|-------------------|
| AAA | 0.6% |
| AA | 1.5% |
| A | 2.9% |
| BBB | 7.2% |
| BB | 19.0% |
| B | 35.0% |
| CCC | 54.0% |

### Loss Given Default by Seniority

| Seniority | Recovery Rate |
|-----------|--------------|
| Senior Secured | 65% |
| Senior Unsecured | 45% |
| Subordinated | 25% |

---

## 4. Portfolio Duration Management

### Target Duration

```python
def futures_contracts_needed(portfolio_value, portfolio_duration,
                             target_duration, futures_duration,
                             futures_price, multiplier=1000):
    duration_gap = target_duration - portfolio_duration
    return round((duration_gap * portfolio_value) /
                 (futures_duration * futures_price * multiplier))
```

### Immunization Conditions

1. PV(Assets) = PV(Liabilities)
2. Duration(Assets) = Duration(Liabilities)
3. Convexity(Assets) > Convexity(Liabilities)

---

## 5. Risk Metrics Summary

| Metric | Purpose | Formula |
|--------|---------|---------|
| Modified Duration | Rate sensitivity | MacD / (1 + y/freq) |
| Dollar Duration | $ impact per 1bp | ModD × Price × 0.0001 |
| Convexity | Curvature correction | Second derivative |
| Credit Spread | Default compensation | Corporate YTM - Treasury YTM |
| OAS | Option-adjusted spread | Z-spread - Option cost |

---

## Skill Cross-References

- [bond-pricing](../../../../.claude/skills/bond-pricing/SKILL.md) - Valuation fundamentals
- [duration-convexity](../../../../.claude/skills/duration-convexity/SKILL.md) - Rate risk management
- [credit-risk](../../../../.claude/skills/credit-risk/SKILL.md) - Default analysis

---

**Template**: KB Skills Integration v1.0
