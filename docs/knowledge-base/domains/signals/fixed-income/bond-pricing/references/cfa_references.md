# Bond Pricing - CFA Institute References

## Primary Source

**CFA Institute.** *CFA Program Curriculum* - Fixed Income Section

## Relevant Readings

### Level I: Introduction to Fixed-Income Valuation

**Learning Outcome Statements (LOS)**:
- Calculate bond price given discount rate and cash flows
- Identify relationships among bond price, coupon rate, maturity, and market discount rate
- Calculate and interpret yield to maturity (YTM)
- Distinguish between spot rates and YTM
- Calculate bond price given spot rates

**Key Concepts**:
- Full price vs. flat price (dirty vs. clean)
- Accrued interest calculation
- Matrix pricing for bonds without recent trades
- Arbitrage-free valuation

**Practice Problems**: End-of-reading questions provide validation cases

### Level II: Advanced Fixed-Income Valuation

**Topics**:
- Term structure models
- Binomial interest rate trees
- Arbitrage-free valuation framework
- Option-embedded bonds (preview for OAS skill)

### CFA Institute Standards

**Ethical Requirements**:
- Material non-public information handling
- Fair dealing in bond recommendations
- Performance presentation standards (GIPS)

**Documentation Standards**:
- Maintaining audit trail of pricing assumptions
- Source attribution for market data
- Disclosure of model limitations

## Integration with Python

**CFA Formulas → Code Mapping**:

```python
# CFA Institute Full Price Formula
def full_price_cfa(PV, AI):
    """
    Calculate full price per CFA standards.

    Parameters:
    -----------
    PV : float
        Present value of bond (flat price)
    AI : float
        Accrued interest

    Returns:
    --------
    float : Full price (dirty price)
    """
    return PV + AI
```

## Key Formulas

### Bond Pricing Formula (CFA Level I)
```
PV = PMT/(1+r)^1 + PMT/(1+r)^2 + ... + (PMT+FV)/(1+r)^N

Where:
  PV = Present Value (bond price)
  PMT = Coupon payment per period
  r = Market discount rate per period
  FV = Face value (par value)
  N = Number of periods to maturity
```

### Accrued Interest (CFA Convention)
```
AI = (t/T) × PMT

Where:
  t = Days from last coupon payment to settlement
  T = Days in coupon period
```

## Practice with CFA Problems

**Recommended Questions**:
1. CFA Level I Reading 44, Question 1: Basic bond pricing
2. CFA Level I Reading 44, Question 5: YTM calculation
3. CFA Level I Reading 44, Question 8: Clean vs. dirty price

**Expected Competency**:
- Solve CFA-style problems within 3 minutes
- Accuracy within ±0.01% of provided solutions

## Reference Materials

**Official CFA Resources**:
- CFA Institute website (www.cfainstitute.org)
- CFA Program Curriculum (update annually)
- CFA Institute Fixed Income Essential Learning Modules

**Study Notes**:
- Kaplan Schweser Notes (supplementary)
- Wiley CFA Study Guides (supplementary)

---

**Status**: Reference placeholder - specify CFA curriculum year
**Last Updated**: 2025-12-07
