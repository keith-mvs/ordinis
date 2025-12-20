# Credit Risk Assessment and Rating Analysis

**Evaluate issuer default risk, credit spreads, and rating migrations for systematic credit analysis**

## Quick Reference

**Skill Level**: Advanced
**Time Commitment**: 20-25 hours
**Prerequisites**: Bond Pricing, Yield Measures

## Core Capabilities

- Fundamental credit analysis (coverage ratios, leverage)
- Market-implied default probability calculation
- Credit rating interpretation (Moody's, S&P, Fitch)
- PD/LGD modeling (Merton, hazard rate approaches)
- Credit spread analysis and decomposition
- Credit migration monitoring

## Key Formulas

```
Credit Spread = YTM_Corporate - YTM_Treasury

Implied PD = Credit Spread / (1 - Recovery Rate)

Expected Loss = PD × LGD × Exposure

Merton PD = N(-d2) where d2 = [ln(V/D) + (r - σ²/2)t] / (σ√t)
```

## Python Quick Start

```python
def implied_default_prob(credit_spread, recovery_rate=0.40):
    """Calculate annual default probability from spread."""
    lgd = 1 - recovery_rate
    return credit_spread / lgd

def interest_coverage(ebit, interest_expense):
    """Times Interest Earned ratio."""
    return ebit / interest_expense
```

## Deliverables

- `credit_risk_model.py`: Complete PD/LGD modeling framework
- `credit_spread_dashboard.md`: Spread tracking and analysis

## Essential References

- Lando: *Credit Risk Modeling*
- Duffie & Singleton: *Credit Risk: Pricing, Measurement, and Management*
- Rating agency methodologies (Moody's, S&P, Fitch)

## Validation Checkpoint

For a BB bond with 300 bps spread and 40% recovery rate:
**Expected Implied Annual PD**: 5.0%

---

**Full Documentation**: See `SKILL.md`
