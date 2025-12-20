# Bond Benchmarking and Relative Performance Analysis

**Compare bonds against benchmarks and peers for relative value identification and performance measurement**

## Quick Reference

**Skill Level**: Advanced
**Time Commitment**: 12-15 hours
**Prerequisites**: Bond Pricing, Yield Measures, Duration, Credit Risk

## Core Capabilities

- Benchmark selection (Treasury curve, corporate indices)
- Spread calculation (G-spread, I-spread, Z-spread)
- Peer comparison by sector and rating
- Excess return attribution (α vs. β decomposition)
- Tracking error and information ratio
- Relative value identification

## Key Formulas

```
G-Spread = YTM_Corporate - YTM_Treasury (bps)

Tracking Error = σ(R_portfolio - R_benchmark)

Information Ratio = Excess Return / Tracking Error

Z-Spread: P = Σ[CF_t / (1 + r_t + Z)^t]
```

## Python Quick Start

```python
def g_spread(corporate_ytm, treasury_ytm):
    """Calculate G-spread in basis points."""
    return (corporate_ytm - treasury_ytm) * 10000

def tracking_error(portfolio_returns, benchmark_returns):
    """Calculate annualized tracking error."""
    excess = np.array(portfolio_returns) - np.array(benchmark_returns)
    return excess.std() * np.sqrt(12)  # Annualize monthly data

def information_ratio(excess_return_annual, tracking_error):
    """Calculate IR."""
    return excess_return_annual / tracking_error
```

## Deliverables

- `benchmark_comparison.ipynb`: Complete benchmarking toolkit
- `bond_benchmark_report.md`: Performance attribution and visualization

## Essential References

- Fabozzi: *Handbook of Fixed Income Securities*
- Bloomberg/ICE BofA: Index methodologies
- Dynkin: *Quantitative Management of Bond Portfolios*

## Validation Checkpoint

Portfolio return 6%, Benchmark 5.2%, TE 1.5%:
**Expected IR**: 0.53

---

**Full Documentation**: See `SKILL.md`
