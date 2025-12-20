# Fixed Income Analysis and Benchmarking

**Section**: 02_signals/fundamental
**Last Updated**: 2025-12-12
**Source Skills**: [bond-benchmarking](../../../../.claude/skills/bond-benchmarking/SKILL.md)

---

## Overview

Systematic framework for comparing bond performance against market benchmarks, calculating yield spreads, and identifying relative value opportunities.

---

## Spread Measures

### G-Spread (Government Spread)

```python
def g_spread(corporate_ytm, treasury_ytm):
    """Calculate G-spread in basis points."""
    return (corporate_ytm - treasury_ytm) * 10000
```

**Interpretation**: Compensation over risk-free rate for credit and liquidity risk.

### I-Spread (Interpolated Spread)

```python
def i_spread(corporate_ytm, swap_rate):
    """Calculate I-spread vs. swap curve (bps)."""
    return (corporate_ytm - swap_rate) * 10000
```

**Use case**: Banks and dealers who fund at swap rates.

### Z-Spread (Zero-Volatility Spread)

```python
from scipy.optimize import newton

def z_spread(bond_price, cash_flows, spot_rates, times):
    """
    Calculate Z-spread: constant spread over spot curve.

    Parameters:
        bond_price: Current price
        cash_flows: Future cash flows
        spot_rates: Risk-free spot rates
        times: Time to each cash flow (years)

    Returns:
        Z-spread as decimal
    """
    def price_diff(z):
        pv = sum(cf / (1 + spot + z)**t
                for cf, spot, t in zip(cash_flows, spot_rates, times))
        return pv - bond_price

    return newton(price_diff, 0.01, maxiter=100)
```

---

## Spread Comparison

| Measure | Benchmark | Considers | Best For |
|---------|-----------|-----------|----------|
| G-Spread | Treasury YTM | Credit + Liquidity | Quick comparison |
| I-Spread | Swap Rate | Credit + Liquidity | Bank funding |
| Z-Spread | Spot Curve | Full term structure | Accurate pricing |
| OAS | Spot + Options | Credit only | Callable bonds |

---

## Relative Value Analysis

### Spread Per Duration

```python
def spread_per_duration(spread_bps, duration):
    """Yield pickup per unit duration risk."""
    return spread_bps / duration

# Example: Compare two bonds
bond_a = {'spread': 120, 'duration': 5.0}  # 24 bps per year
bond_b = {'spread': 180, 'duration': 8.0}  # 22.5 bps per year
# Bond A offers better risk-adjusted yield
```

### Peer Comparison

```python
import numpy as np

def sector_peer_analysis(bond, peer_bonds):
    """Compare bond against sector peers."""
    peer_spreads = np.array([b['spread'] for b in peer_bonds])

    return {
        'bond_spread': bond['spread'],
        'peer_median': np.median(peer_spreads),
        'peer_mean': np.mean(peer_spreads),
        'percentile': (peer_spreads < bond['spread']).sum() / len(peer_spreads) * 100,
        'cheapness': bond['spread'] - np.median(peer_spreads)
    }
```

### Cheapness Score

| Score | Interpretation | Action |
|-------|----------------|--------|
| > +20 bps | Cheap vs. peers | Potential buy |
| -20 to +20 bps | Fair value | Hold |
| < -20 bps | Rich vs. peers | Avoid / Sell |

---

## Performance Attribution

### Total Return Components

```
Total Return = Coupon Income + Price Change + Reinvestment
```

### Excess Return Decomposition

```python
def attribution_decomposition(portfolio_return, benchmark_return,
                              duration_effect, spread_effect,
                              selection_effect):
    """Decompose excess return into components."""
    total_alpha = portfolio_return - benchmark_return

    return {
        'total_excess': total_alpha,
        'duration_contribution': duration_effect,
        'spread_contribution': spread_effect,
        'selection_contribution': selection_effect,
        'residual': total_alpha - (duration_effect + spread_effect + selection_effect)
    }
```

### Attribution Framework

| Component | Source | Example |
|-----------|--------|---------|
| Duration | Yield curve positioning | Overweight long duration in bull market |
| Spread | Sector/quality allocation | Overweight BBB vs. benchmark |
| Selection | Individual bond selection | Pick cheaper bonds within sector |

---

## Benchmark Tracking

### Tracking Error

```python
def tracking_error(portfolio_returns, benchmark_returns):
    """Calculate annualized tracking error."""
    excess = np.array(portfolio_returns) - np.array(benchmark_returns)
    return excess.std() * np.sqrt(12)  # Annualize monthly

def information_ratio(portfolio_returns, benchmark_returns):
    """Calculate information ratio (higher = better)."""
    excess = np.array(portfolio_returns) - np.array(benchmark_returns)
    mean_excess = excess.mean() * 12
    te = excess.std() * np.sqrt(12)
    return mean_excess / te if te > 0 else 0
```

### Tracking Quality Targets

| Metric | Passive | Enhanced Index | Active |
|--------|---------|----------------|--------|
| Tracking Error | < 50 bps | 50-150 bps | > 150 bps |
| Information Ratio | N/A | > 0.5 | > 0.5 |

---

## Common Benchmarks

| Index | Provider | Focus |
|-------|----------|-------|
| US Aggregate | Bloomberg Barclays | Investment grade universe |
| US Corporate | ICE BofA | Corporate bonds only |
| High Yield | Bloomberg Barclays | BB and below |
| Treasury | Bloomberg | US Government |

---

## Portfolio Replication

### Stratified Sampling

```python
def sample_benchmark(benchmark_holdings, target_holdings=50):
    """Select representative subset for index replication."""
    # Group by sector, rating, duration bucket
    # Select largest bonds from each cell
    pass
```

### Cell-Matching Targets

| Dimension | Tolerance |
|-----------|-----------|
| Duration | +/- 0.25 years |
| Sector Weight | +/- 2% |
| Quality Distribution | +/- 3% |

---

## Cross-References

- [Yield Analysis](yield_analysis.md)
- [Fixed Income Risk](../../03_risk/fixed_income_risk.md)
- [Credit Risk](../../../../.claude/skills/credit-risk/SKILL.md)

---

**Template**: KB Skills Integration v1.0
