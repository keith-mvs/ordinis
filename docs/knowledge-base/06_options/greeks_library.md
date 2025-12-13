# Greeks Library and Options Trading Framework

**Section**: 06_options
**Last Updated**: 2025-12-12
**Source Skills**: [options-strategies](../../../.claude/skills/options-strategies/SKILL.md)

---

## Overview

Institutional-grade framework for options Greeks calculation, multi-leg strategy execution, and programmatic trading via Alpaca Markets APIs.

---

## Greeks Fundamentals

| Greek | Measures | Formula | Impact |
|-------|----------|---------|--------|
| Delta | Price sensitivity | dV/dS | ~0-1 for calls, ~0 to -1 for puts |
| Gamma | Delta rate of change | d²V/dS² | Higher near ATM, increases near expiry |
| Theta | Time decay | dV/dt | Negative for long options |
| Vega | Volatility sensitivity | dV/dσ | Higher for ATM options |
| Rho | Interest rate sensitivity | dV/dr | Usually minor for short-dated options |

---

## Black-Scholes Implementation

```python
from scipy.stats import norm
import numpy as np

def black_scholes_call(S, K, T, r, sigma):
    """Calculate call option price."""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    """Calculate option delta."""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1

def calculate_gamma(S, K, T, r, sigma):
    """Calculate option gamma (same for calls and puts)."""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def calculate_theta(S, K, T, r, sigma, option_type='call'):
    """Calculate option theta (per day)."""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == 'call':
        term2 = -r * K * np.exp(-r*T) * norm.cdf(d2)
    else:
        term2 = r * K * np.exp(-r*T) * norm.cdf(-d2)

    return (term1 + term2) / 365

def calculate_vega(S, K, T, r, sigma):
    """Calculate option vega (per 1% IV change)."""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100
```

---

## Strategy Decision Matrix

| Strategy | Market View | IV View | Max Risk | Max Profit | Best For |
|----------|-------------|---------|----------|------------|----------|
| Long Straddle | Neutral | Expansion | Premium | Unlimited | Earnings, events |
| Long Strangle | Neutral | Expansion | Premium | Unlimited | Lower cost vol play |
| Iron Butterfly | Neutral | Contraction | Wing-Premium | Premium | High IV, range-bound |
| Iron Condor | Range-bound | Contraction | Wing-Premium | Premium | Wide range, high IV |
| Bull Call Spread | Bullish | Low | Debit | Width-Debit | Directional, limited capital |
| Bear Put Spread | Bearish | Low | Debit | Width-Debit | Directional downside |
| Calendar Spread | Neutral | Term structure | Debit | Limited | Theta decay arbitrage |

---

## Volatility Analysis

### Implied Volatility Metrics

```python
def expected_move(stock_price, iv, dte):
    """Calculate expected move based on IV."""
    return stock_price * iv * np.sqrt(dte / 365)

def iv_percentile(current_iv, historical_ivs):
    """Calculate IV percentile rank."""
    return (historical_ivs < current_iv).sum() / len(historical_ivs) * 100

def iv_rank(current_iv, high_iv, low_iv):
    """Calculate IV rank."""
    return (current_iv - low_iv) / (high_iv - low_iv) * 100
```

### Entry Signals

| Signal | Condition | Strategy Type |
|--------|-----------|---------------|
| IV Rank > 50% | High IV environment | Premium selling |
| IV Rank < 30% | Low IV environment | Premium buying |
| Pre-earnings | IV expansion expected | Long volatility |
| Post-earnings | IV crush expected | Short volatility |

---

## Portfolio Greeks Management

```python
class PortfolioGreeks:
    """Aggregate portfolio-level Greeks."""

    def __init__(self):
        self.positions = []

    def add_position(self, delta, gamma, theta, vega, qty):
        self.positions.append({
            'delta': delta * qty * 100,
            'gamma': gamma * qty * 100,
            'theta': theta * qty * 100,
            'vega': vega * qty * 100
        })

    def net_greeks(self):
        return {
            'delta': sum(p['delta'] for p in self.positions),
            'gamma': sum(p['gamma'] for p in self.positions),
            'theta': sum(p['theta'] for p in self.positions),
            'vega': sum(p['vega'] for p in self.positions)
        }

    def delta_hedge_shares(self):
        """Shares needed to delta-neutralize."""
        return -round(self.net_greeks()['delta'])
```

---

## Risk Controls

### Position Limits

| Metric | Conservative | Moderate | Aggressive |
|--------|--------------|----------|------------|
| Max position size | 1% capital | 2% capital | 5% capital |
| Portfolio delta | < 50 | < 100 | < 200 |
| Portfolio gamma | < 10 | < 25 | < 50 |
| Max loss per trade | 1% | 2% | 5% |

### Pre-Trade Checklist

- [ ] Calculate maximum loss
- [ ] Verify sufficient buying power
- [ ] Confirm expiration and time decay profile
- [ ] Check IV rank/percentile
- [ ] Review upcoming earnings/events
- [ ] Calculate breakeven points
- [ ] Set profit target and stop-loss

---

## Execution Framework

### Alpaca API Multi-Leg Orders

```python
def submit_iron_condor(api, symbol, expiration, put_spread, call_spread):
    """
    Submit iron condor via Alpaca.
    put_spread: (long_put, short_put)
    call_spread: (short_call, long_call)
    """
    legs = [
        {'symbol': f'{symbol}{expiration}P{put_spread[0]}', 'qty': 1, 'side': 'buy'},
        {'symbol': f'{symbol}{expiration}P{put_spread[1]}', 'qty': 1, 'side': 'sell'},
        {'symbol': f'{symbol}{expiration}C{call_spread[0]}', 'qty': 1, 'side': 'sell'},
        {'symbol': f'{symbol}{expiration}C{call_spread[1]}', 'qty': 1, 'side': 'buy'},
    ]

    return api.submit_order(
        order_class='mleg',
        legs=legs,
        type='limit',
        time_in_force='day'
    )
```

---

## Performance Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Win Rate | Winning Trades / Total | > 50% |
| Profit Factor | Gross Profit / Gross Loss | > 1.5 |
| Sharpe Ratio | (Return - Rf) / StdDev | > 1.0 |
| Max Drawdown | Peak-to-Trough Loss | < 20% |
| Avg Win/Loss | Avg Win / Avg Loss | > 1.0 |

---

## Cross-References

- [Strategy Implementations](strategy_implementations/)
- [Iron Condors](strategy_implementations/iron_condors.md)
- [Volatility Strategies](strategy_implementations/volatility_strategies.md)
- [Technical Analysis](../02_signals/technical/README.md)

---

**Template**: KB Skills Integration v1.0
