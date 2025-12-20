# Options, Futures, and Other Derivatives

> **Industry Standard for Derivatives Pricing**

---

## Metadata

| Field | Value |
|-------|-------|
| **ID** | `pub_hull_options_futures` |
| **Author** | John C. Hull |
| **Published** | 2021 (10th Edition) |
| **Publisher** | Pearson |
| **ISBN** | 978-0136939979 |
| **Domain** | 6 (Options & Derivatives) |
| **Type** | Textbook |
| **Audience** | Intermediate |
| **Practical/Theoretical** | 0.5 |
| **Status** | `pending_review` |
| **Version** | v1.0.0 |

---

## Overview

The industry-standard textbook on derivatives pricing and risk management. Used in CFA, FRM, and MBA programs worldwide. Covers everything from basic options to exotic derivatives and counterparty credit risk.

---

## Key Topics

### Part 1: Fundamentals (Ch 1-11)
- Futures and forwards mechanics
- Hedging strategies with futures
- Interest rate markets
- Swaps (interest rate, currency, commodity)
- **Options fundamentals** - calls, puts, payoffs
- **Trading strategies** - spreads, straddles, strangles

### Part 2: Options Pricing (Ch 12-17)
- **Binomial Trees** - Discrete-time pricing
- **Black-Scholes-Merton** - Continuous-time model
- **Greeks** - Delta, Gamma, Vega, Theta, Rho
- Dividend treatment
- American options pricing

### Part 3: Advanced Topics (Ch 18-30)
- Volatility smiles and surfaces
- Exotic options
- Value at Risk (VaR)
- Credit derivatives
- Real options

---

## Critical Formulas

### Black-Scholes Call Option Price
```
C = S₀N(d₁) - Ke^(-rT)N(d₂)

Where:
d₁ = [ln(S₀/K) + (r + σ²/2)T] / (σ√T)
d₂ = d₁ - σ√T

S₀ = Current stock price
K = Strike price
T = Time to expiration
r = Risk-free rate
σ = Volatility
N(·) = Cumulative standard normal distribution
```

### The Greeks

#### Delta (Δ)
```
Δ = ∂C/∂S ≈ N(d₁)  (for calls)
```
**Meaning:** Change in option price per $1 change in underlying

#### Gamma (Γ)
```
Γ = ∂²C/∂S² = N'(d₁) / (S₀σ√T)
```
**Meaning:** Rate of change of delta

#### Vega (ν)
```
ν = ∂C/∂σ = S₀N'(d₁)√T
```
**Meaning:** Change in option price per 1% change in volatility

#### Theta (Θ)
```
Θ = ∂C/∂T
```
**Meaning:** Time decay (typically negative for long options)

#### Rho (ρ)
```
ρ = ∂C/∂r = KTe^(-rT)N(d₂)
```
**Meaning:** Change in option price per 1% change in interest rates

---

## Integration with Intelligent Investor System

### SignalCore Options Module

| Concept | Application | Priority |
|---------|------------|----------|
| Black-Scholes | Theoretical value calculation | Critical |
| Greeks | Risk metrics | Critical |
| Volatility Surface | Implied vol interpolation | High |
| Binomial Trees | American options pricing | Medium |

**Implementation:**
```python
class OptionsSignalModel:
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate all Greeks for an option."""
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        if option_type == 'call':
            delta = norm.cdf(d1)
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:  # put
            delta = -norm.cdf(-d1)
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        theta = ... # Implementation
        rho = ... # Implementation

        return {
            'price': price, 'delta': delta, 'gamma': gamma,
            'vega': vega, 'theta': theta, 'rho': rho
        }
```

### RiskGuard Options Risk

| Risk Metric | Application | Priority |
|------------|------------|----------|
| Portfolio Delta | Directional risk | Critical |
| Portfolio Gamma | Convexity risk | Critical |
| Portfolio Vega | Volatility exposure | High |
| Greeks Limits | Position limits | High |

**Risk Rules:**
- `RO001`: Portfolio delta within [-1000, +1000] shares
- `RO002`: Max vega exposure ±$10,000 per 1% vol move
- `RO003`: Gamma limits to avoid pin risk near expiration

---

## Key Options Strategies

### Covered Call
- **Structure:** Long stock + Short call
- **Max Profit:** Strike - Entry + Premium
- **Max Loss:** Entry - Premium (if stock → 0)
- **Use Case:** Generate income on existing holdings

### Protective Put
- **Structure:** Long stock + Long put
- **Max Profit:** Unlimited
- **Max Loss:** Entry - Strike + Premium paid
- **Use Case:** Downside protection (insurance)

### Vertical Spread
- **Bull Call Spread:** Buy low strike call + Sell high strike call
- **Bear Put Spread:** Buy high strike put + Sell low strike put
- **Max Profit:** Limited to strike difference - net premium
- **Use Case:** Directional view with defined risk

### Iron Condor
- **Structure:** OTM put spread + OTM call spread
- **Max Profit:** Net premium received
- **Max Loss:** Width of wider spread - premium
- **Use Case:** Profit from low volatility

---

## Critical Insights

### Volatility Smile
- **Problem:** Black-Scholes assumes constant volatility
- **Reality:** Implied volatility varies by strike
- **Pattern:** Often higher IV for OTM puts (crash risk)

**Implications:**
- Use market IV, not historical vol
- IV surface interpolation for off-market strikes
- Skew trading opportunities

### Put-Call Parity
```
C - P = S - Ke^(-rT)
```

**Use:**
- Arbitrage detection
- Synthetic position creation
- Pricing consistency check

### Early Exercise
- **American calls on non-dividend stocks:** Never optimal
- **American puts:** May be optimal to exercise early
- **Use binomial trees** for American option pricing

---

## Related Publications

- **Natenberg, "Option Volatility and Pricing"** - Practitioner focus
- **Taleb, "Dynamic Hedging"** - Options trading
- **Wilmott, "Paul Wilmott on Quantitative Finance"** - Mathematical depth

---

## Tags

`options`, `derivatives`, `black_scholes`, `greeks`, `hedging`, `volatility`, `options_strategies`, `risk_neutral_valuation`

---

**Document Version:** v1.0.0
**Last Updated:** 2025-01-28
**Status:** indexed
