# Protective Strategies Implementation

**Section**: 06_options/strategy_implementations
**Last Updated**: 2025-12-12
**Source Skills**: [married-put](../../../../.claude/skills/married-put/SKILL.md), [protective-collar](../../../../.claude/skills/protective-collar/SKILL.md)

---

## Overview

Protective strategies hedge stock positions against downside risk. This document covers:
1. **Married Put**: Stock + protective put (unlimited upside, defined downside)
2. **Protective Collar**: Stock + put + short call (defined range)

---

## Married Put

### Position Structure

| Leg | Action | Purpose |
|-----|--------|---------|
| Stock | Own 100 shares | Long exposure |
| Long Put | Buy put | Downside insurance |

**Same expiration** for coordinated protection.

### Risk Profile

```python
total_cost = (stock_price * shares) + (put_premium * 100) + transaction_cost
max_loss = (stock_price - put_strike + put_premium) * 100
max_profit = unlimited  # Stock can rise indefinitely
breakeven = stock_price + put_premium
protection_pct = (stock_price - put_strike) / stock_price * 100
```

### Strike Selection

| Type | Strike | Cost | Protection |
|------|--------|------|------------|
| OTM | Below current | Lower | Less protection |
| ATM | At current | Moderate | Full protection |
| ITM | Above current | Higher | Maximum protection |

### Example: AAPL Married Put

- Buy 100 shares @ $175.50
- Buy $165 put @ $2.30 (5.9% OTM)
- Total cost: $175.50 + $2.30 = $177.80/share
- Max loss: $12.80/share ($1,280)
- Breakeven: $177.80

### Expiration Comparison

| Cycle | Premium | Monthly Cost | Management |
|-------|---------|--------------|------------|
| 30-day | Lower | Higher | Frequent rolling |
| 60-day | Moderate | Moderate | Balanced |
| 90-day | Higher | Lower | Infrequent |

---

## Protective Collar

### Position Structure

| Leg | Action | Strike | Purpose |
|-----|--------|--------|---------|
| Stock | Own 100 shares | - | Long exposure |
| Long Put | Buy OTM | Below current | Downside protection |
| Short Call | Sell OTM | Above current | Income to offset put |

**Same expiration** for all options.

### Risk Profile

```python
net_premium = put_premium - call_premium  # Often near zero
max_loss = (stock_price - put_strike + net_premium) * 100
max_profit = (call_strike - stock_price - net_premium) * 100
breakeven = stock_price + net_premium
protected_range = (put_strike, call_strike)
```

### Example: AAPL Collar

- Own 100 shares @ $175
- Buy $165 put @ $3.50 (5.7% OTM)
- Sell $185 call @ $3.25 (5.7% OTM)
- Net debit: $0.25/share ($25)
- Max loss: $10.25/share ($1,025)
- Max profit: $9.75/share ($975)
- Protected range: $165-$185

### Zero-Cost Collar

Adjust strikes to achieve net premium = $0:

```python
# Find strikes where put premium = call premium
# Trade-off: wider strikes (less protection, more cap)
```

**Example**:
- Put $162 @ $2.80
- Call $188 @ $2.80
- Net: $0

---

## Strategy Comparison

| Metric | Married Put | Protective Collar |
|--------|-------------|-------------------|
| Max Loss | Defined | Defined |
| Max Profit | Unlimited | Capped |
| Net Cost | Debit | Often zero-cost |
| Best For | Bullish + cautious | Neutral protection |

---

## Greeks Comparison

| Greek | Married Put | Protective Collar |
|-------|-------------|-------------------|
| Delta | ~0.50-0.70 | ~0.30-0.60 |
| Theta | Negative | Near zero |
| Vega | Positive | Near zero |

---

## Management at Expiration

### Married Put

| Scenario | Action |
|----------|--------|
| Stock > Strike | Put expires worthless, keep stock |
| Stock < Strike | Exercise put or sell at market |

### Protective Collar

| Scenario | Action |
|----------|--------|
| Stock between strikes | Both expire worthless, keep stock |
| Stock < Put strike | Exercise put, limit loss |
| Stock > Call strike | Stock called away, realize capped profit |

---

## When to Use

### Married Put
- New positions in uncertain markets
- Protecting unrealized gains
- Risk-averse investors
- Maximum upside preservation

### Protective Collar
- Large unrealized gains
- Cost-conscious protection
- Acceptable upside cap
- Earnings/event hedging

---

## Risk Warnings

**Married Put**:
- Premium reduces returns
- Time decay erodes value
- Transaction costs on rolling

**Protective Collar**:
- Capped upside (opportunity cost)
- Assignment risk on call
- Tax implications (qualified call rules)

---

**Template**: KB Skills Integration v1.0
