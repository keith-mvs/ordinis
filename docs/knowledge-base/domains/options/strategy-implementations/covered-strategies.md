# Covered Strategies Implementation

**Section**: 06_options/strategy_implementations
**Last Updated**: 2025-12-12
**Source Skills**: [covered-call](../../../../.claude/skills/covered-call/SKILL.md)

---

## Covered Call Strategy

### Position Structure

| Leg | Action | Purpose |
|-----|--------|---------|
| Stock | Own 100+ shares | Underlying position |
| Short Call | Sell OTM call | Income generation |

**Requirement**: Must own stock as collateral for short call.

### Strike Selection

| Approach | Strike | Delta | Best For |
|----------|--------|-------|----------|
| Conservative | 5-10% OTM | 0.15-0.20 | Moderately bullish |
| Moderate | 2-5% OTM | 0.25-0.35 | Neutral outlook |
| Aggressive | ATM | 0.40-0.50 | Maximum income |

### Risk Profile

```python
# Key metrics
premium_received = call_premium * 100
max_profit = (strike - stock_price) * 100 + premium_received
max_loss = (stock_price - premium) * 100  # Stock to zero
breakeven = stock_price - call_premium
```

### Example: AAPL Conservative Income

- Own 100 shares AAPL @ $180
- Sell $194.40 call (8% OTM) @ $1.80
- Premium: $180
- Max profit if assigned: $14.40 + $1.80 = $16.20/share (9%)
- Breakeven: $178.20

---

## Buy-Write Order

Execute stock purchase and call sale simultaneously:

```python
# Simultaneous execution
net_debit = (stock_price - call_premium) * 100

# Example: AAPL
# Buy 100 shares @ $180 = $18,000
# Sell 1 call @ $1.80 = $180 credit
# Net investment: $17,820
```

**Advantages**:
- Single execution
- Better pricing
- Lower commissions

---

## Rolling Strategies

### When to Roll

- DTE < 7 days
- Stock > 10% above strike
- Captured > 80% max profit

### Roll Types

| Type | When | Action |
|------|------|--------|
| Out | Same strike, later expiration | Collect more premium |
| Up and Out | Higher strike, later expiration | Capture more upside |
| Down and Out | Lower strike, later expiration | Defensive move |

```python
# Roll calculation
roll_credit = new_call_premium - buyback_cost
# Profitable if roll_credit > 0
```

---

## Management Guidelines

### Profit Target
- 50-80% of max premium
- Close and reestablish vs. hold to expiration

### Assignment Handling
1. Stock sold at strike price
2. Keep premium received
3. Realize capital gain: (strike - cost basis) + premium
4. Decide: repurchase and repeat?

### Risk Mitigation
- Stop-loss on stock: 10-15% below entry
- Diversify across 10+ positions
- Max 10% per position
- Monitor ex-dividend dates

---

## Tax Considerations

### Qualified Covered Call Requirements
- Expiration > 30 days
- Strike >= 95% of stock price (not deep ITM)
- Preserves long-term capital gains if stock held > 1 year

### Unqualified Covered Call
- Holding period tolled
- Gains may convert to short-term
- Consult tax advisor

---

## Greeks Summary

| Greek | Value | Impact |
|-------|-------|--------|
| Delta | Stock delta - call delta | ~0.50-0.85 |
| Theta | Positive | Earns time decay |
| Vega | Negative | Benefits from IV decrease |

---

## When to Use

**Use covered calls when**:
- Own stock, want income
- Neutral to moderately bullish
- VIX 12-25
- Willing to sell at strike

**Avoid when**:
- Strong breakout expected
- Before major catalysts
- Unwilling to part with stock

---

**Template**: KB Skills Integration v1.0
