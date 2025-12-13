# Volatility Strategies Implementation

**Section**: 06_options/strategy_implementations
**Last Updated**: 2025-12-12
**Source Skills**: [iron-butterfly](../../../../.claude/skills/iron-butterfly/SKILL.md), [long-straddle](../../../../.claude/skills/long-straddle/SKILL.md), [long-strangle](../../../../.claude/skills/long-strangle/SKILL.md)

---

## Overview

Volatility strategies profit from expectations about price movement magnitude rather than direction. This document covers neutral strategies (iron butterfly) and volatility expansion strategies (long straddle, long strangle).

---

## Iron Butterfly

### Position Structure

| Leg | Action | Strike | Purpose |
|-----|--------|--------|---------|
| Short Call | Sell | ATM | Collect premium (straddle) |
| Short Put | Sell | ATM | Collect premium (straddle) |
| Long Call | Buy | OTM | Upside protection |
| Long Put | Buy | OTM | Downside protection |

**All legs share same expiration.**

```python
# Risk profile
net_credit = atm_call_premium + atm_put_premium - otm_call_premium - otm_put_premium
max_profit = net_credit * 100  # At ATM strike
max_loss = (wing_width - net_credit) * 100
profit_zone = (atm_strike - net_credit, atm_strike + net_credit)
```

### Ideal Conditions

- IV Rank > 70% (very high)
- Expect stock to pin at ATM strike
- Post-earnings or post-event IV crush
- High open interest at ATM strike
- Short DTE (7-21 days)

### Example

SPY at $450, IV Rank 85:
- Sell $450 straddle @ $10.75
- Buy $440/$460 strangle @ $1.90
- Net credit: $8.85
- Max profit: $885 (if SPY exactly $450)
- Max loss: $115 (if SPY beyond wings)
- Profit zone: $441.15 - $458.85

---

## Long Straddle

### Position Structure

| Leg | Action | Strike | Purpose |
|-----|--------|--------|---------|
| Long Call | Buy | ATM | Upside exposure |
| Long Put | Buy | ATM | Downside exposure |

```python
# Risk profile
total_cost = call_premium + put_premium
max_loss = total_cost * 100
max_profit = unlimited
upper_breakeven = strike + total_cost
lower_breakeven = strike - total_cost
required_move = total_cost / underlying_price
```

### Ideal Conditions

- IV Rank 50-90% (elevated but not extreme)
- Binary event imminent (earnings, FDA, Fed)
- Historical moves > straddle cost
- 1-7 days before catalyst

### Example

NVDA at $475, earnings in 3 days:
- Buy $475 call @ $22.50
- Buy $475 put @ $21.00
- Total cost: $43.50 ($4,350)
- Breakevens: $431.50 / $518.50
- Required move: 9.2%

---

## Long Strangle

### Position Structure

| Leg | Action | Strike | Purpose |
|-----|--------|--------|---------|
| Long Call | Buy | OTM | Upside exposure |
| Long Put | Buy | OTM | Downside exposure |

```python
# Risk profile
total_cost = call_premium + put_premium
max_loss = total_cost * 100
max_profit = unlimited
upper_breakeven = call_strike + total_cost
lower_breakeven = put_strike - total_cost
```

### vs. Straddle Comparison

| Metric | Straddle | Strangle |
|--------|----------|----------|
| Cost | Higher | 30-50% lower |
| Breakevens | Narrower | Wider |
| Gamma | Higher | Lower |
| Required Move | Smaller | Larger |

### Example

TSLA at $250, expecting large move:
- Buy $265 call @ $8.50
- Buy $235 put @ $7.75
- Total cost: $16.25 ($1,625)
- Breakevens: $218.75 / $281.25
- Required move: ~12%

---

## Greeks Comparison

| Greek | Iron Butterfly | Long Straddle | Long Strangle |
|-------|---------------|---------------|---------------|
| Delta | ~0 (neutral) | ~0 (neutral) | ~0 (neutral) |
| Gamma | Very negative | Very positive | Positive |
| Theta | High positive | Very negative | Negative |
| Vega | High negative | Very positive | Positive |

---

## Strategy Selection Matrix

| Condition | Iron Butterfly | Long Straddle | Long Strangle |
|-----------|---------------|---------------|---------------|
| IV Rank > 70% | Preferred | Avoid | Avoid |
| IV Rank 50-70% | Consider | Preferred | Consider |
| Expect pin at strike | Preferred | Avoid | Avoid |
| Expect 10-15% move | Avoid | Preferred | Consider |
| Expect >15% move | Avoid | Consider | Preferred |
| Cost sensitivity | Low cost (credit) | High cost | Moderate cost |

---

## Management Guidelines

### Iron Butterfly

- **Profit target**: 50-70% of max credit
- **Loss limit**: 2× credit received
- **Adjustment**: Convert to iron condor if threatened
- **Exit**: Day after event to capture IV crush

### Long Straddle/Strangle

- **Profit target**: 20-50% of premium
- **Loss limit**: 50-70% of premium
- **Exit**: Day after event (avoid IV crush)
- **Hold rule**: Cut if no move by event

---

## Code Library

See: `code/options/volatility_strategies.py` for implementation.

---

**Template**: KB Skills Integration v1.0
**Lines**: ~200
