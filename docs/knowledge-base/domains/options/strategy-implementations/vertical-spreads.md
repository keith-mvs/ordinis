# Vertical Spreads Implementation

**Section**: 06_options/strategy_implementations
**Last Updated**: 2025-12-12
**Source Skills**: [bull-call-spread](../../../../.claude/skills/bull-call-spread/SKILL.md), [bear-put-spread](../../../../.claude/skills/bear-put-spread/SKILL.md)

---

## Bull Call Spread

### Structure
- Buy lower strike call (ITM/ATM)
- Sell higher strike call (OTM)
- Same expiration

### Risk Profile
```python
net_debit = long_premium - short_premium
max_profit = spread_width - net_debit
max_loss = net_debit
breakeven = long_strike + net_debit
```

### Example: SPY $445/$455 Bull Call
- Buy $445 call @ $8.50
- Sell $455 call @ $3.20
- Net debit: $5.30
- Max profit: $4.70 (if SPY >= $455)
- Max loss: $5.30 (if SPY <= $445)
- Breakeven: $450.30

---

## Bear Put Spread

### Structure
- Buy higher strike put (ATM/ITM)
- Sell lower strike put (OTM)
- Same expiration

### Risk Profile
```python
net_debit = long_premium - short_premium
max_profit = spread_width - net_debit
max_loss = net_debit
breakeven = long_strike - net_debit
```

### Example: SPY $450/$445 Bear Put
- Buy $450 put @ $7.50
- Sell $445 put @ $5.00
- Net debit: $2.50
- Max profit: $2.50 (if SPY <= $445)
- Max loss: $2.50 (if SPY >= $450)
- Breakeven: $447.50

---

## Spread Width Guidelines

| Width | Risk | Reward | Best For |
|-------|------|--------|----------|
| $2.50-$5 | Low | Low | Small accounts |
| $5-$10 | Moderate | Moderate | Standard |
| $10-$20 | High | High | High conviction |

---

## Management
- **Profit target**: 50-75% max profit
- **Stop loss**: 100-150% of debit
- **DTE sweet spot**: 30-45 days
- **Roll**: Extend expiration or adjust strikes

---

**Template**: KB Skills Integration v1.0
