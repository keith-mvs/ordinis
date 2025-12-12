# Long Call Butterfly - Quickstart

**Parent**: [Long Call Butterfly](../SKILL.md) | **Related**: [Strategy Mechanics](strategy-mechanics.md) | [Examples](examples.md)

---

## Overview

The long call butterfly is a multi-leg debit spread strategy. It is designed for traders with a neutral market outlook.

**Key Characteristics**:
- Neutral directional bias
- Defined maximum risk
- Defined maximum reward
- Suitable for moderate volatility environments

---

## When to Use

**Market Outlook**: Neutral with clear price targets

**Ideal Conditions**:
- Clear technical or fundamental catalyst
- Moderate implied volatility (IV rank 25-75)
- Sufficient time for thesis to develop (30-60 DTE recommended)
- Liquid options market

**Avoid When**:
- IV at extremes (>80 or <20 percentile)
- Insufficient time (<15 DTE for beginners)
- No clear catalyst or setup

---

## Position Sizing

**Capital Requirements**:
```
Risk 1-3% of portfolio per position
Maintain portfolio heat below 15%
Account for correlation with existing positions
```

**Example Position Sizing**:
```python
portfolio_value = 50000
risk_per_trade = portfolio_value * 0.02  # 2%
risk_amount = 1000

# Size position to risk amount
# Details vary by strategy structure
```

---

## Basic Setup

### Step 1: Select Underlying

**Screening Criteria**:
- High liquidity (>500K avg daily volume)
- Tight bid-ask spreads
- Active options market
- Clear directional catalyst

### Step 2: Choose Expiration

**Recommended: 30-60 DTE**
- Balance of time vs. cost
- Manageable theta decay
- Liquid monthly options

### Step 3: Select Strikes

**Strike Selection** varies by strategy:
- Consider price target
- Evaluate delta ranges
- Assess risk/reward ratio
- Check technical levels

See [Strike Selection](strike-selection.md) for detailed guidance.

### Step 4: Execute Trade

**Best Practices**:
- Use limit orders
- Enter as spread order (all legs together)
- Aim for mid-price or better
- Execute during liquid market hours

---

## Position Management

### Profit Targets

**Recommended**: Take profits at 50% of maximum profit
- Captures majority of gains
- Avoids late-stage risks
- Frees capital for new opportunities

### Stop Loss

**Recommended**: Exit at 50% of maximum loss
- Preserves capital
- Maintains discipline
- Allows multiple attempts

### Time Management

**Exit or Roll by 21 DTE**:
- Theta accelerates sharply below 21 days
- Pin risk increases near expiration
- Assignment risk rises for ITM options

---

## Quick Reference Checklist

### Pre-Trade
- [ ] Catalyst identified
- [ ] Technical confirmation
- [ ] IV rank checked (25-75 preferred)
- [ ] 30-60 DTE selected
- [ ] Position size within limits (1-3%)
- [ ] Profit/loss targets defined

### During Trade
- [ ] Daily monitoring
- [ ] P&L vs. targets
- [ ] Technical invalidation watch
- [ ] Portfolio heat check

### Exit
- [ ] Close as spread order
- [ ] Confirm full exit
- [ ] Log trade details
- [ ] Review lessons learned

---

## Common Mistakes

1. **Oversizing positions** - Risk too much per trade
2. **Ignoring IV levels** - Enter when IV too high/low
3. **No exit plan** - No predefined targets
4. **Holding too long** - Past 21 DTE threshold
5. **Emotional decisions** - Override discipline



---

## See Also

**Within This Skill**:
- [Quickstart](quickstart.md) - Getting started guide
- [Strategy Mechanics](strategy-mechanics.md) - Position structure and P&L
- [Examples](examples.md) - Real-world scenarios

**Master Resources**:
- [Options Greeks](../../options-strategies/references/greeks.md) - Comprehensive Greeks guide
- [Volatility Analysis](../../options-strategies/references/volatility.md) - IV metrics

**Related Strategies**:
- [Bull Call Spread](../../bull-call-spread/SKILL.md) - Bullish vertical spread
- [Iron Condor](../../iron-condor/SKILL.md) - Neutral range-bound strategy
- [Married Put](../../married-put/SKILL.md) - Stock protection

---

**Last Updated**: 2025-12-12
