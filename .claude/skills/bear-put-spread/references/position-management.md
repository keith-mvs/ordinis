# Bear Put Spread - Position Management

**Parent**: [Bear Put Spread](../SKILL.md) | **Related**: [Strategy Mechanics](strategy-mechanics.md) | [Greeks Analysis](greeks-analysis.md)

---

## Management Philosophy

Effective position management separates profitable traders from the rest. For bear put spread positions, disciplined management is essential.

**Core Principles**:
1. Take profits when available
2. Cut losses when thesis invalidated
3. Manage time decay actively
4. Avoid pin risk at expiration

---

## Profit Targets

### 50% Rule (Recommended)

**Target**: Close at 50% of maximum profit

**Why 50%?**
- Captures majority of theoretical gains
- Avoids late-stage risks (theta, pin risk)
- Frees capital for new opportunities
- Statistically optimal for repeated trades

**Implementation**:
```python
max_profit = [calculated max profit]
target = max_profit * 0.50
# Close when position reaches target
```

### Alternative Targets

**75% of Max Profit** (Aggressive):
- Higher return per trade
- More time/pin risk
- Use with strong conviction only

**25-30% of Max Profit** (Conservative):
- Quick profits
- High frequency
- Lower per-trade returns

---

## Stop Loss Guidelines

### 50% Loss Threshold

**Standard Stop**: Exit at 50% of maximum loss

**Why 50%?**
- Preserves capital for recovery
- Limits emotional damage
- Allows multiple attempts
- Maintains discipline

**Time-Based Stop**:
```
If underwater at 10 DTE:
- Exit regardless of loss %
- Theta too severe to recover
```

### Thesis Invalidation

**Exit Immediately When**:
- Technical setup breaks (reversal through key level)
- Fundamental catalyst changes
- Sector correlation breaks

Don't wait for numerical stop loss if thesis clearly wrong.

---

## Rolling Strategies

### When to Roll

**Roll Out** (Extend Time):
- Thesis still valid
- Need more time
- 15-21 DTE remaining

**Roll for Credit**:
- Collect additional premium
- Extend duration
- Adjust strikes if needed

**Don't Roll If**:
- Thesis invalidated
- Already rolled once
- Better opportunities exist

---

## Adjustment Strategies

### Favorable Price Movement

**Options**:
1. **Take Profits**: Close at target (recommended)
2. **Roll Forward**: Lock in gains, extend exposure
3. **Tighten Stops**: Protect unrealized gains

### Adverse Price Movement

**Options**:
1. **Hold**: If within tolerance and thesis intact
2. **Close**: If stop loss triggered
3. **Roll**: If confident in thesis, need more time

---

## Monitoring Schedule

### Daily Checks

**Morning** (Market Open):
- Check underlying price
- Calculate current P&L
- Note overnight news
- Set price alerts

**Evening** (Market Close):
- Log closing price
- Update P&L
- Calculate DTE
- Plan next day

### Weekly Review

**Every Monday**:
- Review all positions
- Check DTE remaining
- Assess rolling needs
- Validate thesis
- Calculate portfolio heat

---

## Special Situations

### Early Assignment

**If Short Option Assigned**:
1. Don't panic
2. Exercise long option to offset (if applicable)
3. Contact broker
4. Understand P&L impact

**Prevention**:
- Close ITM positions before expiration week
- Monitor dividend dates
- Watch for low extrinsic value

### Pin Risk

**Definition**: Stock closes very near strike at expiration

**Problem**: Uncertain assignment status

**Prevention**:
- Close all positions by Friday 3 PM (expiration week)
- Never hold through 4 PM close if near strikes
- Especially critical if position near max profit/loss

---

## Exit Execution

### How to Close

**Preferred**: Close as multi-leg order
```
Order Type: [Strategy name] spread
Action: Sell to Close (or Buy to Close)
All legs together
Limit price at mid or better
```

**Avoid**: Closing legs separately
- Higher slippage
- Execution risk
- Poor fills

### Timing

**Best Times**:
- 9:45-10:30 AM ET (post-open, volatility settled)
- 2:00-3:30 PM ET (pre-close, good liquidity)

**Avoid**:
- First/last 15 minutes
- Low volume periods
- Right before major news

---

## Performance Tracking

### Metrics to Log

**Per Trade**:
- Entry/exit dates
- Strikes and expiration
- Net debit/credit
- P&L ($$ and %)
- Days held
- Exit reason

**Portfolio Level**:
- Win rate
- Average win/loss
- Profit factor
- Expected value

---

## Management Checklist

### Entry
- [ ] Position size within limits
- [ ] Profit target defined (50% max)
- [ ] Stop loss set (50% max loss)
- [ ] Exit plan documented
- [ ] Calendar alert set (21 DTE)

### During Trade
- [ ] Daily price monitoring
- [ ] Weekly thesis review
- [ ] Portfolio heat check
- [ ] Adjustment triggers identified

### Exit
- [ ] Close as spread order
- [ ] Verify all legs closed
- [ ] Log in trade journal
- [ ] Review lessons learned



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
