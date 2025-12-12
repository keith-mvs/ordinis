# Long Call Butterfly - Spread Width Optimization

**Parent**: [Long Call Butterfly](../SKILL.md) | **Related**: [Strike Selection](strike-selection.md) | [Examples](examples.md)

---

## Spread Width Optimization for Long Call Butterfly

Choosing the optimal spread width significantly impacts risk/reward, capital efficiency, and probability of profit for long call butterfly positions.

---

## Spread Width Fundamentals

### What Is Spread Width?

**Definition**: Distance between strikes in the spread

**For Long Call Butterfly**:
[Strategy-specific spread width definition]

### Impact on Position

**Wider Spreads**:
- Higher capital requirement
- Higher maximum profit potential
- Wider profit zone
- More room for error

**Narrower Spreads**:
- Lower capital requirement
- Lower maximum profit potential
- Tighter profit zone
- More precision required

---

## Standard Spread Widths

### By Underlying Price

**Stock $0-$50**:
- $2.50 spreads (standard)
- $5.00 spreads (wider)

**Stock $50-$200**:
- $5.00 spreads (standard)
- $10.00 spreads (wider)

**Stock $200-$500**:
- $10.00 spreads (standard)
- $15-$20 spreads (wider)

**Stock >$500**:
- $25+ spreads (standard)
- Proportional to stock price

### Strike Increment Consideration

**Use Available Strikes**:
```python
# SPY at $450, strike increments of $1-$5
available_widths = [1, 2, 5, 10, 15, 20]

# AAPL at $175, strike increments of $2.50-$5
available_widths = [2.50, 5, 7.50, 10, 15]
```

**Liquidity**: Stick to standard widths for better fills

---

## Risk/Reward Analysis

### Calculating Risk/Reward

**For Debit Spreads**:
```python
spread_width = 10.00
net_debit = 6.50

max_loss = net_debit * 100  # $650
max_profit = (spread_width - net_debit) * 100  # $350
risk_reward_ratio = max_profit / max_loss  # 0.54:1
```

**For Credit Spreads**:
```python
spread_width = 5.00
net_credit = 2.00

max_profit = net_credit * 100  # $200
max_loss = (spread_width - net_credit) * 100  # $300
risk_reward_ratio = max_profit / max_loss  # 0.67:1
```

### Target Risk/Reward Ratios

**Aggressive**: 1.5:1 or better
```
Risk $200 to make $300+
Requires narrow spread or favorable pricing
```

**Balanced**: 1:1
```
Risk $250 to make $250
Standard 50% of width as debit/credit
```

**Conservative**: 0.75:1
```
Risk $300 to make $225
Higher probability, lower return
```

---

## Width Selection by Strategy Goal

### For Income Generation

**Goal**: Consistent premium collection

**Recommended Width**:
- Narrow to moderate (maximize frequency)
- Target high probability of profit
- Accept lower profit per trade

**Example**:
```
Stock at $100
Use $5 wide spreads
Collect $2.00 credit (40% of width)
High probability of keeping credit
```

### For Directional Plays

**Goal**: Capitalize on price movement

**Recommended Width**:
- Moderate to wide (capture full move)
- Align with price target
- Balance cost vs. profit potential

**Example**:
```
Stock at $150, target $135
Use $15 wide spread
Long $150 / Short $135
Captures full expected move
```

### For Volatility Trading

**Goal**: Profit from IV expansion/contraction

**Recommended Width**:
- Varies by volatility regime
- Wider in low IV (cheaper to establish)
- Narrower in high IV (expensive options)

---

## Capital Efficiency

### Comparing Spread Widths

**Example: $100 Stock**

**$5 Wide Spread**:
```
Cost: $250 (debit spread example)
Max Profit: $250
Return on Capital: 100%
Probability: 40%
Expected Value: $100 (40% × $250)
```

**$10 Wide Spread**:
```
Cost: $400
Max Profit: $600
Return on Capital: 150%
Probability: 30%
Expected Value: $180 (30% × $600)
```

**$15 Wide Spread**:
```
Cost: $650
Max Profit: $850
Return on Capital: 131%
Probability: 22%
Expected Value: $187 (22% × $850)
```

**Analysis**: Wider spread has better EV but requires more capital

### Portfolio Allocation

**Deploying Limited Capital**:
```python
available_capital = 10000

# Option 1: Narrow spreads
narrow_cost = 250
narrow_positions = available_capital / narrow_cost  # 40 positions
narrow_total_profit = 40 * 250  # $10,000 if all win

# Option 2: Wide spreads
wide_cost = 650
wide_positions = available_capital / wide_cost  # ~15 positions
wide_total_profit = 15 * 850  # $12,750 if all win

# But: probability matters
narrow_expected = 40 * (0.40 * 250)  # $4,000
wide_expected = 15 * (0.22 * 850)  # $2,807

# Narrow spreads win on expected value with high frequency
```

---

## Width and Time Decay

### Theta Impact by Width

**Narrow Spreads**:
- Less theta exposure (smaller position)
- Can trade frequently
- Time decay less impactful per position

**Wide Spreads**:
- More theta exposure
- Hold longer for full profit
- Time decay more significant

**Optimal DTE by Width**:
```
Narrow ($5): 30-45 DTE (can roll frequently)
Moderate ($10): 45-60 DTE (standard)
Wide ($15+): 60-90 DTE (need time for large move)
```

---

## Width Optimization Examples

### Example 1: Conservative ($5 Width)

**Setup**:
```
Stock: SPY at $450
Outlook: Moderately bullish
Capital: $500 per position

Width: $5
Strikes: $450/$455 (call spread example)
Cost: $2.50
Max Profit: $2.50
Risk/Reward: 1:1
```

**Pros**:
- Lower capital requirement
- Can open multiple positions
- High frequency strategy

**Cons**:
- Lower profit per trade
- Requires more precision
- More affected by slippage

### Example 2: Moderate ($10 Width)

**Setup**:
```
Stock: AAPL at $175
Outlook: Strong bullish conviction
Capital: $600 per position

Width: $10
Strikes: $175/$185
Cost: $4.00
Max Profit: $6.00
Risk/Reward: 1.5:1
```

**Pros**:
- Good risk/reward balance
- Standard, liquid width
- Reasonable capital requirement

**Cons**:
- More capital than narrow
- Fewer positions possible

### Example 3: Wide ($20 Width)

**Setup**:
```
Stock: TSLA at $250
Outlook: Very strong directional view
Capital: $1,200 per position

Width: $20
Strikes: $250/$270
Cost: $8.00
Max Profit: $12.00
Risk/Reward: 1.5:1
```

**Pros**:
- Excellent risk/reward
- Room for large move
- High profit potential

**Cons**:
- High capital requirement
- Lower position count
- Need strong conviction

---

## Adjusting Width for Volatility

### High IV Environment

**When IV Rank > 70**:
```
Options expensive
Use narrower spreads to reduce cost
OR
Use credit spreads (sell expensive premium)

Example:
Instead of $15 debit spread @ $10
Use $10 debit spread @ $6
Lower cost, still profitable
```

### Low IV Environment

**When IV Rank < 30**:
```
Options cheap
Can afford wider spreads
Better risk/reward available

Example:
$15 debit spread @ $6
Max profit: $9 (150% return)
Good opportunity due to cheap options
```

---

## Width Selection Checklist

### Pre-Selection
- [ ] Stock price level identified
- [ ] Price target determined
- [ ] Available capital calculated
- [ ] IV environment assessed

### Width Evaluation
- [ ] Standard widths for this stock checked
- [ ] Risk/reward calculated for each width
- [ ] Capital efficiency compared
- [ ] Liquidity verified

### Final Decision
- [ ] Width aligns with conviction level
- [ ] Within capital constraints
- [ ] Risk/reward acceptable (≥1:1)
- [ ] Liquidity adequate for entry/exit

---

## Common Width Mistakes

### Mistake 1: Too Narrow

**Problem**: $2 spread on $300 stock
**Impact**: High slippage, poor risk/reward
**Fix**: Use minimum $5-$10 width

### Mistake 2: Too Wide

**Problem**: $50 spread on $100 stock
**Impact**: Excessive capital, alternatives better
**Fix**: Use proportional width (5-15% of stock price)

### Mistake 3: Ignoring Liquidity

**Problem**: Using $7.50 width when only $5 and $10 liquid
**Impact**: Poor fills, wide bid-ask
**Fix**: Stick to standard increments

### Mistake 4: Fixed Width Only

**Problem**: Always using $5 regardless of stock/situation
**Impact**: Missing optimization opportunities
**Fix**: Adjust width based on conviction and capital



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
