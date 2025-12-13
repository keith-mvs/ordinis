# Butterfly Spreads Implementation

**Section**: 06_options/strategy_implementations
**Last Updated**: 2025-12-12
**Source Skills**: [long-call-butterfly](../../../../.claude/skills/long-call-butterfly/SKILL.md)

---

## Long Call Butterfly

### Position Structure

| Leg | Action | Strike | Quantity |
|-----|--------|--------|----------|
| Long Call | Buy | Lower (ITM) | 1 |
| Short Call | Sell | Middle (ATM) | 2 |
| Long Call | Buy | Upper (OTM) | 1 |

**Requirements**:
- Same expiration
- Equal wing spacing (Upper-Middle = Middle-Lower)

### Risk Profile

```python
net_debit = lower_premium + upper_premium - (2 * middle_premium)
max_profit = wing_width - net_debit  # At middle strike
max_loss = net_debit  # Outside wings
lower_breakeven = lower_strike + net_debit
upper_breakeven = upper_strike - net_debit
profit_zone = (lower_breakeven, upper_breakeven)
```

### Example: SPY Butterfly

SPY at $450, expect minimal movement:
- Buy $445 call @ $8.50
- Sell 2x $450 calls @ $5.00 = $10.00 credit
- Buy $455 call @ $2.50
- Net debit: $8.50 + $2.50 - $10.00 = $1.00 ($100)
- Max profit: $5.00 - $1.00 = $4.00 ($400)
- Max loss: $1.00 ($100)
- Breakevens: $446 / $454

---

## Wing Width Selection

| Width | Profit Zone | Max Profit | Best For |
|-------|-------------|------------|----------|
| $2.50-$5 | Narrow | Lower | High probability |
| $5-$10 | Standard | Moderate | Balanced |
| $10-$20 | Wide | Higher | Lower probability |

**Trade-off**: Wider wings = wider profit zone but lower probability of max profit.

---

## Probability Analysis

```python
# Key probability metrics
prob_in_zone = probability_stock_between(lower_breakeven, upper_breakeven)
prob_max_profit = probability_at_strike(middle_strike)
expected_value = (prob_in_zone * avg_profit) - (prob_outside * max_loss)
```

**Decision Criteria**:
- Prob in zone > 60%: Good setup
- Expected value > 0: Positive expectancy
- Max profit/Max loss > 3:1: Favorable R:R

---

## Ideal Conditions

- IV Rank < 40% (low volatility)
- Stock consolidating in tight range
- Post-earnings quiet period
- Near strong technical support/resistance
- Low expected move

---

## Greeks Profile

| Greek | At Middle | Outside Wings |
|-------|-----------|---------------|
| Delta | ~0 | Increases |
| Gamma | Positive | Negative |
| Theta | Positive | Lower |
| Vega | Negative | Less negative |

**Key insight**: Benefits from time decay and volatility decrease.

---

## Expiration Timing

| Cycle | Cost | Theta Benefit | Gamma Risk |
|-------|------|---------------|------------|
| 7-21 days | Low | Fast | High |
| 30-45 days | Moderate | Moderate | Moderate |
| 60-90 days | Higher | Slow | Low |

---

## Management

### Profit Targets
- 50-70% max profit: Excellent, reduce risk
- 80% max profit: Near optimal
- Hold to expiration: Only if confident in pin

### Stop Loss
- 100% of debit: Full max loss
- Stock outside wings: Exit (low recovery probability)
- IV spike > 50%: Exit (vega loss)

### Adjustments

**Stock moves up (above upper strike)**:
- Close position, accept loss
- Roll all strikes up
- Convert to iron condor

**Stock moves down (below lower strike)**:
- Close position, accept loss
- Roll all strikes down
- Convert to iron condor

---

## Comparison to Other Strategies

| vs. Strategy | Advantage | Disadvantage |
|--------------|-----------|--------------|
| Iron Butterfly | All long options, no margin | Pay debit vs. credit |
| Straddle | Lower cost, profits from no movement | Limited profit |
| Iron Condor | Higher max profit | Narrower profit zone |

---

## Execution

**Order Type**: Enter as single butterfly order (4 legs, 1 ticket)

**Best Practices**:
- Set limit at mid-point
- Adjust by $0.05 if not filled in 60 seconds
- Verify equal wing spacing
- Avoid wide bid/ask spreads (>15%)

---

## Risk Warnings

- **Narrow Profit Zone**: Small range for profitability
- **Gamma Risk**: Rapid position changes near wings
- **Assignment Risk**: Short middle calls can be assigned
- **Pin Risk**: Complex scenarios at exact strike
- **Complexity**: 4 legs = higher commissions

---

## When to Use

**Use butterfly when**:
- Expect minimal movement
- IV rank < 40%
- Clear technical pivot level
- Want defined risk, low capital

**Avoid when**:
- Expecting large moves
- High volatility environment
- Trending market
- Wide bid/ask spreads

---

**Template**: KB Skills Integration v1.0
