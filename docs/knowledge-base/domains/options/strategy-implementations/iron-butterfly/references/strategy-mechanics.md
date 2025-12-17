# Iron Butterfly - Strategy Mechanics

**Parent**: [Iron Butterfly](../SKILL.md) | **Related**: [Quickstart](quickstart.md) | [Greeks Analysis](greeks-analysis.md)

---

## Position Structure

The iron butterfly consists of multiple option legs working together to create a specific risk/reward profile.

### Components

[Detailed component description varies by strategy]

---

## Payoff Analysis

### Maximum Profit

**Formula**:
```
[Strategy-specific max profit formula]
```

**Occurs When**: [Price condition for max profit]

### Maximum Loss

**Formula**:
```
[Strategy-specific max loss formula]
```

**Occurs When**: [Price condition for max loss]

### Breakeven Points

**Formula**:
```
[Strategy-specific breakeven formula]
```

---

## Risk/Reward Profile

**Risk Metrics**:
- Maximum Risk: [Description]
- Maximum Reward: [Description]
- Risk/Reward Ratio: [Typical ratio]
- Probability of Profit: [Factors affecting probability]

---

## Time Decay (Theta)

### Theta Dynamics

**Net Theta Impact**:
- [How theta affects this strategy]
- [Theta behavior over time]
- [When theta works for/against you]

**Theta Management**:
- Exit or roll by 21 DTE
- Monitor daily theta decay
- Accelerated decay < 15 DTE

---

## Implied Volatility Impact

### Vega Exposure

**Net Vega**:
- [How IV changes affect position]
- [Vega sign and magnitude]

**IV Scenarios**:
- IV Expansion: [Effect on position]
- IV Contraction: [Effect on position]

**Best Practice**: Enter when IV rank 25-75

---

## Comparison to Alternatives

### vs. [Alternative Strategy 1]

**This Strategy**:
- [Advantage 1]
- [Advantage 2]
- [Disadvantage 1]

**Alternative**:
- [Its advantages]
- [Its disadvantages]



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
