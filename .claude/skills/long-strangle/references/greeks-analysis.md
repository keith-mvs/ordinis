# Long Strangle - Greeks Analysis

**Parent**: [Long Strangle](../SKILL.md) | **Related**: [Strategy Mechanics](strategy-mechanics.md) | [Position Management](position-management.md)

---

## Overview

Understanding Greeks is essential for managing long strangle positions effectively. This guide covers delta, gamma, theta, and vega impacts.

---

## Delta

### Position Delta

**What It Measures**: Expected P&L change for $1 move in underlying

**For This Strategy**:
[Strategy-specific delta characteristics]

**Typical Range**: [Delta range for this strategy]

### Delta Examples

```python
# Example position
[Strategy-specific delta calculation example]

# Expected P&L for $1 move
expected_profit = position_delta * 100
```

### Delta Management

**High Delta** (large directional exposure):
- Consider taking profits
- May want to reduce position
- Higher P&L swings

**Low Delta** (minimal directional exposure):
- Unlikely to profit from price movement
- Consider closing if thesis unchanged
- Time decay becomes dominant factor

---

## Gamma

### Position Gamma

**What It Measures**: Rate of delta change

**For This Strategy**:
[How gamma affects this specific strategy]

### Gamma Through Time

**Early** (45+ DTE):
- Low gamma
- Delta changes slowly
- More predictable

**Mid** (20-30 DTE):
- Gamma increasing
- Delta more sensitive
- Position can accelerate

**Late** (<10 DTE):
- Very high gamma
- Extremely sensitive
- Dangerous to hold

---

## Theta

### Position Theta

**What It Measures**: Daily time decay

**For This Strategy**:
[Net theta characteristics for this strategy]

### Theta Decay Curve

**45 DTE**:
```
Daily decay: $[X]-$[Y]
Status: Manageable
Action: Monitor
```

**30 DTE**:
```
Daily decay: $[X]-$[Y]
Status: Accelerating
Action: Plan exit or roll
```

**15 DTE**:
```
Daily decay: $[X]-$[Y]
Status: Severe
Action: Exit or roll immediately
```

**7 DTE**:
```
Daily decay: $[X]-$[Y]
Status: Critical
Action: Must exit
```

### Managing Theta

**Key Thresholds**:
- Exit or roll by 21 DTE
- Never hold past 7 DTE
- Monitor daily decay rate
- Compare to profit potential

---

## Vega

### Position Vega

**What It Measures**: Sensitivity to IV changes

**For This Strategy**:
[Vega characteristics and sign]

### IV Scenarios

**IV Expansion** (+10 points):
```
Impact: [$ impact on position]
Action: [Recommended response]
```

**IV Contraction** (-10 points):
```
Impact: [$ impact on position]
Action: [Recommended response]
```

### Vega Considerations

**When Entering**:
- Check IV rank/percentile
- Prefer IV rank 25-75
- Avoid extremes

**During Trade**:
- Monitor IV changes
- IV spike = profit opportunity
- IV crush = may need to adjust

---

## Combined Greeks Analysis

### Example Position

```python
# Position setup
[Specific position example]

# Greeks
delta = [value]
gamma = [value]
theta = [value]
vega = [value]
```

### Scenario Analysis

**Favorable Move**:
```python
# Stock moves in favorable direction
delta_pl = [calculation]
theta_pl = [calculation]
net_pl = [result]
```

**Adverse Move**:
```python
# Stock moves against position
delta_pl = [calculation]
theta_pl = [calculation]
net_pl = [result]
```

**Flat Price**:
```python
# No price movement
theta_pl = [calculation over time period]
decision = [hold, roll, or exit]
```

---

## Greeks-Based Management Rules

### Rule 1: Delta Alert

```
If delta too low/high:
→ Reassess position
→ Consider adjustment
```

### Rule 2: Theta Warning

```
If DTE < 15 and significant theta:
→ Exit or roll immediately
→ Don't fight theta decay
```

### Rule 3: Vega Opportunity

```
If IV changes dramatically:
→ Reassess profit targets
→ May exit early or hold longer
```

### Rule 4: Gamma Caution

```
If DTE < 10:
→ Very high gamma
→ Don't hold through expiration
→ Close by Thursday of exp week
```

---

## Greeks Monitoring

### Daily
- [ ] Check current delta
- [ ] Calculate theta decay
- [ ] Note IV changes
- [ ] Compare to entry Greeks

### Weekly
- [ ] Portfolio net delta
- [ ] Total theta across positions
- [ ] Identify extreme Greeks
- [ ] Plan adjustments

### Critical Thresholds
- [ ] DTE < 15: Exit/roll
- [ ] Theta > $[X]/day: Stressed
- [ ] IV change > 15 points: Reassess
- [ ] Gamma acceleration: Caution



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
