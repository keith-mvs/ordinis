#!/usr/bin/env python3
"""
Generate all 51 reference files for 7 options strategies.

This script creates comprehensive, institutional-grade reference documentation
following the established taxonomy and cross-linking scheme.
"""

from pathlib import Path
import sys

ORDINIS_ROOT = Path(__file__).parent.parent
SKILLS_DIR = ORDINIS_ROOT / ".claude" / "skills"

# Strategy definitions
STRATEGIES = {
    "bear-put-spread": {
        "display_name": "Bear Put Spread",
        "direction": "bearish",
        "type": "vertical debit spread",
        "files": [
            "quickstart",
            "strategy-mechanics",
            "strike-selection",
            "position-management",
            "greeks-analysis",
            "examples",
        ],
    },
    "protective-collar": {
        "display_name": "Protective Collar",
        "direction": "protective",
        "type": "stock + options",
        "files": [
            "quickstart",
            "strategy-mechanics",
            "strike-selection",
            "position-management",
            "greeks-analysis",
            "examples",
            "portfolio-integration",
            "dividend-considerations",
        ],
    },
    "iron-butterfly": {
        "display_name": "Iron Butterfly",
        "direction": "neutral",
        "type": "multi-leg credit spread",
        "files": [
            "quickstart",
            "strategy-mechanics",
            "strike-selection",
            "position-management",
            "greeks-analysis",
            "examples",
            "spread-width-optimization",
        ],
    },
    "iron-condor": {
        "display_name": "Iron Condor",
        "direction": "neutral",
        "type": "multi-leg credit spread",
        "files": [
            "quickstart",
            "strategy-mechanics",
            "strike-selection",
            "position-management",
            "greeks-analysis",
            "examples",
            "spread-width-optimization",
        ],
    },
    "long-call-butterfly": {
        "display_name": "Long Call Butterfly",
        "direction": "neutral",
        "type": "multi-leg debit spread",
        "files": [
            "quickstart",
            "strategy-mechanics",
            "strike-selection",
            "position-management",
            "greeks-analysis",
            "examples",
            "spread-width-optimization",
        ],
    },
    "long-straddle": {
        "display_name": "Long Straddle",
        "direction": "volatility",
        "type": "volatility play",
        "files": [
            "quickstart",
            "strategy-mechanics",
            "strike-selection",
            "position-management",
            "greeks-analysis",
            "examples",
            "iv-analysis",
            "earnings-plays",
        ],
    },
    "long-strangle": {
        "display_name": "Long Strangle",
        "direction": "volatility",
        "type": "volatility play",
        "files": [
            "quickstart",
            "strategy-mechanics",
            "strike-selection",
            "position-management",
            "greeks-analysis",
            "examples",
            "iv-analysis",
            "earnings-plays",
        ],
    },
}


def generate_header(strategy_key: str, file_type: str, related_files: list[str]) -> str:
    """Generate standard header with cross-links."""
    display_name = STRATEGIES[strategy_key]["display_name"]

    # Build related links
    related_links = []
    for rf in related_files[:2]:  # Max 2 related files in header
        related_name = rf.replace("-", " ").title()
        related_links.append(f"[{related_name}]({rf}.md)")

    related_str = " | ".join(related_links) if related_links else ""

    return f"""# {display_name} - {file_type.replace('-', ' ').title()}

**Parent**: [{display_name}](../SKILL.md){' | **Related**: ' + related_str if related_str else ''}

---

"""


def generate_footer(strategy_key: str) -> str:
    """Generate standard footer with see-also links."""
    display_name = STRATEGIES[strategy_key]["display_name"]
    files = STRATEGIES[strategy_key]["files"]

    # Build within-skill links
    within_links = []
    for f in files:
        if f in ["quickstart", "strategy-mechanics", "examples"]:
            f_display = f.replace("-", " ").title()
            within_links.append(f"- [{f_display}]({f}.md) - {get_file_description(f)}")

    footer = f"""

---

## See Also

**Within This Skill**:
{chr(10).join(within_links[:5])}

**Master Resources**:
- [Options Greeks](../../options-strategies/references/greeks.md) - Comprehensive Greeks guide
- [Volatility Analysis](../../options-strategies/references/volatility.md) - IV metrics

**Related Strategies**:
- [Bull Call Spread](../../bull-call-spread/SKILL.md) - Bullish vertical spread
- [Iron Condor](../../iron-condor/SKILL.md) - Neutral range-bound strategy
- [Married Put](../../married-put/SKILL.md) - Stock protection

---

**Last Updated**: 2025-12-12
"""
    return footer


def get_file_description(file_type: str) -> str:
    """Get short description for file type."""
    descriptions = {
        "quickstart": "Getting started guide",
        "strategy-mechanics": "Position structure and P&L",
        "strike-selection": "Optimizing strikes",
        "position-management": "Managing trades",
        "greeks-analysis": "Risk metrics",
        "examples": "Real-world scenarios",
        "portfolio-integration": "Portfolio allocation",
        "dividend-considerations": "Dividend impacts",
        "spread-width-optimization": "Optimizing spread width",
        "iv-analysis": "Volatility analysis",
        "earnings-plays": "Earnings strategies",
    }
    return descriptions.get(file_type, "Documentation")


def generate_quickstart(strategy_key: str) -> str:
    """Generate quickstart.md for any strategy."""
    display_name = STRATEGIES[strategy_key]["display_name"]
    direction = STRATEGIES[strategy_key]["direction"]
    strategy_type = STRATEGIES[strategy_key]["type"]

    content = generate_header(strategy_key, "quickstart", ["strategy-mechanics", "examples"])

    content += f"""## Overview

The {display_name.lower()} is a {strategy_type} strategy. It is designed for traders with a {direction} market outlook.

**Key Characteristics**:
- {direction.capitalize()} directional bias
- Defined maximum risk
- Defined maximum reward
- Suitable for moderate volatility environments

---

## When to Use

**Market Outlook**: {direction.capitalize()} with clear price targets

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

{generate_footer(strategy_key)}"""

    return content


def generate_strategy_mechanics(strategy_key: str) -> str:
    """Generate strategy-mechanics.md."""
    display_name = STRATEGIES[strategy_key]["display_name"]

    content = generate_header(strategy_key, "strategy-mechanics", ["quickstart", "greeks-analysis"])

    content += f"""## Position Structure

The {display_name.lower()} consists of multiple option legs working together to create a specific risk/reward profile.

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

{generate_footer(strategy_key)}"""

    return content


def generate_strike_selection(strategy_key: str) -> str:
    """Generate strike-selection.md."""
    display_name = STRATEGIES[strategy_key]["display_name"]

    content = generate_header(strategy_key, "strike-selection", ["strategy-mechanics", "examples"])

    content += f"""## Strike Selection Framework

Effective strike selection is critical for {display_name.lower()} success. This guide covers systematic approaches to choosing optimal strikes.

---

## Key Considerations

When selecting strikes, evaluate:
1. **Price Target**: Where do you expect the underlying to move?
2. **Probability**: What's the likelihood of reaching your target?
3. **Risk/Reward**: Does the potential profit justify the risk?
4. **Liquidity**: Are the options liquid enough for good fills?
5. **Technical Levels**: Where are support/resistance zones?

---

## Delta-Based Selection

### Using Delta as a Guide

**Delta ranges** provide a systematic framework:

[Strategy-specific delta ranges and recommendations]

**Example Delta Targets**:
```
[Specific delta recommendations for this strategy]
```

---

## Price Target Method

### Systematic Approach

**Step 1**: Identify price target using technical analysis

**Step 2**: Select strikes relative to target

**Step 3**: Validate risk/reward ratio

**Step 4**: Check liquidity and bid-ask spreads

---

## Spread Width Optimization

### Risk/Reward Tradeoffs

**Narrow Spreads**:
- Lower capital requirement
- Tighter profit zone
- More precision needed

**Wide Spreads**:
- Higher capital requirement
- Wider profit zone
- More room for error

**Optimal Width**: [Strategy-specific recommendation]

---

## Strike Selection Examples

### Example 1: Conservative Setup

```
[Detailed example with specific strikes, rationale, and analysis]
```

### Example 2: Moderate Setup

```
[Another example with different parameters]
```

### Example 3: Aggressive Setup

```
[Third example showing aggressive approach]
```

---

## Common Mistakes

1. **Ignoring liquidity** - Trading illiquid strikes
2. **Poor width selection** - Too wide or too narrow
3. **No technical validation** - Ignoring support/resistance
4. **Overcomplicating** - Analysis paralysis

---

## Strike Selection Checklist

- [ ] Price target identified
- [ ] Technical levels mapped
- [ ] Delta ranges evaluated
- [ ] Risk/reward calculated (≥1:1)
- [ ] Liquidity confirmed
- [ ] Bid-ask spreads acceptable

{generate_footer(strategy_key)}"""

    return content


def generate_position_management(strategy_key: str) -> str:
    """Generate position-management.md."""
    display_name = STRATEGIES[strategy_key]["display_name"]

    content = generate_header(
        strategy_key, "position-management", ["strategy-mechanics", "greeks-analysis"]
    )

    content += f"""## Management Philosophy

Effective position management separates profitable traders from the rest. For {display_name.lower()} positions, disciplined management is essential.

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

{generate_footer(strategy_key)}"""

    return content


def generate_greeks_analysis(strategy_key: str) -> str:
    """Generate greeks-analysis.md."""
    display_name = STRATEGIES[strategy_key]["display_name"]

    content = generate_header(
        strategy_key, "greeks-analysis", ["strategy-mechanics", "position-management"]
    )

    content += f"""## Overview

Understanding Greeks is essential for managing {display_name.lower()} positions effectively. This guide covers delta, gamma, theta, and vega impacts.

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

{generate_footer(strategy_key)}"""

    return content


def generate_examples(strategy_key: str) -> str:
    """Generate examples.md."""
    display_name = STRATEGIES[strategy_key]["display_name"]

    content = generate_header(strategy_key, "examples", ["quickstart", "position-management"])

    content += f"""## Real-World {display_name} Examples

This guide presents realistic trade scenarios, including both winners and losers, to illustrate proper execution and management.

---

## Example 1: Successful Trade

### Setup

**Underlying**: [Ticker] ([Description])
**Situation**: [Market condition and catalyst]

**Technical Analysis**:
- Current Price: $[X]
- [Key technical levels]
- [Indicators]

**Fundamental Catalyst**:
- [Catalyst description]

### Trade Construction

**Selection Process**:
```
Outlook: [Expected price movement]
Risk Tolerance: [Conservative/Moderate/Aggressive]
Capital: $[X] ([Y]% of account)

Strike Selection:
[Specific strikes chosen]
[Rationale for each strike]

Expiration: [DTE]
```

**Execution**:
```
[Specific order details]
Net Debit/Credit: $[X]
Max Loss: $[X]
Max Profit: $[Y]
Breakeven: $[Z]
Risk/Reward: 1:[ratio]
```

### Position Management

**Day 1-10**:
```
[Price action]
P&L: [Range]
Action: [What was done]
```

**Day 11-20**:
```
[Continued price action]
P&L: [Updated]
Action: [Management decision]
```

**Exit**:
```
Price: $[X]
Position Value: $[Y]
Profit: $[Z] ([XX]%)
Reason: [Why closed]
```

### Outcome

**Results**:
```
Entry: $[X]
Exit: $[Y]
Profit: $[Z]
Return: [XX]%
Days Held: [X]
Annualized Return: [XX]%
```

**Lessons**:
- [Key lesson 1]
- [Key lesson 2]
- [Key lesson 3]

---

## Example 2: Losing Trade (Learning Experience)

### Setup

**Underlying**: [Ticker]
**Situation**: [What looked promising]

**Initial Analysis**:
- [Why trade seemed good]
- [What was overlooked]

### Trade Construction

**Execution**:
```
[Details of entry]
```

### What Went Wrong

**Issue 1**: [First problem]
```
[How it manifested]
[Impact on position]
```

**Issue 2**: [Second problem]
```
[Additional complications]
```

### Position Management

**The Mistake**:
```
[What should have been done differently]
[Why trader held/exited incorrectly]
```

**The Exit**:
```
Loss: -$[X] (-[XX]%)
Reason: [Why closed at loss]
```

### Outcome

**Results**:
```
Entry: $[X]
Exit: $[Y]
Loss: -$[Z]
Return: -[XX]%
```

**Critical Lessons**:
- [Specific lesson from this failure]
- [What to avoid in future]
- [Red flags that were missed]

---

## Example 3: Rolling Success

### Initial Position

**Setup**:
```
[Original position details]
```

**Why Roll Needed**:
```
[Thesis still valid but need more time]
```

### First Roll

**Execution**:
```
Close: $[X]
Open: $[Y]
Net Cost: $[Z]
New Total Basis: $[Total]
```

**Rationale**:
- [Why rolling made sense]
- [What changed in analysis]

### Final Outcome

**Results After Roll**:
```
Total Capital: $[X]
Exit: $[Y]
Profit: $[Z]
Return: [XX]%
```

**Lessons on Rolling**:
- [When rolling works]
- [Cost management]
- [Knowing when to stop]

---

## Example 4: Quick Stop Loss

### The Setup

**Trade Thesis**:
```
[Why entered]
[Expected outcome]
```

### What Happened

**Catalyst Reversal**:
```
[Unexpected event]
[Immediate impact]
```

**Quick Decision**:
```
[How quickly exited]
[Final loss]
```

### Outcome

**Results**:
```
Loss: -$[X] (-[XX]%)
Time Held: [X] days
```

**Why This Was Correct**:
- [Thesis invalidated]
- [Cut loss quickly]
- [Preserved capital]

---

## Example 5: Perfect Setup

### The Ideal Scenario

**Everything Aligned**:
- [Technical setup]
- [Fundamental catalyst]
- [IV environment]
- [Time horizon]

### Execution

**Position**:
```
[Perfect entry]
[Optimal strikes]
[Good risk/reward]
```

### Result

**Outcome**:
```
Profit: $[X] ([XX]%)
Captured: [XX]% of max profit
Days Held: [X]
```

**What Made It Perfect**:
- [Factor 1]
- [Factor 2]
- [Factor 3]

---

## Key Takeaways

### What Works

1. **Clear Thesis**: [Importance]
2. **Proper Sizing**: [Why critical]
3. **Take Profits**: [50% rule]
4. **Cut Losses**: [Discipline]
5. **Patience**: [Time in market]

### What Doesn't Work

1. **No Plan**: [Consequences]
2. **Oversizing**: [Risks]
3. **Greed**: [Holding too long]
4. **Hope**: [Not a strategy]
5. **Ignoring Risk**: [Disaster]

### Statistical Reality

**Across These Examples**:
```
Win Rate: [XX]%
Average Win: $[X]
Average Loss: $[Y]
Profit Factor: [Z]
Net P&L: $[Total]
```

This aligns with realistic trading:
- Expect 50-65% win rate
- Average winners > average losers
- Consistency requires discipline

{generate_footer(strategy_key)}"""

    return content


# Generate strategy-specific files


def generate_portfolio_integration(strategy_key: str) -> str:
    """Generate portfolio-integration.md (for protective strategies)."""
    display_name = STRATEGIES[strategy_key]["display_name"]

    content = generate_header(
        strategy_key, "portfolio-integration", ["position-management", "examples"]
    )

    content += f"""## Portfolio Integration for {display_name}

Integrating {display_name.lower()} positions into a broader portfolio requires careful consideration of sizing, correlation, and overall risk management.

---

## Position Sizing Within Portfolio

### Capital Allocation

**Maximum Per Position**: 5-10% of portfolio value
```python
portfolio_value = 100000
max_position_size = portfolio_value * 0.10  # $10,000
```

**Total Hedge Allocation**: 10-25% of portfolio
```python
# If hedging $50K of stock
hedge_allocation = 50000 * 0.20  # $10,000 in hedges
# ~2-5% cost for protection
```

### Position Heat Tracking

**Portfolio Heat**: Sum of all risk across positions

```python
total_risk = sum([pos.max_loss for pos in positions])
portfolio_heat = total_risk / portfolio_value

# Keep below 15-20%
if portfolio_heat > 0.20:
    # Reduce exposure or hedge
```

---

## Correlation Management

### Stock vs. Hedge Correlation

**Direct Correlation**:
- Hedging specific positions (protective put on AAPL shares)
- High correlation required
- Size hedge to stock position

**Portfolio-Level Correlation**:
- Hedging overall portfolio (SPY options for tech stocks)
- Moderate correlation acceptable
- Beta-weighted position sizing

**Beta-Weighted Hedging**:
```python
# Stock position
aapl_shares = 100
aapl_price = 175
aapl_value = 17500
aapl_beta = 1.20

# SPY equivalent
spy_price = 450
spy_beta_adjusted_value = aapl_value * (aapl_beta / 1.0)
spy_shares_equivalent = spy_beta_adjusted_value / spy_price

# Size SPY hedge accordingly
```

---

## Multi-Position Management

### Aggregate Risk Metrics

**Net Portfolio Delta**:
```python
# Sum all position deltas
net_delta = sum([pos.delta * pos.quantity for pos in positions])

# Interpret
if net_delta > 100:  # Net bullish
    # Consider hedging
elif net_delta < -100:  # Net bearish
    # Consider reducing hedges
```

**Net Portfolio Theta**:
```python
# Sum all theta
net_theta = sum([pos.theta * pos.quantity for pos in positions])

# Negative theta = paying for hedges/time
# Positive theta = collecting premium
```

### Portfolio Greeks Dashboard

```python
# Daily monitoring
portfolio_metrics = {{
    'net_delta': calculate_net_delta(),
    'net_gamma': calculate_net_gamma(),
    'net_theta': calculate_net_theta(),
    'net_vega': calculate_net_vega(),
    'total_capital_at_risk': sum_max_losses(),
    'portfolio_heat': total_risk / portfolio_value
}}
```

---

## Sector and Concentration Risk

### Sector Exposure

**Track by Sector**:
```python
sector_exposure = {{
    'Technology': {{'value': 50000, 'hedged': 10000, 'net': 40000}},
    'Healthcare': {{'value': 25000, 'hedged': 0, 'net': 25000}},
    'Finance': {{'value': 15000, 'hedged': 3000, 'net': 12000}},
}}

# Identify overconcentration
for sector, exposure in sector_exposure.items():
    sector_pct = exposure['net'] / portfolio_value
    if sector_pct > 0.25:  # > 25%
        print(f"Overconcentrated in {{sector}}: {{sector_pct:.1%}}")
```

### Single-Stock Concentration

**Maximum Position Size**: 5-10% per stock
```python
for ticker, position in stock_positions.items():
    position_pct = position.value / portfolio_value
    if position_pct > 0.10:
        # Consider hedging or reducing
```

---

## Rebalancing and Rolling

### When to Rebalance Hedges

**Triggers**:
1. Stock position changes significantly (>20%)
2. Hedge expires
3. Market volatility regime changes
4. Correlation breaks down

**Rebalancing Example**:
```python
# Initial: $50K stock, $2K in protective options (4%)
# After rally: $65K stock, options unchanged

# Recalculate
current_hedge_pct = 2000 / 65000  # 3%
target_hedge_pct = 0.04  # 4%

# Need additional hedging
additional_hedge = 65000 * 0.04 - 2000  # $600
```

### Rolling Hedges

**Quarterly Rolling**:
- Roll hedges every 90 days
- Maintain consistent protection level
- Adjust strikes based on new prices

**Cost Management**:
```python
# Track cumulative hedge costs
total_hedge_cost = sum([hedge.cost for hedge in hedge_history])
annualized_cost = total_hedge_cost / years * 100 / avg_portfolio_value

# Target: 2-5% annualized cost for protection
```

---

## Tax Considerations

### Wash Sale Rules

**Avoid Wash Sales**:
- Don't buy back within 30 days of closing at loss
- Applies to substantially identical securities
- Can affect cost basis

**Strategies to Avoid**:
1. Wait 31+ days before repurchasing
2. Use different strike/expiration
3. Use correlated but different underlying (e.g., SPY vs. QQQ)

### Straddle Rules

**Tax Straddle Considerations**:
- Positions that offset each other
- Can defer losses
- Special rules for options vs. stock

**Consult Tax Professional** for specific situations

---

## Performance Attribution

### Separate Returns

**Track Separately**:
```python
stock_only_return = stock_pl / stock_investment
hedge_cost = hedge_pl / stock_investment
net_return = stock_only_return + hedge_cost

# Example:
# Stock: +15%
# Hedges: -3%
# Net: +12%
```

### Cost of Protection

**Insurance Value**:
```python
# What did protection save in downturns?
max_drawdown_unhedged = -25%  # Portfolio would have dropped 25%
max_drawdown_hedged = -10%    # Actually dropped 10%

protection_value = (-0.10) - (-0.25)  # 15% saved
hedge_cost_annualized = -3%

# Net benefit in downturn year: 15% - 3% = 12% saved
```

---

## Scenario Analysis

### Portfolio Stress Testing

**Run Scenarios**:
1. **Market Drop 20%**: What's portfolio impact?
2. **Volatility Spike**: How do hedges perform?
3. **Sector Rotation**: Correlation breakdown?

**Example Stress Test**:
```python
scenarios = {{
    'market_crash': {{'spy_change': -0.20, 'vix_change': +20}},
    'slow_grind_down': {{'spy_change': -0.10, 'vix_change': +5}},
    'volatility_spike': {{'spy_change': 0.00, 'vix_change': +15}},
}}

for scenario_name, params in scenarios.items():
    portfolio_pl = calculate_scenario_pl(params)
    print(f"{{scenario_name}}: {{portfolio_pl:,.0f}}")
```

---

## Best Practices

### Daily Monitoring

- [ ] Check portfolio net delta
- [ ] Verify hedge ratios still appropriate
- [ ] Monitor sector concentrations
- [ ] Review total capital at risk

### Weekly Review

- [ ] Analyze hedge effectiveness
- [ ] Calculate cost of protection
- [ ] Rebalance if needed
- [ ] Update correlation assumptions

### Monthly Deep Dive

- [ ] Full scenario analysis
- [ ] Performance attribution
- [ ] Cost-benefit analysis of hedges
- [ ] Strategic adjustments

---

## Integration Checklist

### Setup
- [ ] Define portfolio protection goals (max drawdown tolerance)
- [ ] Calculate appropriate hedge sizes
- [ ] Establish position limits
- [ ] Set up Greeks tracking

### Ongoing
- [ ] Monitor portfolio heat daily
- [ ] Rebalance hedges as needed
- [ ] Track hedge costs vs. value
- [ ] Adjust for concentration changes

### Review
- [ ] Quarterly hedge review
- [ ] Annual cost-benefit analysis
- [ ] Strategy effectiveness evaluation
- [ ] Adjust protection levels as needed

{generate_footer(strategy_key)}"""

    return content


def generate_dividend_considerations(strategy_key: str) -> str:
    """Generate dividend-considerations.md (for stock + option strategies)."""
    display_name = STRATEGIES[strategy_key]["display_name"]

    content = generate_header(
        strategy_key, "dividend-considerations", ["position-management", "examples"]
    )

    content += f"""## Dividend Considerations for {display_name}

Dividends significantly impact {display_name.lower()} positions. Understanding ex-dividend dates, early assignment risk, and dividend capture strategies is essential.

---

## Ex-Dividend Date Basics

### Key Dates

**Declaration Date**: Company announces dividend
**Ex-Dividend Date**: Must own stock before this date to receive dividend
**Record Date**: Shareholder of record
**Payment Date**: Dividend paid

**Critical**: Ex-dividend date determines ownership for dividend

### Stock Price Adjustment

**On Ex-Dividend Date**:
```
Stock typically drops by dividend amount

Example:
Pre-ex-dividend: $100
Dividend: $0.50
Ex-dividend open: ~$99.50
```

This drop is automatic and expected.

---

## Impact on Options

### Put Options

**ITM Puts**: High early assignment risk
```
Example:
Stock: $100
Put Strike: $105 (ITM)
Ex-Dividend: Tomorrow
Dividend: $0.75

Risk: Put holder may exercise early to:
1. Get short stock position
2. Capture dividend ($0.75)
3. Avoid dividend loss on short put
```

**Assignment Probability**:
- High if put deep ITM (>$1 intrinsic value over dividend)
- Very high if little time value remaining
- Increases as expiration approaches

### Call Options

**ITM Calls**: Moderate early assignment risk (less common than puts)
```
Call owner would exercise to:
1. Own stock
2. Receive dividend

But usually not optimal unless:
- Call deep ITM
- Little extrinsic value
- Dividend > time value
```

---

## Managing Dividend Risk

### Before Ex-Dividend Date

**Check Upcoming Dividends**:
```python
# For each position
if days_to_ex_dividend < 7:
    if short_put_itm and dividend > time_value:
        # High assignment risk
        consider_closing()
```

**Assignment Risk Assessment**:
```python
intrinsic_value = max(0, strike - stock_price)  # for puts
time_value = option_price - intrinsic_value
dividend_amount = upcoming_dividend

if dividend_amount > time_value:
    assignment_probability = "HIGH"
    action = "Close or roll position"
```

### During Ex-Dividend Week

**Daily Monitoring**:
- Check time value on short options
- Compare dividend to time value
- Watch for assignment notices (usually evening)

**If Assigned Early** (Short Put):
```
Result: Now short stock
Exposure: Stock dropped, but you're short
Dividend: Owe dividend to stock lender

Actions:
1. Buy stock to close short (if protective strategy)
2. Or: Exercise long put if available
3. Confirm with broker
4. Understand P&L impact
```

---

## Dividend Capture Strategies

### Intentional Dividend Capture

**Strategy**: Own stock through ex-dividend, protected by options

**Example** ({display_name}):
```
Buy 100 shares at $100
Dividend: $0.60 (quarterly)
[Add appropriate options protection]

On ex-dividend:
- Stock drops to ~$99.40
- Receive $0.60 dividend
- Net: Neutral (ignoring protection cost)
```

**Considerations**:
- Dividend must exceed protection cost
- Stock may drop more than dividend
- Short-term capital gains tax on quick sales
- Early assignment risk on short options

### High-Yield Dividend Stocks

**Special Considerations**:
```
Stocks with >3% annual yield
Quarterly dividends >$0.75

Implications:
- Higher early assignment risk
- More frequent ex-dividend dates
- Premium adjustments in options pricing
```

**Examples**:
- Utilities (often 3-5% yield)
- REITs (often 4-7% yield)
- Telecom (often 3-6% yield)

---

## Adjusting for Dividends

### Strike Selection

**Account for Dividends**:
```python
# If stock pays $0.50 dividend before expiration
# Stock expected to drop by $0.50 on ex-div

adjusted_price_target = target - expected_dividends
# Use adjusted target for strike selection
```

### Premium Adjustments

**Options Pricing**:
- Puts more expensive (dividend reduces stock price)
- Calls less expensive (dividend reduces stock price)
- Larger dividends = larger adjustments

**Example**:
```
Stock: $100
Without dividend: $105 call = $2.50
With $1 dividend: $105 call = $1.50
(Less valuable since stock drops by dividend)
```

---

## Tax Implications

### Qualified Dividends

**Requirements for Qualified Dividend Tax Rate**:
- Hold stock > 60 days during 121-day period around ex-div
- For preferred stock: > 90 days during 181-day period

**Tax Rates**:
- Qualified: 0%, 15%, or 20% (based on income)
- Ordinary: Marginal tax rate (up to 37%)

### Hedged Positions

**Wash Sale Risk**:
- Selling stock at loss within 30 days of buying call/put
- Can disallow loss deduction

**Constructive Sale**:
- Certain hedged positions may trigger constructive sale
- Consult tax professional

### Dividend Received Deduction

**Corporations**: May deduct portion of dividends
**Individuals**: No deduction

---

## Calendar Planning

### Ex-Dividend Tracking

**Track Upcoming Dividends**:
```python
# Maintain calendar
positions_with_upcoming_divs = {{
    'AAPL': {{'ex_date': '2025-05-15', 'amount': 0.24}},
    'MSFT': {{'ex_date': '2025-05-20', 'amount': 0.68}},
    'JNJ': {{'ex_date': '2025-05-25', 'amount': 1.13}},
}}

# Alert 7 days before
for ticker, div_info in positions_with_upcoming_divs.items():
    if days_to_ex_div <= 7:
        review_assignment_risk(ticker)
```

### Seasonal Patterns

**Common Dividend Months**:
- March, June, September, December (quarterly)
- Some companies: January, April, July, October

**Plan Around**:
- Heavy dividend months
- Coordinate rolling with ex-div dates
- Adjust strategies for high-yield periods

---

## Special Situations

### Large Special Dividends

**One-Time Payments**:
```
Company announces special dividend: $2.50
Stock: $80
Impact: Stock drops $2.50 on ex-div

Options:
- Strikes adjust (rare but possible)
- Early assignment risk very high
- May need special handling
```

**Action**:
- Check with broker on strike adjustments
- Consider closing before ex-div
- Understand contract specifications

### Dividend Cuts/Suspensions

**Announcement Impact**:
```
Company cuts dividend
Stock typically drops significantly

Effect on options:
- Put values increase
- Call values decrease
- May benefit bearish strategies
```

---

## Best Practices

### Before Opening Position

- [ ] Check dividend history
- [ ] Note upcoming ex-dividend dates
- [ ] Calculate dividend impact on strikes
- [ ] Plan management around dividends

### Position Management

- [ ] Set calendar alerts (7 days before ex-div)
- [ ] Monitor time value vs. dividend
- [ ] Be prepared to close ITM shorts
- [ ] Understand assignment consequences

### Tax Planning

- [ ] Track holding periods
- [ ] Consider qualified dividend requirements
- [ ] Understand wash sale rules
- [ ] Consult tax professional for complex situations

---

## Dividend Checklist

### Pre-Trade
- [ ] Dividend yield checked
- [ ] Ex-dividend dates identified
- [ ] Strikes adjusted for expected drop
- [ ] Assignment risk assessed

### During Position
- [ ] 7-day ex-div alert set
- [ ] Time value monitored
- [ ] Assignment risk daily check
- [ ] Closing plan if needed

### Ex-Dividend Week
- [ ] Daily time value check
- [ ] Assignment watch (evening)
- [ ] Broker account monitoring
- [ ] Ready to act on assignment

{generate_footer(strategy_key)}"""

    return content


def generate_spread_width_optimization(strategy_key: str) -> str:
    """Generate spread-width-optimization.md (for spreads and multi-leg strategies)."""
    display_name = STRATEGIES[strategy_key]["display_name"]

    content = generate_header(
        strategy_key, "spread-width-optimization", ["strike-selection", "examples"]
    )

    content += f"""## Spread Width Optimization for {display_name}

Choosing the optimal spread width significantly impacts risk/reward, capital efficiency, and probability of profit for {display_name.lower()} positions.

---

## Spread Width Fundamentals

### What Is Spread Width?

**Definition**: Distance between strikes in the spread

**For {display_name}**:
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

{generate_footer(strategy_key)}"""

    return content


def generate_iv_analysis(strategy_key: str) -> str:
    """Generate iv-analysis.md (for volatility strategies)."""
    display_name = STRATEGIES[strategy_key]["display_name"]

    content = generate_header(strategy_key, "iv-analysis", ["strategy-mechanics", "earnings-plays"])

    content += f"""## Implied Volatility Analysis for {display_name}

For {display_name.lower()} positions, implied volatility (IV) is arguably the most important factor. Understanding IV rank, percentile, and expected move is essential for success.

---

## IV Fundamentals

### What Is Implied Volatility?

**Definition**: Market's expectation of future volatility

**Key Points**:
- Forward-looking (not historical)
- Derived from option prices
- Higher IV = higher option prices
- Changes constantly based on supply/demand

**For {display_name}**:
[Strategy-specific IV impact]

---

## IV Metrics

### IV Rank

**Definition**: Current IV relative to 52-week range

**Formula**:
```python
iv_rank = ((current_iv - iv_52w_low) /
           (iv_52w_high - iv_52w_low)) * 100

# Example:
# Current IV: 35%
# 52-week low: 20%
# 52-week high: 60%
iv_rank = ((35 - 20) / (60 - 20)) * 100 = 37.5
```

**Interpretation**:
- **0-25**: Very low IV (avoid buying premium)
- **25-50**: Low to moderate IV
- **50-75**: Moderate to high IV (good for {display_name})
- **75-100**: Very high IV (best opportunities)

### IV Percentile

**Definition**: Percentage of days in past year where IV was lower

**Formula**:
```python
# Count days where IV < current IV
days_below = count(historical_iv < current_iv)
total_days = 252  # trading days
iv_percentile = (days_below / total_days) * 100
```

**Difference from IV Rank**:
- IV Rank: Based on high/low
- IV Percentile: Based on distribution
- Percentile more robust to outliers

### Expected Move

**Definition**: Market's expectation for stock movement by expiration

**Calculation**:
```python
stock_price = 100
days_to_expiration = 30
iv = 0.40  # 40%

# Simplified expected move (1 standard deviation)
expected_move_annual = stock_price * iv
expected_move_period = expected_move_annual * sqrt(days/365)

# Example:
expected_move_30d = 100 * 0.40 * sqrt(30/365)
                  = 100 * 0.40 * 0.287
                  = $11.48

# Expected range: $88.52 to $111.48 (68% probability)
```

**Using Expected Move**:
- Compare to strike selection
- Assess probability of profit
- Size positions appropriately

---

## IV Regimes

### Low IV Environment

**Characteristics**:
- IV Rank < 25
- Quiet markets
- Low option prices
- Compressed volatility

**For {display_name}**:
```
Avoid or reduce size
Options too cheap
Limited profit potential
Wait for IV expansion
```

**Alternative Strategies**:
- Sell credit spreads instead
- Wait for better IV environment
- Use other strategies

### Normal IV Environment

**Characteristics**:
- IV Rank 25-75
- Typical market conditions
- Reasonable option prices
- Standard volatility

**For {display_name}**:
```
Good environment
Standard position sizing
Normal profit expectations
Monitor for IV changes
```

**Management**:
- Enter positions normally
- Use standard profit targets
- Watch for regime changes

### High IV Environment

**Characteristics**:
- IV Rank > 75
- Market stress or uncertainty
- Elevated option prices
- Expanded volatility

**For {display_name}**:
```
OPTIMAL environment
Increase position size
Higher profit potential
Best risk/reward
```

**Opportunities**:
- IV likely to contract (profit source)
- Larger expected moves
- Premium decay works for you

---

## IV Expansion and Contraction

### IV Expansion Triggers

**Common Catalysts**:
1. **Earnings announcements**
   - IV peaks day before earnings
   - Can double or triple base IV

2. **Economic data**
   - FOMC meetings
   - Jobs reports
   - CPI/inflation data

3. **Company-specific events**
   - FDA decisions (biotech)
   - Product launches
   - Legal proceedings

4. **Market stress**
   - Geopolitical events
   - Market selloffs
   - Sector rotations

**Trading Around Expansion**:
```python
# Before event
base_iv = 25%
event_iv = 60%

# Enter when IV expanding
# Plan exit after event (IV contraction)
```

### IV Contraction (Crush)

**What Happens**:
```
Event passes
Uncertainty resolves
IV collapses quickly

Example:
Pre-earnings IV: 80%
Post-earnings IV: 30%
IV crushed: -50 points (62.5% reduction)
```

**Impact on {display_name}**:
```
[Strategy-specific impact of IV crush]
```

**Managing Through Crush**:
- Exit before event if timing unclear
- Understand position vega exposure
- Have exit plan for post-event

---

## IV Skew and Term Structure

### IV Skew

**Definition**: IV variation across strikes

**Typical Patterns**:
```
OTM Puts: Higher IV (demand for downside protection)
ATM Options: Mid-range IV
OTM Calls: Lower IV

Example:
$95 Put IV: 35%
$100 ATM IV: 30%
$105 Call IV: 28%
```

**Using Skew**:
- Affects strike selection
- Can indicate market sentiment
- May create opportunities

### IV Term Structure

**Definition**: IV variation across expirations

**Normal Pattern**:
```
Front month: Higher IV (events, uncertainty)
Back months: Lower IV (uncertainty averages out)

Example:
30 DTE: IV = 40%
60 DTE: IV = 35%
90 DTE: IV = 32%
```

**Inverted Structure** (bullish for volatility):
```
Front month IV > Back month IV significantly
Often before major events
```

---

## Using IV for Entry Timing

### Optimal Entry Conditions

**For {display_name}**:
```
Target IV Rank: 50-85
Target IV Percentile: 60-90
Catalyst: Upcoming (IV likely to stay elevated)
```

**Entry Checklist**:
- [ ] IV rank > 50
- [ ] IV expanding or near peak
- [ ] Catalyst identified
- [ ] Expected move calculated
- [ ] Position sized for volatility

### Avoiding Poor Entries

**Red Flags**:
```
IV rank < 25 (too cheap, likely to stay low)
IV just crushed (already contracted)
No catalyst (no reason for IV to stay high)
```

**Wait For**:
- IV to expand to acceptable levels
- Clear catalyst approaching
- Better risk/reward setup

---

## IV-Based Position Management

### Profit-Taking Based on IV

**IV Expansion**:
```
Position benefits from IV increase
Consider taking profits early
IV may not stay elevated

Example:
Entered at IV Rank 60
Now at IV Rank 85
Already profitable from vega
Take profits before crush
```

**IV Contraction**:
```
Position hurt by IV decrease
Reassess thesis
May need to extend time or exit

Example:
Entered at IV Rank 70
Now at IV Rank 40
Vega loss significant
Consider exiting or rolling
```

### Rolling Based on IV

**High IV**: Roll to lock in higher premium
**Low IV**: Avoid rolling (expensive to adjust)

---

## Volatility Metrics Dashboard

### Daily Monitoring

```python
metrics = {{
    'current_iv': get_current_iv(ticker),
    'iv_rank': calculate_iv_rank(ticker),
    'iv_percentile': calculate_iv_percentile(ticker),
    'expected_move_30d': calculate_expected_move(ticker, 30),
    'vix_level': get_vix(),
    'vix_term_structure': get_vix_term_structure(),
}}

# Decision logic
if metrics['iv_rank'] > 60 and metrics['iv_percentile'] > 70:
    signal = "STRONG ENTRY"
elif metrics['iv_rank'] < 30:
    signal = "AVOID / WAIT"
else:
    signal = "MONITOR"
```

### Historical IV Analysis

**Track Over Time**:
```python
# Plot IV history
iv_history = get_iv_history(ticker, days=365)

# Identify patterns
earnings_dates = get_earnings_dates(ticker)
iv_at_earnings = [iv_history[date] for date in earnings_dates]

avg_earnings_iv = mean(iv_at_earnings)
current_iv = iv_history[-1]

if current_iv / avg_earnings_iv > 0.90:
    print("IV near typical earnings level")
```

---

## Advanced IV Concepts

### Realized vs. Implied Volatility

**Realized Volatility** (Historical):
```python
# Calculate from price history
price_returns = calculate_returns(price_history)
realized_vol = std(price_returns) * sqrt(252)
```

**Comparison**:
```
If IV > Realized Vol:
  → Options "expensive"
  → May be overstating future volatility
  → Consider selling premium

If IV < Realized Vol:
  → Options "cheap"
  → May be understating future volatility
  → Consider buying premium ({display_name})
```

### Volatility Risk Premium

**Concept**: IV typically overstates realized volatility

**Implication**:
- Selling volatility profitable over time
- Buying volatility needs timing
- Enter {display_name} when IV elevated and likely to realize

---

## IV Analysis Checklist

### Before Entry
- [ ] Current IV calculated
- [ ] IV rank determined (target >50)
- [ ] IV percentile checked (target >60)
- [ ] Expected move calculated
- [ ] Catalyst identified
- [ ] Compared to historical IV
- [ ] Skew and term structure reviewed

### During Position
- [ ] Daily IV monitoring
- [ ] Track IV changes
- [ ] Vega P&L attribution
- [ ] Compare to expectations
- [ ] Watch for IV crush signals

### Exit Planning
- [ ] Know catalyst date
- [ ] Plan for IV contraction
- [ ] Set IV-based profit targets
- [ ] Prepare for crush scenario

{generate_footer(strategy_key)}"""

    return content


def generate_earnings_plays(strategy_key: str) -> str:
    """Generate earnings-plays.md (for volatility strategies)."""
    display_name = STRATEGIES[strategy_key]["display_name"]

    content = generate_header(strategy_key, "earnings-plays", ["iv-analysis", "examples"])

    content += f"""## Earnings Plays with {display_name}

Earnings announcements create significant volatility opportunities for {display_name.lower()} traders. This guide covers timing, sizing, and managing earnings-driven positions.

---

## Earnings Basics

### The Earnings Cycle

**Quarterly Cycle**:
- Q1 Earnings: April/May
- Q2 Earnings: July/August
- Q3 Earnings: October/November
- Q4 Earnings: January/February

**Key Dates**:
1. **Announcement Date**: Company announces earnings date
2. **Earnings Date**: After market close or before market open
3. **Conference Call**: Management discusses results
4. **Guidance**: Forward-looking statements

### IV Behavior Around Earnings

**Typical Pattern**:
```
T-30 days: IV starts rising
T-14 days: IV acceleration
T-7 days: IV rapid increase
T-1 day: IV peaks (80-100+ rank common)
T+0 (after): IV CRUSHES (drops 50-70%)
T+1: IV stabilizes at lower level
```

**Example**:
```
Stock: AAPL
30 days before: IV = 25%
7 days before: IV = 35%
1 day before: IV = 55%
After earnings: IV = 22%

IV Crush: -33 points (-60% reduction)
```

---

## Why Earnings Are Ideal for {display_name}

### Volatility Expansion

**Pre-Earnings**:
```
Uncertainty drives option buying
IV expands significantly
{display_name} benefits from elevated premiums
```

**Key Advantage**:
[Strategy-specific advantage in earnings plays]

### Expected Move

**Calculating**:
```python
# Using ATM straddle price as proxy
atm_call_price = 8.50
atm_put_price = 8.25
straddle_price = atm_call_price + atm_put_price  # 16.75

expected_move_pct = (straddle_price / stock_price) * 100
# If stock = $150: (16.75 / 150) * 100 = 11.2%

expected_range = {{
    'upper': 150 * 1.112,  # $166.80
    'lower': 150 * 0.888,  # $133.20
}}
```

**Using Expected Move**:
- Stock has ~68% probability of staying in range
- ~32% probability of moving beyond range
- Size {display_name} positions accordingly

---

## Timing Entry and Exit

### Optimal Entry Timing

**Early Entry** (T-14 to T-21):
```
Pros:
- Lower IV = cheaper entry
- IV expansion benefits position
- More time for thesis

Cons:
- Longer hold time
- More theta decay
- Uncertain IV path
```

**Late Entry** (T-3 to T-7):
```
Pros:
- IV already elevated
- Shorter hold time
- Clear timeline

Cons:
- Higher entry cost
- Less IV expansion benefit
- Options already expensive
```

**Recommended for {display_name}**:
```
Sweet spot: T-7 to T-10
- IV elevated but still expanding
- Reasonable hold period
- Good risk/reward
```

### Exit Timing

**Before Earnings** (sell premium before event):
```
Exit: Day before or morning of earnings

Pros:
- Capture IV expansion
- Avoid IV crush
- Keep vega profits

Cons:
- Miss potential directional move
- May leave money on table
```

**After Earnings** (hold through announcement):
```
Exit: Morning after earnings

Pros:
- Capture full move if large
- Potential huge profits
- Full speculation on direction

Cons:
- IV crush hurts badly
- Theta acceleration
- High risk
```

**Recommended for {display_name}**:
```
[Strategy-specific timing recommendation]

Rationale:
[Why this timing works for this strategy]
```

---

## Position Sizing for Earnings

### Risk Management

**Standard Sizing**: 2-3% risk per trade

**Earnings Plays**: REDUCE to 1-1.5%

**Why Smaller**:
```
Higher uncertainty
Binary outcome
IV crush risk
Possible total loss
```

**Example**:
```python
portfolio_value = 50000
normal_risk = 50000 * 0.02  # $1,000

earnings_risk = 50000 * 0.015  # $750 (reduced)

# If max loss per contract = $250
contracts = 750 / 250  # 3 contracts max
```

### Portfolio Heat

**Limit Concurrent Earnings**:
```
Max 3-5 earnings plays open simultaneously
Total earnings exposure: <10% of portfolio
Diversify across sectors
```

**Example**:
```python
earnings_positions = {{
    'AAPL': {{'risk': 750, 'date': '2025-04-30'}},
    'GOOGL': {{'risk': 750, 'date': '2025-04-28'}},
    'MSFT': {{'risk': 1000, 'date': '2025-04-25'}},
}}

total_earnings_risk = sum([p['risk'] for p in earnings_positions.values()])
# $2,500 (5% of $50K portfolio) - OK
```

---

## Stock Selection for Earnings Plays

### Screening Criteria

**Ideal Candidates**:
```
Market Cap: >$10B (liquid options)
Average Volume: >1M shares/day
Options Volume: >10K contracts/day
IV Rank: Currently >60
Historical Move: >5% on earnings
Sector: Technology, biotech, retail (movers)
```

**Avoid**:
```
Small caps (<$1B) - illiquid, manipulated
No clear catalyst
IV already crushed recently
Thinly traded options
```

### Historical Earnings Move Analysis

**Research Past Quarters**:
```python
# Collect data
past_earnings = {{
    '2024-Q4': {{'expected': 8.5, 'actual': 12.3, 'direction': 'up'}},
    '2024-Q3': {{'expected': 7.2, 'actual': 3.1, 'direction': 'down'}},
    '2024-Q2': {{'expected': 9.1, 'actual': 11.8, 'direction': 'up'}},
    '2024-Q1': {{'expected': 6.8, 'actual': 2.4, 'direction': 'down'}},
}}

# Analyze
avg_expected = mean([q['expected'] for q in past_earnings.values()])
avg_actual = mean([q['actual'] for q in past_earnings.values()])

print(f"Average expected: {{avg_expected:.1f}}%")
print(f"Average actual: {{avg_actual:.1f}}%")

# Does stock typically exceed expected move?
if avg_actual > avg_expected:
    print("Stock tends to move more than expected")
```

---

## Managing Through Earnings

### Pre-Earnings Checklist

**24-48 Hours Before**:
- [ ] Confirm exact announcement time (pre/post market)
- [ ] Check current IV vs. entry IV
- [ ] Calculate current P&L
- [ ] Decide: hold through or exit before
- [ ] Set exit orders if holding

**Day Of Earnings**:
- [ ] Monitor IV (may spike further)
- [ ] Last chance to exit before announcement
- [ ] Prepare for binary outcome
- [ ] Have exit plan ready

### Post-Earnings Actions

**Immediately After** (within 1 hour):
```
1. Check stock reaction
   - Gap up/down?
   - How much?

2. Check position value
   - IV crushed by how much?
   - Current P&L?

3. Decide action
   - Take profits if good move
   - Cut loss if bad move
   - NEVER hope for recovery
```

**Morning After**:
```
1. Check overnight movement
2. Assess IV levels
3. Calculate final P&L
4. Execute exit if not done
5. Log trade results
```

---

## Earnings Play Scenarios

### Scenario 1: Large Move in Favor

**What Happens**:
```
Entered {display_name}
Expected move: ±10%
Actual move: +15% (favorable direction)

Result:
- Directional gain: +++
- IV crush: --
- Net: Likely profitable
```

**Action**: Take profits immediately after announcement

### Scenario 2: Small Move (Stuck in Range)

**What Happens**:
```
Expected move: ±10%
Actual move: +3% (small)

Result:
- Minimal directional gain
- IV crush: --
- Theta decay: -
- Net: Likely loss for {display_name}
```

**Action**: Exit quickly, accept small loss

### Scenario 3: Large Move Against

**What Happens**:
```
Expected move: ±10%
Actual move: -15% (wrong direction)

Result:
- Directional loss: ---
- IV crush: --
- Net: Significant loss
```

**Action**: Cut loss immediately, don't hold hoping

### Scenario 4: Minimal Move (Pin)

**What Happens**:
```
Expected move: ±10%
Actual move: +0.5% (dead flat)

Result:
- No directional gain
- IV crush: ---
- Theta decay: --
- Net: Worst case for {display_name}
```

**Action**: Exit immediately at market open

---

## Advanced Earnings Strategies

### Earnings Calendars

**Planning Ahead**:
```python
# Track upcoming earnings (30 days out)
earnings_calendar = {{
    '2025-04-25': ['MSFT', 'GOOGL'],
    '2025-04-28': ['META', 'AMZN'],
    '2025-04-30': ['AAPL', 'INTC'],
}}

# Plan entries 7-14 days before
for date, tickers in earnings_calendar.items():
    entry_window = date - timedelta(days=10)
    print(f"Plan {{tickers}} entries around {{entry_window}}")
```

### Sector Rotation

**Tech Earnings Season**:
```
Late April: MSFT, GOOGL, META
Early May: AAPL, AMZN

Strategy:
- Open positions sequentially
- Close after each announcement
- Rotate capital to next opportunity
- Avoid too many concurrent positions
```

### IV Expansion Capture

**Strategy**: Enter early, exit before earnings

**Example**:
```
T-21: Enter when IV = 30%
T-7: IV expands to 50%
T-1: Exit (before earnings)

Profit source: IV expansion (+vega)
Avoid: IV crush risk
```

**For {display_name}**:
```
[Strategy-specific IV expansion approach]
```

---

## Earnings Playbook

### Stock Research

**Week Before Entry**:
- [ ] Confirm earnings date
- [ ] Research analyst expectations
- [ ] Check past earnings results (4 quarters)
- [ ] Calculate historical moves
- [ ] Identify support/resistance levels

### Entry Execution

**T-10 to T-7 Days**:
- [ ] Check IV rank (target >60)
- [ ] Calculate expected move
- [ ] Size position (1-1.5% risk)
- [ ] Execute limit order
- [ ] Document entry rationale

### Position Monitoring

**Daily Checks**:
- [ ] IV tracking
- [ ] P&L calculation
- [ ] Time to earnings
- [ ] Technical levels

### Exit Execution

**Choose Path**:
- [ ] Exit T-1 (before earnings) - safer
- [ ] Hold through (after earnings) - riskier
- [ ] Set exit orders
- [ ] Monitor execution
- [ ] Log results

---

## Common Earnings Mistakes

### Mistake 1: Holding Too Long

**Problem**: Holding through earnings "just to see"
**Impact**: IV crush destroys position
**Fix**: Have clear exit plan BEFORE entry

### Mistake 2: Oversizing

**Problem**: Treating earnings plays like normal trades
**Impact**: Excessive risk, emotional decisions
**Fix**: Reduce size to 1-1.5% risk

### Mistake 3: Chasing IV

**Problem**: Entering T-1 or T-0 when IV peaked
**Impact**: Overpaying for options, poor risk/reward
**Fix**: Enter T-7 to T-10 when IV still expanding

### Mistake 4: Ignoring Historical Moves

**Problem**: Not researching past earnings reactions
**Impact**: Surprised by typical move patterns
**Fix**: Analyze 4+ past quarters before entry

### Mistake 5: No Exit Plan

**Problem**: Deciding during announcement
**Impact**: Emotional decisions, poor execution
**Fix**: Decide BEFORE entry: hold through or exit before

---

## Earnings Checklist

### Pre-Entry
- [ ] Earnings date confirmed
- [ ] Historical moves researched
- [ ] IV rank checked (>60)
- [ ] Expected move calculated
- [ ] Position sized (1-1.5% risk)
- [ ] Entry timing planned (T-7 to T-10)
- [ ] Exit strategy decided

### During Position
- [ ] Daily IV monitoring
- [ ] P&L tracking
- [ ] Days to earnings countdown
- [ ] Technical level watching
- [ ] Exit plan confirmed

### Exit
- [ ] Before or after decision finalized
- [ ] Exit order placed
- [ ] Execution confirmed
- [ ] Results logged
- [ ] Lessons documented

{generate_footer(strategy_key)}"""

    return content


# Main execution


def main():
    """Generate all reference files."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate all reference files")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't write files")
    parser.add_argument("--strategy", help="Generate only for specific strategy")
    args = parser.parse_args()

    print("=" * 80)
    print("REFERENCE FILE GENERATOR")
    print("=" * 80)
    print()
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE FILES'}")
    print(f"Skills Directory: {SKILLS_DIR}")
    print()

    # Filter strategies if specific one requested
    strategies_to_process = STRATEGIES
    if args.strategy:
        if args.strategy in STRATEGIES:
            strategies_to_process = {args.strategy: STRATEGIES[args.strategy]}
        else:
            print(f"Error: Strategy '{args.strategy}' not found")
            print(f"Available: {', '.join(STRATEGIES.keys())}")
            return 1

    # Statistics
    files_created = 0
    total_size = 0

    # Generate files for each strategy
    for strategy_key, strategy_info in strategies_to_process.items():
        print(
            f"\n--- {strategy_info['display_name'].upper()} ({len(strategy_info['files'])} files) ---\n"
        )

        strategy_dir = SKILLS_DIR / strategy_key / "references"
        if not args.dry_run:
            strategy_dir.mkdir(parents=True, exist_ok=True)

        # Generate each file type
        for file_type in strategy_info["files"]:
            filename = f"{file_type}.md"
            filepath = strategy_dir / filename

            # Generate content based on file type
            if file_type == "quickstart":
                content = generate_quickstart(strategy_key)
            elif file_type == "strategy-mechanics":
                content = generate_strategy_mechanics(strategy_key)
            elif file_type == "strike-selection":
                content = generate_strike_selection(strategy_key)
            elif file_type == "position-management":
                content = generate_position_management(strategy_key)
            elif file_type == "greeks-analysis":
                content = generate_greeks_analysis(strategy_key)
            elif file_type == "examples":
                content = generate_examples(strategy_key)
            elif file_type == "portfolio-integration":
                content = generate_portfolio_integration(strategy_key)
            elif file_type == "dividend-considerations":
                content = generate_dividend_considerations(strategy_key)
            elif file_type == "spread-width-optimization":
                content = generate_spread_width_optimization(strategy_key)
            elif file_type == "iv-analysis":
                content = generate_iv_analysis(strategy_key)
            elif file_type == "earnings-plays":
                content = generate_earnings_plays(strategy_key)
            else:
                print(f"  [SKIP] {filename} - no generator")
                continue

            size_kb = len(content) / 1024

            if args.dry_run:
                print(f"  [DRY RUN] {filename} ({size_kb:.1f} KB)")
            else:
                filepath.write_text(content, encoding="utf-8")
                print(f"  [CREATED] {filename} ({size_kb:.1f} KB)")

            files_created += 1
            total_size += len(content)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Strategies processed: {len(strategies_to_process)}")
    print(f"Files created: {files_created}")
    print(f"Total size: {total_size / 1024:.1f} KB")

    if args.dry_run:
        print("\nDRY RUN COMPLETE - No files written")
        print("Run without --dry-run to create files")
    else:
        print("\nALL FILES CREATED SUCCESSFULLY")

    return 0


if __name__ == "__main__":
    sys.exit(main())
