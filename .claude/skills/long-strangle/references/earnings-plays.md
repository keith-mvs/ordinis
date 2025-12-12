# Long Strangle - Earnings Plays

**Parent**: [Long Strangle](../SKILL.md) | **Related**: [Iv Analysis](iv-analysis.md) | [Examples](examples.md)

---

## Earnings Plays with Long Strangle

Earnings announcements create significant volatility opportunities for long strangle traders. This guide covers timing, sizing, and managing earnings-driven positions.

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

## Why Earnings Are Ideal for Long Strangle

### Volatility Expansion

**Pre-Earnings**:
```
Uncertainty drives option buying
IV expands significantly
Long Strangle benefits from elevated premiums
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

expected_range = {
    'upper': 150 * 1.112,  # $166.80
    'lower': 150 * 0.888,  # $133.20
}
```

**Using Expected Move**:
- Stock has ~68% probability of staying in range
- ~32% probability of moving beyond range
- Size Long Strangle positions accordingly

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

**Recommended for Long Strangle**:
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

**Recommended for Long Strangle**:
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
earnings_positions = {
    'AAPL': {'risk': 750, 'date': '2025-04-30'},
    'GOOGL': {'risk': 750, 'date': '2025-04-28'},
    'MSFT': {'risk': 1000, 'date': '2025-04-25'},
}

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
past_earnings = {
    '2024-Q4': {'expected': 8.5, 'actual': 12.3, 'direction': 'up'},
    '2024-Q3': {'expected': 7.2, 'actual': 3.1, 'direction': 'down'},
    '2024-Q2': {'expected': 9.1, 'actual': 11.8, 'direction': 'up'},
    '2024-Q1': {'expected': 6.8, 'actual': 2.4, 'direction': 'down'},
}

# Analyze
avg_expected = mean([q['expected'] for q in past_earnings.values()])
avg_actual = mean([q['actual'] for q in past_earnings.values()])

print(f"Average expected: {avg_expected:.1f}%")
print(f"Average actual: {avg_actual:.1f}%")

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
Entered Long Strangle
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
- Net: Likely loss for Long Strangle
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
- Net: Worst case for Long Strangle
```

**Action**: Exit immediately at market open

---

## Advanced Earnings Strategies

### Earnings Calendars

**Planning Ahead**:
```python
# Track upcoming earnings (30 days out)
earnings_calendar = {
    '2025-04-25': ['MSFT', 'GOOGL'],
    '2025-04-28': ['META', 'AMZN'],
    '2025-04-30': ['AAPL', 'INTC'],
}

# Plan entries 7-14 days before
for date, tickers in earnings_calendar.items():
    entry_window = date - timedelta(days=10)
    print(f"Plan {tickers} entries around {entry_window}")
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

**For Long Strangle**:
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
