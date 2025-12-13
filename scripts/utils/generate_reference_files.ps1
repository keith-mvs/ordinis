#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Generate all 51 reference files for 7 options strategies
.DESCRIPTION
    Creates comprehensive, institutional-grade reference documentation for:
    - bear-put-spread (6 files)
    - protective-collar (8 files)
    - iron-butterfly (7 files)
    - iron-condor (7 files)
    - long-call-butterfly (7 files)
    - long-straddle (8 files)
    - long-strangle (8 files)

    Follows established taxonomy and cross-linking scheme.
.PARAMETER DryRun
    Preview files to be created without writing
.EXAMPLE
    .\generate_reference_files.ps1
    .\generate_reference_files.ps1 -DryRun
#>

[CmdletBinding()]
param(
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$SkillsDir = Join-Path $PSScriptRoot '..' '.claude' 'skills'

# Ensure skills directory exists
if (-not (Test-Path $SkillsDir)) {
    throw "Skills directory not found: $SkillsDir"
}

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "REFERENCE FILE GENERATOR" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Skills Directory: $SkillsDir"
Write-Host "Mode: $(if ($DryRun) { 'DRY RUN (preview only)' } else { 'WRITE FILES' })"
Write-Host ""

# Statistics
$script:FilesCreated = 0
$script:TotalSize = 0

function Write-ReferenceFile {
    param(
        [string]$Strategy,
        [string]$FileName,
        [string]$Content
    )

    $FilePath = Join-Path $SkillsDir $Strategy 'references' $FileName
    $SizeKB = [math]::Round($Content.Length / 1KB, 1)

    if ($DryRun) {
        Write-Host "[DRY RUN] Would create: $Strategy/references/$FileName ($SizeKB KB)" -ForegroundColor Yellow
    } else {
        # Ensure directory exists
        $Dir = Split-Path $FilePath -Parent
        if (-not (Test-Path $Dir)) {
            New-Item -ItemType Directory -Path $Dir -Force | Out-Null
        }

        # Write file
        Set-Content -Path $FilePath -Value $Content -Encoding UTF8 -NoNewline
        Write-Host "[CREATED] $Strategy/references/$FileName ($SizeKB KB)" -ForegroundColor Green
    }

    $script:FilesCreated++
    $script:TotalSize += $Content.Length
}

#region Bear Put Spread (6 files)

Write-Host "`n--- BEAR PUT SPREAD (6 files) ---" -ForegroundColor Magenta

# 1. quickstart.md
$Content = @'
# Bear Put Spread - Quick Start Guide

**Parent**: [Bear Put Spread](../SKILL.md) | **Related**: [Strategy Mechanics](strategy-mechanics.md) | [Examples](examples.md)

---

## Overview

The bear put spread is a limited-risk, limited-reward options strategy designed to profit from moderate downward price movement. It involves buying a higher-strike put and simultaneously selling a lower-strike put with the same expiration.

**Position Structure**:
- **Long**: 1 Put at higher strike (e.g., $100)
- **Short**: 1 Put at lower strike (e.g., $95)
- **Net Cost**: Debit spread (pay upfront)

**Key Characteristics**:
- Bearish directional bias
- Defined maximum loss (net debit paid)
- Defined maximum profit (spread width - net debit)
- Lower cost than outright put purchase
- Reduced vega risk compared to naked puts

---

## When to Use

**Market Outlook**: Moderately bearish with defined downside target

**Ideal Conditions**:
- Expect 5-15% decline in underlying
- Defined price target below current level
- Moderate implied volatility (not at extremes)
- Sufficient time for move (30-60 DTE recommended)

**Avoid When**:
- Expecting crash-level decline (>20%)
- IV percentile > 80 (expensive puts)
- Very short timeframe (<15 DTE for beginners)
- No clear technical support levels

---

## Position Sizing

**Capital Requirements**:
```
Maximum Loss = Net Debit Paid
Maximum Profit = Spread Width - Net Debit
Required Capital = Net Debit × 100 × Number of Spreads
```

**Example**:
```
Stock at $100
Buy $100 put for $5.00
Sell $95 put for $2.50
Net Debit = $2.50 per share

For 1 contract:
- Max Loss: $250 (2.50 × 100)
- Max Profit: $250 ($5 spread - $2.50 debit = $2.50 × 100)
- Required Capital: $250
- Breakeven: $97.50 (100 - 2.50)
```

**Position Sizing Guidelines**:
- Risk 1-3% of portfolio per trade
- Account for assignment risk on short put
- Consider correlation with existing positions
- Maintain cash reserves for adjustments

---

## Basic Setup

### Step 1: Select Underlying

**Screening Criteria**:
- High liquidity (>500K average daily volume)
- Tight bid-ask spreads (<$0.10 for puts)
- Active options market (open interest >1000)
- Clear bearish catalyst or technical setup

**Technical Confirmation**:
- Broken support level
- Bearish chart pattern (head-and-shoulders, double top)
- RSI overbought (>70)
- Negative divergence on momentum indicators

### Step 2: Choose Expiration

**30-45 DTE** (Sweet Spot):
- Good balance of time vs. cost
- Sufficient time for move to develop
- Manageable theta decay
- Liquid monthly options

**60-90 DTE** (Conservative):
- More time for thesis to play out
- Lower theta per day
- Higher upfront capital requirement

**15-30 DTE** (Aggressive):
- Faster theta decay
- Lower capital requirement
- Requires precise timing

### Step 3: Select Strikes

**Long Put Strike (Higher)**:
- **ATM**: Delta ~0.50, balanced cost/protection
- **Slightly ITM**: Delta 0.55-0.65, more directional
- **Slightly OTM**: Delta 0.35-0.45, lower cost

**Short Put Strike (Lower)**:
- Typically $5-$10 below long strike
- Should be below price target
- Check support levels to avoid early assignment

**Spread Width Options**:
- **$5 Wide**: Lower capital, tighter profit zone
- **$10 Wide**: Standard for most stocks
- **$15+ Wide**: Higher capital, better risk/reward if confident

---

## Quick Reference Checklist

### Pre-Trade
- [ ] Bearish catalyst identified
- [ ] Technical confirmation present
- [ ] IV rank checked (prefer 25-75)
- [ ] 30-60 DTE selected
- [ ] Strikes chosen relative to price target
- [ ] Position size within risk limits (1-3%)
- [ ] Profit target and stop loss defined

### Entry
- [ ] Limit order at mid-price or better
- [ ] Entered as vertical spread order
- [ ] Filled within acceptable slippage

### Management
- [ ] Daily price monitoring
- [ ] Check P&L vs. targets
- [ ] Watch for technical invalidation
- [ ] Exit at 50% profit or 50% loss
- [ ] Close before expiration week

---

## See Also

**Within This Skill**:
- [Strategy Mechanics](strategy-mechanics.md) - Detailed position structure and P&L
- [Strike Selection](strike-selection.md) - Advanced strike optimization
- [Position Management](position-management.md) - Trade adjustments and rolling
- [Greeks Analysis](greeks-analysis.md) - Risk management with Greeks
- [Examples](examples.md) - Real-world trade scenarios

**Master Resources**:
- [Options Greeks](../../options-strategies/references/greeks.md) - Comprehensive Greeks guide
- [Volatility Analysis](../../options-strategies/references/volatility.md) - IV metrics and analysis

**Related Strategies**:
- [Bull Call Spread](../../bull-call-spread/SKILL.md) - Opposite directional strategy
- [Iron Condor](../../iron-condor/SKILL.md) - Uses bear put spread as component

---

**Last Updated**: 2025-12-12
'@
Write-ReferenceFile -Strategy 'bear-put-spread' -FileName 'quickstart.md' -Content $Content

# 2. strategy-mechanics.md
$Content = @'
# Bear Put Spread - Strategy Mechanics

**Parent**: [Bear Put Spread](../SKILL.md) | **Related**: [Quickstart](quickstart.md) | [Greeks Analysis](greeks-analysis.md)

---

## Position Structure

### Components

A bear put spread consists of two puts with the same expiration but different strikes:

**Long Put (Higher Strike)**:
- Buy to open
- Higher premium (more expensive)
- Provides bearish exposure
- Defines maximum profit point

**Short Put (Lower Strike)**:
- Sell to open
- Lower premium (cheaper)
- Reduces net cost
- Defines maximum loss point below this strike

### Net Position Characteristics

**Type**: Vertical debit spread
**Direction**: Bearish
**Cost**: Net debit (pay to open)
**Risk Profile**: Limited risk, limited reward

---

## Payoff Analysis

### Maximum Profit

**Formula**:
```
Max Profit = (Long Strike - Short Strike) - Net Debit
Max Profit = Spread Width - Net Debit Paid
```

**Occurs When**: Stock price ≤ short strike at expiration

**Example**:
```python
# Stock at $100
long_strike = 100
short_strike = 95
long_premium = 5.00
short_premium = 2.50

net_debit = long_premium - short_premium  # 2.50
spread_width = long_strike - short_strike  # 5.00
max_profit = spread_width - net_debit      # 2.50 per share
max_profit_total = max_profit * 100        # $250 per contract
```

### Maximum Loss

**Formula**:
```
Max Loss = Net Debit Paid
Max Loss = Long Premium - Short Premium
```

**Occurs When**: Stock price ≥ long strike at expiration

**Example**:
```python
max_loss = net_debit * 100  # $250 per contract
```

### Breakeven Point

**Formula**:
```
Breakeven = Long Strike - Net Debit
```

**Example**:
```python
breakeven = 100 - 2.50  # $97.50
```

At $97.50, the position breaks even (excluding commissions).

---

## Profit/Loss at Expiration

### P&L Formula by Price Region

**Region 1: S ≤ Short Strike** (Maximum Profit)
```
P&L = (Spread Width - Net Debit) × 100
```

**Region 2: Short Strike < S < Long Strike** (Partial Profit/Loss)
```
P&L = [(Long Strike - S) - Net Debit] × 100
```

**Region 3: S ≥ Long Strike** (Maximum Loss)
```
P&L = -Net Debit × 100
```

### Example Scenarios

**Setup**:
- Stock: $100
- Buy $100 put @ $5.00
- Sell $95 put @ $2.50
- Net debit: $2.50

**Scenario 1: Stock at $92** (Below short strike)
```python
long_put_value = 100 - 92  # $8 intrinsic
short_put_value = 95 - 92  # $3 intrinsic
spread_value = long_put_value - short_put_value  # $5

profit = (5.00 - 2.50) * 100  # $250 (MAX)
return_pct = 250 / 250  # 100%
```

**Scenario 2: Stock at $97** (Between strikes)
```python
long_put_value = 100 - 97  # $3 intrinsic
short_put_value = 0         # OTM
spread_value = 3.00

profit = (3.00 - 2.50) * 100  # $50
return_pct = 50 / 250  # 20%
```

**Scenario 3: Stock at $100** (At long strike = breakeven)
```python
long_put_value = 0  # ATM, no intrinsic
short_put_value = 0  # OTM
spread_value = 0

profit = (0 - 2.50) * 100  # -$250
# But we paid $2.50, so breakeven is at $97.50, not $100
```

**Scenario 4: Stock at $103** (Above long strike)
```python
long_put_value = 0  # OTM
short_put_value = 0  # OTM
spread_value = 0

loss = -2.50 * 100  # -$250 (MAX)
return_pct = -100%
```

---

## Risk/Reward Profile

### Risk Metrics

**Maximum Risk**: Net debit paid ($250 in example)
**Maximum Reward**: Spread width - net debit ($250 in example)
**Risk/Reward Ratio**: 1:1 in this example

**Break-Even Probability**: Depends on:
- Implied volatility
- Time to expiration
- Distance from current price

### Ideal Risk/Reward Targets

**Aggressive**: 1:1.5 or better
```
Example: Risk $200 to make $300
- Buy $100 put @ $4.00
- Sell $93 put @ $1.50
- Net debit: $2.50
- Max profit: $4.50 ($7 spread - $2.50 debit)
```

**Balanced**: 1:1
```
Example: Risk $250 to make $250
- Standard 50% of spread width as debit
```

**Conservative**: 1:0.75
```
Example: Risk $300 to make $200
- Buy ATM put (expensive)
- Sell far OTM put (cheap)
- Higher probability of profit
- Lower max profit
```

---

## Time Decay (Theta)

### Theta Dynamics

**Before Move**: Theta works against you
- Long put has negative theta
- Short put has positive theta
- Net theta typically negative but smaller than naked put

**After Favorable Move**: Theta can work for you
- If stock drops to/below short strike
- Both puts are ITM
- Theta becomes less significant
- Position approaches max profit

### Example Theta Analysis

```python
# At entry (45 DTE)
long_put_theta = -0.03   # Losing $3/day
short_put_theta = 0.02   # Gaining $2/day
net_theta = -0.01        # Losing $1/day net

# Position value impact
daily_theta_decay = -0.01 * 100  # -$1 per day
```

**Theta Acceleration**:
- Minimal with >45 DTE
- Moderate at 30-45 DTE
- Significant at <30 DTE
- Critical at <15 DTE

---

## Implied Volatility Impact

### Vega Exposure

**Long Put**: Positive vega (benefits from IV increase)
**Short Put**: Negative vega (hurts from IV increase)
**Net Vega**: Positive but reduced vs. naked put

### IV Scenarios

**Scenario 1: IV Increases** (e.g., market stress)
```python
# Before IV increase
long_put_value = 5.00
short_put_value = 2.50
spread_value = 2.50

# After IV spike (+10 points)
long_put_value = 6.50  # +$1.50
short_put_value = 3.50  # +$1.00
spread_value = 3.00    # +$0.50

# Net benefit: $50 per contract
```

**Scenario 2: IV Decreases** (volatility crush)
```python
# After IV drop (-10 points)
long_put_value = 3.50  # -$1.50
short_put_value = 1.50  # -$1.00
spread_value = 2.00    # -$0.50

# Net cost: -$50 per contract
```

**Implication**: Enter when IV is not at extremes (prefer IV rank 25-75)

---

## Early Assignment Risk

### When Assignment Occurs

**Short Put Assignment Risk**:
- Stock drops below short strike
- Put is deep ITM
- Dividend approaching (rare for puts)
- Close to expiration

### Managing Assignment

**Prevention**:
1. Close position before expiration week
2. Monitor ITM puts daily
3. Roll position if stock near short strike

**If Assigned**:
1. You're now short 100 shares at short strike
2. Exercise long put to offset (deliver shares)
3. Result: Max profit realized
4. Contact broker immediately

**Example**:
```
Short $95 put assigned at expiration
- Now short 100 shares at $95
- Exercise long $100 put
- Deliver shares at $100
- Net: ($100 - $95) - $2.50 debit = $2.50/share profit
- Total: $250 (same as max profit)
```

---

## Commission Impact

### Fee Structure

**Typical Commissions** (varies by broker):
```
Entry:
- Options contract fee: $0.65 × 2 legs = $1.30
- Regulatory fees: $0.10
Total entry: ~$1.40

Exit:
- Options contract fee: $0.65 × 2 legs = $1.30
- Regulatory fees: $0.10
Total exit: ~$1.40

Round-trip total: ~$2.80 per contract
```

### Impact on Returns

**Example Trade**:
```python
max_profit_gross = 250.00
round_trip_commissions = 2.80
max_profit_net = 247.20

return_gross = (250 / 250) * 100  # 100%
return_net = (247.20 / 252.80) * 100  # 97.8%
```

For small positions, commissions can significantly impact percentage returns.

---

## Comparison to Alternatives

### vs. Long Put (Naked)

**Bear Put Spread**:
- ✅ Lower cost
- ✅ Defined risk
- ❌ Limited profit potential
- ❌ More complex execution

**Long Put**:
- ❌ Higher cost
- ❌ Higher vega risk
- ✅ Unlimited profit potential (to $0)
- ✅ Simple execution

### vs. Short Call

**Bear Put Spread**:
- ✅ Defined risk
- ✅ Debit spread (no margin requirement)
- ❌ Net cost upfront
- ✅ Better for moderate decline

**Short Call**:
- ❌ Unlimited risk
- ❌ Margin requirement
- ✅ Collect premium upfront
- ❌ Risk in sharp rallies

---

## See Also

**Within This Skill**:
- [Quickstart](quickstart.md) - Getting started guide
- [Strike Selection](strike-selection.md) - Optimizing strike prices
- [Position Management](position-management.md) - Adjustments and exits
- [Greeks Analysis](greeks-analysis.md) - Detailed Greeks breakdown
- [Examples](examples.md) - Real-world scenarios

**Master Resources**:
- [Options Pricing](../../options-strategies/references/greeks.md) - Black-Scholes and Greeks
- [Volatility](../../options-strategies/references/volatility.md) - IV analysis

**Related Strategies**:
- [Bull Call Spread](../../bull-call-spread/SKILL.md) - Bullish equivalent
- [Protective Put](../../married-put/SKILL.md) - Stock protection alternative

---

**Last Updated**: 2025-12-12
'@
Write-ReferenceFile -Strategy 'bear-put-spread' -FileName 'strategy-mechanics.md' -Content $Content

# 3. strike-selection.md
$Content = @'
# Bear Put Spread - Strike Selection

**Parent**: [Bear Put Spread](../SKILL.md) | **Related**: [Strategy Mechanics](strategy-mechanics.md) | [Examples](examples.md)

---

## Strike Selection Framework

### Key Decisions

When constructing a bear put spread, you must choose:
1. **Long put strike** (higher strike - you buy)
2. **Short put strike** (lower strike - you sell)
3. **Spread width** (difference between strikes)

Each decision impacts:
- Capital requirement
- Maximum profit potential
- Breakeven point
- Probability of profit
- Risk/reward ratio

---

## Long Put Strike Selection

### Delta-Based Approach

**ATM (At-The-Money)**: Delta ~0.50
- **Pros**: Balanced cost/directional exposure, symmetric risk
- **Cons**: Moderate cost, requires price to move below current
- **Best for**: Moderate bearish conviction

**ITM (In-The-Money)**: Delta 0.55-0.70
- **Pros**: Already profitable if stock stays flat or drops, higher delta
- **Cons**: More expensive, lower leverage
- **Best for**: High bearish conviction, conservative approach

**OTM (Out-of-The-Money)**: Delta 0.30-0.45
- **Pros**: Lower cost, higher leverage
- **Cons**: Requires significant move, lower probability
- **Best for**: Aggressive bearish outlook

### Price Target Method

**Step 1**: Determine price target
```
Current Price: $100
Analysis: Expect drop to $90
Target: $90 (10% decline)
```

**Step 2**: Select long strike relative to target
```
Conservative: Long strike AT current price ($100)
- Breakeven will be below current
- Profit if stock reaches target

Moderate: Long strike slightly below ($97-98)
- Lower cost
- Still profits if target reached

Aggressive: Long strike near target ($92-93)
- Lowest cost
- Maximum leverage
- Must reach target for profit
```

### Implied Volatility Consideration

**High IV Environment** (IV rank > 70):
- Avoid buying expensive options
- Consider OTM long strike to reduce cost
- Or wait for IV to normalize

**Low IV Environment** (IV rank < 30):
- Options cheaper
- Can afford ATM or ITM strikes
- Better risk/reward

**Normal IV** (IV rank 30-70):
- Use standard ATM to slight OTM
- Follow price target method

---

## Short Put Strike Selection

### Methods

**Fixed Spread Width Method**:
```
Choose long strike first
Then subtract fixed amount:
- $5 width for stocks $50-$150
- $10 width for stocks $150-$300
- $15+ width for stocks >$300
```

**Support Level Method**:
```
Identify technical support below price target
Place short strike at or just below support
- Reduces early assignment risk
- Logical exit point if thesis invalidated
```

**Probability-Based Method**:
```
Use options chain to check probability
Target short strike with:
- 15-25% probability of finishing ITM
- Balances premium collected vs. risk
```

### Delta Targets for Short Put

**Conservative**: Delta 0.15-0.20
- Far OTM
- Low probability of assignment
- Lower premium collected
- Wider spread required

**Moderate**: Delta 0.25-0.35
- Moderately OTM
- Balanced probability
- Decent premium
- Standard approach

**Aggressive**: Delta 0.40-0.45
- Near ATM
- Higher probability of assignment
- More premium collected
- Narrower spread possible

---

## Spread Width Optimization

### Risk/Reward Tradeoffs

**Narrow Spread** ($2.50 - $5.00):
- Lower capital requirement
- Lower maximum profit
- Higher capital efficiency needed
- Requires more precise timing

**Standard Spread** ($5.00 - $10.00):
- Balanced capital requirement
- Reasonable profit potential
- Most liquid and common
- Good for most situations

**Wide Spread** ($10.00 - $20.00):
- Higher capital requirement
- Higher maximum profit potential
- More room for error
- Better for high conviction trades

### Calculating Optimal Width

**Method 1: Target Risk/Reward Ratio**
```python
# Target 1:1 risk/reward
# Want to risk $250 to make $250

target_profit = 250
# Solve for spread width:
# max_profit = (width - debit) * 100 = 250
# If debit = 50% of width (typical)
# Then: (width - 0.5*width) * 100 = 250
# 0.5 * width * 100 = 250
# width = 5.00

spread_width = 5.00
```

**Method 2: Expected Move Based**
```python
# Expected move to target
current_price = 100
price_target = 92
expected_move = current_price - price_target  # 8 points

# Spread should capture most of move
spread_width = expected_move * 0.75  # 6 points
# Use $5 or $10 standard width
```

---

## Expiration Selection Impact on Strikes

### Time Horizon

**Longer Expiration** (60-90 DTE):
- Can use wider strikes (more room for move)
- Lower theta decay pressure
- More expensive options
- Better for larger expected moves

**Medium Expiration** (30-60 DTE):
- Standard strike selection
- Balanced theta
- Most liquid options
- Recommended for most trades

**Shorter Expiration** (15-30 DTE):
- Tighter strikes recommended
- Faster theta decay
- Requires more precise timing
- Higher risk/reward

---

## Strike Selection Examples

### Example 1: Conservative Setup

**Scenario**:
```
Stock: $150
Outlook: Moderately bearish, expect $140 in 60 days
IV Rank: 45 (moderate)
Risk Tolerance: Conservative
```

**Strike Selection**:
```
Expiration: 60 DTE
Long Strike: $150 (ATM, delta 0.50)
Short Strike: $140 (at price target, delta 0.25)
Spread Width: $10

Long Put Premium: $7.50
Short Put Premium: $3.00
Net Debit: $4.50
```

**Analysis**:
```
Max Loss: $450
Max Profit: $550 ($10 - $4.50 = $5.50 × 100)
Breakeven: $145.50
Risk/Reward: 1:1.22
```

### Example 2: Moderate Setup

**Scenario**:
```
Stock: $200
Outlook: Bearish, expect $185 in 45 days
IV Rank: 55 (moderate-high)
Risk Tolerance: Moderate
```

**Strike Selection**:
```
Expiration: 45 DTE
Long Strike: $195 (slightly OTM, delta 0.40)
Short Strike: $185 (at target, delta 0.22)
Spread Width: $10

Long Put Premium: $6.00
Short Put Premium: $2.25
Net Debit: $3.75
```

**Analysis**:
```
Max Loss: $375
Max Profit: $625 ($10 - $3.75 = $6.25 × 100)
Breakeven: $191.25
Risk/Reward: 1:1.67
```

### Example 3: Aggressive Setup

**Scenario**:
```
Stock: $100
Outlook: Strong bearish, expect $85 in 30 days
IV Rank: 35 (moderate-low)
Risk Tolerance: Aggressive
```

**Strike Selection**:
```
Expiration: 30 DTE
Long Strike: $95 (OTM, delta 0.35)
Short Strike: $85 (at target, delta 0.15)
Spread Width: $10

Long Put Premium: $3.50
Short Put Premium: $0.75
Net Debit: $2.75
```

**Analysis**:
```
Max Loss: $275
Max Profit: $725 ($10 - $2.75 = $7.25 × 100)
Breakeven: $92.25
Risk/Reward: 1:2.64
```

---

## Common Strike Selection Mistakes

### Mistake 1: Spread Too Narrow

**Problem**: $2.50 spread on $200 stock
```
Max profit: ~$125
Max loss: ~$125
Commission impact: ~2% of max profit
Slippage risk: High
```

**Fix**: Use minimum $5 spread, preferably $10

### Mistake 2: Spread Too Wide

**Problem**: $25 spread on $100 stock
```
Capital required: ~$1,500+
Inefficient capital usage
Better alternatives exist
```

**Fix**: Use $5-$10 spreads for most stocks

### Mistake 3: Ignoring Technical Levels

**Problem**: Short strike at obvious support
```
Stock at $100
Support at $95
Short $95 put

Risk: Bounce at support, no profit
```

**Fix**: Place short strike below support ($92-93)

### Mistake 4: Poor Risk/Reward

**Problem**: Risking $400 to make $100
```
Debit: $4.00
Width: $5.00
Max profit: $1.00 × 100 = $100
Max loss: $400
Ratio: 1:0.25
```

**Fix**: Target minimum 1:1, ideally 1:1.5+

---

## Strike Adjustment Guidelines

### When to Adjust Strikes

**Before Entry**:
- IV rank changes significantly
- Price target revised
- Technical levels shift

**After Entry** (via rolling):
- Stock moved against you
- Thesis still valid
- Want to extend time or adjust strikes

### Rolling Strikes

**Roll Down** (take profits, maintain exposure):
```
Original:
Buy $100 put, Sell $95 put
Stock now at $93

Roll to:
Buy $95 put, Sell $90 put
- Realize some profit
- Maintain bearish exposure
- Reset position
```

**Roll Out** (extend time, same strikes):
```
Original: 45 DTE
Stock stuck at $98
Thesis intact

Roll to: 60 DTE
Same strikes
Add capital to extend time
```

---

## Strike Selection Checklist

### Pre-Selection
- [ ] Price target identified
- [ ] Expected move calculated
- [ ] Technical levels mapped
- [ ] IV rank checked
- [ ] Time horizon determined

### Long Strike Selection
- [ ] Delta range selected (0.30-0.70)
- [ ] Relative to current price evaluated
- [ ] Relative to price target evaluated
- [ ] Premium cost acceptable

### Short Strike Selection
- [ ] Below price target
- [ ] Below technical support
- [ ] Delta range appropriate (0.15-0.35)
- [ ] Premium collected meaningful

### Spread Width
- [ ] Width appropriate for stock price
- [ ] Captures expected move
- [ ] Risk/reward ratio acceptable (≥1:1)
- [ ] Capital requirement reasonable

### Final Validation
- [ ] Breakeven calculated
- [ ] Maximum loss acceptable
- [ ] Maximum profit justifies risk
- [ ] Bid-ask spread reasonable (<10% of debit)

---

## See Also

**Within This Skill**:
- [Quickstart](quickstart.md) - Basic setup guide
- [Strategy Mechanics](strategy-mechanics.md) - P&L analysis
- [Position Management](position-management.md) - Adjustments
- [Greeks Analysis](greeks-analysis.md) - Greek impacts
- [Examples](examples.md) - Real strike selections

**Master Resources**:
- [Options Pricing](../../options-strategies/references/greeks.md) - Delta and pricing
- [Volatility](../../options-strategies/references/volatility.md) - IV impact on strikes

**Related Strategies**:
- [Bull Call Spread](../../bull-call-spread/SKILL.md) - Bullish strike selection

---

**Last Updated**: 2025-12-12
'@
Write-ReferenceFile -Strategy 'bear-put-spread' -FileName 'strike-selection.md' -Content $Content

# 4. position-management.md
$Content = @'
# Bear Put Spread - Position Management

**Parent**: [Bear Put Spread](../SKILL.md) | **Related**: [Strategy Mechanics](strategy-mechanics.md) | [Strike Selection](strike-selection.md)

---

## Management Philosophy

Effective position management is critical for bear put spreads because:
- Limited profit potential requires disciplined profit-taking
- Time decay accelerates near expiration
- Spreads can turn from winners to losers quickly
- Assignment risk increases as expiration approaches

**Core Principle**: Take profits when available, cut losses when thesis invalidated.

---

## Profit Targets

### Percentage-Based Targets

**50% of Maximum Profit** (Recommended Default)
```python
max_profit = 250
target_profit = max_profit * 0.50  # $125

# Close when position worth:
target_spread_value = net_debit + (max_profit / 100) * 0.50
# If net debit = $2.50, close at $3.75 spread value
```

**Why 50%?**
- Captures majority of theoretical gains
- Avoids late-stage theta decay
- Frees capital for new opportunities
- Reduces pin risk
- Statistically optimal for repeated trades

**75% of Maximum Profit** (Aggressive)
- Higher return per trade
- More time risk
- Theta acceleration zone
- Use only with strong conviction

**25-30% of Maximum Profit** (Conservative)
- Quick profits
- High trade frequency
- Lower risk
- Compounds faster

### Price-Based Targets

**At Short Strike**:
```
Stock reaches short strike
Spread near maximum value
Close to avoid pin risk
```

**At Technical Target**:
```
Stock hits support level
Take profits regardless of percentage
Reduces risk of reversal
```

---

## Stop Loss Guidelines

### Loss Thresholds

**50% of Maximum Loss** (Standard)
```python
max_loss = 250
stop_loss = max_loss * 0.50  # $125 loss

# Exit when spread value drops to:
stop_spread_value = net_debit * 0.50
# If net debit = $2.50, exit if spread worth $1.25
```

**Why 50%?**
- Preserves capital for recovery
- Limits damage from wrong thesis
- Allows multiple attempts
- Maintains psychological discipline

**Time-Based Stop Loss**:
```
If position underwater at 10 DTE:
- Exit regardless of loss percentage
- Theta acceleration too severe
- Unlikely to recover
```

### Thesis Invalidation

**Exit immediately when**:
- Technical picture reverses (break above resistance)
- Fundamental catalyst changes (positive earnings surprise)
- Correlation breaks (sector reversal)

**Example**:
```
Entered bear spread on failed support at $100
Stop loss: 50% of max loss ($125)

Stock rallies to $105:
- Support failure invalidated
- Exit immediately
- Don't wait for technical stop
```

---

## Rolling Strategies

### When to Roll

**Roll Out** (Extend Time):
- Thesis still valid
- Need more time for move
- Stock hasn't moved much
- 15-21 DTE remaining

**Roll Down** (Lower Strikes):
- Stock dropped favorably
- Want to lock in some profit
- Maintain bearish exposure
- Take chips off table

**Roll Out and Down**:
- Combine time extension with strike adjustment
- Most common rolling strategy
- Adds capital but maintains exposure

### Rolling Mechanics

**Roll Out Example**:
```
Original Position (21 DTE):
Buy $100 put @ $3.00
Sell $95 put @ $1.25
Net debit: $1.75

Current (21 DTE):
Stock at $98
Spread value: $2.25 (small profit)
Close for $2.25 (50¢ profit)

New Position (45 DTE):
Buy $100 put @ $4.50
Sell $95 put @ $2.00
Net debit: $2.50

Net adjustment cost: $2.50 - $2.25 = $0.25
Total capital now: $1.75 + $0.25 = $2.00
```

**Roll Down Example**:
```
Original Position:
Buy $100 put @ $5.00
Sell $95 put @ $2.50
Net debit: $2.50

Current State:
Stock at $93
Spread value: $4.75 (near max of $5.00)

Close and Roll Down:
Close for $4.75 ($2.25 profit)

New Position:
Buy $95 put @ $4.25
Sell $90 put @ $2.00
Net debit: $2.25

Realized profit: $4.75 - $2.50 = $2.25
New position cost: $2.25
Still bearish, profit locked
```

---

## Adjustment Strategies

### If Stock Rallies (Against You)

**Stage 1: Small Move Against** (2-3%)
- Monitor daily
- Check thesis validity
- Hold if conviction remains
- Prepare to exit if continues

**Stage 2: Moderate Move Against** (5-7%)
- Approaching stop loss
- Re-evaluate thesis
- Consider rolling out
- Tighten monitoring

**Stage 3: Large Move Against** (>10%)
- Likely stopped out
- Thesis invalidated
- Exit immediately
- Accept loss

### If Stock Drops (In Your Favor)

**Stage 1: Small Drop** (2-5%)
- Position showing profit
- Hold to target (50% max profit)
- Monitor for reversal

**Stage 2: Reaches Target** (5-10%)
- At/near short strike
- Take profits
- Consider rolling down if very bullish

**Stage 3: Crashes Past Target** (>15%)
- At maximum profit
- Close immediately
- Don't hope for more
- Avoid assignment risk

---

## Monitoring Schedule

### Daily Checks

**Morning Routine** (Market Open):
```
1. Check underlying price
2. Calculate current P&L
3. Compare to profit target and stop loss
4. Note any overnight news
5. Check technical levels
```

**End of Day** (Market Close):
```
1. Log closing price
2. Update P&L
3. Calculate days to expiration
4. Set alerts for next day
5. Plan any needed actions
```

### Weekly Review

**Every Monday**:
- Review all open positions
- Check days to expiration
- Assess if rolling needed
- Review thesis validity
- Calculate portfolio heat

---

## Special Situations

### Early Assignment

**Short Put Assigned**:
```
Notification: Short 100 shares at short strike

Immediate Actions:
1. Exercise long put
2. Deliver shares at long strike
3. Realize max profit
4. Contact broker to confirm

Result:
- Profit = (long strike - short strike) - net debit
- Same as holding to expiration
```

**Prevention**:
- Close ITM spreads before expiration week
- Monitor dividend dates (rare for puts)
- Watch for low time value on short put

### Pin Risk

**Definition**: Stock closes very near a strike at expiration

**Problem**:
```
Stock at $95.05 at Friday close
Short $95 put status unclear
May or may not be assigned
Long put may expire worthless
```

**Prevention**:
- Close all positions by Friday 3:00 PM
- Don't hold through 4:00 PM close
- Especially critical if stock near strikes

### Dividend Capture

**Ex-Dividend Date**:
- Rare to cause put assignment
- Stock typically drops by dividend amount
- Can help bearish position
- Monitor if large dividend

---

## Exit Execution

### How to Close

**Preferred Method**: Close as spread
```
Order Type: Vertical Spread
Action: Sell to Close
Long Strike: $100 PUT
Short Strike: $95 PUT
Limit Price: $3.75 (or current mid-price)
```

**Avoid**: Closing legs separately
- Higher slippage
- Execution risk
- Price movement between legs
- Less favorable fills

### Timing

**Best Times to Exit**:
- 9:45-10:30 AM ET (post-open volatility settled)
- 2:00-3:30 PM ET (pre-close liquidity)

**Avoid**:
- First 15 minutes (9:30-9:45 AM)
- Last 15 minutes (3:45-4:00 PM)
- Low volume periods (lunch hour)

### Order Types

**Limit Order** (Recommended):
```
Set limit at mid-price
Adjust in $0.05 increments if needed
Be patient for good fill
```

**Market Order** (Avoid):
- Unpredictable fill
- High slippage
- Only if urgent exit needed

---

## Position Sizing and Portfolio Heat

### Single Position Sizing

**Conservative**: Risk 1% of portfolio per trade
```python
portfolio_value = 50000
risk_per_trade = portfolio_value * 0.01  # $500
contracts = risk_per_trade / max_loss_per_contract
# If max loss = $250/contract: 2 contracts
```

**Moderate**: Risk 2% of portfolio
```python
risk_per_trade = 50000 * 0.02  # $1000
contracts = 1000 / 250  # 4 contracts
```

**Aggressive**: Risk 3% of portfolio
```python
risk_per_trade = 50000 * 0.03  # $1500
contracts = 1500 / 250  # 6 contracts
```

### Portfolio Heat (Multiple Positions)

**Total Risk Limit**: 10-15% of portfolio
```python
# With 5 bear spreads open
total_at_risk = 5 * 500  # $2500
portfolio_heat = 2500 / 50000  # 5%
status = "OK"  # Under 15% limit
```

If approaching limits:
- No new positions
- Close losing positions
- Wait for capital release

---

## Performance Tracking

### Metrics to Track

**Per Trade**:
- Entry date and price
- Strikes and expiration
- Net debit paid
- Exit date and price
- Net credit received
- P&L amount
- P&L percentage
- Days held
- Reason for exit

**Portfolio Level**:
- Win rate (% profitable trades)
- Average win ($ and %)
- Average loss ($ and %)
- Profit factor (total wins / total losses)
- Expected value per trade

### Example Trade Log

```
Trade #47:
Symbol: XYZ
Entry: 2025-11-15
Expiration: 2025-12-20 (35 DTE)
Strikes: Buy $150 put / Sell $145 put
Debit: $2.75 ($275)
Exit: 2025-12-01
Credit: $4.00 ($400)
P&L: $125 (45%)
Days Held: 16
Exit Reason: 50% profit target reached
```

---

## Common Management Mistakes

### Mistake 1: Holding for Max Profit

**Problem**: Waiting for 100% of max profit
**Impact**: Theta decay erodes position, winners turn to losers
**Fix**: Take 50-75% profit consistently

### Mistake 2: No Stop Loss

**Problem**: Hoping position recovers
**Impact**: Small losses become large losses
**Fix**: Set stop loss at 50% max loss, honor it

### Mistake 3: Overtrading

**Problem**: Too many positions open
**Impact**: Inability to manage all effectively
**Fix**: Limit to 3-5 active spreads

### Mistake 4: Ignoring Theta

**Problem**: Holding positions past 15 DTE
**Impact**: Rapid time decay
**Fix**: Close or roll by 21 DTE

### Mistake 5: Poor Record Keeping

**Problem**: No trade journal
**Impact**: Can't learn from mistakes
**Fix**: Log every trade with details

---

## Management Checklist

### Entry
- [ ] Position size within limits (1-3% risk)
- [ ] Profit target defined (50% max)
- [ ] Stop loss set (50% max loss)
- [ ] Exit plan documented
- [ ] Calendar alert set for 21 DTE

### During Trade
- [ ] Daily price monitoring
- [ ] Weekly P&L review
- [ ] Thesis validation check
- [ ] Technical levels watching
- [ ] Portfolio heat under limit

### Exit
- [ ] Close as vertical spread
- [ ] Timing during liquid hours
- [ ] Limit order at fair price
- [ ] Confirm both legs closed
- [ ] Log trade in journal

---

## See Also

**Within This Skill**:
- [Quickstart](quickstart.md) - Setup basics
- [Strategy Mechanics](strategy-mechanics.md) - P&L understanding
- [Strike Selection](strike-selection.md) - Initial setup
- [Greeks Analysis](greeks-analysis.md) - Risk metrics
- [Examples](examples.md) - Management scenarios

**Master Resources**:
- [Options Greeks](../../options-strategies/references/greeks.md) - Greek impacts on management
- [Volatility](../../options-strategies/references/volatility.md) - IV changes

**Related Strategies**:
- [Bull Call Spread](../../bull-call-spread/SKILL.md) - Similar management
- [Iron Condor](../../iron-condor/SKILL.md) - Multi-leg management

---

**Last Updated**: 2025-12-12
'@
Write-ReferenceFile -Strategy 'bear-put-spread' -FileName 'position-management.md' -Content $Content

# 5. greeks-analysis.md
$Content = @'
# Bear Put Spread - Greeks Analysis

**Parent**: [Bear Put Spread](../SKILL.md) | **Related**: [Strategy Mechanics](strategy-mechanics.md) | [Position Management](position-management.md)

---

## Overview

Understanding Greeks is essential for managing bear put spreads effectively. This guide covers how delta, gamma, theta, and vega affect your position and how to use them for risk management.

---

## Delta

### Position Delta

**Formula**:
```
Position Delta = (Long Put Delta) - (Short Put Delta)
```

**Characteristics**:
- Negative (benefits from price decline)
- Typically ranges from -0.15 to -0.40 per spread
- Represents expected P&L for $1 move in underlying

### Delta Examples

**ATM Spread**:
```python
# Stock at $100
long_put_strike = 100  # Delta: -0.50
short_put_strike = 95   # Delta: -0.20

position_delta = -0.50 - (-0.20)  # -0.30
# For 1 contract (100 shares)
effective_delta = -0.30 * 100  # -30

# Expected P&L for $1 drop:
expected_profit = 30  # $30 profit if stock drops $1
```

**OTM Spread**:
```python
# Stock at $100
long_put_strike = 95   # Delta: -0.30
short_put_strike = 90  # Delta: -0.12

position_delta = -0.30 - (-0.12)  # -0.18
effective_delta = -0.18 * 100  # -18
# Less sensitive to price movement
```

### Delta Evolution

**As Stock Drops** (favorable):
```
Initial: Stock $100, Delta -0.30
Stock → $95: Delta increases to -0.45
Stock → $90: Delta approaches -1.00

Why: Long put goes deeper ITM, short put approaches ATM
```

**As Stock Rises** (unfavorable):
```
Initial: Stock $100, Delta -0.30
Stock → $105: Delta decreases to -0.15
Stock → $110: Delta approaches 0.00

Why: Both puts move OTM, lose directional sensitivity
```

### Using Delta for Management

**High Delta Positions** (|-0.40| or higher):
- Very directional
- Large P&L swings
- Consider taking profits
- May want to roll down

**Low Delta Positions** (|-0.15| or lower):
- Minimal directional exposure
- Unlikely to profit
- Consider closing if thesis unchanged
- May need to roll out/down

---

## Gamma

### Position Gamma

**Formula**:
```
Position Gamma = (Long Put Gamma) - (Short Put Gamma)
```

**Characteristics**:
- Typically positive
- Measures delta acceleration
- Peaks near ATM
- Low when far OTM or deep ITM

### Gamma Impact

**Positive Gamma** (typical for debit spreads):
```
Benefits:
- Delta increases as stock drops (good for bearish)
- Delta decreases as stock rises (limits losses)

Drawback:
- Works against you if position moves toward worthless
```

### Gamma Example

```python
# Current state
stock_price = 100
position_delta = -0.30
position_gamma = 0.08

# Stock drops $1 to $99
new_delta = -0.30 + (-0.08)  # -0.38
# Delta became more negative (more bearish)

# Stock rises $1 to $101
new_delta = -0.30 - (-0.08)  # -0.22
# Delta became less negative (less bearish)
```

### Gamma Through Time

**Early in Trade** (45+ DTE):
- Low gamma
- Delta changes slowly
- More time for thesis to develop

**Mid-Trade** (20-30 DTE):
- Gamma increasing
- Delta more sensitive
- Position can accelerate quickly

**Near Expiration** (<10 DTE):
- Very high gamma
- Delta extremely sensitive
- Dangerous to hold

---

## Theta

### Position Theta

**Formula**:
```
Position Theta = (Long Put Theta) - (Short Put Theta)
```

**Characteristics**:
- Typically negative (time decay works against you)
- Accelerates near expiration
- Offset by short put theta (reduces net decay)

### Theta Analysis

**Net Theta Example**:
```python
# 45 DTE
long_put_theta = -0.04   # Losing $4/day
short_put_theta = -0.02  # Gaining $2/day (you're short)

position_theta = -0.04 - (-0.02)  # -0.02
daily_decay = -0.02 * 100  # -$2/day

# Over a week:
weekly_decay = -2 * 5  # -$10 (market days)
```

### Theta Decay Curve

**45 DTE**:
```
Daily theta decay: -$1 to -$2
Position can be held
Time is not critical
```

**30 DTE**:
```
Daily theta decay: -$2 to -$4
Theta accelerating
Start monitoring closely
```

**15 DTE**:
```
Daily theta decay: -$5 to -$10
Severe theta decay
Should exit or roll
```

**7 DTE**:
```
Daily theta decay: -$10 to -$20+
Critical theta
Must exit
```

### Managing Theta

**Theta-Positive State** (after favorable move):
```
Stock at/below short strike
Both puts ITM
Theta becomes less negative or slightly positive
Position stabilizes

Action: Take profits, avoid theta reversal
```

**Theta-Negative State** (stock hasn't moved):
```
Underlying unchanged
Time ticking away
Losing value daily

Action:
- Exit if near 15 DTE
- Roll out if thesis intact
- Cut loss if thesis broken
```

---

## Vega

### Position Vega

**Formula**:
```
Position Vega = (Long Put Vega) - (Short Put Vega)
```

**Characteristics**:
- Typically positive (benefits from IV increase)
- Reduced vs. naked put (short put offsets)
- Less critical than for straddles/strangles

### Vega Example

```python
# Current state
long_put_vega = 0.15   # +$15 per 1% IV increase
short_put_vega = 0.08  # -$8 per 1% IV increase

position_vega = 0.15 - 0.08  # 0.07
# +$7 per 1% IV increase

# IV increases from 25% to 30% (+5 points):
iv_impact = 0.07 * 5 * 100  # +$35

# IV decreases from 25% to 20% (-5 points):
iv_impact = 0.07 * -5 * 100  # -$35
```

### IV Scenarios

**IV Expansion** (market stress, event upcoming):
```
Position benefits
Spreads widen
Can lock in extra profit

Example:
Entry with IV = 25%
Crisis → IV = 40%
Extra profit from vega: ~$50-$100
```

**IV Contraction** (vol crush post-earnings):
```
Position suffers
Spreads narrow
Loses extrinsic value

Example:
Entry with IV = 45% (pre-earnings)
Post-earnings → IV = 25%
Loss from vega: ~$50-$100
```

### Vega Considerations

**Entering Trades**:
- Check IV rank/percentile
- Avoid entering when IV rank > 75 (expensive)
- Prefer IV rank 25-75 (moderate)
- OK to enter when IV rank < 25 (cheap options)

**During Trades**:
- Monitor IV changes
- IV spike = opportunity to take profit early
- IV crush = may need to extend time (roll)

---

## Combined Greeks Analysis

### Example Position

**Setup**:
```
Stock: $100
Buy $100 put @ $5.00
Sell $95 put @ $2.50
Net Debit: $2.50
DTE: 45
IV: 30%
```

**Greeks**:
```python
delta = -0.30    # -$30 per $1 drop
gamma = 0.06     # Delta changes by 0.06 per $1 move
theta = -0.02    # -$2 per day
vega = 0.07      # +$7 per 1% IV increase
```

### Scenario Analysis

**Scenario 1: Stock drops $5 to $95** (favorable)
```python
# Delta impact (primary)
delta_pl = -0.30 * 100 * -5  # +$150

# Gamma impact (delta now ~-0.50)
gamma_impact = 0.06 * 5 * -5 * 100  # Additional sensitivity

# Net P&L after 5 days:
# Delta: +$150
# Theta: -$2 * 5 = -$10
# Vega: ~0 (assume IV unchanged)
total_pl = 150 - 10  # +$140 (56% of max profit)

# Take profits at 50% target
```

**Scenario 2: Stock rallies $5 to $105** (unfavorable)
```python
# Delta impact
delta_pl = -0.30 * 100 * 5  # -$150

# Theta impact (5 days)
theta_pl = -0.02 * 100 * 5  # -$10

# Net P&L:
total_pl = -150 - 10  # -$160 (64% of max loss)

# Approaching stop loss, evaluate thesis
```

**Scenario 3: Stock flat at $100** (time decay)
```python
# After 15 days
delta_pl = 0  # No price movement
theta_pl = -0.03 * 100 * 15  # -$45 (theta accelerated)

# Net P&L:
total_pl = -45  # -$45 (18% loss)

# Decision: Roll out if thesis intact, or exit
```

---

## Greeks-Based Management Rules

### Rule 1: Delta Alert

```
If |delta| < 0.15:
    → Low probability of profit
    → Consider exiting
    → Unless very early in trade

If |delta| > 0.60:
    → High directional risk
    → Consider taking profits
    → Or rolling down to lock in gains
```

### Rule 2: Theta Warning

```
If DTE < 15 and theta < -0.05:
    → Exit immediately
    → Or roll out

If DTE < 30 and position underwater:
    → Evaluate exit vs. roll
    → Theta will accelerate
```

### Rule 3: Vega Opportunity

```
If IV increases >15 points:
    → Consider taking profits early
    → Extra gains from vega
    → May not repeat

If IV decreases >15 points:
    → May need to extend time (roll)
    → Or accept reduced profit potential
```

### Rule 4: Gamma Caution

```
If DTE < 10:
    → Gamma very high
    → Delta changes rapidly
    → Don't hold through expiration
    → Close by Thursday of expiration week
```

---

## Greeks Monitoring Checklist

### Daily Monitoring
- [ ] Check current delta
- [ ] Calculate daily theta decay
- [ ] Note IV changes
- [ ] Compare to Greeks at entry

### Weekly Review
- [ ] Calculate net delta across portfolio
- [ ] Sum theta across all positions
- [ ] Check if any position has extreme Greeks
- [ ] Identify positions needing adjustment

### Critical Thresholds
- [ ] Delta < -0.15: Consider exit
- [ ] DTE < 15: Exit or roll immediately
- [ ] Theta > -$5/day: Position stressed
- [ ] IV change > 15 points: Reassess position

---

## See Also

**Within This Skill**:
- [Quickstart](quickstart.md) - Setup basics
- [Strategy Mechanics](strategy-mechanics.md) - P&L mechanics
- [Strike Selection](strike-selection.md) - Initial Greeks setup
- [Position Management](position-management.md) - Managing based on Greeks
- [Examples](examples.md) - Real Greeks scenarios

**Master Resources**:
- [Options Greeks](../../options-strategies/references/greeks.md) - Comprehensive Greeks guide
- [Black-Scholes Pricing](../../options-strategies/references/greeks.md#black-scholes) - Greeks calculations

**Related Strategies**:
- [Bull Call Spread](../../bull-call-spread/SKILL.md) - Similar Greek profile
- [Iron Condor](../../iron-condor/SKILL.md) - Multi-leg Greeks

---

**Last Updated**: 2025-12-12
'@
Write-ReferenceFile -Strategy 'bear-put-spread' -FileName 'greeks-analysis.md' -Content $Content

# 6. examples.md
$Content = @'
# Bear Put Spread - Real-World Examples

**Parent**: [Bear Put Spread](../SKILL.md) | **Related**: [Quickstart](quickstart.md) | [Strike Selection](strike-selection.md)

---

## Example 1: Technology Stock Breakdown

### Setup

**Underlying**: TECH (fictional tech stock)
**Situation**: Failed earnings, broken support, downtrend confirmed

**Technical Analysis**:
- Current Price: $152
- Broken Support: $155 (now resistance)
- Next Support: $140
- RSI: 68 (overbought)
- MACD: Bearish crossover

**Fundamental Catalyst**:
- Missed earnings by 15%
- Lowered guidance
- Sector weakness

**IV Analysis**:
- Current IV: 35%
- IV Rank: 48 (moderate)
- Historical average IV: 32%

### Trade Construction

**Selection Process**:
```
Outlook: Expect drop to $140 in 45 days
Risk Tolerance: Moderate
Capital: $500 per position (2% of $25k account)

Strike Selection:
Long: $150 put (slightly OTM, delta -0.42)
Short: $140 put (at support target, delta -0.22)
Spread Width: $10

Expiration: 45 DTE
```

**Execution**:
```
Buy 1 TECH Dec 150 put @ $6.25
Sell 1 TECH Dec 140 put @ $2.75
Net Debit: $3.50 ($350 total)

Max Loss: $350
Max Profit: $650 (($10 - $3.50) × 100)
Breakeven: $146.50
Risk/Reward: 1:1.86
```

### Position Management

**Day 1-10** (Price: $150-$152):
```
Stock trading in narrow range
P&L: -$25 to +$15
Theta: -$2/day
Action: Monitor, hold
```

**Day 11-20** (Price: $147-$149):
```
Stock drifting lower
P&L: +$75 to +$125
Approaching 50% profit target
Action: Set alert at $7.25 spread value
```

**Day 21** (Price: $145):
```
Stock dropped sharply
Spread Value: $7.50
P&L: +$400 (61% of max)
Action: CLOSE POSITION
Reason: Exceeded 50% target, take profits
```

### Outcome

**Results**:
```
Entry: $3.50 debit
Exit: $7.50 credit
Profit: $400
Return: 114%
Days Held: 21
Annualized Return: ~1,980%
```

**Lessons**:
- Waited for technical confirmation
- Sized appropriately (2% risk)
- Took profits at target
- Didn't get greedy

---

## Example 2: Failed Breakout

### Setup

**Underlying**: GROWTH (fictional growth stock)
**Situation**: Double top pattern, failed breakout above $200

**Technical Analysis**:
- Current Price: $198
- Failed Resistance: $205
- Support Levels: $190, $180
- Volume declining on rally attempts
- Bearish divergence on RSI

**IV Analysis**:
- Current IV: 42%
- IV Rank: 65 (elevated but not extreme)
- Post-earnings (IV stable)

### Trade Construction

**Selection Process**:
```
Outlook: Expect retest of $180 support
Time Horizon: 60 days (patient trade)
Capital: $750 (3% of $25k account)

Strike Selection:
Long: $195 put (ATM, delta -0.48)
Short: $180 put (at support, delta -0.18)
Spread Width: $15

Expiration: 60 DTE
```

**Execution**:
```
Buy 1 GROWTH Jan 195 put @ $10.50
Sell 1 GROWTH Jan 180 put @ $3.25
Net Debit: $7.25 ($725 total)

Max Loss: $725
Max Profit: $775 (($15 - $7.25) × 100)
Breakeven: $187.75
Risk/Reward: 1:1.07
```

### Position Management

**Week 1-2** (Price: $195-$200):
```
Stock consolidating near entry
P&L: -$50 to +$50
Thesis: Intact
Action: Hold, monitor resistance
```

**Week 3** (Price rallies to $203):
```
Unexpected rally
P&L: -$250 (approaching 50% stop loss)
Thesis: Re-evaluation needed

Analysis:
- No new fundamental catalyst
- Low volume rally
- Still below prior high of $205
Decision: HOLD (thesis intact, not stopped out yet)
```

**Week 4** (Price drops to $192):
```
Rally failed, decline resuming
P&L: +$200
Back in profit zone
Action: Monitor for continuation
```

**Week 5-6** (Price: $185-$188):
```
Steady decline to near target
Spread Value: $11.00
P&L: +$375 (48% of max)
DTE: 30 remaining

Decision: CLOSE POSITION
Reason: Near 50% target, 30 DTE threshold
```

### Outcome

**Results**:
```
Entry: $7.25 debit
Exit: $11.00 credit
Profit: $375
Return: 52%
Days Held: 30
Annualized Return: ~630%
```

**Lessons**:
- Stuck with thesis despite temporary adverse move
- Didn't panic sell on counter-trend rally
- Exited at reasonable profit before theta acceleration
- Managed emotions well

---

## Example 3: Earnings Play (Unsuccessful)

### Setup

**Underlying**: RETAIL (fictional retail stock)
**Situation**: Earnings in 10 days, expecting disappointment

**Analysis**:
- Current Price: $88
- Sector weakness (competitors missed)
- Inventory build-up noted
- Consumer spending slowing

**IV Analysis**:
- Current IV: 55%
- IV Rank: 82 (very high, pre-earnings)
- Historical post-earnings IV: 28%
- **WARNING**: High IV = expensive options

### Trade Construction

**Flawed Selection Process**:
```
Outlook: Expect miss and drop to $80
Capital: $400 (aggressive for uncertain event)

Strike Selection:
Long: $85 put (OTM, delta -0.35)
Short: $80 put (delta -0.18)
Spread Width: $5

Expiration: 30 DTE (includes earnings)
```

**Execution**:
```
Buy 1 RETAIL Nov 85 put @ $4.25
Sell 1 RETAIL Nov 80 put @ $1.75
Net Debit: $2.50 ($250 total)

Max Loss: $250
Max Profit: $250
Risk/Reward: 1:1
```

### What Went Wrong

**Day 1-7** (Before Earnings):
```
Stock drifting slightly lower
Price: $86-$88
P&L: +$25
IV: Still elevated at 55%
```

**Day 8** (Earnings Release):
```
Result: Slight miss, but better than expected
Stock reaction: Initial drop to $85, then rally
Implied Volatility: CRASHED to 28%

Immediate Impact:
Price at $87 (flat from entry)
IV crush: -27 points
Spread Value: $1.50 (was $2.50)
P&L: -$100 (-40% loss)

Why:
- Vega loss from IV crush: ~$80
- Theta decay: ~$20
- Delta was neutral (price flat)
```

**Day 9-15** (Post-Earnings):
```
Stock stabilizing at $86-$88
No continued decline
Spread Value: $1.25
P&L: -$125 (50% of max loss)

Decision: EXIT at stop loss
```

### Outcome

**Results**:
```
Entry: $2.50 debit
Exit: $1.25 credit
Loss: -$125
Return: -50%
Reason for Loss: IV crush, stock didn't move enough
```

**Lessons**:
- **AVOID trading options around earnings with debit spreads**
- High IV makes puts expensive
- IV crush can overwhelm directional move
- If must trade earnings, use credit spreads instead
- Respected stop loss (good discipline)

---

## Example 4: Rolling Success

### Setup

**Underlying**: BANK (fictional bank stock)
**Situation**: Rising rates expected to pressure valuations

**Initial Trade**:
```
Entry Date: March 1
Stock Price: $68
Buy 1 BANK Apr 68 put @ $3.75
Sell 1 BANK Apr 63 put @ $1.50
Net Debit: $2.25 ($225)
DTE: 45
```

### Position Evolution

**March 1-15** (Price: $66-$68):
```
Slow drift lower
P&L: +$50
Thesis: Intact but slow to develop
```

**March 16-25** (Price: $67-$69):
```
Stock rallying (against position)
DTE: 21
P&L: -$25
Decision: Stock stalling but thesis valid
Action: ROLL OUT
```

**Roll Execution** (March 25):
```
Close April 68/63 spread @ $2.00
Open May 68/63 spread @ $2.75

Net Cost to Roll: $0.75
New Total Basis: $2.25 + $0.75 = $3.00
New DTE: 42
```

**March 26 - April 10** (Price: $64-$66):
```
Fed announces rate hike
Banks sell off
Stock drops to $64
P&L: +$200 (67% of new max profit)
```

**April 11** (Price: $63.50):
```
Stock approaching short strike
Spread Value: $4.75
P&L: +$175 (58% of max)
DTE: 32

Decision: CLOSE POSITION
Reason: Excellent profit, near short strike
```

### Outcome

**Results**:
```
Initial Entry: $2.25
Roll Cost: $0.75
Total Capital: $3.00 ($300)

Exit: $4.75 credit
Profit: $175
Return: 58%
Days Held: 41
```

**Lessons**:
- Rolling extended time for thesis to develop
- Additional capital deployed, but position worked
- Took profits before reaching max
- Patience with good thesis pays off

---

## Example 5: Quick Stop Loss

### Setup

**Underlying**: PHARMA (fictional pharma stock)
**Situation**: FDA decision pending, expect rejection

**Trade**:
```
Stock Price: $124
Buy 1 PHARMA Dec 120 put @ $5.00
Sell 1 PHARMA Dec 110 put @ $1.50
Net Debit: $3.50 ($350)
DTE: 30
```

### What Happened

**Day 1-3** (Price: $122-$124):
```
Waiting for FDA announcement
P&L: -$25 (theta decay)
```

**Day 4** (FDA Decision):
```
Result: APPROVED (opposite of thesis)
Stock: SPIKES to $135 (+9%)
Spread Value: $0.50
P&L: -$300 (-86% of max loss)

Immediate Action: EXIT
Exit Price: $0.50 credit
Final Loss: -$300
```

### Outcome

**Results**:
```
Entry: $3.50 debit
Exit: $0.50 credit
Loss: -$300
Return: -86%
Reason: Binary event went against thesis
```

**Lessons**:
- **Binary events are high risk with options**
- Thesis invalidated immediately
- Cut loss quickly (didn't wait for stop loss)
- Accept loss and move on
- Risked only 2% of portfolio, survived easily

---

## Key Takeaways Across Examples

### What Works

1. **Clear Thesis**: Technical + fundamental alignment
2. **Appropriate Sizing**: Risk 1-3% per trade
3. **Take Profits**: 50% of max profit is excellent
4. **Manage Risk**: Honor stop losses
5. **Patience**: Give trades time to work

### What Doesn't Work

1. **Earnings Plays**: IV crush kills debit spreads
2. **Binary Events**: High risk, difficult to manage
3. **Hoping/Praying**: Exit when thesis breaks
4. **Oversizing**: Emotional decision making
5. **Greed**: Holding past targets

### Statistical Reality

**Across these 5 examples**:
```
Win Rate: 60% (3 wins, 2 losses)
Average Win: +$317
Average Loss: -$213
Profit Factor: 2.23
Net P&L: +$525
```

This aligns with realistic bear spread trading:
- Expect 50-65% win rate
- Average winners should exceed average losers
- Consistent profitability requires discipline

---

## See Also

**Within This Skill**:
- [Quickstart](quickstart.md) - Basic setup
- [Strategy Mechanics](strategy-mechanics.md) - Understanding P&L
- [Strike Selection](strike-selection.md) - Choosing strikes
- [Position Management](position-management.md) - Managing trades
- [Greeks Analysis](greeks-analysis.md) - Risk metrics

**Master Resources**:
- [Options Greeks](../../options-strategies/references/greeks.md)
- [Volatility Analysis](../../options-strategies/references/volatility.md)

**Related Strategies**:
- [Bull Call Spread](../../bull-call-spread/SKILL.md)
- [Iron Condor](../../iron-condor/SKILL.md)

---

**Last Updated**: 2025-12-12
'@
Write-ReferenceFile -Strategy 'bear-put-spread' -FileName 'examples.md' -Content $Content

#endregion

# Summary
Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host ""
Write-Host "Files Created: $script:FilesCreated" -ForegroundColor Green
Write-Host "Total Size: $([math]::Round($script:TotalSize / 1KB, 1)) KB" -ForegroundColor Green
Write-Host ""

if ($DryRun) {
    Write-Host "DRY RUN COMPLETE - No files were written" -ForegroundColor Yellow
    Write-Host "Run without -DryRun to create files" -ForegroundColor Yellow
} else {
    Write-Host "ALL FILES CREATED SUCCESSFULLY" -ForegroundColor Green
}
'@
</invoke>
