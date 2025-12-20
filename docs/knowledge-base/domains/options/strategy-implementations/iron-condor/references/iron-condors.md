# Iron Condor Strategy Implementation

**Section**: 06_options/strategy_implementations
**Last Updated**: 2025-12-12
**Source Skill**: [iron-condor](../../../../.claude/skills/iron-condor/SKILL.md)

---

## Overview

An iron condor is a neutral credit spread strategy that profits from range-bound price action. By selling an OTM put spread and an OTM call spread simultaneously, the trader collects premium upfront and profits when the underlying remains between the short strikes through expiration.

**Strategy Type**: Credit spread, defined risk, neutral outlook
**Market Condition**: Range-bound, elevated IV
**Complexity**: Intermediate

---

## Position Structure

### Components

| Leg | Action | Strike | Purpose |
|-----|--------|--------|---------|
| Long Put | Buy | Lowest | Downside protection |
| Short Put | Sell | Lower-middle | Collect credit (put spread) |
| Short Call | Sell | Upper-middle | Collect credit (call spread) |
| Long Call | Buy | Highest | Upside protection |

**All legs share the same expiration date.**

```python
# Iron condor structure
iron_condor = {
    'long_put': {'action': 'buy', 'strike': lowest},
    'short_put': {'action': 'sell', 'strike': lower_middle},
    'short_call': {'action': 'sell', 'strike': upper_middle},
    'long_call': {'action': 'buy', 'strike': highest}
}

# Wing width = distance between strikes on each side
put_wing_width = short_put - long_put
call_wing_width = long_call - short_call
```

---

## Risk/Reward Profile

### Key Metrics

```python
# Net credit received
net_credit = put_spread_credit + call_spread_credit

# Maximum profit (stock stays between short strikes)
max_profit = net_credit * 100 * contracts

# Maximum loss (stock breaches either wing)
max_loss = (max(put_wing_width, call_wing_width) - net_credit) * 100 * contracts

# Breakeven points
lower_breakeven = short_put - net_credit
upper_breakeven = short_call + net_credit

# Profit zone width
profit_zone = upper_breakeven - lower_breakeven

# Risk/reward ratio
risk_reward_ratio = max_loss / max_profit
```

### Example Calculation

```python
# SPY at $450, expect to stay $440-$460
position = {
    'underlying': 450.00,
    'short_put': 440.00,
    'long_put': 435.00,      # $5 wide
    'short_call': 460.00,
    'long_call': 465.00,     # $5 wide
    'put_credit': 0.75,
    'call_credit': 0.80,
    'contracts': 10
}

net_credit = 0.75 + 0.80  # $1.55
max_profit = 1.55 * 100 * 10  # $1,550
max_loss = (5.00 - 1.55) * 100 * 10  # $3,450
lower_be = 440.00 - 1.55  # $438.45
upper_be = 460.00 + 1.55  # $461.55
profit_zone = 461.55 - 438.45  # 23.1 points (5.1% range)
```

---

## Strike Selection Framework

### Delta-Based Approach

| Profile | Short Delta | Probability of Profit | Risk/Reward |
|---------|-------------|----------------------|-------------|
| Conservative | 0.10 | ~80% | Higher ratio |
| Standard | 0.20 | ~70% | Balanced |
| Aggressive | 0.30 | ~60% | Lower ratio |

```python
def select_condor_strikes(
    underlying_price: float,
    target_delta: float,
    wing_width: float,
    option_chain: pd.DataFrame
) -> dict:
    """
    Select iron condor strikes using delta-based framework.

    Args:
        underlying_price: Current stock price
        target_delta: Target delta for short strikes (0.10-0.30)
        wing_width: Width of each spread
        option_chain: Available options with Greeks

    Returns:
        Dictionary with all four strikes
    """
    # Find short put at negative target delta
    short_put = option_chain[
        (option_chain['type'] == 'put') &
        (option_chain['delta'].abs() - target_delta).abs().idxmin()
    ]['strike']

    # Find short call at positive target delta
    short_call = option_chain[
        (option_chain['type'] == 'call') &
        (option_chain['delta'] - target_delta).abs().idxmin()
    ]['strike']

    return {
        'long_put': short_put - wing_width,
        'short_put': short_put,
        'short_call': short_call,
        'long_call': short_call + wing_width
    }
```

### Technical Level Alignment

Align strikes with support/resistance:

```python
def align_with_technicals(
    base_strikes: dict,
    support_levels: list,
    resistance_levels: list
) -> dict:
    """Adjust strikes to align with technical levels."""
    adjusted = base_strikes.copy()

    # Place short put above nearest support
    for support in sorted(support_levels, reverse=True):
        if support < base_strikes['short_put']:
            adjusted['short_put'] = support + 1  # Just above support
            adjusted['long_put'] = adjusted['short_put'] - wing_width
            break

    # Place short call below nearest resistance
    for resistance in sorted(resistance_levels):
        if resistance > base_strikes['short_call']:
            adjusted['short_call'] = resistance - 1  # Just below resistance
            adjusted['long_call'] = adjusted['short_call'] + wing_width
            break

    return adjusted
```

---

## Wing Width Selection

### Tradeoff Analysis

| Width | Max Risk | Credit | R:R Ratio | Best For |
|-------|----------|--------|-----------|----------|
| Narrow ($2.50-$5) | $250-500 | Lower | Better | Small accounts |
| Standard ($5-$10) | $500-1000 | Moderate | Balanced | Most traders |
| Wide ($10-$20) | $1000-2000 | Higher | Worse | Premium collection |

```python
def optimize_wing_width(
    account_size: float,
    risk_per_trade: float,
    available_widths: list = [2.50, 5.00, 7.50, 10.00]
) -> float:
    """
    Select optimal wing width based on risk tolerance.
    """
    max_risk_dollars = account_size * risk_per_trade

    # Find largest width within risk tolerance
    # Assume ~30% credit collection (conservative)
    for width in sorted(available_widths, reverse=True):
        estimated_risk = width * 100 * 0.70  # 70% of width
        if estimated_risk <= max_risk_dollars:
            return width

    return available_widths[0]  # Smallest width
```

---

## Greeks Profile

### Net Greeks Characteristics

| Greek | Sign | Interpretation |
|-------|------|----------------|
| Delta | ~0 | Directionally neutral |
| Gamma | Negative | Accelerates against outside range |
| Theta | Positive | Earns time decay daily |
| Vega | Negative | Benefits from IV contraction |

```python
def calculate_condor_greeks(
    short_put: float,
    long_put: float,
    short_call: float,
    long_call: float,
    underlying: float,
    iv: float,
    dte: int
) -> dict:
    """
    Calculate net Greeks for iron condor position.
    """
    # Individual leg Greeks (simplified Black-Scholes)
    # In practice, use proper options pricing library

    greeks = {
        'delta': 0.0,   # Near zero (neutral)
        'gamma': 0.0,   # Negative (short gamma)
        'theta': 0.0,   # Positive (collecting decay)
        'vega': 0.0     # Negative (short vega)
    }

    # Short options contribute negative gamma, positive theta, negative vega
    # Long options provide protection but reduce net credit

    return greeks
```

### Greeks Management Rules

```python
# Delta alert
if abs(position_delta) > 0.30:
    action = "reassess_position"
    reason = "directional_bias_developing"

# Gamma caution
if dte < 10:
    action = "close_or_roll"
    reason = "gamma_risk_elevated"

# Theta optimization
if dte > 21:
    theta_collection = "steady"
elif dte > 7:
    theta_collection = "accelerating"
else:
    theta_collection = "maximum_but_risky"
```

---

## IV Analysis

### Optimal Entry Conditions

```python
# IV environment assessment
def assess_iv_for_condor(
    current_iv: float,
    iv_52wk_high: float,
    iv_52wk_low: float,
    hv_20: float
) -> dict:
    """
    Evaluate IV environment for iron condor entry.
    """
    iv_rank = (current_iv - iv_52wk_low) / (iv_52wk_high - iv_52wk_low)
    iv_premium = current_iv / hv_20

    return {
        'iv_rank': iv_rank,
        'iv_premium': iv_premium,
        'favorable': iv_rank > 0.50,
        'optimal': iv_rank > 0.50 and iv_premium > 1.10,
        'avoid': iv_rank < 0.30,
        'recommendation': (
            'ENTER' if iv_rank > 0.50 else
            'WAIT' if iv_rank > 0.30 else
            'AVOID'
        )
    }
```

### IV Thresholds

| IV Rank | Action | Rationale |
|---------|--------|-----------|
| > 70% | Strong entry | Premium rich |
| 50-70% | Good entry | Favorable |
| 30-50% | Marginal | Reduced premium |
| < 30% | Avoid | Insufficient credit |

---

## Expiration Selection

### DTE Guidelines

| DTE Range | Theta Decay | Gamma Risk | Use Case |
|-----------|-------------|------------|----------|
| 21-30 | Fast | Higher | Active management |
| 30-45 | Balanced | Moderate | Standard (recommended) |
| 45-60 | Slower | Lower | Less active traders |

```python
def select_expiration(
    iv_rank: float,
    management_style: str = 'standard'
) -> int:
    """
    Select optimal DTE for iron condor.
    """
    if management_style == 'active':
        return 21 if iv_rank > 0.60 else 30
    elif management_style == 'standard':
        return 45 if iv_rank > 0.50 else 30
    else:  # passive
        return 60
```

---

## Position Sizing

```python
def size_iron_condor(
    account_value: float,
    risk_per_trade: float,
    wing_width: float,
    net_credit: float
) -> int:
    """
    Calculate appropriate number of contracts.

    Args:
        account_value: Total account equity
        risk_per_trade: Max risk as decimal (e.g., 0.02 for 2%)
        wing_width: Spread width in dollars
        net_credit: Total credit received per contract

    Returns:
        Number of contracts to trade
    """
    max_risk_dollars = account_value * risk_per_trade
    max_loss_per_contract = (wing_width - net_credit) * 100

    contracts = int(max_risk_dollars / max_loss_per_contract)

    return max(1, contracts)

# Example
contracts = size_iron_condor(
    account_value=100000,
    risk_per_trade=0.02,    # 2%
    wing_width=5.00,
    net_credit=1.55
)
# Returns: 5 contracts (max risk $1,725)
```

---

## Trade Management

### Profit Targets

| Target | When to Use | Rationale |
|--------|-------------|-----------|
| 50% max profit | Standard | Optimal risk/reward |
| 75% max profit | High conviction | More reward, more risk |
| 21 DTE | Time-based | Avoid gamma acceleration |

```python
def check_profit_target(
    current_value: float,
    max_profit: float,
    dte: int,
    target_pct: float = 0.50
) -> dict:
    """
    Evaluate if profit target is reached.
    """
    profit_pct = 1 - (current_value / max_profit)

    if profit_pct >= target_pct:
        return {'action': 'CLOSE', 'reason': f'{target_pct*100}% profit target'}
    elif dte <= 21 and profit_pct >= 0.30:
        return {'action': 'CLOSE', 'reason': '21 DTE with 30%+ profit'}
    else:
        return {'action': 'HOLD', 'profit_pct': profit_pct}
```

### Loss Management

```python
def check_loss_limit(
    current_loss: float,
    net_credit: float,
    max_loss: float,
    multiplier: float = 2.0
) -> dict:
    """
    Evaluate loss limit breach.
    """
    loss_ratio = current_loss / (net_credit * 100)

    if loss_ratio >= multiplier:
        return {'action': 'CLOSE', 'reason': f'{multiplier}x credit loss'}
    elif current_loss >= max_loss * 0.75:
        return {'action': 'EVALUATE', 'reason': 'approaching max loss'}
    else:
        return {'action': 'HOLD', 'loss_ratio': loss_ratio}
```

### Adjustment Strategies

#### Roll Untested Side

When one side is threatened:

```python
def roll_untested_side(
    position: dict,
    current_price: float,
    threatened_side: str
) -> dict:
    """
    Roll the untested side closer to collect more credit.
    """
    if threatened_side == 'call':
        # Stock moving up, call side threatened
        # Roll put side up (closer to money)
        new_short_put = position['short_put'] + adjustment
        additional_credit = estimate_new_credit(new_short_put)
    else:
        # Stock moving down, put side threatened
        # Roll call side down
        new_short_call = position['short_call'] - adjustment
        additional_credit = estimate_new_credit(new_short_call)

    return {
        'action': 'roll_untested',
        'new_strikes': new_strikes,
        'additional_credit': additional_credit
    }
```

#### Convert to Iron Butterfly

Tighten the position when threatened:

```python
def convert_to_butterfly(position: dict, atm_strike: float) -> dict:
    """
    Convert iron condor to iron butterfly for credit.

    Sell ATM options on both sides.
    """
    return {
        'action': 'convert',
        'new_short_put': atm_strike,
        'new_short_call': atm_strike,
        'keep_wings': True,
        'rationale': 'collect_maximum_credit_at_atm'
    }
```

---

## Entry Checklist

```markdown
Pre-Trade:
- [ ] IV Rank > 50%
- [ ] No major catalysts within expiration
- [ ] Underlying in defined range (support/resistance)
- [ ] Liquidity adequate (tight bid/ask)
- [ ] Account has margin for position

Strike Selection:
- [ ] Short deltas: 0.15-0.25
- [ ] Wing widths match risk tolerance
- [ ] Strikes align with technical levels
- [ ] Credit provides acceptable R:R (target 3:1)

Position Sizing:
- [ ] Max loss < 2% of account
- [ ] Buying power impact acceptable
- [ ] Room for adjustment if needed

Order Entry:
- [ ] Enter as single iron condor order (4 legs)
- [ ] Limit order at mid-point or better
- [ ] Verify credit (not debit!)
```

---

## Exit Checklist

```markdown
Profit Exit:
- [ ] 50% max profit reached
- [ ] Or 21 DTE with 30%+ profit
- [ ] Close as single order (all 4 legs)

Loss Exit:
- [ ] Loss exceeds 2x credit received
- [ ] Or price breaches short strike
- [ ] Or thesis invalidated

Management:
- [ ] Close by Friday 3 PM expiration week
- [ ] Avoid pin risk near strikes
- [ ] Log trade in journal
```

---

## Risk Warnings

**Primary Risks**:
- Limited profit potential (capped at credit)
- Gamma acceleration near expiration
- Gap risk on earnings/events
- Assignment risk on short options
- Pin risk at expiration

**Mitigation**:
- Only trade in high IV (rank > 50)
- Close or roll by 21 DTE
- Avoid holding through binary events
- Size conservatively (1-3% risk)
- Have adjustment plan before entry

---

## Related Skills

- [iron-condor skill](../../../../.claude/skills/iron-condor/SKILL.md) - Interactive analysis
- [iron-butterfly skill](../../../../.claude/skills/iron-butterfly/SKILL.md) - ATM variant
- [options-strategies skill](../../../../.claude/skills/options-strategies/SKILL.md) - General framework

## Related KB Sections

- [Greeks Library](../README.md#3-the-greeks) - Detailed Greeks reference
- [IV Analysis](../README.md#2-implied-volatility) - Volatility metrics
- [Risk Management](../../03_risk/README.md) - Portfolio risk

---

## Academic References

1. Hull, J.C. (2022). "Options, Futures, and Other Derivatives" - Chapter 12
2. Natenberg, S. (2015). "Option Volatility and Pricing" - Spread strategies
3. CBOE Options Institute - Iron Condor education materials

---

**Template**: KB Skills Integration v1.0
**Lines**: ~500
