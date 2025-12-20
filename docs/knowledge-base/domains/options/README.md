# Options & Derivatives - Knowledge Base

## Purpose

This section provides options knowledge required for **automated options strategies** including strike/expiry selection rules, Greeks management, and defined-risk strategy specifications.

---

## 1. Options Fundamentals

### 1.1 Core Definitions

| Term | Definition |
|------|------------|
| **Call** | Right to buy underlying at strike price |
| **Put** | Right to sell underlying at strike price |
| **Strike** | Price at which option can be exercised |
| **Expiration** | Date option contract expires |
| **Premium** | Price paid for option contract |
| **Exercise** | Converting option to underlying position |
| **Assignment** | Obligation to fulfill option (seller) |

---

### 1.2 Moneyness

```python
# Moneyness calculation
def moneyness(option_type: str, strike: float, underlying: float) -> str:
    if option_type == 'CALL':
        if strike < underlying:
            return 'ITM'  # In The Money
        elif strike > underlying:
            return 'OTM'  # Out of The Money
        else:
            return 'ATM'  # At The Money
    else:  # PUT
        if strike > underlying:
            return 'ITM'
        elif strike < underlying:
            return 'OTM'
        else:
            return 'ATM'

# Moneyness percentage
ITM_PCT = (underlying - strike) / underlying  # For calls
OTM_PCT = (strike - underlying) / underlying  # For calls
```

**Rule Templates**:
```python
# Strike selection by moneyness
ATM_STRIKE = round(underlying / strike_increment) * strike_increment
ITM_1_STRIKE = ATM_STRIKE - strike_increment  # 1 strike ITM (call)
OTM_1_STRIKE = ATM_STRIKE + strike_increment  # 1 strike OTM (call)

# Delta-based strike selection
DEEP_ITM = delta > 0.80
ITM = 0.55 < delta <= 0.80
ATM = 0.45 <= delta <= 0.55
OTM = 0.20 <= delta < 0.45
DEEP_OTM = delta < 0.20
```

---

### 1.3 Option Pricing Basics

**Intrinsic Value**:
```python
# Call intrinsic
CALL_INTRINSIC = max(0, underlying - strike)

# Put intrinsic
PUT_INTRINSIC = max(0, strike - underlying)
```

**Extrinsic Value (Time Value)**:
```python
EXTRINSIC = premium - intrinsic

# Extrinsic is highest for ATM options
# Extrinsic decays as expiration approaches (theta)
```

**Black-Scholes Inputs** (conceptual):
- Underlying price
- Strike price
- Time to expiration
- Risk-free rate
- Implied volatility

---

## 2. Implied Volatility

### 2.1 IV Metrics

| Metric | Definition | Use |
|--------|------------|-----|
| **IV** | Market's expected volatility | Option pricing |
| **IV Rank** | Current IV percentile over 1 year | Relative IV level |
| **IV Percentile** | % of days with lower IV | Distribution position |
| **IV Skew** | IV difference across strikes | Market sentiment |
| **Term Structure** | IV across expirations | Time expectations |

**Rule Templates**:
```python
# IV Rank calculation
IV_RANK = (current_iv - iv_52wk_low) / (iv_52wk_high - iv_52wk_low)

# IV Percentile
IV_PERCENTILE = count(historical_iv < current_iv) / total_days

# IV thresholds
HIGH_IV = IV_RANK > 0.50
LOW_IV = IV_RANK < 0.30
ELEVATED_IV = IV_RANK > 0.70
DEPRESSED_IV = IV_RANK < 0.20
```

---

### 2.2 IV Trading Rules

```python
# Premium selling environment
SELL_PREMIUM_FAVORABLE = (
    IV_RANK > 0.50 AND
    IV > HV_20 * 1.10  # IV premium to realized
)

# Premium buying environment
BUY_PREMIUM_FAVORABLE = (
    IV_RANK < 0.30 AND
    IV < HV_20 * 0.90  # IV discount to realized
)

# IV crush expectation (post-earnings)
EXPECT_IV_CRUSH = days_to_earnings < 5 AND IV_elevated

# Strategy selection based on IV
IF HIGH_IV:
    preferred_strategies = ['credit_spreads', 'iron_condors', 'short_strangles']
IF LOW_IV:
    preferred_strategies = ['debit_spreads', 'long_straddles', 'calendar_spreads']
```

---

### 2.3 Volatility Skew

```python
# Put skew (typical for equities)
PUT_SKEW = IV_25delta_put - IV_25delta_call
NORMAL_SKEW = PUT_SKEW > 0  # Puts more expensive

# Skew analysis
ELEVATED_PUT_SKEW = PUT_SKEW > historical_avg_skew * 1.5  # Fear elevated
FLAT_SKEW = abs(PUT_SKEW) < threshold  # Complacency

# Term structure
BACKWARDATION = front_month_iv > back_month_iv  # Near-term fear
CONTANGO = back_month_iv > front_month_iv  # Normal structure
```

---

## 3. The Greeks

### 3.1 Delta

**Definition**: Rate of change of option price with respect to underlying price.

| Delta | Interpretation |
|-------|----------------|
| +1.0 | Moves 1:1 with stock (deep ITM call, 100 shares) |
| +0.5 | ATM call, 50% chance of expiring ITM |
| +0.2 | OTM call, 20% chance of expiring ITM |
| -0.5 | ATM put |
| -1.0 | Deep ITM put |

**Rule Templates**:
```python
# Position delta
POSITION_DELTA = option_delta * contracts * multiplier

# Portfolio delta
PORTFOLIO_DELTA = sum(position_delta for all positions)

# Delta limits
MAX_POSITION_DELTA = equity * 0.02  # 2% of equity as delta exposure
MAX_PORTFOLIO_DELTA = equity * 0.10  # 10% total delta exposure

# Delta-neutral target
DELTA_NEUTRAL = abs(PORTFOLIO_DELTA) < threshold
NEED_HEDGE = abs(PORTFOLIO_DELTA) > MAX_PORTFOLIO_DELTA
```

---

### 3.2 Gamma

**Definition**: Rate of change of delta with respect to underlying price.

```python
# Gamma characteristics
# - Highest for ATM options near expiration
# - Creates convexity (good for buyers, bad for sellers)

# Gamma risk
HIGH_GAMMA = near_expiration AND near_ATM
GAMMA_RISK = short_gamma AND HIGH_GAMMA  # Dangerous

# Gamma thresholds
MAX_SHORT_GAMMA = portfolio_threshold  # Limit gamma exposure

# Position management
IF HIGH_GAMMA AND SHORT_POSITION:
    consider_rolling_out()  # More time = less gamma
```

---

### 3.3 Theta

**Definition**: Rate of time decay (daily loss in option value).

```python
# Theta characteristics
# - Options lose value each day (theta decay)
# - Decay accelerates near expiration
# - ATM options have highest theta

# Theta as income
DAILY_THETA_INCOME = abs(position_theta)  # For short options

# Theta acceleration
THETA_ACCELERATES = days_to_expiration < 21  # Last 3 weeks

# Strategy implications
IF SELLING_PREMIUM:
    target_dte = 30 to 45 days  # Balance theta vs gamma
IF BUYING_PREMIUM:
    target_dte = 45+ days  # Reduce theta decay impact
```

---

### 3.4 Vega

**Definition**: Sensitivity to implied volatility changes.

```python
# Vega characteristics
# - Long options = long vega (benefit from IV increase)
# - Short options = short vega (benefit from IV decrease)
# - ATM options have highest vega

# Vega position
PORTFOLIO_VEGA = sum(position_vega for all positions)

# Vega exposure limits
MAX_VEGA_EXPOSURE = equity * vega_limit_pct

# Strategy selection
IF EXPECTING_IV_INCREASE:
    prefer_long_vega_strategies()  # Straddles, strangles
IF EXPECTING_IV_DECREASE:
    prefer_short_vega_strategies()  # Credit spreads, iron condors
```

---

## 4. Strategy Archetypes

### 4.1 Directional Strategies

**Long Call**:
```python
LONG_CALL = {
    'structure': 'buy_call',
    'outlook': 'bullish',
    'max_loss': premium_paid,
    'max_gain': unlimited,
    'breakeven': strike + premium,
    'greeks': {
        'delta': positive,
        'gamma': positive,
        'theta': negative,
        'vega': positive
    },
    'ideal_conditions': 'low_iv, expecting_move_up'
}
```

**Long Put**:
```python
LONG_PUT = {
    'structure': 'buy_put',
    'outlook': 'bearish',
    'max_loss': premium_paid,
    'max_gain': strike - premium (stock to zero),
    'breakeven': strike - premium,
    'ideal_conditions': 'low_iv, expecting_move_down'
}
```

---

### 4.2 Credit Spreads (Defined Risk)

**Bull Put Spread (Credit)**:
```python
BULL_PUT_SPREAD = {
    'structure': [
        {'action': 'sell', 'type': 'put', 'strike': 'higher'},
        {'action': 'buy', 'type': 'put', 'strike': 'lower'}
    ],
    'outlook': 'neutral_to_bullish',
    'max_gain': net_credit,
    'max_loss': (strike_width - net_credit) * multiplier,
    'breakeven': short_strike - net_credit,
    'probability_profit': delta_of_short_strike,  # Approximate
    'selection_rules': {
        'short_strike_delta': 0.25 to 0.35,  # 65-75% POP
        'width': 1 to 5 dollars,
        'dte': 30 to 45 days,
        'iv_rank': > 0.30
    }
}
```

**Bear Call Spread (Credit)**:
```python
BEAR_CALL_SPREAD = {
    'structure': [
        {'action': 'sell', 'type': 'call', 'strike': 'lower'},
        {'action': 'buy', 'type': 'call', 'strike': 'higher'}
    ],
    'outlook': 'neutral_to_bearish',
    'max_gain': net_credit,
    'max_loss': (strike_width - net_credit) * multiplier,
    'breakeven': short_strike + net_credit,
    'selection_rules': {
        'short_strike_delta': 0.25 to 0.35,
        'width': 1 to 5 dollars,
        'dte': 30 to 45 days
    }
}
```

---

### 4.3 Iron Condor

```python
IRON_CONDOR = {
    'structure': [
        # Put spread (lower)
        {'action': 'buy', 'type': 'put', 'strike': 'lowest'},
        {'action': 'sell', 'type': 'put', 'strike': 'lower_middle'},
        # Call spread (upper)
        {'action': 'sell', 'type': 'call', 'strike': 'upper_middle'},
        {'action': 'buy', 'type': 'call', 'strike': 'highest'}
    ],
    'outlook': 'neutral, range-bound',
    'max_gain': total_credit,
    'max_loss': max(put_spread_width, call_spread_width) - credit,
    'breakevens': [
        short_put - credit,
        short_call + credit
    ],
    'selection_rules': {
        'short_strike_delta': 0.15 to 0.20,  # ~70-85% POP
        'width': symmetric or asymmetric,
        'dte': 30 to 45 days,
        'iv_rank': > 0.40,  # Higher IV preferred
        'underlying': low_trend_strength  # ADX < 25
    },
    'management': {
        'profit_target': 50% of max profit,
        'loss_limit': 100-200% of credit received,
        'roll_trigger': tested_strike (price near short strike)
    }
}
```

---

### 4.4 Straddle/Strangle

**Long Straddle**:
```python
LONG_STRADDLE = {
    'structure': [
        {'action': 'buy', 'type': 'call', 'strike': 'ATM'},
        {'action': 'buy', 'type': 'put', 'strike': 'ATM'}
    ],
    'outlook': 'expecting_big_move, direction_unknown',
    'max_loss': total_premium_paid,
    'max_gain': unlimited,
    'breakevens': [
        strike - total_premium,
        strike + total_premium
    ],
    'ideal_conditions': {
        'iv_rank': < 0.30,
        'expected_move': > implied_move,
        'catalyst': earnings, events
    }
}
```

**Short Strangle**:
```python
SHORT_STRANGLE = {
    'structure': [
        {'action': 'sell', 'type': 'call', 'strike': 'OTM_call'},
        {'action': 'sell', 'type': 'put', 'strike': 'OTM_put'}
    ],
    'outlook': 'neutral, low_volatility',
    'max_gain': total_premium,
    'max_loss': unlimited,  # DANGEROUS without hedge
    'selection_rules': {
        'strike_delta': 0.15 to 0.20 each side,
        'dte': 30 to 45 days,
        'iv_rank': > 0.50,
        'margin_requirement': significant
    },
    'risk_note': 'undefined_risk - use with caution'
}
```

---

### 4.5 Covered Strategies

**Covered Call**:
```python
COVERED_CALL = {
    'structure': [
        {'action': 'hold', 'type': 'shares', 'quantity': 100},
        {'action': 'sell', 'type': 'call', 'quantity': 1}
    ],
    'outlook': 'neutral_to_slightly_bullish',
    'max_gain': (strike - stock_cost) + premium,
    'max_loss': stock_cost - premium,  # Stock to zero
    'breakeven': stock_cost - premium,
    'selection_rules': {
        'strike': 'ATM to 1-2 strikes OTM',
        'delta': 0.30 to 0.40,
        'dte': 30 to 45 days,
        'when': 'IV elevated, willing to sell at strike'
    },
    'management': {
        'if_assigned': 'sell shares at strike - acceptable',
        'if_expires_worthless': 'keep premium, repeat',
        'roll_trigger': 'stock approaches strike with time left'
    }
}
```

**Cash-Secured Put**:
```python
CASH_SECURED_PUT = {
    'structure': [
        {'action': 'hold', 'type': 'cash', 'amount': strike * 100},
        {'action': 'sell', 'type': 'put', 'quantity': 1}
    ],
    'outlook': 'neutral_to_bullish, willing_to_own',
    'max_gain': premium,
    'max_loss': (strike - premium) * 100,  # If stock to zero
    'breakeven': strike - premium,
    'selection_rules': {
        'strike': 'ATM to 1-2 strikes OTM',
        'delta': 0.25 to 0.40,
        'dte': 30 to 45 days,
        'stock': 'willing to own at strike'
    }
}
```

---

## 5. Strike & Expiration Selection

### 5.1 Strike Selection Rules

```python
def select_strike(strategy: str, underlying: float, iv_rank: float) -> Strike:
    """
    Rule-based strike selection.
    """
    if strategy == 'credit_spread':
        # Select based on probability (delta)
        target_delta = 0.30 if iv_rank > 0.50 else 0.25
        return find_strike_by_delta(target_delta)

    elif strategy == 'debit_spread':
        # More aggressive, closer to money
        target_delta = 0.45 to 0.50
        return find_strike_by_delta(target_delta)

    elif strategy == 'iron_condor':
        # Wide, high probability
        target_delta = 0.16 if iv_rank > 0.50 else 0.20
        return find_strike_by_delta(target_delta)

    elif strategy == 'covered_call':
        # Willing to sell, some upside
        target_delta = 0.30 to 0.35
        return find_strike_by_delta(target_delta)
```

---

### 5.2 Expiration Selection Rules

```python
def select_expiration(strategy: str, iv_rank: float) -> int:
    """
    Return target days to expiration.
    """
    if strategy in ['credit_spread', 'iron_condor']:
        # Balance theta decay vs gamma risk
        if iv_rank > 0.50:
            return 45  # More premium available
        else:
            return 30  # Less time for things to go wrong

    elif strategy in ['debit_spread', 'long_call', 'long_put']:
        # More time, less theta decay
        return 60 to 90

    elif strategy == 'covered_call':
        # Monthly cycle
        return 30 to 45

    elif strategy == 'earnings_play':
        # Capture the event
        return first_expiry_after_earnings
```

---

### 5.3 Width Selection (Spreads)

```python
def select_spread_width(strategy: str, account_size: float, risk_per_trade: float) -> float:
    """
    Determine spread width based on risk parameters.
    """
    # Max loss per spread
    max_loss_target = account_size * risk_per_trade

    # For credit spreads: max_loss = width - credit
    # So: width = max_loss + credit (approximately)

    if strategy == 'credit_spread':
        # Common widths: $1, $2, $2.50, $5
        suggested_width = round(max_loss_target / 100 / contracts)
        return min(suggested_width, 5)  # Cap at $5 wide

    elif strategy == 'iron_condor':
        # Often symmetric, $2-5 wings
        return 2 to 5
```

---

## 6. Position Sizing for Options

### 6.1 Max Loss Based Sizing

```python
def size_option_position(
    max_risk_pct: float,
    account_equity: float,
    strategy: dict
) -> int:
    """
    Size position based on maximum loss.
    """
    max_risk_dollars = account_equity * max_risk_pct

    if strategy['type'] == 'credit_spread':
        max_loss_per_contract = (strategy['width'] - strategy['credit']) * 100
        contracts = int(max_risk_dollars / max_loss_per_contract)

    elif strategy['type'] == 'long_option':
        max_loss_per_contract = strategy['premium'] * 100
        contracts = int(max_risk_dollars / max_loss_per_contract)

    elif strategy['type'] == 'iron_condor':
        max_loss_per_contract = strategy['max_width'] * 100 - strategy['credit'] * 100
        contracts = int(max_risk_dollars / max_loss_per_contract)

    return max(1, contracts)  # At least 1 contract
```

---

### 6.2 Delta-Based Sizing

```python
def size_by_delta(
    max_delta_exposure: float,
    account_equity: float,
    option_delta: float
) -> int:
    """
    Size based on delta exposure limits.
    """
    delta_limit_dollars = account_equity * max_delta_exposure
    delta_per_contract = abs(option_delta) * 100  # Per 100 shares

    contracts = int(delta_limit_dollars / delta_per_contract)
    return max(1, contracts)
```

---

## 7. Options Management Rules

### 7.1 Profit Taking

```python
# Credit spreads / Iron condors
IF profit_pct >= 0.50:  # 50% of max profit
    action = "close_position"
    reason = "profit_target_reached"

IF profit_pct >= 0.75 AND dte > 14:
    action = "close_position"
    reason = "high_profit_with_time_remaining"

# Long options
IF profit_pct >= 1.00:  # 100% gain
    action = "close_or_trail_stop"
```

---

### 7.2 Loss Management

```python
# Credit spreads
IF loss_pct >= 1.00 to 2.00:  # 100-200% of credit
    action = "close_position"
    reason = "loss_limit_reached"

IF price_breaches_short_strike:
    action = "evaluate_roll_or_close"

# Long options
IF loss_pct >= 0.50:  # 50% of premium
    action = "close_position"
    reason = "stop_loss"
```

---

### 7.3 Rolling Strategies

```python
def should_roll(position: OptionsPosition) -> RollDecision:
    """
    Determine if position should be rolled.
    """
    # Time-based roll
    if position.dte < 7 and position.profit_pct < 0.50:
        return RollDecision(
            action='roll_out',
            new_dte=30,
            reason='avoid_gamma_risk'
        )

    # Tested roll (price near short strike)
    if position.is_tested():
        if position.has_profit():
            return RollDecision(action='close', reason='take_profit')
        else:
            return RollDecision(
                action='roll_out_and_up' if call_tested else 'roll_out_and_down',
                reason='defend_position'
            )

    return RollDecision(action='hold')
```

---

## 8. PDT Considerations (Under $25K)

```python
# Pattern Day Trader rule constraints
IF account_value < 25000 AND margin_account:
    day_trades_available = 3 per 5 rolling days

# Options implications
OPTIONS_PDT_RULES = {
    'spread_open_close_same_day': counts_as_day_trade,
    'single_leg_open_close': counts_as_day_trade,
    'workaround': 'open_today_close_tomorrow'
}

# Strategies for sub-25K accounts
SUB_25K_PREFERRED = [
    'multi_day_swings',  # Open and hold overnight
    'spreads_held_overnight',
    'cash_account'  # No PDT but T+1 settlement on options
]
```

---

## Academic References

1. **Hull, J.C.**: "Options, Futures, and Other Derivatives" - Definitive textbook
2. **Natenberg, S.**: "Option Volatility and Pricing" - Practical volatility focus
3. **Sinclair, E.**: "Volatility Trading" - Trading volatility systematically
4. **Taleb, N.N.**: "Dynamic Hedging" - Advanced risk management
5. **CBOE Education**: Options Institute materials
6. **Black, F. & Scholes, M. (1973)**: Original options pricing model

---

## Key Takeaways

1. **Defined risk preferred**: Spreads over naked positions
2. **IV matters**: Sell high IV, buy low IV
3. **Position size by max loss**: Never risk more than acceptable
4. **Manage actively**: Take profits, cut losses, roll when necessary
5. **Greeks awareness**: Understand exposure across all dimensions
6. **Time is money**: Theta works for sellers, against buyers
7. **Probability focus**: Higher probability = smaller profits (trade-off)
