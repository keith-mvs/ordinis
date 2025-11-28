# Risk Management & Position Sizing - Knowledge Base

## Purpose

Risk management provides the **hard constraints** that protect capital and ensure survival. These rules are **non-negotiable** in an automated system and must be enforced at every level.

---

## 1. Core Risk Principles

### 1.1 Survival First

```python
CORE_PRINCIPLES = {
    'survival': 'Never risk catastrophic loss',
    'consistency': 'Apply rules without exception',
    'adaptability': 'Reduce risk in adverse conditions',
    'accountability': 'Log all decisions for review'
}

# The cardinal rule
NEVER_RISK_RUIN = True  # Non-negotiable
```

### 1.2 Risk Hierarchy

```
Level 1: Trade Risk      → Individual position limits
Level 2: Daily Risk      → Daily loss limits
Level 3: Portfolio Risk  → Total exposure limits
Level 4: Account Risk    → Drawdown circuit breakers
Level 5: System Risk     → Kill switches and safeguards
```

---

## 2. Per-Trade Risk Limits

### 2.1 Fixed Percentage Risk

**The 1% Rule** (Foundational):
```python
# Maximum risk per trade as percentage of equity
MAX_RISK_PER_TRADE_PCT = 0.01  # 1% of account equity

def calculate_max_risk(equity: float) -> float:
    return equity * MAX_RISK_PER_TRADE_PCT

# Example: $20,000 account
# Max risk per trade = $20,000 * 0.01 = $200
```

**Adjustable Risk Levels**:
```python
RISK_LEVELS = {
    'conservative': 0.005,  # 0.5% per trade
    'moderate': 0.01,       # 1% per trade
    'aggressive': 0.02,     # 2% per trade
    'maximum': 0.03         # 3% per trade (not recommended)
}

# Dynamic adjustment
def adjusted_risk_pct(base_risk: float, conditions: dict) -> float:
    risk = base_risk

    # Reduce in adverse conditions
    if conditions['drawdown'] > 0.10:
        risk *= 0.5  # Half size after 10% drawdown
    if conditions['losing_streak'] > 3:
        risk *= 0.75
    if conditions['volatility_elevated']:
        risk *= 0.75

    return max(risk, 0.0025)  # Floor at 0.25%
```

---

### 2.2 Fixed Dollar Risk

```python
# Alternative: Fixed dollar amount
MAX_RISK_DOLLARS = 200  # Fixed regardless of account size

# Useful for:
# - Small accounts (where 1% is too small)
# - Consistent bet sizing
# - Psychological comfort

def risk_per_trade(equity: float, method: str) -> float:
    if method == 'percentage':
        return equity * MAX_RISK_PER_TRADE_PCT
    elif method == 'fixed':
        return min(MAX_RISK_DOLLARS, equity * 0.02)  # Cap at 2%
```

---

## 3. Position Sizing Methods

### 3.1 Risk-Based Position Sizing

**Core Formula**:
```python
def calculate_position_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_price: float
) -> int:
    """
    Size position so stop loss equals acceptable risk.
    """
    risk_amount = equity * risk_pct
    risk_per_share = abs(entry_price - stop_price)

    if risk_per_share == 0:
        return 0  # Invalid stop

    shares = int(risk_amount / risk_per_share)
    return shares

# Example:
# Equity: $20,000, Risk: 1%, Entry: $50, Stop: $48
# Risk amount: $200, Risk per share: $2
# Position size: 100 shares
```

---

### 3.2 ATR-Based Position Sizing

```python
def atr_position_size(
    equity: float,
    risk_pct: float,
    atr: float,
    atr_multiplier: float = 2.0
) -> int:
    """
    Size based on volatility (ATR).
    Stop is set at ATR multiplier.
    """
    risk_amount = equity * risk_pct
    stop_distance = atr * atr_multiplier
    shares = int(risk_amount / stop_distance)
    return shares

# Benefits:
# - Volatility-adjusted (smaller size in volatile markets)
# - Consistent risk across different stocks
# - Adapts to market conditions
```

---

### 3.3 Kelly Criterion (Fractional)

```python
def kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly Criterion for optimal bet sizing.
    Use fractional Kelly (25-50%) for safety.
    """
    if avg_loss == 0:
        return 0

    win_loss_ratio = avg_win / avg_loss
    kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

    # Apply fractional Kelly
    KELLY_FRACTION = 0.25  # Use 25% of Kelly
    return max(0, kelly * KELLY_FRACTION)

# Warning: Full Kelly is too aggressive
# Always use fractional (10-50% of full Kelly)
```

---

### 3.4 Volatility-Adjusted Sizing

```python
def volatility_adjusted_size(
    base_size: int,
    current_volatility: float,
    normal_volatility: float
) -> int:
    """
    Adjust size inversely to volatility.
    """
    volatility_ratio = normal_volatility / current_volatility
    adjusted_size = int(base_size * volatility_ratio)

    # Caps
    min_size = base_size * 0.25  # At least 25% of base
    max_size = base_size * 2.0   # At most 200% of base

    return int(min(max(adjusted_size, min_size), max_size))
```

---

## 4. Stop Loss Methods

### 4.1 Fixed Percentage Stop

```python
# Simple percentage-based stop
STOP_LOSS_PCT = 0.02  # 2% below entry

def fixed_pct_stop(entry_price: float, direction: str) -> float:
    if direction == 'long':
        return entry_price * (1 - STOP_LOSS_PCT)
    else:  # short
        return entry_price * (1 + STOP_LOSS_PCT)
```

---

### 4.2 ATR-Based Stop

```python
def atr_stop(
    entry_price: float,
    atr: float,
    multiplier: float = 2.0,
    direction: str = 'long'
) -> float:
    """
    Stop based on volatility.
    """
    stop_distance = atr * multiplier

    if direction == 'long':
        return entry_price - stop_distance
    else:
        return entry_price + stop_distance

# Common multipliers:
# Tight: 1.5 ATR
# Normal: 2.0 ATR
# Wide: 3.0 ATR
```

---

### 4.3 Support/Resistance Stop

```python
def structure_stop(
    entry_price: float,
    support_level: float,
    buffer_pct: float = 0.005,
    direction: str = 'long'
) -> float:
    """
    Stop below support (long) or above resistance (short).
    """
    if direction == 'long':
        return support_level * (1 - buffer_pct)
    else:
        return support_level * (1 + buffer_pct)

# Validation
IF stop_distance > max_acceptable_risk:
    action = "reduce_position_size" OR "skip_trade"
```

---

### 4.4 Trailing Stop

```python
def trailing_stop(
    current_price: float,
    highest_since_entry: float,
    trail_pct: float = 0.05,
    direction: str = 'long'
) -> float:
    """
    Dynamic stop that follows price.
    """
    if direction == 'long':
        return highest_since_entry * (1 - trail_pct)
    else:
        return highest_since_entry * (1 + trail_pct)

# ATR-based trailing
def atr_trailing_stop(high_water: float, atr: float, mult: float = 2.5) -> float:
    return high_water - (atr * mult)
```

---

### 4.5 Time-Based Stop

```python
# Exit if trade doesn't perform in time window
def time_stop(
    entry_time: datetime,
    current_time: datetime,
    max_hold_days: int,
    profit_threshold: float
) -> bool:
    """
    Exit if time exceeded without hitting profit target.
    """
    days_held = (current_time - entry_time).days

    if days_held >= max_hold_days:
        return True  # Trigger exit

    return False
```

---

## 5. Portfolio-Level Limits

### 5.1 Maximum Open Positions

```python
# Diversification limits
MAX_OPEN_POSITIONS = 10  # Maximum concurrent positions

def can_open_new_position(current_positions: int) -> bool:
    return current_positions < MAX_OPEN_POSITIONS

# Scaling with account size
def max_positions_by_account(equity: float) -> int:
    if equity < 10000:
        return 3
    elif equity < 25000:
        return 5
    elif equity < 100000:
        return 10
    else:
        return 20
```

---

### 5.2 Sector Concentration Limits

```python
MAX_SECTOR_CONCENTRATION = 0.25  # 25% max in any sector

def sector_exposure(positions: List[Position]) -> Dict[str, float]:
    total_value = sum(p.value for p in positions)
    sector_values = defaultdict(float)

    for p in positions:
        sector_values[p.sector] += p.value

    return {s: v / total_value for s, v in sector_values.items()}

def can_add_to_sector(sector: str, current_exposure: Dict) -> bool:
    return current_exposure.get(sector, 0) < MAX_SECTOR_CONCENTRATION
```

---

### 5.3 Correlation Limits

```python
# Avoid highly correlated positions
MAX_CORRELATED_POSITIONS = 3  # Max positions with correlation > 0.7

def check_correlation_limit(
    new_position: Position,
    existing_positions: List[Position],
    correlation_matrix: pd.DataFrame
) -> bool:
    """
    Check if new position would exceed correlation limits.
    """
    correlated_count = 0

    for pos in existing_positions:
        correlation = correlation_matrix.loc[new_position.ticker, pos.ticker]
        if abs(correlation) > 0.7:
            correlated_count += 1

    return correlated_count < MAX_CORRELATED_POSITIONS
```

---

### 5.4 Beta/Delta Exposure

```python
# Portfolio-level market exposure
MAX_PORTFOLIO_BETA = 1.5  # Max beta-weighted exposure
MAX_PORTFOLIO_DELTA = 0.10  # Max delta as % of equity

def portfolio_beta(positions: List[Position]) -> float:
    total_value = sum(p.value for p in positions)
    weighted_beta = sum(p.beta * p.value for p in positions)
    return weighted_beta / total_value if total_value > 0 else 0

def check_beta_limit(current_beta: float, new_position: Position) -> bool:
    projected_beta = calculate_new_portfolio_beta(current_beta, new_position)
    return projected_beta <= MAX_PORTFOLIO_BETA
```

---

## 6. Daily & Drawdown Limits

### 6.1 Daily Loss Limit

```python
# Maximum loss allowed per day
DAILY_LOSS_LIMIT_PCT = 0.03  # 3% of equity

def check_daily_limit(
    starting_equity: float,
    current_equity: float
) -> bool:
    """
    Returns True if daily limit breached.
    """
    daily_pnl_pct = (current_equity - starting_equity) / starting_equity

    if daily_pnl_pct <= -DAILY_LOSS_LIMIT_PCT:
        return True  # STOP TRADING

    return False

# Actions when limit hit
DAILY_LIMIT_ACTIONS = {
    'close_all_positions': optional,  # May close or just stop new trades
    'no_new_trades': True,  # Mandatory
    'alert': True,
    'log_incident': True,
    'resume': 'next_trading_day'
}
```

---

### 6.2 Drawdown Circuit Breakers

```python
# Equity curve circuit breakers
DRAWDOWN_LEVELS = {
    'warning': 0.05,    # 5% drawdown - alert
    'reduce': 0.10,     # 10% drawdown - reduce position sizes
    'pause': 0.15,      # 15% drawdown - pause new trades
    'halt': 0.20        # 20% drawdown - halt all trading
}

def check_drawdown(current_equity: float, peak_equity: float) -> str:
    drawdown = (peak_equity - current_equity) / peak_equity

    if drawdown >= DRAWDOWN_LEVELS['halt']:
        return 'HALT'
    elif drawdown >= DRAWDOWN_LEVELS['pause']:
        return 'PAUSE'
    elif drawdown >= DRAWDOWN_LEVELS['reduce']:
        return 'REDUCE'
    elif drawdown >= DRAWDOWN_LEVELS['warning']:
        return 'WARNING'
    else:
        return 'NORMAL'

# Drawdown response
DRAWDOWN_RESPONSES = {
    'HALT': {
        'action': 'close_all_positions',
        'new_trades': False,
        'review_required': True,
        'resume': 'manual_approval_only'
    },
    'PAUSE': {
        'action': 'no_new_trades',
        'existing': 'manage_normally',
        'resume': 'after_N_days_or_recovery'
    },
    'REDUCE': {
        'action': 'reduce_position_size_50pct',
        'max_positions': 'reduce_by_half'
    }
}
```

---

### 6.3 Losing Streak Management

```python
# Consecutive loss limits
MAX_CONSECUTIVE_LOSSES = 5

def check_losing_streak(trade_results: List[bool]) -> int:
    """
    Count consecutive losses from most recent.
    """
    streak = 0
    for result in reversed(trade_results):
        if not result:  # Loss
            streak += 1
        else:
            break
    return streak

# Actions
def losing_streak_response(streak: int) -> dict:
    if streak >= 5:
        return {'action': 'pause_1_day', 'size_reduction': 0.5}
    elif streak >= 3:
        return {'action': 'reduce_size', 'size_reduction': 0.75}
    else:
        return {'action': 'continue', 'size_reduction': 1.0}
```

---

## 7. Risk Metrics

### 7.1 Portfolio Heat

```python
def portfolio_heat(positions: List[Position]) -> float:
    """
    Total portfolio risk as percentage of equity.
    Heat = sum of all position risks.
    """
    total_risk = sum(p.risk_amount for p in positions)
    return total_risk / equity

# Limits
MAX_PORTFOLIO_HEAT = 0.06  # 6% max total risk
# With 1% risk per trade, max 6 positions at full heat
```

---

### 7.2 Value at Risk (VaR)

```python
def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR at confidence level.
    """
    return returns.quantile(1 - confidence)

def parametric_var(
    portfolio_value: float,
    portfolio_std: float,
    confidence: float = 0.95
) -> float:
    """
    Parametric VaR assuming normal distribution.
    """
    from scipy.stats import norm
    z_score = norm.ppf(1 - confidence)
    return portfolio_value * portfolio_std * z_score

# Usage
MAX_VAR_95 = 0.02  # Max 2% VaR at 95% confidence
```

---

### 7.3 Expected Shortfall (CVaR)

```python
def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Average loss beyond VaR (tail risk).
    """
    var = returns.quantile(1 - confidence)
    return returns[returns <= var].mean()
```

---

## 8. Position Management Rules

### 8.1 Scaling In

```python
# Rules for adding to positions
SCALING_RULES = {
    'max_adds': 2,  # Maximum times to add
    'required_profit': True,  # Only add to winning positions
    'add_size': 'half_initial',  # Each add is half of initial
    'stop_adjustment': 'trail_to_breakeven'
}

def can_scale_in(position: Position) -> bool:
    return (
        position.adds_count < SCALING_RULES['max_adds'] and
        position.unrealized_pnl > 0 and
        position.risk_reward_still_valid
    )
```

---

### 8.2 Scaling Out

```python
# Partial profit taking
SCALE_OUT_RULES = {
    'first_target': {'pct': 0.50, 'action': 'close_33pct'},
    'second_target': {'pct': 1.00, 'action': 'close_33pct'},
    'remainder': 'trail_stop'
}

def scale_out_check(position: Position) -> Optional[Action]:
    profit_pct = position.unrealized_pnl_pct

    if profit_pct >= 1.00 and position.scale_outs < 2:
        return Action('close_partial', pct=0.33)
    elif profit_pct >= 0.50 and position.scale_outs < 1:
        return Action('close_partial', pct=0.33)

    return None
```

---

## 9. System Safeguards

### 9.1 Kill Switches

```python
KILL_SWITCH_TRIGGERS = {
    'daily_loss_exceeded': True,
    'drawdown_halt_level': True,
    'system_error': True,
    'connectivity_loss': True,
    'unusual_activity': True,
    'manual_trigger': True
}

def kill_switch_check(state: SystemState) -> bool:
    """
    Check if kill switch should be triggered.
    """
    if state.daily_pnl_pct <= -DAILY_LOSS_LIMIT_PCT:
        return True
    if state.drawdown >= DRAWDOWN_LEVELS['halt']:
        return True
    if state.error_count > MAX_ERRORS:
        return True
    if not state.broker_connected:
        return True

    return False

def execute_kill_switch():
    """
    Emergency shutdown procedure.
    """
    actions = [
        'cancel_all_pending_orders',
        'close_all_positions',  # Optional
        'disable_new_orders',
        'send_alert_notification',
        'log_all_state_data',
        'require_manual_restart'
    ]
    for action in actions:
        execute(action)
```

---

### 9.2 Order Sanity Checks

```python
def validate_order(order: Order) -> Tuple[bool, str]:
    """
    Pre-flight checks before sending order.
    """
    checks = []

    # Price sanity
    if order.type == 'LIMIT':
        deviation = abs(order.price - market_price) / market_price
        if deviation > 0.05:  # 5% from market
            checks.append(('FAIL', 'price_deviation_too_large'))

    # Size sanity
    if order.quantity > MAX_SHARES_PER_ORDER:
        checks.append(('FAIL', 'size_exceeds_maximum'))

    if order.quantity * order.price > MAX_DOLLAR_PER_ORDER:
        checks.append(('FAIL', 'dollar_amount_exceeds_maximum'))

    # Risk check
    if order.risk_amount > equity * MAX_RISK_PER_TRADE_PCT:
        checks.append(('FAIL', 'risk_exceeds_limit'))

    # Concentration check
    if would_exceed_concentration(order):
        checks.append(('FAIL', 'concentration_limit'))

    if any(c[0] == 'FAIL' for c in checks):
        return False, checks
    return True, []
```

---

### 9.3 Fat Finger Protection

```python
FAT_FINGER_LIMITS = {
    'max_shares_per_order': 10000,
    'max_dollar_per_order': 50000,
    'max_pct_of_equity_per_order': 0.20,
    'max_orders_per_minute': 10,
    'price_deviation_limit': 0.03  # 3% from last trade
}

def fat_finger_check(order: Order) -> bool:
    if order.quantity > FAT_FINGER_LIMITS['max_shares_per_order']:
        return False
    if order.quantity * order.price > FAT_FINGER_LIMITS['max_dollar_per_order']:
        return False
    if abs(order.price - last_price) / last_price > FAT_FINGER_LIMITS['price_deviation_limit']:
        return False
    return True
```

---

## 10. Risk Management Templates

### 10.1 Conservative Profile

```python
CONSERVATIVE_RISK_PROFILE = {
    'risk_per_trade': 0.005,      # 0.5%
    'max_positions': 5,
    'max_sector_concentration': 0.20,
    'daily_loss_limit': 0.02,     # 2%
    'max_drawdown': 0.10,         # 10%
    'position_sizing': 'atr_based',
    'stop_method': 'atr_3x',
    'leverage': None
}
```

### 10.2 Moderate Profile

```python
MODERATE_RISK_PROFILE = {
    'risk_per_trade': 0.01,       # 1%
    'max_positions': 10,
    'max_sector_concentration': 0.25,
    'daily_loss_limit': 0.03,     # 3%
    'max_drawdown': 0.15,         # 15%
    'position_sizing': 'atr_based',
    'stop_method': 'atr_2x',
    'leverage': None
}
```

### 10.3 Aggressive Profile

```python
AGGRESSIVE_RISK_PROFILE = {
    'risk_per_trade': 0.02,       # 2%
    'max_positions': 15,
    'max_sector_concentration': 0.30,
    'daily_loss_limit': 0.05,     # 5%
    'max_drawdown': 0.25,         # 25%
    'position_sizing': 'fixed_pct',
    'stop_method': 'atr_1.5x',
    'leverage': 'margin_available'
}
```

---

## Academic References

1. **Vince, R.**: "The Mathematics of Money Management" - Position sizing theory
2. **Tharp, V.K.**: "Trade Your Way to Financial Freedom" - Expectancy and sizing
3. **Kelly, J.L. (1956)**: "A New Interpretation of Information Rate" - Kelly Criterion
4. **Markowitz, H.**: Modern Portfolio Theory - Diversification
5. **Taleb, N.N.**: "Fooled by Randomness" - Risk and uncertainty

---

## Key Takeaways

1. **Risk per trade is paramount**: Never exceed your limit
2. **Position sizing determines survival**: Size for the worst case
3. **Stops are mandatory**: No trade without a defined exit
4. **Portfolio limits prevent concentration**: Diversify
5. **Drawdown rules prevent ruin**: Reduce risk in losing streaks
6. **Kill switches are essential**: Have emergency procedures
7. **All limits are hard**: No exceptions in automated systems
