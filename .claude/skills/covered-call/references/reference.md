# Covered Call Strategy - Reference Guide

## Table of Contents
- [Complete Strike Selection Methodology](#strike-selection)
- [Detailed Greeks Analysis](#greeks-analysis)
- [P&L Formulas and Examples](#pnl-formulas)
- [Tax Considerations](#tax-considerations)
- [Performance Evaluation](#performance-evaluation)
- [Execution Guidelines](#execution)
- [Advanced Topics](#advanced-topics)

## Strike Selection

### Expected Move Calculation

Calculate one standard deviation expected move:
```python
expected_move = stock_price * implied_volatility * sqrt(days_to_exp / 365)
strike_target = stock_price + expected_move
```

Set strike above 1 standard deviation to reduce assignment probability to ~16%.

### Technical Analysis Integration

**Resistance-Based Strike Selection**:
1. Identify key resistance levels (previous highs, moving averages, Fibonacci)
2. Place strike at or above nearest resistance
3. Reduces breach probability while maximizing premium

**Example**:
- Stock at $100
- Resistance at $105 (52-week high)
- Select $105 strike for maximum premium with natural cap

### Return Optimization

**Target Monthly Returns**:
```python
monthly_target = 0.02  # 2% monthly = 24% annualized
required_premium = stock_value * monthly_target
# Select strike that yields required premium
```

**Strike-Premium Trade-off Matrix**:
| Strike OTM | Delta | Premium | Assignment Risk | Upside Capture |
|-----------|-------|---------|-----------------|----------------|
| ATM | 0.50 | 4-5% | High (50%) | Minimal |
| 2.5% OTM | 0.35 | 2.5-3% | Moderate (35%) | Limited |
| 5% OTM | 0.25 | 1.5-2% | Low (25%) | Good |
| 10% OTM | 0.15 | 0.8-1% | Very Low (15%) | Excellent |

## Greeks Analysis

### Detailed Delta Management

**Net Delta Formula**:
```
Net Delta = (Shares / 100) + (Short Call Delta × Contracts)
```

**Example Calculation**:
- Long 100 shares = +1.0 delta
- Short 1 call at 0.30 delta = -0.30 delta
- Net delta = 1.0 - 0.30 = 0.70

**Delta Interpretation**:
- Net delta 0.70 = 70% of stock's directional exposure
- For every $1 move in stock, position gains/loses $0.70
- 30% of upside is "given up" for premium collection

### Theta Decay Dynamics

**Time Decay Curve**:
```
Days to Expiration | Daily Theta | % of Value
90 days            | 0.02        | 0.5%
60 days            | 0.03        | 1.0%
45 days            | 0.04        | 1.5%
30 days            | 0.06        | 2.0%
21 days            | 0.08        | 3.0%
14 days            | 0.12        | 4.0%
7 days             | 0.20        | 6.0%
```

**Optimal Roll Timing**: Roll at 21 DTE to capture 70-80% of maximum theta while avoiding acceleration zone.

### Vega Impact Analysis

**IV Change Scenarios**:
```python
# Scenario 1: IV Spike (+5 points)
call_vega = 0.15
iv_change = 5
call_value_change = -0.15 * 5 = -$0.75 per share (loss)
# Short call increased in value

# Scenario 2: IV Crush (-5 points)
call_value_change = -0.15 * (-5) = +$0.75 per share (gain)
# Short call decreased in value
```

**Strategy**: Write calls when IV elevated (high premiums), roll or close when IV declines (cheap buybacks).

### Gamma Considerations

**Gamma Acceleration**:
- ATM options have highest gamma
- As stock approaches strike, delta changes accelerate
- Near expiration, gamma spikes dramatically

**Risk Management**:
```python
if days_to_expiration < 14 and stock_near_strike:
    # High gamma risk - delta changing rapidly
    consider_rolling_early = True
```

## P&L Formulas

### Break-Even Analysis

**Break-Even Price**:
```
BE = Stock Entry Price - Premium Received

Example:
- Stock purchased at $100
- Call premium $2
- Break-even = $100 - $2 = $98
```

**Multiple Scenario P&L**:
```python
def calculate_pnl_scenarios(entry, strike, premium):
    scenarios = []
    
    for exit_price in range(int(entry * 0.80), int(entry * 1.20), 5):
        if exit_price >= strike:
            # Stock called away
            pnl = (strike - entry) + premium
        else:
            # Keep stock
            pnl = (exit_price - entry) + premium
        
        scenarios.append({
            'exit_price': exit_price,
            'pnl': pnl,
            'return_pct': pnl / entry
        })
    
    return scenarios
```

### Return Metrics

**Time-Weighted Returns**:
```python
def annualized_return(premium, stock_value, days_held):
    """Calculate annualized return from covered call."""
    period_return = premium / stock_value
    periods_per_year = 365 / days_held
    annualized = period_return * periods_per_year
    return annualized

# Example:
# Premium: $2, Stock: $100, Held 35 days
# Period return = 2%
# Annualized = 2% * (365/35) = 20.9%
```

### Risk-Adjusted Metrics

**Sharpe Ratio for Covered Calls**:
```python
import numpy as np

def calculate_sharpe(monthly_returns, risk_free_rate=0.04):
    """
    Calculate Sharpe ratio for covered call strategy.
    
    Covered calls typically have:
    - Higher Sharpe than buy-and-hold in flat/down markets
    - Lower Sharpe in strong bull markets
    """
    excess_returns = np.array(monthly_returns) - (risk_free_rate / 12)
    sharpe = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
    return sharpe * np.sqrt(12)  # Annualized
```

## Tax Considerations

### Qualified vs. Unqualified Covered Calls

**Qualified Covered Call Criteria**:
1. Stock held >1 year (long-term)
2. Call expiration >30 days
3. Strike price not too deep ITM:
   - Stock >$150: Strike ≥ (stock - $10)
   - Stock $50-150: Strike ≥ 95% of stock price
   - Stock <$50: Strike ≥ (stock - $5)

**Tax Impact**:
```python
def tax_treatment(stock_basis, stock_sale, call_premium, holding_period, is_qualified):
    """
    Calculate tax on covered call assignment.
    
    Qualified call: Preserves long-term capital gains on stock
    Unqualified call: May reset to short-term gains
    """
    capital_gain = stock_sale - stock_basis
    
    if is_qualified and holding_period > 365:
        # Long-term capital gains rate (0%, 15%, or 20%)
        ltcg_rate = 0.15  # Example
        stock_tax = capital_gain * ltcg_rate
        premium_tax = call_premium * 0.37  # Short-term rate
    else:
        # All short-term
        total_gain = capital_gain + call_premium
        stock_tax = total_gain * 0.37
        premium_tax = 0
    
    return {
        'stock_tax': stock_tax,
        'premium_tax': premium_tax,
        'total_tax': stock_tax + premium_tax,
        'after_tax_return': (capital_gain + call_premium - stock_tax - premium_tax) / stock_basis
    }
```

### Wash Sale Rules

**Wash Sale Trigger**:
- Sell stock at loss
- Repurchase same/substantially identical security within 30 days
- Loss disallowed, added to cost basis of replacement

**Covered Call Impact**:
- Writing call against stock does NOT trigger wash sale
- Selling stock at loss, then buying back within 30 days DOES trigger
- Solution: Wait 31 days before repurchasing

## Performance Evaluation

### Benchmarking Methodology

**Comparison Framework**:
```python
def benchmark_performance(cc_returns, buy_hold_returns, market_returns):
    """
    Compare covered call to alternatives.
    
    Best practice: Compare over multiple market cycles
    """
    # Cumulative returns
    cc_cumulative = (1 + np.array(cc_returns)).prod() - 1
    bh_cumulative = (1 + np.array(buy_hold_returns)).prod() - 1
    market_cumulative = (1 + np.array(market_returns)).prod() - 1
    
    # Volatility
    cc_volatility = np.std(cc_returns) * np.sqrt(12)
    bh_volatility = np.std(buy_hold_returns) * np.sqrt(12)
    
    # Sharpe ratios
    cc_sharpe = calculate_sharpe(cc_returns)
    bh_sharpe = calculate_sharpe(buy_hold_returns)
    
    return {
        'covered_call_return': cc_cumulative,
        'buy_hold_return': bh_cumulative,
        'market_return': market_cumulative,
        'cc_volatility': cc_volatility,
        'bh_volatility': bh_volatility,
        'cc_sharpe': cc_sharpe,
        'bh_sharpe': bh_sharpe,
        'outperformance': cc_cumulative - bh_cumulative
    }
```

### Market Regime Analysis

**Performance by Market Condition**:

| Market Condition | CC Performance | Buy-Hold | Winner |
|-----------------|----------------|----------|---------|
| Bull Market (>15% up) | Moderate gains | Strong gains | Buy-Hold |
| Mild Bull (5-15% up) | Good gains | Moderate gains | Covered Call |
| Sideways (±5%) | Positive income | Flat | Covered Call |
| Mild Bear (-5 to -15%) | Small loss | Moderate loss | Covered Call |
| Bear Market (<-15%) | Significant loss | Severe loss | Covered Call |

**Conclusion**: Covered calls outperform in sideways/moderately bullish markets but underperform in strong bull markets.

## Execution

### Advanced Order Types

**Limit Order Strategy**:
```python
def intelligent_limit_pricing(bid, ask, market_conditions):
    """
    Dynamic limit pricing based on market conditions.
    
    Tight market: Start at mid, walk toward bid
    Wide market: Start above mid, be patient
    """
    mid = (bid + ask) / 2
    spread_pct = (ask - bid) / bid
    
    if spread_pct < 0.05:  # Tight market
        initial_limit = mid
        walk_increment = 0.01  # Walk down $0.01
    else:  # Wide market
        initial_limit = mid + (ask - mid) * 0.5
        walk_increment = 0.05  # Walk down $0.05
    
    return {
        'initial_limit': initial_limit,
        'walk_increment': walk_increment,
        'patience_minutes': 5 if spread_pct < 0.05 else 15
    }
```

### Slippage Minimization

**Expected Slippage**:
- Liquid stocks/options: 0.5-1% of premium
- Less liquid: 2-5% of premium
- Illiquid: 5-10% of premium

**Slippage Reduction Tactics**:
1. Trade during high-volume periods
2. Use limit orders, never market orders
3. Break large orders into smaller chunks
4. Avoid earnings week (wide spreads)

## Advanced Topics

### Repair Strategies

**Underwater Position Recovery**:
```python
def covered_call_repair(current_price, entry_price, underwater_amount):
    """
    Use covered calls to lower break-even on losing positions.
    
    Strategy: Write multiple shorter-term calls to collect premium
    faster and reduce effective break-even.
    """
    # Calculate premium needed to break even
    premium_needed = entry_price - current_price
    
    # Aggressive repair: Weekly calls for 4 weeks
    weekly_premium_target = premium_needed / 4
    
    # Conservative repair: Monthly calls for 3 months
    monthly_premium_target = premium_needed / 3
    
    return {
        'aggressive_weekly_target': weekly_premium_target,
        'conservative_monthly_target': monthly_premium_target,
        'estimated_recovery_days_aggressive': 28,
        'estimated_recovery_days_conservative': 90
    }
```

### Portfolio-Level Implementation

**Diversified Covered Call Portfolio**:
```python
def construct_cc_portfolio(account_value, positions_target=10):
    """
    Build diversified covered call portfolio.
    
    Best practices:
    - 10+ positions across sectors
    - No single position >10% of account
    - Mix of dividend/growth stocks
    - Stagger expirations (avoid all expiring same week)
    """
    position_size = account_value / positions_target
    
    portfolio = {
        'total_value': account_value,
        'num_positions': positions_target,
        'position_size': position_size,
        'sector_limits': {
            'technology': 0.30,
            'healthcare': 0.20,
            'financials': 0.20,
            'consumer': 0.15,
            'industrials': 0.15
        },
        'expiration_ladder': [
            {'week': 1, 'pct': 0.20},
            {'week': 2, 'pct': 0.30},
            {'week': 3, 'pct': 0.30},
            {'week': 4, 'pct': 0.20}
        ]
    }
    
    return portfolio
```

### Automated Execution Systems

**Rule-Based Rolling**:
```python
def auto_roll_decision(position, market_data, rules):
    """
    Automated decision system for rolling calls.
    
    Rules-based approach removes emotion, ensures consistency.
    """
    days_remaining = position['days_to_exp']
    delta = position['call_delta']
    profit_pct = position['profit_captured_pct']
    
    # Rule 1: Always roll at 7 DTE
    if days_remaining <= 7:
        return {'action': 'roll', 'reason': 'expiration_threshold'}
    
    # Rule 2: Roll if captured 80% of max profit
    if profit_pct >= 0.80:
        return {'action': 'roll', 'reason': 'profit_target_achieved'}
    
    # Rule 3: Roll if deep ITM (delta > 0.70)
    if abs(delta) > 0.70:
        return {'action': 'roll', 'reason': 'deep_itm'}
    
    # Otherwise hold
    return {'action': 'hold', 'reason': 'criteria_not_met'}
```

---

**For implementation code, see scripts/ directory.**
**For additional questions, refer to SKILL.md or consult referenced texts.**
