# Protective Collar - Portfolio Integration

**Parent**: [Protective Collar](../SKILL.md) | **Related**: [Position Management](position-management.md) | [Examples](examples.md)

---

## Portfolio Integration for Protective Collar

Integrating protective collar positions into a broader portfolio requires careful consideration of sizing, correlation, and overall risk management.

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
portfolio_metrics = {
    'net_delta': calculate_net_delta(),
    'net_gamma': calculate_net_gamma(),
    'net_theta': calculate_net_theta(),
    'net_vega': calculate_net_vega(),
    'total_capital_at_risk': sum_max_losses(),
    'portfolio_heat': total_risk / portfolio_value
}
```

---

## Sector and Concentration Risk

### Sector Exposure

**Track by Sector**:
```python
sector_exposure = {
    'Technology': {'value': 50000, 'hedged': 10000, 'net': 40000},
    'Healthcare': {'value': 25000, 'hedged': 0, 'net': 25000},
    'Finance': {'value': 15000, 'hedged': 3000, 'net': 12000},
}

# Identify overconcentration
for sector, exposure in sector_exposure.items():
    sector_pct = exposure['net'] / portfolio_value
    if sector_pct > 0.25:  # > 25%
        print(f"Overconcentrated in {sector}: {sector_pct:.1%}")
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
scenarios = {
    'market_crash': {'spy_change': -0.20, 'vix_change': +20},
    'slow_grind_down': {'spy_change': -0.10, 'vix_change': +5},
    'volatility_spike': {'spy_change': 0.00, 'vix_change': +15},
}

for scenario_name, params in scenarios.items():
    portfolio_pl = calculate_scenario_pl(params)
    print(f"{scenario_name}: {portfolio_pl:,.0f}")
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
