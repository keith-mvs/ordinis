# Protective Collar - Dividend Considerations

**Parent**: [Protective Collar](../SKILL.md) | **Related**: [Position Management](position-management.md) | [Examples](examples.md)

---

## Dividend Considerations for Protective Collar

Dividends significantly impact protective collar positions. Understanding ex-dividend dates, early assignment risk, and dividend capture strategies is essential.

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

**Example** (Protective Collar):
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
positions_with_upcoming_divs = {
    'AAPL': {'ex_date': '2025-05-15', 'amount': 0.24},
    'MSFT': {'ex_date': '2025-05-20', 'amount': 0.68},
    'JNJ': {'ex_date': '2025-05-25', 'amount': 1.13},
}

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
