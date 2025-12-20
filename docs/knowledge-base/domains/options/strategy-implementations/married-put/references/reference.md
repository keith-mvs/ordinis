# Married-Put Strategy: Advanced Reference

Detailed mathematical formulas, Greeks calculations, and advanced scenarios for the married-put options strategy.

## Table of Contents

- [Options Greeks](#options-greeks)
- [Implied Volatility Impact](#implied-volatility-impact)
- [Time Decay Analysis](#time-decay-analysis)
- [Rolling Strategies](#rolling-strategies)
- [Tax Implications](#tax-implications)
- [Advanced Scenarios](#advanced-scenarios)
- [Mathematical Formulas](#mathematical-formulas)

## Options Greeks

### Delta (Δ)

**Definition**: Rate of change in option price relative to $1 change in stock price.

**For Put Options**: Delta ranges from 0 to -1
- OTM puts: -0.10 to -0.40 (less sensitive)
- ATM puts: -0.45 to -0.55 (moderate sensitivity)
- ITM puts: -0.60 to -0.95 (high sensitivity)

**Married-Put Position Delta**:
```
Position Delta = Stock Delta + Put Delta
Position Delta = 1.00 + Put Delta
```

**Example**:
- Stock: 100 shares (Delta = +1.00)
- ATM Put: Delta = -0.50
- **Net Position Delta = +0.50**

This means for every $1 move in stock:
- Stock up $1 → Position gains ~$50
- Stock down $1 → Position loses ~$50

The put provides 50% downside protection while maintaining 50% upside exposure.

### Gamma (Γ)

**Definition**: Rate of change in Delta relative to $1 change in stock price.

**For Put Options**: Gamma is highest for ATM options
- OTM puts: Lower gamma (0.01-0.03)
- ATM puts: Higher gamma (0.04-0.08)
- ITM puts: Lower gamma (0.01-0.03)

**Significance for Married Puts**:
- High gamma near expiration increases protection sensitivity
- ATM puts provide dynamic hedging as stock moves
- Gamma risk minimal compared to selling options

### Theta (Θ)

**Definition**: Rate of option value decay per day (time decay).

**For Put Options**: Always negative for long puts
- 30 days to expiration: -$0.03 to -$0.08 per day
- 60 days to expiration: -$0.02 to -$0.05 per day
- 90 days to expiration: -$0.01 to -$0.03 per day

**Monthly Theta Cost**:
```python
# Calculate monthly time decay cost
monthly_theta = put_theta * 30  # Daily theta × 30 days
annualized_theta_cost = monthly_theta * 12
theta_percentage = (annualized_theta_cost / put_premium) * 100
```

**Example**:
- Put premium: $3.50
- Daily theta: -$0.045
- Monthly cost: $0.045 × 30 = $1.35
- Annualized: $1.35 × 12 = $16.20
- **Annual theta drag: 463% of premium** (typical for ATM options)

### Vega (ν)

**Definition**: Change in option price for 1% change in implied volatility.

**For Put Options**: Positive vega for long puts
- 30 days: Vega = 0.05 to 0.12
- 60 days: Vega = 0.08 to 0.18
- 90 days: Vega = 0.10 to 0.22

**Impact on Married-Put Cost**:
```python
# Calculate premium change from IV shift
new_premium = current_premium + (vega * iv_change)

# Example: IV increases 5%
# Current premium: $3.50, Vega: 0.15
# New premium: $3.50 + (0.15 × 5) = $4.25
```

**Volatility Scenarios**:
- **IV Expansion** (market stress): Put becomes more expensive but provides needed protection
- **IV Contraction** (calm markets): Put loses value faster, consider rolling to cheaper strikes

### Greeks Summary Table

| Greek | Symbol | Measures | Long Put | Married-Put Impact |
|-------|--------|----------|----------|-------------------|
| Delta | Δ | Price sensitivity | -0.50 typical | Reduces position delta by 50% |
| Gamma | Γ | Delta change | +0.05 typical | Increases protection as stock falls |
| Theta | Θ | Time decay | -$0.04/day | Steady cost of protection |
| Vega | ν | Volatility sensitivity | +0.15 typical | Benefits from IV increase |
| Rho | ρ | Interest rate sensitivity | Negative | Minimal impact for short-term |

## Implied Volatility Impact

### IV Levels by Stock Type

**Low IV Stocks (10-20%)**:
- Large-cap blue chips
- Utilities, staples
- Premium cost: 1-3% of stock price for ATM puts

**Medium IV Stocks (20-35%)**:
- Mid-cap growth
- Cyclical stocks
- Premium cost: 3-6% of stock price for ATM puts

**High IV Stocks (35-60%)**:
- Small-cap
- Biotech, tech
- Premium cost: 6-12% of stock price for ATM puts

**Very High IV Stocks (60%+)**:
- Pre-earnings
- Crisis stocks
- Premium cost: 12%+ of stock price for ATM puts

### IV Rank and Percentile

**IV Rank**: Where current IV stands relative to 52-week range
```
IV Rank = (Current IV - 52W Low IV) / (52W High IV - 52W Low IV)
```

**Strategy Adjustments**:
- **Low IV Rank (0-25)**: Consider buying protection, premiums cheap
- **Mid IV Rank (25-75)**: Standard pricing, normal strategy
- **High IV Rank (75-100)**: Protection expensive, use wider strikes or shorter duration

### Calculating Put Premium from IV

**Black-Scholes approximation** for ATM put:
```python
import numpy as np
from scipy.stats import norm

def estimate_put_premium(S, K, T, r, sigma):
    """
    S: Stock price
    K: Strike price
    T: Time to expiration (years)
    r: Risk-free rate (annual)
    sigma: Implied volatility (annual)
    """
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return put_price

# Example: $50 stock, $48 strike, 60 days, 4% rate, 30% IV
premium = estimate_put_premium(50, 48, 60/365, 0.04, 0.30)
print(f"Estimated put premium: ${premium:.2f}")
```

## Time Decay Analysis

### Theta Decay Curve

Time decay accelerates as expiration approaches:

**Days to Expiration vs. Time Value**:
- 90 days: 100% of extrinsic value remains
- 60 days: ~80% remains (20% decay)
- 45 days: ~65% remains (35% decay)
- 30 days: ~45% remains (55% decay)
- 15 days: ~20% remains (80% decay)
- 7 days: ~5% remains (95% decay)

### Rolling to Maintain Protection

**When to Roll**:
- 21-30 days before expiration (capturing remaining theta)
- When stock has moved significantly from strike
- After IV spike when premiums are elevated

**Rolling Mechanics**:
```python
def calculate_roll_cost(current_put_value, new_put_premium):
    """
    Calculate net cost to roll put forward
    """
    sell_proceeds = current_put_value
    buy_cost = new_put_premium
    transaction_costs = 2 * 0.65  # Sell old + buy new

    net_roll_cost = buy_cost - sell_proceeds + transaction_costs
    return net_roll_cost

# Example: Roll 30-day $48 put to 60-day $50 put
current_value = 1.85  # Current put worth
new_premium = 3.20    # New put costs
roll_cost = calculate_roll_cost(current_value, new_premium)
print(f"Net cost to roll: ${roll_cost:.2f}")
```

**Rolling Strategies**:
1. **Roll same strike**: Maintain same protection level
2. **Roll up**: Increase protection (higher strike)
3. **Roll down**: Reduce cost (lower strike)
4. **Roll and split**: Buy 2 contracts at lower strikes

## Tax Implications

### Holding Period Rules

**Qualified Dividends and Long-Term Gains**:
- Stock must be held >60 days without protective puts
- Protective put "suspends" holding period
- Clock restarts when put position closed

**Tax Impact**:
```
Without married put:
- Hold stock 1 year → Long-term capital gains (15-20% rate)

With married put:
- Hold stock 1 year with continuous put protection
- Holding period suspended while protected
- Gains taxed as short-term (ordinary income rate: 22-37%)
```

### Constructive Sales

Holding stock + deep ITM put may trigger constructive sale:
- If put delta approaches -1.00
- Substantially eliminates risk and opportunity
- May trigger immediate tax recognition

**Safe Harbor**: Stay with ATM or OTM puts (delta -0.60 or higher)

### Put Premium Treatment

**Loss on expired puts**:
- Deductible as capital loss
- Can offset capital gains
- $3,000 annual limit for excess losses

**Embedded premium if assigned**:
- Put premium adds to cost basis if exercised
- Reduces capital gain or increases loss

**Example**:
```
Buy 100 shares at $50: Cost = $5,000
Buy put strike $48, premium $2.50: Cost = $250

Stock falls to $45, exercise put:
- Sell stock at $48: Proceeds = $4,800
- Adjusted cost basis: $5,000 + $250 = $5,250
- Capital loss: $5,250 - $4,800 = $450
```

## Advanced Scenarios

### Scenario 1: Earnings Protection

**Setup**: Hold 100 shares at $52.30 ahead of earnings

**Risk**: Historical post-earnings moves: ±12%

**Strategy**:
- Buy 1x $50 put, expiring week after earnings
- Premium: $3.20 (elevated IV)
- Protection level: 4.4% below current price

**Outcomes**:
1. **Stock rallies 8% to $56.50**:
   - Stock gain: $420
   - Put expires worthless: -$320
   - Net gain: $100 (1.9% return)

2. **Stock drops 10% to $47.07**:
   - Stock loss: -$523
   - Put gain: $293 (intrinsic: $2.93)
   - Net loss: -$230 (4.4% loss, limited)

### Scenario 2: Rolling Up After Rally

**Initial Position**:
- Buy 100 shares at $45.00
- Buy $43 put, premium $2.10
- Total cost: $4,710

**Stock rallies to $52.00** after 30 days:
- Stock gain: $700
- Put now worth: $0.40

**Decision**: Roll up to protect gains
- Sell $43 put for $0.40
- Buy $50 put (60 days), premium $2.85
- Net roll cost: $2.45 + $1.30 = $3.75

**New Position**:
- Protected at $50 (vs. $43 previously)
- Locked in $5.00 gain on stock (minus premiums)
- Total protection cost: $2.10 + $3.75 = $5.85

### Scenario 3: Volatility Spike Management

**Position**:
- 200 shares at $68.25
- 2x $65 puts, premium $3.15 each
- IV: 32%

**Market correction**, IV spikes to 55%:
- Stock drops to $62.50
- Puts now worth $5.20 (intrinsic: $2.50, extrinsic: $2.70)

**Options**:
1. **Hold**: Maintain protection through volatility
2. **Sell premium**: Close puts, capture $4.10 profit to offset stock loss
3. **Roll to longer term**: Sell $65 puts, buy 90-day $60 puts at elevated IV

### Scenario 4: Small Position Management ($5K-$10K)

**Capital**: $7,500

**Position**:
- Stock price: $38.50
- Target allocation: 80% = $6,000
- Shares: 155 (rounded to 1.55 contracts)
- Choose: Buy 100 shares (1 contract)

**Protection**:
- Buy 1x $37 put (OTM), 45 days
- Premium: $1.45
- Total cost: $3,850 + $145 + $0.65 = $3,995.65

**Characteristics**:
- Capital efficient (53% deployed)
- 3.9% protection buffer
- $2.10 max loss per share (5.5%)
- Remaining capital: $3,504 for other positions

### Scenario 5: High-Volatility Small Cap ($45K position)

**Stock**: Small-cap biotech at $23.80, IV: 68%

**Position**:
- Target: $45,000
- Shares: 1,890 (18.9 contracts, round to 1,800)
- Cost: $42,840

**Protection Challenge**:
- ATM put ($24): $4.25 (18% of stock price!)
- 60-day protection: $7,650 (17.9% of position)

**Strategy Adjustment**:
- Use OTM put: $22 strike
- Premium: $2.65 (11% of stock price)
- Protection cost: $4,770 (11.1% of position)
- Accept 7.6% unprotected decline
- Total position cost: $47,610

**Rationale**: High IV makes ATM protection prohibitively expensive; OTM provides catastrophic protection while accepting moderate losses.

## Mathematical Formulas

### Breakeven Price

```
Breakeven = S₀ + (Pₚ + Tₓ)

Where:
S₀ = Initial stock price
Pₚ = Put premium per share
Tₓ = Transaction cost per share = 0.65/100
```

### Maximum Loss

```
Max Loss = (S₀ - K) + Pₚ + Tₓ

Where:
K = Put strike price
Limited loss zone: Any price ≤ K
```

### Maximum Gain

```
Max Gain = ∞ (unlimited)

Actual gain at price Sₜ:
Gain = (Sₜ - S₀) - Pₚ - Tₓ
```

### Profit/Loss at Any Price

```
For Sₜ > K (put expires worthless):
P/L = (Sₜ - S₀) - Pₚ - Tₓ

For Sₜ ≤ K (put has value):
P/L = (K - S₀) - Pₚ - Tₓ = -Max Loss
```

### Return on Investment (ROI)

```
ROI = (Final Value - Initial Investment) / Initial Investment × 100%

Final Value = max(Sₜ, K) × Shares
Initial Investment = (S₀ × Shares) + (Pₚ × 100) + Tₓ
```

### Protection Percentage

```
Protection % = (S₀ - K) / S₀ × 100%

Example: $50 stock, $48 strike
Protection % = (50 - 48) / 50 = 4.0%
```

### Annualized Protection Cost

```
Annual Cost % = (Pₚ / S₀) × (365 / Days) × 100%

Example: $50 stock, $2.50 premium, 60 days
Annual Cost = (2.50 / 50) × (365 / 60) = 30.4%
```

### Effective Cost Basis

```
Effective Basis = S₀ + Pₚ + Tₓ

If exercised:
Realized Loss per Share = Effective Basis - K
```

## Performance Metrics

### Sharpe Ratio Adjustment

```
Traditional: Sharpe = (Return - Risk-Free Rate) / Std Dev

With protection: Adjusted Std Dev reduced by put delta

Married-Put Sharpe = (Return - Rₓ) / (σ × (1 + δₚ))

Where:
δₚ = Put delta (negative, e.g., -0.50)
Effect: Reduces volatility by ~50% for ATM put
```

### Risk-Adjusted Return

```
Risk-Adjusted Return = Expected Return / Max Drawdown

Without protection:
Max Drawdown = Potential 100% loss

With married put:
Max Drawdown = (S₀ - K + Pₚ) / S₀

Example: $50 stock, $48 put, $2.50 premium
Max Drawdown = (50 - 48 + 2.50) / 50 = 9.0%
```

## Comparison to Alternatives

### Married Put vs. Collar

**Married Put**:
- Long stock + Long put
- Cost: Put premium (net debit)
- Upside: Unlimited
- Downside: Limited to max loss

**Collar**:
- Long stock + Long put + Short call
- Cost: Reduced or zero (net credit possible)
- Upside: Capped at call strike
- Downside: Limited to max loss

**When to Choose Married Put**:
- Want unlimited upside potential
- Bullish on stock continuation
- Willing to pay premium for full upside

### Married Put vs. Stop-Loss Order

**Married Put**:
- Guaranteed minimum exit price
- No gap risk
- Time decay cost
- Works during market closures

**Stop-Loss Order**:
- No upfront cost
- Gap risk (may execute below stop)
- Requires monitoring
- Fails in fast markets or halts

**When to Choose Married Put**:
- Volatile stock prone to gaps
- Overnight/weekend holding
- Important technical support level
- Want "insurance" not discipline

## Further Reading

**Books**:
- *Options as a Strategic Investment* - Lawrence G. McMillan
- *Options Volatility & Pricing* - Sheldon Natenberg
- *The Options Playbook* - Brian Overby

**Online Resources**:
- CBOE Options Education: https://www.cboe.com/education/
- OIC (Options Industry Council): https://www.optionseducation.org/
- Tastytrade Options Education: https://www.tastytrade.com/

**Academic Papers**:
- Black-Scholes-Merton Option Pricing Model
- "Volatility and the Alchemy of Risk" - Peter Carr
- "Dynamic Hedging" - Nassim Nicholas Taleb

**Regulatory**:
- Options Disclosure Document (ODD)
- FINRA Options Education
- SEC Investor Publications on Options
