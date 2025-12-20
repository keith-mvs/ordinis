# Example Married-Put Positions

This directory contains real-world example positions demonstrating married-put strategies across different market conditions and volatility levels.

## Sample Positions Overview

The `sample_positions.csv` file contains 5 example positions:

### Position 1: COST - Low Volatility (18% IV)
**Large-Cap Defensive**
- Stock: $52.75
- Investment: $10,550 (200 shares)
- Put Strike: $50.00 (5.2% protection)
- Premium: $1.85 (3.5% of stock price)
- Expiration: 45 days
- **Use Case**: Conservative protection on stable blue-chip stock
- **Characteristics**: Low premiums, small protection buffer, ideal for quality holdings

### Position 2: SNAP - High Volatility (58% IV)
**Small-Cap Growth**
- Stock: $23.80
- Investment: $9,520 (400 shares)
- Put Strike: $22.00 (7.6% protection)
- Premium: $2.65 (11.1% of stock price)
- Expiration: 60 days
- **Use Case**: Protecting speculative position in volatile small-cap
- **Characteristics**: Elevated premiums due to high IV, wider protection needed

### Position 3: TDOC - Medium Volatility (32% IV)
**Mid-Cap Healthcare**
- Stock: $38.50
- Investment: $11,550 (300 shares)
- Put Strike: $37.00 (3.9% protection)
- Premium: $2.15 (5.6% of stock price)
- Expiration: 60 days
- **Use Case**: Standard hedging on growth stock with moderate risk
- **Characteristics**: Balanced cost-benefit, suitable for most strategies

### Position 4: PLTR - Medium-High Volatility (38% IV)
**Mid-Cap Technology**
- Stock: $28.40
- Investment: $14,200 (500 shares)
- Put Strike: $27.00 (4.9% protection)
- Premium: $2.40 (8.5% of stock price)
- Expiration: 45 days
- **Use Case**: Short-term protection around earnings or events
- **Characteristics**: Higher premium reflects event risk, shorter duration

### Position 5: MRNA - High Volatility (52% IV)
**Large-Cap Biotech**
- Stock: $175.50
- Investment: $17,550 (100 shares)
- Put Strike: $170.00 (3.1% protection)
- Premium: $8.25 (4.7% of stock price)
- Expiration: 90 days
- **Use Case**: Protecting significant position with binary event risk
- **Characteristics**: High absolute premium, longer duration for event coverage

## Loading and Analyzing Examples

### Load Sample Data

```python
import pandas as pd
from scripts.married_put_calculator import MarriedPut

# Load sample positions
df = pd.read_csv('examples/sample_positions.csv')

# Create position objects
positions = []
for _, row in df.iterrows():
    pos = MarriedPut(
        stock_price=row['stock_price'],
        shares=row['shares'],
        put_strike=row['put_strike'],
        put_premium=row['put_premium'],
        days_to_expiration=row['days_to_expiration']
    )
    positions.append({
        'ticker': row['ticker'],
        'position': pos,
        'iv_level': row['iv_level'],
        'scenario': row['scenario_description']
    })

# Analyze each position
for p in positions:
    print(f"\n{p['ticker']} - {p['iv_level'].upper()} IV")
    print(f"Scenario: {p['scenario']}")
    print(f"Breakeven: ${p['position'].breakeven_price:.2f}")
    print(f"Max Loss: ${p['position'].max_loss:.2f}")
    print(f"Protection: {p['position'].protection_percentage:.1f}%")
```

### Compare Across Volatility Levels

```python
# Group by volatility level
iv_groups = df.groupby('iv_level').agg({
    'put_premium': 'mean',
    'iv_pct': 'mean'
})

print("\nAverage Premium by Volatility Level:")
print(iv_groups)
```

### Calculate Portfolio Metrics

```python
from scripts.position_sizer import portfolio_heat_check

# Calculate total portfolio risk
positions_data = []
for _, row in df.iterrows():
    pos = MarriedPut(
        stock_price=row['stock_price'],
        shares=row['shares'],
        put_strike=row['put_strike'],
        put_premium=row['put_premium']
    )
    positions_data.append({
        'name': row['ticker'],
        'max_loss': pos.max_loss
    })

# Check portfolio heat
total_investment = df['investment_size'].sum()
heat = portfolio_heat_check(positions_data, total_investment, max_portfolio_risk=0.02)

print(f"\nPortfolio Heat Analysis:")
print(f"Total Investment: ${total_investment:,.2f}")
print(f"Total Risk: ${heat['total_risk_dollars']:,.2f}")
print(f"Portfolio Heat: {heat['total_heat_pct']:.2f}%")
print(f"Risk Status: {heat['risk_status']}")
```

## Investment Size Guidelines

These examples follow the $5,000 to $50,000 position sizing requirements:

- **Small positions** ($5K-$10K): SNAP, COST
- **Medium positions** ($10K-$15K): TDOC, PLTR
- **Larger positions** ($15K-$50K): MRNA

Each position represents standard 100-share option contracts (or multiples thereof).

## Volatility Impact on Premiums

Notice how implied volatility affects premium costs:

| IV Level | Avg IV | Avg Premium % of Stock |
|----------|--------|------------------------|
| Low      | 18%    | 3.5%                   |
| Medium   | 35%    | 7.0%                   |
| High     | 55%    | 8.0%                   |

**Key Insight**: High IV stocks require accepting either:
1. Higher premium costs for ATM protection, or
2. Wider protection gaps (deeper OTM puts) to control costs

## Usage Recommendations

### For Beginners
Start with **Position 1 (COST)** or **Position 3 (TDOC)**:
- Lower volatility = more predictable costs
- Standard protection levels
- Easy to understand P/L scenarios

### For Advanced Traders
Study **Position 2 (SNAP)** and **Position 5 (MRNA)**:
- High-volatility cost management
- Strike selection in elevated IV
- Trade-offs between cost and protection

### For Portfolio Managers
Analyze the **complete portfolio** across all 5 positions:
- Diversification benefits
- Total risk exposure
- Correlation effects on protection costs

## Additional Resources

See [reference.md](../reference.md) for:
- Greeks calculations for these positions
- Volatility analysis frameworks
- Advanced scenario modeling

## Data Sources

Position data represents realistic scenarios based on:
- Actual market volatility levels (2023-2024)
- Real option chain pricing structures
- Standard market conventions
- Typical retail investor position sizes

*Note: Prices and IV levels are illustrative examples. Always use current market data for actual trading decisions.*
