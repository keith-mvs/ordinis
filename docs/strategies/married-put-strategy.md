# Married Put Strategy - Implementation Guide

Comprehensive implementation of the married put protective options strategy for the Ordinis trading platform.

## Overview

A **married put** combines a long stock position with a long protective put option, providing downside insurance while maintaining unlimited upside potential. This is the options equivalent of purchasing insurance on your stock holdings.

### Strategy Profile

- **Type**: Protective/Hedging Strategy
- **Market Outlook**: Bullish but cautious
- **Max Loss**: Limited to (Stock Price - Put Strike + Put Premium)
- **Max Gain**: Unlimited (stock can rise indefinitely)
- **Break-even**: Stock Price + Put Premium paid
- **Risk Level**: Conservative (protected downside, unlimited upside)

## Components

### 1. Strategy Class (`src/strategies/options/married_put.py`)

Core implementation with opportunity analysis and P&L calculations.

**Key Methods**:
- `analyze_opportunity()` - Evaluate protective put scenarios
- `calculate_payoff()` - Calculate P&L at different stock prices
- `compare_strikes()` - Compare multiple put strike prices

**Configuration Parameters**:
```python
MarriedPutStrategy(
    name="Conservative Protection",
    min_protection_pct=0.05,   # Minimum 5% downside protection
    max_protection_pct=0.15,   # Maximum 15% downside protection
    max_premium_pct=0.05,      # Max 5% premium cost as % of stock
    min_delta=0.30,            # Minimum put delta (OTM)
    max_delta=0.70,            # Maximum put delta (ITM)
    days_to_expiration=45,     # Target 45-day options
    dte_tolerance=15,          # Allow ±15 days variance
    transaction_cost=0.65      # Per contract fee
)
```

### 2. Demo Script (`examples/married_put_demo.py`)

Interactive demonstration showing three scenarios:

**Scenario 1: Conservative Protection (5% OTM Put)**
```
Stock Price:           $100.00
Put Strike:            $95.00 (5% below)
Put Premium:           $2.00
Protection:            5.0% downside
Annual Cost:           16.2%
Max Loss:              $700.65 (6.9%)
Meets Criteria:        YES
```

**Scenario 2: Moderate Protection (ATM Put)**
```
Stock Price:           $100.00
Put Strike:            $100.00 (at price)
Put Premium:           $3.50
Protection:            0.0% (immediate)
Annual Cost:           28.4%
Max Loss:              $350.65 (3.4%)
Meets Criteria:        NO (higher cost)
```

**Scenario 3: Maximum Protection (10% ITM Put)**
```
Stock Price:           $100.00
Put Strike:            $110.00 (10% above)
Put Premium:           $12.00
Protection:            Guaranteed exit at $110
Annual Cost:           97.3%
Max Loss:              $200.65 (1.8%)
Meets Criteria:        NO (exceeds max premium)
```

### 3. Backtest Engine (`examples/backtest_married_put.py`)

Historical simulation with rolling protective puts.

**AAPL Results (2023-2025, 2 years)**:
```
Initial Capital:       $100,000.00
Final Equity:          $115,738.41
Total Return:          15.74%
Premium Paid:          $20,334.93 (20.33%)
Max Drawdown:          -24.38%
Sharpe Ratio:          0.48

Comparison:
Buy-and-Hold Return:   44.69%
Buy-and-Hold Final:    $144,686.82
Protection Cost:       28.95%

Trade Statistics:
Total Puts:            17
Expired Worthless:     13 (76.5%)
Protection Events:     3 (17.6%)
Avg Premium/Put:       $1,196.17
Annual Protection Cost: 10.17%
```

### 4. Unit Tests (`tests/test_strategies/test_options/test_married_put.py`)

Comprehensive test coverage (17 tests, 100% pass rate):
- Configuration validation
- Strategy initialization
- Opportunity analysis (OTM, ATM, ITM puts)
- P&L calculations
- Strike comparisons
- Cost metrics

## Usage Examples

### Basic Analysis

```python
from src.strategies.options.married_put import MarriedPutStrategy

# Create strategy
strategy = MarriedPutStrategy(name="AAPL Protection")

# Analyze opportunity
result = strategy.analyze_opportunity(
    stock_price=100.00,
    put_strike=95.00,      # 5% OTM
    put_premium=2.00,
    put_delta=-0.35,
    days_to_expiration=45,
    shares=100
)

print(f"Breakeven: ${result['breakeven']:.2f}")
print(f"Max Loss: ${result['max_loss']:.2f}")
print(f"Protection: {result['protection_pct']*100:.1f}%")
print(f"Meets Criteria: {result['meets_criteria']}")
```

### Calculate P&L

```python
# Calculate profit/loss at various prices
payoff = strategy.calculate_payoff(
    stock_entry=100.00,
    current_stock_price=85.00,  # Stock down 15%
    put_strike=95.00,
    put_premium=2.00,
    shares=100
)

print(f"Stock P/L: ${payoff['stock_pl']:.2f}")
print(f"Put P/L: ${payoff['put_pl']:.2f}")
print(f"Total P/L: ${payoff['total_pl']:.2f}")
print(f"Protection Active: {payoff['protection_active']}")
```

### Compare Strikes

```python
# Compare multiple strike prices
strikes_data = [
    (90.00, 1.00, -0.25),   # Far OTM
    (95.00, 2.00, -0.35),   # OTM
    (100.00, 3.50, -0.50),  # ATM
]

comparison = strategy.compare_strikes(
    stock_price=100.00,
    strikes_and_premiums=strikes_data,
    days_to_expiration=45,
    shares=100
)

for result in comparison:
    print(f"Strike ${result['put_strike']:.2f}: "
          f"Cost {result['premium_pct']*100:.1f}%, "
          f"Protection {result['protection_pct']*100:.1f}%")
```

### Run Demo

```bash
python examples/married_put_demo.py
```

### Run Backtest

```bash
python examples/backtest_married_put.py
```

## Key Metrics Explained

### Protection Percentage
Distance between stock price and put strike:
- **5% OTM**: Protection starts at 5% decline
- **ATM**: Immediate protection from any decline
- **ITM**: Guaranteed minimum exit price above current

### Annualized Cost
Premium cost extrapolated to annual rate:
```
Annual Cost = (Premium / Stock Price) × (365 / Days to Expiration)
```

Example: $2 premium on $100 stock for 45 days = 16.2% annualized

### Breakeven
Stock price needed to break even:
```
Breakeven = Stock Price + Put Premium + (Transaction Cost / Shares)
```

### Maximum Loss
Worst-case loss if stock crashes:
```
Max Loss = (Stock Price - Put Strike + Put Premium) × Shares
```

## Strike Selection Guidelines

### OTM Puts (5-10% below stock price)
**Pros**:
- Lower premium cost (2-3% of stock value)
- Less drag on returns
- Suitable for high-conviction positions

**Cons**:
- Gap between price and protection
- Doesn't protect against small declines
- Need larger move for protection to activate

**Best for**: Long-term investors, cost-conscious protection

### ATM Puts (at current stock price)
**Pros**:
- Immediate protection from any decline
- Balanced cost/protection ratio
- Good psychological comfort

**Cons**:
- Moderate premium cost (3-5% of stock value)
- Need stock to rise by premium to profit
- Higher theta decay

**Best for**: Uncertain markets, moderate risk tolerance

### ITM Puts (5-10% above stock price)
**Pros**:
- Guaranteed minimum exit price
- Maximum downside protection
- Built-in profit floor

**Cons**:
- Very expensive (10-15% of stock value)
- Significantly reduces upside potential
- Only for extreme risk aversion

**Best for**: Maximum protection needs, earnings uncertainty

## Backtest Insights

### Performance Analysis

The 2-year AAPL backtest (2023-2025) reveals:

1. **Protection Cost**: 28.95% underperformance vs buy-and-hold
   - Annual protection cost: ~10.17%
   - 17 puts purchased, rolling every 45 days
   - $20,334 total premium paid on $100k capital

2. **Risk Reduction**: Drawdown limited to -24.38%
   - Without protection: potentially worse during crashes
   - 3 times protection prevented larger losses
   - 76.5% of puts expired worthless (cost of insurance)

3. **Trade-offs**:
   - Sacrificed 29% return for downside protection
   - Sharpe ratio: 0.48 (risk-adjusted returns)
   - Effective in volatile/declining markets
   - Drag on returns in strong bull markets

### When to Use Married Puts

**Good scenarios**:
- Protecting unrealized gains (tax loss harvesting alternative)
- Holding through earnings announcements
- Uncertain market conditions
- Compliance with risk limits
- New positions in volatile stocks

**Poor scenarios**:
- Strong bull markets (protection rarely needed)
- Long-term buy-and-hold (cost compounds)
- Low-volatility environments (expensive insurance)
- Small accounts (transaction costs significant)

## Cost Analysis

### Real-World Example: $50,000 Position

**Stock**: Trading at $100/share (500 shares)

**45-day 5% OTM Put Protection**:
- Strike: $95
- Premium: $2.00/share
- Total cost: $1,000 (2% of position)
- Annualized: ~16% of position value

**Rolling Strategy (1 year)**:
- 8 rolls per year (45-day cycles)
- Total premium: ~$8,000 (16% drag)
- Protection maintained continuously

**Breakeven Requirement**:
- Stock must gain 16% annually just to cover protection
- Any gains above 16% = net profit
- Losses limited to 7% max (5% gap + 2% premium)

## Files Structure

```
src/strategies/options/
├── __init__.py                  # Module exports
├── married_put.py               # Strategy implementation
└── covered_call.py              # Related strategy

examples/
├── married_put_demo.py          # Interactive scenarios
├── backtest_married_put.py      # Historical backtest
└── README_MARRIED_PUT.md        # This file

tests/test_strategies/test_options/
└── test_married_put.py          # 17 unit tests

data/backtest_results/
├── AAPL_married_put_equity_*.csv     # Equity curve
├── AAPL_married_put_trades_*.csv     # Trade log
└── AAPL_married_put_summary_*.txt    # Performance summary
```

## Integration with Ordinis

### Strategy Registration

```python
from src.strategies.options import MarriedPutStrategy

# Create and configure
strategy = MarriedPutStrategy(
    name="Portfolio Protection",
    min_protection_pct=0.05,
    max_premium_pct=0.04
)

# Analyze opportunities
result = strategy.analyze_opportunity(...)
if result["meets_criteria"]:
    # Execute trade
    pass
```

### With OptionsCore Engine

```python
from src.engines.optionscore import OptionsCoreEngine
from src.strategies.options import MarriedPutStrategy

# Initialize engines
options_engine = OptionsCoreEngine(config)
strategy = MarriedPutStrategy(name="Protection")

# Fetch live options chain
chain = await options_engine.get_option_chain(
    symbol="AAPL",
    contract_type="put"
)

# Filter and analyze puts matching criteria
for contract in chain.puts:
    result = strategy.analyze_opportunity(
        stock_price=chain.underlying_price,
        put_strike=contract.strike,
        put_premium=contract.ask,
        put_delta=contract.delta,
        days_to_expiration=contract.days_to_expiration
    )

    if result["meets_criteria"]:
        print(f"Found opportunity: Strike ${result['put_strike']}")
```

## Comparison: Married Put vs Covered Call

| Aspect | Married Put | Covered Call |
|--------|-------------|--------------|
| **Direction** | Bullish + Protection | Neutral to Slightly Bullish |
| **Max Profit** | Unlimited | Limited to Strike + Premium |
| **Max Loss** | Limited | Unlimited (if stock crashes) |
| **Premium** | Pay (cost) | Collect (income) |
| **Purpose** | Downside protection | Income generation |
| **Best Market** | Volatile/Uncertain | Sideways/Slow rise |
| **Annual Cost** | 10-20% drag | 10-20% income |

## Testing

Run all married put tests:
```bash
pytest tests/test_strategies/test_options/test_married_put.py -v
```

Expected: 17 tests passing

## References

### Related Strategies
- **Protective Put**: Same as married put (different name)
- **Collar**: Married put + covered call (limits upside and downside)
- **Cash-Secured Put**: Sell put to acquire stock below market

### OptionsCore Integration
- Black-Scholes pricing: `src/engines/optionscore/pricing/black_scholes.py`
- Greeks calculation: `src/engines/optionscore/pricing/greeks.py`
- Options chain: `src/engines/optionscore/core/enrichment.py`

### Skill Package
See `skills/married-put/` for standalone calculator and analysis tools from the married-put skill package.

---

**Last Updated**: 2025-12-11
**Version**: 1.0.0
**Status**: Production Ready
