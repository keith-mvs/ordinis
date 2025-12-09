# Portfolio Management Skill - Quick Reference

## Installation

Import the `portfolio-management.skill` file into Claude via the Skills menu.

## What's Included

### 5 Core Scripts

1. **broker_connector.py** - Connect to Alpaca, Interactive Brokers, TD Ameritrade
2. **market_data.py** - Fetch quotes and historical data (Yahoo Finance, Alpha Vantage, Polygon.io)
3. **portfolio_analytics.py** - Calculate performance, risk metrics, and allocation analysis
4. **tax_tracker.py** - Tax lot accounting, wash sales, IRS Form 8949 export
5. **dividend_tracker.py** - Dividend income tracking and future projections

### 3 Reference Files

1. **formulas.md** - Mathematical formulas for all calculations
2. **benchmarks.md** - Major market indices and benchmark information
3. **sectors.md** - GICS sector classification system

## Key Features

### Broker API Connectivity
- Connect to multiple brokers simultaneously
- Retrieve real-time positions and account balances
- Access order history and transaction data
- Support for paper and live trading accounts

### Market Data Integration
- Real-time quotes for any symbol
- Historical OHLCV data with multiple intervals
- Dividend history retrieval
- Free (Yahoo Finance) and premium (Alpha Vantage, Polygon) options

### Performance Analytics
- Total return, annualized return, YTD/MTD returns
- Alpha, tracking error, information ratio
- Win rate, max drawdown, recovery time
- Benchmark comparison and attribution

### Risk Metrics
- Volatility (annualized standard deviation)
- Sharpe ratio, Sortino ratio, Calmar ratio
- Beta, Value at Risk (VaR), Conditional VaR
- Downside deviation and risk-adjusted returns

### Allocation Analysis
- Sector and asset class allocation
- Target allocation vs actual drift monitoring
- Concentration risk (Herfindahl index)
- Top holdings analysis
- Position sizing alerts

### Tax Optimization
- Multiple lot accounting methods (FIFO, LIFO, HIFO, Specific ID)
- Automatic wash sale detection (30-day rule)
- Short-term vs long-term capital gains classification
- Unrealized gains by tax lot
- IRS Form 8949 export ready

### Dividend Income
- Qualified vs non-qualified dividend tracking
- Dividend yield and growth rate analysis
- Future income projections
- Dividend Aristocrat scoring
- Payment calendar and reminders
- 1099-DIV export for tax filing

## Getting Started

### Example 1: Basic Portfolio Analysis

```python
from scripts.broker_connector import create_connector
from scripts.portfolio_analytics import PortfolioAnalytics

# Connect and get positions
connector = create_connector('alpaca')
connector.connect()
account = connector.get_account()

print(f"Total Value: ${account.total_value:,.2f}")
```

### Example 2: Performance vs S&P 500

```python
from scripts.market_data import create_provider
from scripts.portfolio_analytics import PortfolioAnalytics, calculate_returns_from_prices

# Get data
provider = create_provider('yahoo')
spy = provider.get_historical('SPY', start, end)
spy_returns = calculate_returns_from_prices(spy.data['close'])

# Calculate metrics
analytics = PortfolioAnalytics()
perf = analytics.calculate_performance(portfolio_returns, spy_returns)
risk = analytics.calculate_risk_metrics(portfolio_returns, spy_returns)

print(f"Alpha: {perf.alpha:.2%}")
print(f"Sharpe Ratio: {risk.sharpe_ratio:.2f}")
```

### Example 3: Tax Loss Harvesting

```python
from scripts.tax_tracker import TaxTracker, LotMethod

# Use HIFO to maximize losses
tracker = TaxTracker(method=LotMethod.HIFO)

# Import transactions and analyze
unrealized = tracker.get_unrealized_gains(current_prices)
losses = unrealized[unrealized['unrealized_gain'] < 0]

# Export for tax filing
form8949 = tracker.export_form8949(year=2024)
```

### Example 4: Dividend Income Planning

```python
from scripts.dividend_tracker import DividendTracker

tracker = DividendTracker()

# Analyze positions
for symbol in positions:
    metrics = tracker.analyze_position(symbol, shares, price, div_history)
    
# Project future income
projections = tracker.project_future_income(years=5, growth_rate=0.05)

# Get high yielders
high_yield = tracker.get_high_yield_positions(min_yield=0.04)
```

## Dependencies

Install required packages:

```bash
pip install pandas numpy alpaca-trade-api ib_insync yfinance alpha_vantage polygon-api-client
```

## API Keys Setup

Set environment variables for your API keys:

```bash
# Broker APIs
export BROKER_API_KEY="your_alpaca_key"
export BROKER_API_SECRET="your_alpaca_secret"
export TDA_REFRESH_TOKEN="your_tda_token"

# Market Data APIs
export ALPHA_VANTAGE_API_KEY="your_av_key"
export POLYGON_API_KEY="your_polygon_key"
```

Or use a `.env` file with python-dotenv.

## Common Workflows

### Daily Portfolio Monitor
1. Connect to broker(s) and retrieve positions
2. Fetch current prices from market data provider
3. Calculate performance vs benchmark
4. Check position sizing alerts
5. Review dividend calendar for upcoming payments

### End of Month Review
1. Calculate MTD and YTD performance
2. Analyze sector allocation drift
3. Review top holdings concentration
4. Check tax lots for harvesting opportunities
5. Project dividend income for quarter

### Tax Season Preparation
1. Export realized gains via Form 8949
2. Export dividend income via 1099-DIV format
3. Review wash sales and disallowed losses
4. Calculate tax liability estimates
5. Optimize lot selection for Q4 sales

### Quarterly Rebalancing
1. Calculate current allocation vs target
2. Identify drift exceeding threshold (5%)
3. Calculate rebalancing trades needed
4. Consider tax implications of sales
5. Execute rebalancing strategy

## Tips and Best Practices

1. **Cache market data** - Historical data doesn't change, cache it to avoid repeated API calls
2. **Use batch requests** - Get multiple quotes at once instead of individual calls
3. **Monitor rate limits** - Different providers have different limits, add delays if needed
4. **Start with paper trading** - Test broker integrations with paper accounts first
5. **Validate data** - Check for missing data, handle gaps appropriately
6. **Keep credentials secure** - Use environment variables, never commit API keys
7. **Regular backups** - Export tax lot data regularly in case of data loss
8. **Document lot selection** - Keep records of which lots you sold for tax purposes

## Troubleshooting

**Connection Failed**: Check API keys, network connectivity, and rate limits
**Missing Data**: Use different time periods or data providers
**Incorrect Calculations**: Verify date alignment between portfolio and benchmark
**Wash Sale Issues**: Review 30-day window around sales, ensure no duplicate purchases

## Support

Reference the comprehensive SKILL.md documentation within the skill for:
- Detailed script documentation
- Advanced use cases
- Custom metric implementations
- Integration examples
- Complete API references

---

**Created**: December 2024
**Version**: 1.0
**Requires**: Python 3.8+, pandas, numpy
