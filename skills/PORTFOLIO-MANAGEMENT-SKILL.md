# Portfolio Management Skill

## Installation

The portfolio-management skill has been created and packaged. To use it:

1. Download the skill file: `portfolio-management.skill` (available in Claude outputs)
2. Import it into Claude via Settings > Skills > Import Skill
3. The skill will be immediately available for use

## Skill Location

The complete skill package including all scripts and references is available for download from the current Claude session outputs.

## What's Included

### 5 Core Scripts

1. **broker_connector.py** - Connect to Alpaca, Interactive Brokers, TD Ameritrade
   - Unified API interface for multiple brokers
   - Support for paper and live trading accounts
   - Position, order, and transaction history retrieval

2. **market_data.py** - Fetch quotes and historical data
   - Yahoo Finance (free, no API key required)
   - Alpha Vantage (API key required)
   - Polygon.io (API key required)
   - Real-time quotes and historical OHLCV data

3. **portfolio_analytics.py** - Performance and risk analytics
   - Total return, annualized return, alpha
   - Sharpe ratio, Sortino ratio, beta, VaR, CVaR
   - Sector allocation and drift analysis
   - Position sizing alerts

4. **tax_tracker.py** - Tax lot accounting and optimization
   - Multiple lot accounting methods (FIFO, LIFO, HIFO, Specific ID, Average)
   - Automatic wash sale detection (IRS 30-day rule)
   - Short-term vs long-term classification
   - IRS Form 8949 export

5. **dividend_tracker.py** - Dividend income analysis
   - Qualified vs non-qualified dividend tracking
   - Dividend growth rate calculation (CAGR)
   - Dividend Aristocrat scoring
   - Future income projections
   - 1099-DIV export

### 3 Reference Files

1. **formulas.md** - Mathematical formulas for all calculations
   - Performance metrics (Total return, CAGR, TWR, MWR)
   - Risk metrics (Volatility, Sharpe, Sortino, beta, VaR, CVaR)
   - Alpha metrics (Jensen's alpha, Treynor ratio, M²)
   - Tax calculations and dividend metrics

2. **benchmarks.md** - Major market indices and benchmark information
   - US Equity: S&P 500, NASDAQ, Russell 2000, Dow Jones
   - International: MSCI World, MSCI EAFE, MSCI EM
   - Fixed Income: Bloomberg Aggregate, Treasury, High Yield
   - Commodities, Sector, and Factor benchmarks

3. **sectors.md** - GICS sector classification system
   - 11 GICS sectors with detailed descriptions
   - Sector rotation strategies by economic cycle
   - Sector characteristics (cyclicality, volatility, growth)
   - Symbol-to-sector mappings

## Key Features

### Broker Integration
- Connect to multiple brokers simultaneously
- Retrieve real-time positions and account balances
- Access order history and transaction data
- Paper and live trading account support

### Market Data
- Real-time quotes for any symbol
- Historical data with multiple timeframes (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
- Dividend history retrieval
- Free and premium data provider options

### Performance Analytics
- Performance metrics: total return, annualized return, YTD/MTD, alpha
- Risk metrics: Sharpe, Sortino, Calmar ratios, beta, VaR, CVaR
- Benchmark comparison and tracking error
- Win rate and drawdown analysis

### Tax Optimization
- 5 lot accounting methods for optimal tax treatment
- Automatic IRS wash sale detection
- Realized vs unrealized gains tracking
- Tax form export (Form 8949, 1099-DIV)

### Allocation Monitoring
- Sector and asset class allocation tracking
- Target vs actual drift monitoring
- Concentration risk analysis (Herfindahl index)
- Customizable position sizing alerts

### Dividend Management
- Income tracking with tax classification
- Yield and growth rate calculations
- Dividend Aristocrat identification
- Future income projections
- Payment calendar

## Quick Start Examples

### Example 1: Connect to Broker

```python
from scripts.broker_connector import create_connector

connector = create_connector('alpaca')
if connector.connect():
    account = connector.get_account()
    print(f"Total Value: ${account.total_value:,.2f}")
```

### Example 2: Calculate Performance vs S&P 500

```python
from scripts.market_data import create_provider
from scripts.portfolio_analytics import PortfolioAnalytics, calculate_returns_from_prices

provider = create_provider('yahoo')
spy = provider.get_historical('SPY', start, end)
spy_returns = calculate_returns_from_prices(spy.data['close'])

analytics = PortfolioAnalytics()
perf = analytics.calculate_performance(portfolio_returns, spy_returns)

print(f"Alpha: {perf.alpha:.2%}")
print(f"Sharpe Ratio: {risk.sharpe_ratio:.2f}")
```

### Example 3: Tax Loss Harvesting

```python
from scripts.tax_tracker import TaxTracker, LotMethod

tracker = TaxTracker(method=LotMethod.HIFO)
unrealized = tracker.get_unrealized_gains(current_prices)
losses = unrealized[unrealized['unrealized_gain'] < 0]

# Export for tax filing
form8949 = tracker.export_form8949(year=2024)
```

### Example 4: Dividend Income Planning

```python
from scripts.dividend_tracker import DividendTracker

tracker = DividendTracker()
summary = tracker.calculate_summary()
projections = tracker.project_future_income(years=5, growth_rate=0.05)

print(f"Projected Annual Income: ${summary.projected_annual_income:,.2f}")
```

## Dependencies

Install required packages:

```bash
pip install pandas numpy alpaca-trade-api ib_insync yfinance alpha_vantage polygon-api-client
```

## API Keys

Set environment variables for your API keys:

```bash
# Broker APIs
export BROKER_API_KEY="your_alpaca_key"
export BROKER_API_SECRET="your_alpaca_secret"

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
5. Review dividend calendar

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

1. Cache market data to avoid repeated API calls
2. Use batch requests for multiple quotes
3. Monitor rate limits for each API provider
4. Start with paper trading accounts for testing
5. Validate data and handle gaps appropriately
6. Keep credentials secure in environment variables
7. Regular backups of tax lot data
8. Document lot selection for tax purposes

## Support and Documentation

For comprehensive documentation, see the full SKILL.md file within the imported skill, which includes:
- Detailed script API documentation
- Advanced use cases and integration examples
- Custom metric implementations
- Troubleshooting guides
- Extension patterns for adding new features

---

**Created**: December 2024  
**Version**: 1.0  
**Project**: Intelligent Investor  
**Requires**: Python 3.8+, pandas, numpy
