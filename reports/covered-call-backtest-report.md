# Covered Call Strategy Backtest

## Quick Start

Run backtest on AAPL historical data (last 2 years):

```bash
python examples/backtest_covered_call.py
```

## Sample Results

**Performance (AAPL 2023-2025):**
- Initial Capital: $100,000
- Final Equity: $143,164
- **Total Return: 43.16%**
- **Premium Collected: $30,049 (30% of capital)**
- Sharpe Ratio: 0.80
- Max Drawdown: -32.41%

**Trade Statistics:**
- Total Trades: 24 (monthly cycles)
- Assignments: 9 (37.5%)
- Expired OTM: 14 (58.3%)
- Avg Premium/Trade: $1,252
- Win Rate: 60.9%

## Strategy Configuration

Default parameters:
```python
initial_capital=100000.0      # Starting capital
shares_per_contract=100       # Standard options contract
strike_otm_pct=0.05          # 5% OTM strikes
days_to_expiration=30        # Monthly cycles
risk_free_rate=0.03          # 3% risk-free rate
assumed_iv=0.25              # 25% implied volatility
```

## Backtest Methodology

### Position Management
1. **Entry**: Sell covered call at 5% OTM strike every month
2. **Exit**: Close at expiration (30 days)
3. **Assignment**: If ITM at expiration, stock called away at strike
4. **Rollover**: Immediately open new position after expiration

### Pricing Model
- Uses Black-Scholes for option pricing
- Greeks calculated for each position
- Realistic execution assumptions

### Capital Allocation
- Maximum shares = floor(capital / stock_price / 100) * 100
- Number of contracts = shares / 100
- Remaining cash held as reserve

## Output Files

Results saved to `data/backtest_results/`:

### 1. Equity Curve CSV
Daily portfolio value tracking:
- Stock value
- Cash balance
- Total equity
- Returns
- Drawdown

### 2. Trades CSV
Individual trade records:
- Entry/exit dates and prices
- Strike price and premium
- Delta at entry
- Outcome (assigned/expired)
- P&L per trade

### 3. Summary TXT
High-level performance metrics:
- Total return
- Premium collected
- Sharpe ratio
- Max drawdown
- Trade statistics

## Customization

### Run with Different Parameters

```python
from examples.backtest_covered_call import CoveredCallBacktest

backtest = CoveredCallBacktest(
    initial_capital=50000.0,
    strike_otm_pct=0.10,        # 10% OTM (more conservative)
    days_to_expiration=45,      # 45-day cycles
    assumed_iv=0.30,            # 30% IV (higher premium)
)

results = backtest.run_backtest(data, symbol="AAPL")
```

### Run on Different Stocks

```python
import pandas as pd

# Load different stock
data = pd.read_csv("data/historical/MSFT_historical.csv",
                   index_col=0, parse_dates=True)

# Run backtest
backtest = CoveredCallBacktest()
results = backtest.run_backtest(data, symbol="MSFT")
```

## Key Insights

### When Covered Calls Work Best
✓ **Sideways/Slightly Bullish Markets**: AAPL 2023-2024 was ideal
✓ **High Volatility**: Higher IV = larger premiums
✓ **Large Cap Stocks**: Better option liquidity

### Risks Observed
✗ **Capped Upside**: Missed gains when stock rallied >5%
✗ **Drawdowns**: Still exposed to stock decline
✗ **Assignment Risk**: Stock called away 37.5% of the time

### Premium Collection
- **30% of capital** collected in premiums over 2 years
- **15% annualized** premium yield
- Provides downside cushion equivalent to 5% decline

## Next Steps

### Enhance Backtest
1. **Variable Strikes**: Adjust OTM% based on volatility
2. **Early Exit**: Buy back when option drops to 80% profit
3. **Rolling**: Roll up/out when approaching assignment
4. **Multiple Underlyings**: Diversify across stocks

### Compare Strategies
- Buy-and-hold AAPL vs Covered Call
- Monthly vs Weekly vs 45-day cycles
- Different OTM percentages (3%, 5%, 10%)

### Live Trading Integration
- Connect to RiskGuard for position sizing
- Use OptionsCore for real-time Greeks
- Integrate with FlowRoute for execution

## References

- Strategy: `src/strategies/options/covered_call.py`
- Tests: `tests/test_strategies/test_options/test_covered_call.py`
- Examples: `examples/covered_call_*.py`
- OptionsCore: `src/engines/optionscore/`

---

**Generated**: 2025-12-10
**Backtest Period**: 2023-12-07 to 2025-12-10 (2 years)
**Data Source**: Yahoo Finance via yfinance
