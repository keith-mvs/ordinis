# ProofBench - Backtesting Engine Guide

## Overview

ProofBench is an event-driven simulation engine for backtesting trading strategies with realistic execution modeling and comprehensive performance analytics.

## Features

### Event-Driven Architecture
- Priority-based event queue for deterministic simulation
- Multiple event types: market data, orders, fills, positions
- Proper event ordering ensures realistic simulation flow

### Realistic Execution Modeling
- **Slippage Model** with three components:
  - Fixed component (bid-ask spread)
  - Variable component (market impact based on volume)
  - Volatility component (based on bar volatility)
- **Commission Models**:
  - Per-share commission
  - Flat per-trade commission
  - Percentage-based commission
- **Order Types**:
  - Market orders
  - Limit orders
  - Stop orders
  - Stop-limit orders

### Portfolio Management
- Real-time position tracking
- Mark-to-market updates
- Realized and unrealized P&L
- Cash management
- Trade history

### Performance Analytics
- **Returns**: Total return, annualized return (CAGR)
- **Risk Metrics**: Volatility, downside deviation
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown Analysis**: Max drawdown, average drawdown, drawdown duration
- **Trade Statistics**: Win rate, profit factor, average win/loss, expectancy

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy
```

### 2. Basic Usage

```python
import pandas as pd
from src.engines.proofbench import (
    SimulationEngine,
    SimulationConfig,
    Order,
    OrderSide,
    OrderType,
)

# Configure simulation
config = SimulationConfig(
    initial_capital=100000.0,
    bar_frequency="1d",
)

# Create engine
engine = SimulationEngine(config)

# Load historical data (OHLCV DataFrame with datetime index)
engine.load_data("AAPL", ohlcv_data)

# Define strategy
def my_strategy(engine, symbol, bar):
    # Your trading logic here
    pos = engine.get_position(symbol)

    if some_buy_condition:
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        engine.submit_order(order)

# Run backtest
engine.set_strategy(my_strategy)
results = engine.run()

# Analyze results
print(f"Total Return: {results.metrics.total_return:.2f}%")
print(f"Sharpe Ratio: {results.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {results.metrics.max_drawdown:.2f}%")
```

### 3. Access Results

```python
# Performance metrics
metrics = results.metrics
print(f"Annualized Return: {metrics.annualized_return:.2f}%")
print(f"Volatility: {metrics.volatility:.2f}%")
print(f"Win Rate: {metrics.win_rate:.2f}%")
print(f"Profit Factor: {metrics.profit_factor:.2f}")

# Equity curve (DataFrame)
equity_df = results.equity_curve

# Trade history (DataFrame)
trades_df = results.trades

# Final portfolio state
portfolio = results.portfolio
print(f"Cash: ${portfolio.cash:,.2f}")
print(f"Positions: {portfolio.num_positions}")
```

## Configuration

### Simulation Configuration

```python
config = SimulationConfig(
    initial_capital=100000.0,          # Starting capital
    bar_frequency="1d",                 # Bar frequency (1d, 1h, etc.)
    record_equity_frequency=1,          # Record equity every N bars
    risk_free_rate=0.02,               # Annual risk-free rate (2%)
)
```

### Execution Configuration

```python
from src.engines.proofbench import ExecutionConfig

exec_config = ExecutionConfig(
    estimated_spread=0.001,             # Bid-ask spread (0.1%)
    impact_coefficient=0.1,             # Market impact factor
    volatility_factor=0.5,              # Volatility slippage factor
    max_slippage=0.01,                  # Maximum slippage (1%)
    commission_per_share=0.01,          # $0.01 per share
    commission_per_trade=0.0,           # No flat commission
    commission_pct=0.001,               # 0.1% of trade value
    min_commission=0.0,                 # No minimum
)

config = SimulationConfig(
    initial_capital=100000.0,
    execution_config=exec_config,
)
```

## Strategy Development

### Strategy Callback

Your strategy is a callback function that receives:
- `engine`: The simulation engine instance
- `symbol`: Current symbol being processed
- `bar`: Current OHLCV bar data

```python
def my_strategy(engine, symbol, bar):
    """
    Args:
        engine: SimulationEngine instance
        symbol: str - stock symbol
        bar: Bar object with open, high, low, close, volume, timestamp
    """
    # Access current position
    pos = engine.get_position(symbol)

    # Access portfolio state
    cash = engine.get_cash()
    equity = engine.get_equity()

    # Submit orders
    if buy_condition:
        order = Order(
            symbol=symbol,
            side=OrderSide.BUY,
            quantity=100,
            order_type=OrderType.MARKET,
        )
        engine.submit_order(order)
```

### Order Types

#### Market Order
```python
order = Order(
    symbol="AAPL",
    side=OrderSide.BUY,
    quantity=100,
    order_type=OrderType.MARKET,
)
```

#### Limit Order
```python
order = Order(
    symbol="AAPL",
    side=OrderSide.BUY,
    quantity=100,
    order_type=OrderType.LIMIT,
    limit_price=150.0,
)
```

#### Stop Order
```python
order = Order(
    symbol="AAPL",
    side=OrderSide.SELL,
    quantity=100,
    order_type=OrderType.STOP,
    stop_price=145.0,
)
```

### Strategy Examples

#### Simple Moving Average Crossover
```python
def sma_crossover(engine, symbol, bar):
    if not hasattr(engine, 'prices'):
        engine.prices = []

    engine.prices.append(bar.close)

    if len(engine.prices) < 50:
        return

    sma_20 = sum(engine.prices[-20:]) / 20
    sma_50 = sum(engine.prices[-50:]) / 50

    pos = engine.get_position(symbol)

    # Golden cross: buy signal
    if sma_20 > sma_50 and (pos is None or pos.quantity == 0):
        cash_to_invest = engine.get_equity() * 0.9
        quantity = int(cash_to_invest / bar.close)

        if quantity > 0:
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=quantity,
                order_type=OrderType.MARKET,
            )
            engine.submit_order(order)

    # Death cross: sell signal
    elif sma_20 < sma_50 and pos and pos.quantity > 0:
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=pos.quantity,
            order_type=OrderType.MARKET,
        )
        engine.submit_order(order)
```

## Performance Metrics

### Returns
- **total_return**: Total return from start to end (%)
- **annualized_return**: Compound annual growth rate (%)

### Risk Metrics
- **volatility**: Annualized volatility of returns (%)
- **downside_deviation**: Downside deviation (semi-std) (%)

### Risk-Adjusted Returns
- **sharpe_ratio**: (Return - Risk-free) / Volatility
- **sortino_ratio**: (Return - Risk-free) / Downside Deviation
- **calmar_ratio**: Annualized Return / Max Drawdown

### Drawdown Metrics
- **max_drawdown**: Maximum peak-to-trough decline (%)
- **avg_drawdown**: Average drawdown during drawdown periods (%)
- **max_drawdown_duration**: Longest drawdown period (days)

### Trade Statistics
- **num_trades**: Total number of completed trades
- **win_rate**: Percentage of profitable trades
- **profit_factor**: Gross profit / Gross loss
- **avg_win**: Average profit on winning trades ($)
- **avg_loss**: Average loss on losing trades ($)
- **largest_win**: Largest single win ($)
- **largest_loss**: Largest single loss ($)
- **expectancy**: Average P&L per trade ($)

## Data Requirements

### Required Columns
Your data must have these columns:
- `open`: Opening price
- `high`: High price
- `low`: Low price
- `close`: Closing price
- `volume`: Trading volume

### Index
- Must be a pandas DatetimeIndex
- Should be sorted chronologically

### Example
```python
import pandas as pd

dates = pd.date_range("2024-01-01", periods=100, freq="1d")
data = pd.DataFrame({
    "open": [...],
    "high": [...],
    "low": [...],
    "close": [...],
    "volume": [...],
}, index=dates)

engine.load_data("AAPL", data)
```

## Best Practices

1. **Avoid Lookahead Bias**: Only use data available at the time of the bar
2. **Position Sizing**: Always check cash availability before trading
3. **Risk Management**: Implement stop losses and position limits
4. **Transaction Costs**: Configure realistic commission and slippage
5. **Data Quality**: Ensure clean, survivorship-bias-free data
6. **Walk-Forward Testing**: Use separate in-sample/out-of-sample periods
7. **Multiple Symbols**: Test on diverse assets and time periods

## Common Pitfalls

### Lookahead Bias
```python
# BAD: Using future data
def bad_strategy(engine, symbol, bar):
    # This would use bar.close which hasn't happened yet!
    if bar.close > bar.open:  # OK
        pass
```

### Insufficient Capital
```python
# GOOD: Check cash before trading
def good_strategy(engine, symbol, bar):
    quantity = 1000
    cost = quantity * bar.close

    if engine.get_cash() >= cost:
        order = Order(symbol=symbol, side=OrderSide.BUY, quantity=quantity)
        engine.submit_order(order)
```

### Overfitting
- Test on out-of-sample data
- Use walk-forward analysis
- Keep strategies simple
- Validate across multiple time periods

## Advanced Topics

### Multiple Symbols
```python
# Load data for multiple symbols
engine.load_data("AAPL", aapl_data)
engine.load_data("MSFT", msft_data)
engine.load_data("GOOGL", googl_data)

# Strategy receives each symbol separately
def multi_symbol_strategy(engine, symbol, bar):
    # Strategy logic for each symbol
    pass
```

### Portfolio Rebalancing
```python
def rebalance_strategy(engine, symbol, bar):
    if not hasattr(engine, 'last_rebalance'):
        engine.last_rebalance = bar.timestamp

    # Rebalance monthly
    days_since = (bar.timestamp - engine.last_rebalance).days
    if days_since >= 30:
        # Rebalancing logic
        engine.last_rebalance = bar.timestamp
```

## Testing

Run the test suite:
```bash
pytest tests/test_engines/test_proofbench/ -v
```

## Examples

See `docs/examples/proofbench_example.py` for complete working examples.

## Architecture

ProofBench uses an event-driven architecture:
1. **Event Queue**: Priority queue processes events chronologically
2. **Market Events**: Bar updates trigger strategy evaluation
3. **Order Events**: Strategy submits orders
4. **Fill Events**: Execution simulator fills orders realistically
5. **Portfolio Events**: Position and cash updated
6. **Analytics**: Performance calculated on equity curve and trades

This ensures:
- Deterministic behavior
- Realistic execution modeling
- Proper event ordering
- No lookahead bias

## License

Part of the Ordinis project.
