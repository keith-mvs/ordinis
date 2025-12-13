# Trading Strategies Library

Pre-built, production-ready trading strategies for the Intelligent Investor system.

## Overview

This library provides battle-tested trading strategies that can be easily integrated into your trading workflow. Each strategy is:

- **Ready to use** - Just instantiate and generate signals
- **Well-documented** - Complete descriptions and risk considerations
- **Flexible** - Customizable parameters for different markets
- **Type-safe** - Full type hints and validation

## Available Strategies

### 1. RSI Mean Reversion Strategy

**Best For:** Range-bound markets, counter-trend trading

Trades mean reversion using the Relative Strength Index (RSI) indicator.

```python
from application.strategies import RSIMeanReversionStrategy

strategy = RSIMeanReversionStrategy(
    name="RSI-MR",
    rsi_period=14,
    oversold_threshold=30,
    overbought_threshold=70
)

signal = strategy.generate_signal(market_data, timestamp)
print(strategy.get_description())
```

**Parameters:**
- `rsi_period`: Period for RSI calculation (default: 14)
- `oversold_threshold`: RSI level for buy signals (default: 30)
- `overbought_threshold`: RSI level for sell signals (default: 70)
- `extreme_oversold`: Extreme oversold level (default: 20)
- `extreme_overbought`: Extreme overbought level (default: 80)

**Minimum Bars Required:** RSI period + 20 (default: 34 bars)

### 2. Moving Average Crossover Strategy

**Best For:** Trending markets, longer-term positions

Classic trend-following using golden cross and death cross patterns.

```python
from application.strategies import MovingAverageCrossoverStrategy

strategy = MovingAverageCrossoverStrategy(
    name="MA-Cross",
    fast_period=50,
    slow_period=200,
    ma_type="SMA"  # or "EMA"
)

signal = strategy.generate_signal(market_data, timestamp)
```

**Parameters:**
- `fast_period`: Fast moving average period (default: 50)
- `slow_period`: Slow moving average period (default: 200)
- `ma_type`: Type of MA - 'SMA' or 'EMA' (default: 'SMA')

**Minimum Bars Required:** Slow period + 10 (default: 210 bars)

### 3. Momentum Breakout Strategy

**Best For:** Volatility breakouts, range transitions

Identifies and trades price breakouts with volume and momentum confirmation.

```python
from application.strategies import MomentumBreakoutStrategy

strategy = MomentumBreakoutStrategy(
    name="Momentum-BO",
    lookback_period=20,
    atr_period=14,
    volume_multiplier=1.5,
    breakout_threshold=0.02
)

signal = strategy.generate_signal(market_data, timestamp)
```

**Parameters:**
- `lookback_period`: Period for high/low range (default: 20)
- `atr_period`: Average True Range period (default: 14)
- `volume_multiplier`: Volume confirmation threshold (default: 1.5x)
- `breakout_threshold`: Breakout distance threshold (default: 2%)

**Minimum Bars Required:** max(lookback_period, atr_period) + 10 (default: 30 bars)

## Usage Guide

### Basic Usage

```python
from datetime import datetime
import pandas as pd
from application.strategies import RSIMeanReversionStrategy

# Create strategy
strategy = RSIMeanReversionStrategy(name="MyRSI")

# Load market data (must have OHLCV columns and DatetimeIndex)
market_data = pd.read_csv("data.csv", index_col="timestamp", parse_dates=True)

# Generate signal
signal = strategy.generate_signal(market_data, datetime.utcnow())

if signal:
    print(f"Signal: {signal.signal_type.value} {signal.direction.value}")
    print(f"Probability: {signal.probability:.2%}")
    print(f"Expected Return: {signal.expected_return:.2%}")
```

### Integration with ProofBench

```python
from engines.proofbench import SimulationEngine, SimulationConfig
from application.strategies import MovingAverageCrossoverStrategy

# Create strategy
strategy = MovingAverageCrossoverStrategy(name="MA-50-200")

# Setup simulation
config = SimulationConfig(initial_capital=100000.0)
sim = SimulationEngine(config=config)
sim.load_data("SPY", market_data)

# Trading callback
def on_bar(engine, symbol, bar):
    if len(engine.portfolio.equity_curve) < strategy.get_required_bars():
        return

    # Get data up to current bar
    current_data = market_data.iloc[:len(engine.portfolio.equity_curve)]

    # Generate signal
    signal = strategy.generate_signal(current_data, bar.timestamp)

    if signal and signal.signal_type == SignalType.ENTRY:
        # Place order based on signal
        # ... (implement order logic)
        pass

sim.on_bar = on_bar
results = sim.run()
```

### Strategy Comparison

```python
from application.strategies import (
    RSIMeanReversionStrategy,
    MovingAverageCrossoverStrategy,
    MomentumBreakoutStrategy
)

# Create multiple strategies
strategies = [
    RSIMeanReversionStrategy(name="RSI-14"),
    MovingAverageCrossoverStrategy(name="MA-50-200"),
    MomentumBreakoutStrategy(name="Breakout-20")
]

# Test each strategy
for strategy in strategies:
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy.name}")
    print(f"{'='*60}")
    print(strategy.get_description())

    signal = strategy.generate_signal(market_data, datetime.utcnow())
    if signal:
        print(f"\nCurrent Signal: {signal.signal_type.value}")
        print(f"Direction: {signal.direction.value}")
        print(f"Confidence: {signal.probability:.2%}")
```

### Custom Strategy Creation

Create your own strategy by extending `BaseStrategy`:

```python
from application.strategies.base import BaseStrategy
from engines.signalcore.core.signal import Signal, SignalType, Direction
from datetime import datetime
import pandas as pd

class MyCustomStrategy(BaseStrategy):
    def configure(self):
        # Set default parameters
        self.params.setdefault("param1", 10)
        self.params.setdefault("min_bars", 50)

    def generate_signal(self, data: pd.DataFrame, timestamp: datetime) -> Signal | None:
        # Validate data
        is_valid, msg = self.validate_data(data)
        if not is_valid:
            return None

        # Implement your signal logic
        # ...

        return Signal(
            symbol="SYMBOL",
            timestamp=timestamp,
            signal_type=SignalType.ENTRY,
            direction=Direction.LONG,
            probability=0.65,
            score=100.0,
            expected_return=0.05,
            metadata={"strategy": self.name}
        )

    def get_description(self) -> str:
        return "My custom strategy description"

    def get_required_bars(self) -> int:
        return self.params.get("min_bars", 50)
```

## Data Requirements

All strategies require market data with:

1. **OHLCV Columns:**
   - `open`: Opening price
   - `high`: High price
   - `low`: Low price
   - `close`: Closing price
   - `volume`: Trading volume

2. **DatetimeIndex:**
   - Data must use `DatetimeIndex` for time-based operations
   - Set with: `data.set_index("timestamp", inplace=True)`

3. **Minimum Bars:**
   - Each strategy requires minimum historical data
   - Check with: `strategy.get_required_bars()`

## Strategy Selection Guide

| Market Condition | Recommended Strategy | Alternative |
|-----------------|---------------------|-------------|
| Range-bound | RSI Mean Reversion | Momentum Breakout |
| Strong uptrend | MA Crossover | Momentum Breakout |
| Strong downtrend | MA Crossover | - |
| High volatility | Momentum Breakout | RSI Mean Reversion |
| Low volatility | MA Crossover | RSI Mean Reversion |
| Consolidation | Momentum Breakout | - |

## Risk Management

All strategies should be used with proper risk management:

1. **Position Sizing**
   - Never risk more than 1-2% per trade
   - Use ATR or volatility-based sizing

2. **Stop Losses**
   - Always set stops based on strategy logic
   - Check signal metadata for suggested levels

3. **Diversification**
   - Use multiple strategies across different timeframes
   - Combine trend-following and mean-reversion approaches

4. **Backtesting**
   - Test strategies on historical data first
   - Use ProofBench for comprehensive analysis
   - Consider transaction costs and slippage

## Performance Optimization

### Parameter Tuning

```python
# Test different parameter combinations
from itertools import product

rsi_periods = [10, 14, 20]
thresholds = [(25, 75), (30, 70), (35, 65)]

best_sharpe = 0
best_params = None

for period, (oversold, overbought) in product(rsi_periods, thresholds):
    strategy = RSIMeanReversionStrategy(
        name=f"RSI-{period}",
        rsi_period=period,
        oversold_threshold=oversold,
        overbought_threshold=overbought
    )

    # Run backtest
    results = run_backtest(strategy, market_data)

    if results.metrics.sharpe_ratio > best_sharpe:
        best_sharpe = results.metrics.sharpe_ratio
        best_params = (period, oversold, overbought)

print(f"Best parameters: RSI={best_params[0]}, thresholds={best_params[1:]}")
print(f"Sharpe Ratio: {best_sharpe:.2f}")
```

### AI Enhancement

Combine strategies with NVIDIA AI for enhanced insights:

```python
from engines.signalcore.models import LLMEnhancedModel

# Wrap your strategy with AI interpretation
# (This requires adapting the strategy to work with LLMEnhancedModel)
```

## Testing

Run tests to verify strategies:

```bash
pytest tests/test_strategies/ -v
```

## Contributing

To add a new strategy:

1. Create a new file in `src/application/strategies/`
2. Extend `BaseStrategy` class
3. Implement required methods
4. Add to `__init__.py` exports
5. Create tests in `tests/test_strategies/`
6. Update this README

## Support

For issues or questions:
- Check strategy descriptions with `strategy.get_description()`
- Review examples in `examples/`
- See comprehensive docs in `docs/`

---

**Version:** 1.0.0
**Last Updated:** 2025-11-29
