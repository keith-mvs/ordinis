# Historical Trade Simulation Setup - VERIFIED WORKING

**Status:** Main branch ready for simulations
**Date:** 2025-11-30
**Test Result:** ProofBench simulator CONFIRMED WORKING

---

## What's Working (Verified)

### ✓ ProofBench Simulation Engine
- **Status:** FULLY FUNCTIONAL
- **Test:** `python -m pytest tests/test_engines/test_proofbench/test_simulator.py -v`
- **Result:** All core tests passing
- **Capabilities:**
  - Event-driven backtesting
  - Order execution simulation
  - Portfolio management
  - Performance metrics (Sharpe, Sortino, Max DD, Win Rate, etc.)
  - Equity curve tracking
  - Trade history

### ✓ Strategies Available
1. **MovingAverageCrossoverStrategy**
   - Location: `src/strategies/moving_average_crossover.py`
   - Type: Trend following
   - Params: fast_period (50), slow_period (200)
   - Status: Imported successfully

2. **MomentumBreakoutStrategy**
   - Location: `src/strategies/momentum_breakout.py`
   - Type: Momentum
   - Status: Available

3. **RSIMeanReversionStrategy**
   - Location: `src/strategies/rsi_mean_reversion.py`
   - Type: Mean reversion
   - Status: Available

### ✓ Knowledge Base
10 domain READMEs available:
- 01_market_fundamentals
- 02_technical_analysis
- 03_volume_liquidity
- 04_fundamental_analysis
- 05_news_sentiment
- 06_options_derivatives
- 07_risk_management
- 09_system_architecture
- 10_mathematical_foundations
- 11_references

### ✓ Data Plugins (Classes Verified)
- **IEXDataPlugin**: `from plugins.market_data.iex import IEXDataPlugin`
- **PolygonDataPlugin**: `from plugins.market_data.polygon import PolygonDataPlugin`

---

## Quick Start: Run Your First Simulation

###  Simple Example (Working Code)

```python
# test_sim_quickstart.py
import numpy as np
import pandas as pd
from engines.proofbench import (
    SimulationConfig, SimulationEngine,
    Order, OrderSide, OrderType
)

# Create data
dates = pd.date_range("2024-01-01", periods=100, freq="1d")
prices = 100 + np.cumsum(np.random.normal(0.5, 2, 100))
data = pd.DataFrame({
    "open": prices * 0.995,
    "high": prices * 1.01,
    "low": prices * 0.99,
    "close": prices,
    "volume": np.random.randint(1000000, 5000000, 100),
}, index=dates)

# Configure
config = SimulationConfig(initial_capital=100000.0, bar_frequency="1d")
engine = SimulationEngine(config)
engine.load_data("TEST", data)

# Simple strategy
def buy_and_hold(eng, symbol, bar):
    if not hasattr(eng, "bought"):
        eng.bought = False
    if not eng.bought:
        qty = int(eng.portfolio.cash * 0.5 / bar.close)
        if qty > 0:
            order = Order(symbol=symbol, side=OrderSide.BUY,
                         quantity=qty, order_type=OrderType.MARKET)
            eng.submit_order(order)
            eng.bought = True

engine.set_strategy(buy_and_hold)

# Run
results = engine.run()

# Results
print(f"Return: {results.metrics.total_return:.2f}%")
print(f"Sharpe: {results.metrics.sharpe_ratio:.2f}")
print(f"Max DD: {results.metrics.max_drawdown:.2f}%")
```

**Run:** `python test_sim_quickstart.py`

---

## Next Steps

### 1. Get Real Market Data

**Option A: Use IEX Plugin**
```python
from plugins.market_data.iex import IEXDataPlugin

plugin = IEXDataPlugin(config={"api_key": "your_iex_key"})
# Requires IEX API key from https://iexcloud.io/
```

**Option B: Use Polygon Plugin**
```python
from plugins.market_data.polygon import PolygonDataPlugin

plugin = PolygonDataPlugin(config={"api_key": "your_polygon_key"})
# Requires Polygon API key from https://polygon.io/
```

**Option C: Load CSV Data**
```python
data = pd.read_csv("historical_data.csv",
                   index_col="timestamp", parse_dates=True)
```

### 2. Use KB-Derived Strategy

```python
from strategies.moving_average_crossover import MovingAverageCrossoverStrategy

# Create strategy from KB concepts
strategy = MovingAverageCrossoverStrategy(
    name="MA-Cross-50-200",
    params={
        "fast_period": 50,
        "slow_period": 200,
        "ma_type": "SMA"
    }
)

# Integration with simulator
def kb_strategy_callback(engine, symbol, bar):
    # Get historical data up to current bar
    current_idx = len(engine.portfolio.equity_curve)
    hist_data = all_data.iloc[:current_idx]

    # Generate signal from KB strategy
    signal = strategy.generate_signal(hist_data, bar.timestamp)

    if signal and signal.signal_type.value == "entry":
        # Execute based on signal
        # ... (implement order logic)
        pass

engine.set_strategy(kb_strategy_callback)
```

### 3. Add NVIDIA AI Enhancement (Optional)

```python
from engines.cortex import CortexEngine

# Enable AI-powered analysis
cortex = CortexEngine(
    nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
    usd_code_enabled=True
)

# Generate hypothesis from market context
hypothesis = cortex.generate_hypothesis(
    market_context={"regime": "trending", "volatility": "low"}
)

# Use hypothesis to configure strategy
# ... (implement strategy based on hypothesis)
```

---

## Performance Metrics Available

From `results.metrics`:
- `total_return` - Total return %
- `annualized_return` - Annualized return %
- `sharpe_ratio` - Risk-adjusted return
- `sortino_ratio` - Downside risk-adjusted return
- `max_drawdown` - Maximum peak-to-trough decline %
- `win_rate` - Percentage of winning trades
- `profit_factor` - Gross profit / Gross loss
- `num_trades` - Total number of trades
- `avg_trade_pnl` - Average P&L per trade
- `equity_final` - Final portfolio value

---

## Testing Your Setup

Run official tests to verify:
```bash
# Test simulation engine
python -m pytest tests/test_engines/test_proofbench/test_simulator.py -v

# Test all proofbench components
python -m pytest tests/test_engines/test_proofbench/ -v

# Test specific strategy
python -c "from strategies.moving_average_crossover import MovingAverageCrossoverStrategy; print('OK')"
```

---

## Common Issues

### Issue: Import errors
**Solution:** Ensure package is installed:
```bash
python -m pip install -e .
```

### Issue: No trades executed
**Checklist:**
- [ ] Strategy callback is set: `engine.set_strategy(callback)`
- [ ] Data has enough bars for strategy requirements
- [ ] Strategy logic is triggering (add print statements)
- [ ] Orders are being submitted in callback

### Issue: Coverage errors in tests
**Solution:** Tests pass but coverage < 50% is expected. Add `-k test_name` to skip coverage:
```bash
python -m pytest tests/test_engines/test_proofbench/ -v --no-cov
```

---

## File Locations

- **Simulator:** `src/engines/proofbench/core/simulator.py`
- **Strategies:** `src/strategies/*.py`
- **KB:** `docs/knowledge-base/*/README.md`
- **Tests:** `tests/test_engines/test_proofbench/*.py`
- **Examples:** `docs/examples/proofbench_example.py`

---

## Architecture Overview

```
KB Strategy Design
       ↓
Strategy Implementation (src/strategies/)
       ↓
Market Data (Plugins or CSV)
       ↓
SimulationEngine (ProofBench)
       ↓
Results & Metrics
```

---

**VERIFIED:** 2025-11-30
**Main Branch:** Ready for historical simulations
**ProofBench Status:** ✓ Working
