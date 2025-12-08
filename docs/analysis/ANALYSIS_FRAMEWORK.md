# Analysis Framework Documentation

## Overview

The Ordinis Analysis Framework provides a comprehensive suite of tools for stock analysis, organized into three primary approaches:

1. **Technical Analysis** - Studies price/volume data to predict movements
2. **Fundamental Analysis** - Evaluates company financial health
3. **Sentiment Analysis** - Gauges market psychology and investor attitudes

These approaches provide different perspectives and can be combined for robust investment decisions.

---

## 1. Technical Analysis

### Core Belief
All relevant information is already reflected in the stock's price, and historical price movements tend to repeat due to market psychology.

### Module Structure

```
src/analysis/technical/
├── __init__.py
├── indicators/
│   ├── __init__.py
│   ├── moving_averages.py    # SMA, EMA, WMA, VWAP, Hull MA
│   ├── oscillators.py        # RSI, Stochastic, CCI, Williams %R
│   ├── volatility.py         # ATR, Bollinger Bands, Keltner Channels
│   ├── volume.py             # OBV, CMF, VWAP, Force Index
│   └── combined.py           # Unified TechnicalIndicators class
├── patterns/
│   ├── __init__.py
│   ├── chart_patterns.py     # Head & Shoulders, Triangles, etc.
│   ├── candlestick.py        # Doji, Hammer, Engulfing patterns
│   └── fibonacci.py          # Retracement, Extensions
└── trend_analysis.py         # Trend direction and strength
```

### Indicator Categories

#### Moving Averages
Smooth price data to identify trend direction.

| Indicator | Description | Use Case |
|-----------|-------------|----------|
| SMA | Simple Moving Average | Baseline trend identification |
| EMA | Exponential Moving Average | Faster response to price changes |
| WMA | Weighted Moving Average | Linear weight decay |
| VWAP | Volume Weighted Average Price | Institutional fair value |
| Hull MA | Faster, smoother MA | Reduced lag trend following |
| KAMA | Adaptive MA | Adjusts to volatility |

```python
from src.analysis.technical import MovingAverages

# Calculate 20-period EMA
ema = MovingAverages.ema(close_prices, period=20)

# Generate crossover signal
signal = MovingAverages.crossover_signal(data, fast_period=20, slow_period=50)
```

#### Oscillators
Identify overbought/oversold conditions.

| Indicator | Range | Overbought | Oversold |
|-----------|-------|------------|----------|
| RSI | 0-100 | >70 | <30 |
| Stochastic | 0-100 | >80 | <20 |
| CCI | Unbounded | >100 | <-100 |
| Williams %R | -100 to 0 | >-20 | <-80 |
| MFI | 0-100 | >80 | <20 |

```python
from src.analysis.technical import Oscillators

# Calculate RSI
rsi = Oscillators.rsi(close_prices, period=14)

# Get signal with overbought/oversold detection
signal = Oscillators.rsi_signal(data, period=14, overbought=70, oversold=30)
```

#### Volatility Indicators
Measure price dispersion and risk.

| Indicator | Description |
|-----------|-------------|
| ATR | Average True Range - absolute volatility measure |
| Bollinger Bands | Standard deviation bands around SMA |
| Keltner Channels | ATR-based bands around EMA |
| Donchian Channels | High/low breakout channels |
| Historical Volatility | Annualized standard deviation |

```python
from src.analysis.technical import VolatilityIndicators

# Get Bollinger Band signal
bb_signal = VolatilityIndicators.bollinger_signal(data, period=20, std_dev=2.0)

# Comprehensive volatility analysis
vol_metrics = VolatilityIndicators.volatility_analysis(data)
```

#### Volume Indicators
Confirm price movements with volume analysis.

| Indicator | Description |
|-----------|-------------|
| OBV | On-Balance Volume - cumulative volume flow |
| A/D Line | Accumulation/Distribution |
| CMF | Chaikin Money Flow |
| Force Index | Price change × volume |
| Relative Volume | Current vs average volume |

```python
from src.analysis.technical import VolumeIndicators

# Volume confirmation check
vol_signal = VolumeIndicators.volume_confirmation(data)
```

### Chart Patterns

Visual formations that indicate potential price movements:

- **Reversal Patterns**: Head & Shoulders, Double Top/Bottom
- **Continuation Patterns**: Triangles, Flags, Wedges
- **Candlestick Patterns**: Doji, Hammer, Engulfing

### Fibonacci Analysis

Key retracement levels: 23.6%, 38.2%, 50%, 61.8%, 100%

---

## 2. Fundamental Analysis

### Purpose
Evaluates company financial health and long-term growth potential. Ideal for long-term value investing.

### Module Structure

```
src/analysis/fundamental/
├── __init__.py
├── financial_ratios.py       # P/E, P/B, ROE, etc.
├── valuation_metrics.py      # DCF, comparables
├── growth_analysis.py        # Revenue/earnings growth
└── financial_statements.py   # Balance sheet, income, cash flow
```

### Key Metrics

| Category | Metrics |
|----------|---------|
| Valuation | P/E, P/B, P/S, EV/EBITDA |
| Profitability | ROE, ROA, Profit Margin, ROIC |
| Liquidity | Current Ratio, Quick Ratio |
| Solvency | Debt/Equity, Interest Coverage |
| Growth | Revenue Growth, EPS Growth, CAGR |

---

## 3. Sentiment Analysis

### Purpose
Gauges market sentiment to identify buying/selling opportunities based on crowd behavior.

### Module Structure

```
src/analysis/sentiment/
├── __init__.py
├── news_analyzer.py          # News sentiment scoring
├── social_media.py           # Twitter, Reddit analysis
└── analyst_ratings.py        # Analyst consensus
```

### Data Sources

- News articles and headlines
- Social media (Twitter, Reddit, StockTwits)
- Analyst reports and ratings
- Options flow and put/call ratios
- Insider trading activity

---

## 4. Regime-Adaptive Strategies

### Overview

The regime-adaptive system automatically adjusts trading strategies based on detected market conditions.

### Module Structure

```
src/strategies/regime_adaptive/
├── __init__.py
├── regime_detector.py        # Market condition classification
├── trend_following.py        # Bull market strategies
├── mean_reversion.py         # Sideways market strategies
├── volatility_trading.py     # High volatility strategies
└── adaptive_manager.py       # Unified strategy manager
```

### Market Regimes

| Regime | Characteristics | Primary Strategy |
|--------|-----------------|------------------|
| BULL | Strong uptrend, ADX>25, price above 200 MA | Trend Following |
| BEAR | Strong downtrend, defensive positioning | Cash/Defensive |
| SIDEWAYS | Range-bound, low directional movement | Mean Reversion |
| VOLATILE | High VIX, expanded ranges | Volatility Trading |
| TRANSITIONAL | Regime change in progress | Reduced Exposure |

### Strategy Pools

#### Trend-Following (Bull Markets)
- MA Crossover Strategy
- Breakout Strategy
- ADX Trend Strategy

#### Mean-Reversion (Sideways Markets)
- Bollinger Fade Strategy
- RSI Reversal Strategy
- Keltner Channel Strategy
- Z-Score Reversion Strategy

#### Volatility Trading (Volatile Markets)
- Scalping Strategy
- Volatility Breakout Strategy
- ATR Trailing Strategy

### Adaptive Weighting

Default regime weights:

| Regime | Trend | Reversion | Volatility | Cash |
|--------|-------|-----------|------------|------|
| BULL | 80% | 10% | 5% | 5% |
| BEAR | 20% | 20% | 10% | 50% |
| SIDEWAYS | 10% | 70% | 10% | 10% |
| VOLATILE | 20% | 20% | 40% | 20% |
| TRANSITIONAL | 15% | 15% | 10% | 60% |

### Usage

```python
from src.strategies.regime_adaptive import AdaptiveStrategyManager, AdaptiveConfig

# Create manager with custom config
config = AdaptiveConfig(
    use_ensemble=True,
    confidence_scaling=True,
    volatility_scaling=True,
)

manager = AdaptiveStrategyManager(config)

# Generate signal from data
signal = manager.update(ohlcv_data)

# Get regime information
regime_info = manager.get_regime_info()
print(f"Current Regime: {regime_info['regime']}")
print(f"Confidence: {regime_info['confidence']:.1%}")
```

---

## 5. Backtesting Framework

### Training Data Generation

```python
from src.data import TrainingDataGenerator, TrainingConfig

config = TrainingConfig(
    symbols=["SPY", "QQQ"],
    chunk_sizes_months=[2, 3, 4, 6, 8, 10, 12],
    lookback_years=[5, 10, 15, 20],
)

generator = TrainingDataGenerator(config)
chunks = generator.generate_chunks("SPY", num_chunks=100, balance_regimes=True)
```

### Cross-Validation

```python
from src.data import RegimeCrossValidator

validator = RegimeCrossValidator(
    strategy_callback=my_strategy,
    strategy_name="My Strategy"
)

report = validator.validate(chunks)
report.print_report()
```

---

## 6. Quick Reference

### Signal Interpretation

| Signal | Action |
|--------|--------|
| BUY | Enter long position |
| SELL | Exit position or take profit |
| EXIT | Exit on stop loss or time stop |
| HOLD | Maintain current position |

### Position Sizing

- **Base Size**: 80% of available capital
- **Confidence Scaling**: Reduces size when regime uncertain
- **Volatility Scaling**: Reduces size in high volatility
- **Minimum Size**: 30% floor

### Stop Loss Methods

| Method | Description |
|--------|-------------|
| ATR-based | Stop at N × ATR from entry |
| Bollinger | Stop at lower band |
| Trailing | Ratchets up with price |
| Time | Exit after N bars |

---

## 7. Integration with ProofBench

```python
from src.strategies.regime_adaptive import AdaptiveStrategyManager, create_strategy_callback
from src.engines.proofbench.core.simulator import SimulationEngine, SimulationConfig

# Create adaptive manager
manager = AdaptiveStrategyManager()

# Create callback for simulator
callback = create_strategy_callback(manager)

# Run backtest
config = SimulationConfig(initial_capital=100000)
engine = SimulationEngine(config)
engine.load_data("SPY", data)
engine.set_strategy(callback)
results = engine.run()
```

---

## Version History

- **v0.3.0** - Regime-adaptive strategies, technical analysis reorganization
- **v0.2.0** - Multi-timeframe training data, cross-validation
- **v0.1.0** - Initial backtesting framework
