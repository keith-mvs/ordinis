# Analysis Framework Documentation

## Overview

The Ordinis Analysis Framework (OAF) is engineered to provide a robust, multi-dimensional suite of tools for sophisticated stock market analysis. By synthesizing diverse analytical methodologies, the framework enables traders and algorithms to construct a holistic view of market conditions. Central to this architecture is a flexible configuration system—utilizing classes like `AdaptiveConfig` and `TrainingConfig`—which allows precise tuning of parameters such as regime detection thresholds, indicator lookback periods, and ensemble weighting. The system is built upon three primary pillars of analysis, each offering unique insights into asset valuation and price action:

1. **Technical Analysis** - Studies price/volume data to predict movements
    Support and Resistance: Support and resistance levels identify key price zones where historical buying or selling pressure has caused reversals. These "floors" and "ceilings" are critical for the framework to determine optimal entry and exit points and anticipate trend exhaustion.
    Trendlines: Trendlines connect series of highs or lows to visually map the asset's directional momentum. Whether indicating bullish higher lows or bearish lower highs, these lines serve as dynamic support/resistance boundaries used to confirm trend strength within the `trend_analysis` module.
    Candlestick Patterns: Candlestick patterns decode immediate market sentiment through specific formations like Doji, Hammer, and Engulfing. The framework utilizes these patterns (defined in `src/analysis/technical/patterns/candlestick.py`) to anticipate short-term reversals or continuations based on crowd psychology.
    Fibonacci Retracement: Fibonacci retracement applies mathematical ratios (23.6%, 38.2%, 50%, 61.8%) to identify probable reversal zones during market corrections. These levels help predict where price pullbacks might stabilize before the primary trend resumes.
    Moving Averages: Moving averages (SMA, EMA, VWAP) smooth price data to filter noise and highlight the prevailing trend. The framework analyzes crossovers and slopes of these averages to generate buy/sell signals and define the current market regime.
    Price Action: Price action analysis focuses on raw market movements—highs, lows, and formations—without the lag of derived indicators. This approach allows for swift decision-making based on pure price behavior and structural market shifts.
2. **Fundamental Analysis** - Evaluates company financial health
    Financial Ratios: Financial ratios utilize data from financial statements to evaluate a company's operational efficiency, liquidity, and profitability. Metrics like the Price-to-Earnings (P/E) ratio and Return on Equity (ROE) allow for quick comparisons against industry peers to assess relative value.
    Financial Statements: Financial statements—comprising the balance sheet, income statement, and cash flow statement—provide the raw data regarding a company's assets, liabilities, and operational results. Deep analysis of these documents reveals the structural health and cash-generating ability of the business.
    Intrinsic Valuation: Intrinsic valuation methods, such as Discounted Cash Flow (DCF) analysis, estimate the true value of a company based on its projected future cash flows rather than current market sentiment. This approach helps investors identify stocks trading below their calculated worth, signaling a potential buying opportunity.
    Qualitative Factors: Qualitative analysis assesses non-quantifiable aspects of a business, including management quality, brand reputation, and competitive advantages or "moats." These factors are crucial for determining the sustainability of a company's long-term growth and market position.
    Macroeconomic Analysis: Macroeconomic analysis examines broader economic conditions, such as interest rates, inflation, and GDP growth, to understand the external environment affecting a company. This "top-down" perspective helps determine if the economic climate is favorable for specific sectors or industries.
    Growth Metrics: Growth metrics focus on the rate at which a company is expanding its revenue, earnings, and free cash flow over time. Analyzing historical and projected growth rates helps distinguish between stagnant value traps and high-potential compounding investments.
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
