---
name: technical-indicators
description: Master calculation, interpretation, and application of twelve core technical indicators for trend analysis, momentum assessment, volatility measurement, and volume confirmation. Use when analyzing price charts, evaluating market conditions, identifying trading signals, or implementing quantitative trading strategies. Covers ADX, ATR, Bollinger Bands, CCI, Fibonacci retracements, Ichimoku Cloud, moving averages, MACD, OBV, Parabolic SAR, RSI, and Stochastic oscillator.
---

# Technical Indicators Analysis

## Overview

This skill provides expert-level capabilities in technical analysis through mastery of twelve widely-used market indicators. All methodologies derive from established technical analysis references including the CMT curriculum, Edwards & Magee, and Bloomberg Market Concepts.

## Indicator Categories

Technical indicators are organized by analytical purpose:

**Trend Indicators**: ADX, Ichimoku Cloud, Moving Averages, Parabolic SAR
**Momentum Indicators**: CCI, MACD, RSI, Stochastic Oscillator
**Volatility Indicators**: ATR, Bollinger Bands
**Volume Indicators**: OBV
**Static Levels**: Fibonacci Retracement

## Quick Reference

| Indicator | Primary Use | Time Frame | Key Signal |
|-----------|-------------|------------|------------|
| ADX | Trend strength | Medium-long | >25 strong trend |
| ATR | Volatility | Any | Higher = more volatile |
| Bollinger Bands | Volatility, reversals | Short-medium | Price at bands |
| CCI | Overbought/oversold | Short-medium | >100 / <-100 |
| Fibonacci | Support/resistance | Any | 38.2%, 61.8% levels |
| Ichimoku Cloud | Trend direction | Medium-long | Price vs cloud position |
| MA | Trend direction | Any | Price vs MA, crossovers |
| MACD | Momentum shifts | Medium-long | Line crossovers |
| OBV | Volume confirmation | Any | Divergence with price |
| Parabolic SAR | Trend direction | Short-medium | Dot position vs price |
| RSI | Overbought/oversold | Short-medium | >70 / <30 |
| Stochastic | Short-term momentum | Short | >80 / <20 |

## Indicator Selection Workflow

1. **Define analytical objective**:
   - Trend identification → Use ADX, MA, Ichimoku, Parabolic SAR
   - Momentum assessment → Use RSI, Stochastic, MACD, CCI
   - Volatility measurement → Use ATR, Bollinger Bands
   - Volume confirmation → Use OBV
   - Support/resistance → Use Fibonacci, MA

2. **Match time frame**:
   - Short-term trading (minutes to days) → Stochastic, RSI, Parabolic SAR
   - Medium-term positioning (days to weeks) → MACD, Bollinger Bands, CCI
   - Long-term trends (weeks to months) → ADX, Ichimoku, MA

3. **Confirm with multiple indicators**:
   - Combine trend + momentum indicators for signal validation
   - Use volume indicators to confirm price movements
   - Apply volatility indicators for risk management

## Detailed Indicator Analysis

For comprehensive methodology, calculation formulas, and interpretation guidelines:

**Trend Indicators**: See [references/trend_indicators.md](references/trend_indicators.md)
**Momentum Indicators**: See [references/momentum_indicators.md](references/momentum_indicators.md)
**Volatility Indicators**: See [references/volatility_indicators.md](references/volatility_indicators.md)
**Volume Indicators**: See [references/volume_indicators.md](references/volume_indicators.md)
**Static Levels**: See [references/static_levels.md](references/static_levels.md)

## Python Implementation

### Required libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ta import trend, momentum, volatility, volume
from scipy import stats
```

### Calculation scripts

Pre-built calculation functions for all indicators:

```bash
python scripts/calculate_indicators.py --symbol SPY --indicator RSI --period 14
python scripts/calculate_indicators.py --symbol SPY --indicator MACD
python scripts/calculate_indicators.py --symbol SPY --indicator BOLLINGER --period 20
```

Script provides standardized calculations with configurable parameters. Output includes indicator values, signal interpretations, and visualization support.

For complete implementation details: See [scripts/calculate_indicators.py](scripts/calculate_indicators.py)

## Practical Application Workflow

### Step 1: Data preparation

Ensure OHLCV data (Open, High, Low, Close, Volume) is available with consistent time intervals.

### Step 2: Indicator calculation

Calculate indicators appropriate for your analytical objective and time frame.

### Step 3: Signal interpretation

Apply standard interpretation rules:
- Overbought/oversold levels for oscillators
- Crossover signals for moving averages
- Divergence patterns for confirmation

### Step 4: Multi-indicator confirmation

Validate signals across indicator categories:
- Trend indicator confirms directional bias
- Momentum indicator confirms strength
- Volume indicator validates price action

### Step 5: Risk management integration

Use volatility indicators (ATR, Bollinger Bands) for position sizing and stop-loss placement.

## Interpretation Guidelines

### Trend strength assessment

Strong trend (ADX > 25):
- Follow trend direction indicated by price vs moving averages
- Use momentum indicators to time entries
- Trail stops using Parabolic SAR or ATR multiples

Weak trend (ADX < 20):
- Consider mean-reversion strategies
- Use Bollinger Bands for overbought/oversold conditions
- Tighten profit targets

### Momentum divergence detection

Price makes new high but RSI/MACD does not:
- Bearish divergence signals potential reversal
- Reduce position size or take profits
- Wait for confirmation from trend indicators

Price makes new low but RSI/MACD does not:
- Bullish divergence signals potential reversal
- Consider accumulation strategies
- Confirm with volume increase

### Volatility regime adaptation

High volatility (ATR expanding, Bollinger Bands widening):
- Widen stop losses to avoid premature exit
- Reduce position size to maintain consistent risk
- Expect larger price swings and breakout potential

Low volatility (ATR contracting, Bollinger Bands narrowing):
- Anticipate volatility expansion (potential breakout)
- Tighten profit targets
- Consider range-bound strategies

## Case Studies

Real-world application examples across various market conditions:

**Trend-following scenarios**: See [examples/trend_following_cases.md](examples/trend_following_cases.md)
**Mean-reversion scenarios**: See [examples/mean_reversion_cases.md](examples/mean_reversion_cases.md)
**Breakout scenarios**: See [examples/breakout_cases.md](examples/breakout_cases.md)
**Divergence scenarios**: See [examples/divergence_cases.md](examples/divergence_cases.md)

## Validation Framework

### Signal accuracy assessment

Evaluate indicator performance across different market regimes:
- Bull markets: Trend-following indicators excel
- Bear markets: Momentum indicators provide early warnings
- Sideways markets: Oscillators identify range boundaries

### Backtesting methodology

Test indicator signals against historical data:

```python
python scripts/backtest_indicators.py --symbol SPY --start 2020-01-01 --end 2024-12-31 --indicators RSI,MACD,BB
```

Output includes win rate, profit factor, maximum drawdown, and regime-specific performance.

### Parameter optimization

Optimize indicator parameters for specific securities:
- Test multiple lookback periods
- Evaluate sensitivity to parameter changes
- Avoid overfitting to historical data

## Common Pitfalls

**Overreliance on single indicators**: No indicator provides perfect signals. Always use multiple confirmation sources.

**Ignoring market context**: Indicators perform differently in trending vs ranging markets. Assess regime before applying signals.

**Parameter overfitting**: Optimizing parameters for past data often degrades future performance. Use robust, widely-accepted default values.

**Lagging indicators**: Most indicators lag price action. Combine with price action analysis for timely signals.

**False signals in choppy markets**: Oscillators generate frequent whipsaw signals during low-volatility periods. Filter signals with trend confirmation.

## Performance Metrics

Evaluate indicator effectiveness using:

**Accuracy**: Percentage of correct signals
**Sharpe Ratio**: Risk-adjusted returns
**Maximum Drawdown**: Largest peak-to-trough decline
**Win Rate**: Profitable trades / total trades
**Profit Factor**: Gross profits / gross losses

Track metrics across different market regimes and time frames to identify optimal indicator combinations.

## Ethical Considerations

**Market impact**: High-frequency indicator-based strategies may contribute to market microstructure effects and volatility.

**Information asymmetry**: Technical analysis assumes efficient markets with equal information access. Be aware of institutional advantages in data processing and execution speed.

**Behavioral influence**: Popular indicators create self-fulfilling prophecies as traders act on common signals. Consider positioning ahead of obvious technical levels.

**Risk disclosure**: Indicator-based strategies do not guarantee profits. Past performance does not predict future results. Implement robust risk management protocols.

## Authoritative References

All methodologies derive from established sources:

- **CMT Association**: Chartered Market Technician curriculum
- **Edwards & Magee**: Technical Analysis of Stock Trends (11th Edition)
- **Bloomberg**: Market Concepts technical analysis modules
- **Murphy, John J.**: Technical Analysis of the Financial Markets
- **Pring, Martin J.**: Technical Analysis Explained (5th Edition)
- **Academic sources**: Quantitative Finance journals, Journal of Technical Analysis

## Integration with Quantitative Frameworks

Technical indicators integrate with:

**Risk management systems**: ATR-based position sizing
**Algorithmic trading**: Signal generation and execution logic
**Portfolio optimization**: Tactical allocation based on trend strength
**Performance attribution**: Decomposing returns by indicator strategy

## Tools and Resources

**Python libraries**: ta, ta-lib, pandas_ta
**Data providers**: Alpha Vantage, Yahoo Finance, Polygon.io
**Visualization**: matplotlib, plotly, mplfinance
**Backtesting frameworks**: Backtrader, Zipline, VectorBT

## Next Steps

1. **Master calculation methodology**: Review detailed formulas in reference materials
2. **Implement in Python**: Use provided scripts as templates
3. **Analyze case studies**: Study real-world applications in examples directory
4. **Backtest strategies**: Validate indicator combinations on historical data
5. **Monitor performance**: Track signal accuracy across market regimes

For specific indicator details, calculation algorithms, and interpretation frameworks, reference the linked documentation files organized by indicator category.
