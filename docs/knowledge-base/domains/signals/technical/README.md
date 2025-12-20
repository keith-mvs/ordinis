# Technical Analysis - Knowledge Base

## Overview

Technical analysis provides **rule-based, machine-implementable** methods for generating trading signals from price, volume, and derived indicator data.

**Philosophy**: Each indicator/pattern is a modular building block. Combine multiple signals with proper filters for robust strategies.

---

## Directory Structure

```
02_technical_analysis/
├── README.md                    # This file - overview and navigation
├── overlays/                    # Indicators plotted on price chart
│   ├── README.md
│   ├── moving_averages.md       # SMA, EMA, WMA, Hull MA, KAMA
│   ├── bollinger_bands.md       # Bollinger Bands, %B, Bandwidth
│   ├── keltner_channels.md      # ATR-based price channels
│   └── envelopes.md             # Percentage-based envelopes
├── oscillators/                 # Bounded indicators (typically 0-100)
│   ├── README.md
│   ├── rsi.md                   # Relative Strength Index
│   ├── stochastic.md            # Stochastic Oscillator
│   ├── cci.md                   # Commodity Channel Index
│   └── williams_r.md            # Williams %R
├── trend_indicators/            # Trend direction and strength
│   ├── README.md
│   ├── adx_dmi.md               # Average Directional Index
│   ├── parabolic_sar.md         # Parabolic Stop and Reverse
│   └── aroon.md                 # Aroon Up/Down
├── volatility/                  # Volatility measurement
│   ├── README.md
│   ├── atr.md                   # Average True Range
│   └── implied_realized.md      # IV vs Historical Volatility
├── composite/                   # Multi-component indicators
│   ├── README.md
│   ├── macd.md                  # Moving Average Convergence Divergence
│   └── momentum.md              # ROC, Momentum, PPO
├── patterns/                    # Chart and candlestick patterns
│   ├── README.md
│   ├── candlestick.md           # Japanese candlestick patterns
│   ├── chart_patterns.md        # Head & shoulders, triangles, etc.
│   └── support_resistance.md    # S/R level detection
└── advanced/                    # Advanced TA concepts
    ├── README.md
    ├── multi_timeframe.md       # Multiple timeframe analysis
    └── regime_detection.md      # Market regime classification
```

---

## Indicator Categories

### 1. Overlays (Price Chart)
Indicators plotted directly on the price chart. Used for trend identification and dynamic support/resistance.

| Indicator | Type | Best Use |
|-----------|------|----------|
| Moving Averages | Trend | Trend direction, dynamic S/R |
| Bollinger Bands | Volatility | Mean reversion, squeeze breakouts |
| Keltner Channels | Volatility | Trend-following with ATR |
| Envelopes | Range | Fixed % deviation from MA |

### 2. Oscillators (Separate Panel)
Bounded indicators that oscillate between fixed values. Used for overbought/oversold detection.

| Indicator | Range | Primary Use |
|-----------|-------|-------------|
| RSI | 0-100 | Overbought/oversold, divergence |
| Stochastic | 0-100 | Mean reversion, crossovers |
| CCI | Unbounded | Trend strength, extremes |
| Williams %R | -100 to 0 | Quick reversal signals |

### 3. Trend Indicators
Measure trend presence, direction, and strength.

| Indicator | Measures | Strength |
|-----------|----------|----------|
| ADX | Trend strength | >25 trending, <20 ranging |
| Parabolic SAR | Trend + stops | Trailing stop placement |
| Aroon | Trend age | New highs/lows timing |
| Ichimoku Cloud | Trend, momentum, S/R | Cloud position + TK cross |

### 4. Volatility
Measure price dispersion and expected movement.

| Indicator | Measures | Application |
|-----------|----------|-------------|
| ATR | Average range | Position sizing, stops |
| Bollinger Width | Band expansion | Volatility state |
| IV vs RV | Option premium | Volatility trades |

### 5. Composite
Multi-component indicators combining several calculations.

| Indicator | Components | Signal |
|-----------|------------|--------|
| MACD | 2 EMAs + signal | Crossovers, divergence |
| PPO | % MACD | Normalized comparison |
| TRIX | Triple EMA | Filtered momentum |
| CompositeIndicator | Weighted/vote mix | Consensus of multiple signals |

---

## Usage Principles

### 1. No Single Indicator is Reliable
Always combine multiple confirming signals:
```
ENTRY = trend_filter AND momentum_signal AND volume_confirmation
```

### 2. Context Matters
- **Trending markets**: Use trend-following indicators (MA, ADX, MACD)
- **Ranging markets**: Use oscillators (RSI, Stochastic, Bollinger)
- **High volatility**: Widen stops, reduce size

### 3. Multiple Timeframes
Align signals across timeframes for higher probability:
```
HIGHER_TF_TREND = Daily trend direction
ENTRY_TF = 4H or 1H signal alignment
EXECUTION_TF = 15M or 5M precise entry
```

### 4. Volume Confirmation
Especially for breakouts:
```
VALID_BREAKOUT = price_breakout AND volume > 1.5 * avg_volume
```

---

## Implementation in Ordinis

These indicators are implemented in:
- `src/ordinis/analysis/technical/indicators/` - Core indicator calculations
- `src/strategies/regime_adaptive/` - Regime-aware strategy framework

### Available Indicator Classes

```python
from ordinis.analysis.technical import (
    TechnicalIndicators, # Unified interface including Ichimoku Cloud
    MovingAverages,      # SMA, EMA, WMA, VWAP, Hull, KAMA
    Oscillators,         # RSI, Stochastic, CCI, Williams %R
    VolatilityIndicators,# ATR, Bollinger, Keltner, Donchian
    VolumeIndicators,    # OBV, CMF, Force Index
    CompositeIndicator,  # Weighted/voting aggregation
    MultiTimeframeAnalyzer,  # Cross-timeframe alignment
    CandlestickPatterns,     # 15+ candlestick detections
    BreakoutDetector,        # Breakout confirmation helper
)

# CLI shortcut for Phase 3 analytics:
# python -m ordinis.interface.cli analyze --data data/AAPL_historical.csv
```

---

## Academic Validation

| Method | Evidence | Notes |
|--------|----------|-------|
| MA Crossovers | Mixed | Better as filters than signals |
| RSI Extremes | Weak | Context-dependent |
| MACD | Moderate | Works in trending markets |
| Bollinger Bands | Moderate | Mean reversion in ranges |
| ADX | Moderate | Good for regime detection |

**Key Reference**: Aronson, D.R. (2006) "Evidence-Based Technical Analysis"

---

## Quick Links

- [Moving Averages](overlays/moving_averages.md)
- [Bollinger Bands](overlays/bollinger_bands.md)
- [RSI](oscillators/rsi.md)
- [MACD](composite/macd.md)
- [ADX](trend_indicators/adx_dmi.md)
- [ATR](volatility/atr.md)
- [Candlestick Patterns](patterns/candlestick.md)
- [Breakout Detection](patterns/breakout.md)
- [Multi-Timeframe Analysis](advanced/multi_timeframe.md)
