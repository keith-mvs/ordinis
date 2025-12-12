# MACD Strategy

## Overview

The MACD (Moving Average Convergence Divergence) strategy is a momentum-based approach that trades crossover signals between the MACD line and signal line, with additional confirmation from histogram strength and zero-line position.

---

## Strategy Specification

```yaml
strategy:
  # Identification
  id: "STRAT-MACD-001"
  name: "MACD Momentum Crossover"
  version: "1.0.0"
  author: "Intelligent Investor System"
  created: "2024-12-01"
  last_updated: "2024-12-01"
  status: "development"

  # Classification
  category: "momentum"
  style: "swing"
  instruments: ["equities", "etfs", "forex", "crypto"]

  # Description
  description: |
    Trades momentum opportunities using MACD crossover signals.
    Enters long positions on bullish crossovers (MACD crosses above signal line),
    exits on bearish crossovers. Signal strength is enhanced by zero-line position
    and histogram momentum.

  hypothesis: |
    Moving average crossovers capture momentum shifts in price trends. When the
    faster moving average crosses the slower one, it signals a change in momentum
    direction. The MACD histogram provides early warning of crossovers and
    momentum strength, while the zero line acts as a trend confirmation filter.

  # Universe & Filters
  universe:
    market: "US"
    exchanges: ["NYSE", "NASDAQ"]
    filters:
      - min_price: 10
      - max_price: 500
      - min_avg_volume: 500000
      - min_market_cap: 1000000000
      - exclude_sectors: []
      - exclude_tickers: []

  # Entry Rules
  entry:
    conditions:
      - bullish_crossover: |
          prev_macd <= prev_signal AND current_macd > current_signal
      - above_zero_bonus: |
          current_macd > 0  # Stronger signal when above zero
    logic: "bullish_crossover (above_zero_bonus enhances probability)"
    confirmation:
      - histogram_positive: "histogram > 0"
    timing:
      - allowed_hours: "09:30-16:00"
      - avoid_days: ["FOMC", "NFP", "Earnings"]

  # Exit Rules
  exit:
    profit_target:
      method: "indicator"
      condition: "bearish_crossover"
    stop_loss:
      method: "fixed_pct"
      value: 0.05
    trailing_stop:
      enabled: true
      method: "pct"
      value: 0.03
    time_stop:
      enabled: true
      max_hold_days: 30

  # Position Sizing
  position_sizing:
    method: "risk_based"
    risk_per_trade: 0.01
    max_position_pct: 0.10

  # Risk Parameters
  risk:
    max_positions: 5
    max_sector_exposure: 0.25
    max_correlation: 0.70
    max_daily_loss: 0.03
    max_drawdown: 0.15

  # Data Requirements
  data:
    price_data: "daily"
    lookback_period: 55  # slow_period + signal_period + 20
    indicators:
      - name: "MACD"
        params: {fast: 12, slow: 26, signal: 9}
    fundamental_data: false
    news_data: false
    options_data: false

  # Evaluation Metrics
  evaluation:
    primary_metric: "sharpe_ratio"
    minimum_thresholds:
      sharpe: 1.0
      win_rate: 0.45
      profit_factor: 1.4
      max_drawdown: 0.15
    required_sample_size: 100
```

---

## Technical Details

### MACD Calculation

```
MACD Line = EMA(close, fast_period) - EMA(close, slow_period)
Signal Line = EMA(MACD_line, signal_period)
Histogram = MACD Line - Signal Line
```

Default parameters:
- **Fast Period**: 12 bars
- **Slow Period**: 26 bars
- **Signal Period**: 9 bars

### Key Metrics

| Metric | Description | Usage |
|--------|-------------|-------|
| MACD Line | Difference between fast and slow EMAs | Momentum direction |
| Signal Line | Smoothed MACD | Crossover trigger |
| Histogram | MACD - Signal | Momentum strength |
| Zero Line | Reference point | Trend confirmation |

### Signal Generation Logic

1. **BUY Signal (ENTRY/LONG)**
   - MACD line crosses above signal line (bullish crossover)
   - Stronger signal when crossover occurs above zero line
   - Histogram turning positive adds confirmation

2. **SELL Signal (EXIT)**
   - MACD line crosses below signal line (bearish crossover)
   - Stronger signal when crossover occurs below zero line
   - Exit existing long positions

3. **HOLD Signal**
   - No crossover occurring
   - Score reflects current momentum positioning

### Signal Strength Scoring

| Condition | Score Range | Probability | Expected Return |
|-----------|-------------|-------------|-----------------|
| Bullish crossover above zero | 0.7 - 1.0 | 70% - 80% | 6% - 10% |
| Bullish crossover below zero | 0.5 - 0.7 | 60% - 70% | 4% - 6% |
| Hold (bullish positioning) | 0.1 - 0.3 | 50% | - |
| Hold (bearish positioning) | -0.1 to -0.3 | 50% | - |
| Bearish crossover above zero | -0.5 to -0.7 | 60% - 70% | - |
| Bearish crossover below zero | -0.7 to -1.0 | 70% - 80% | - |

---

## Regime Detection

The strategy automatically detects market regimes:

| Regime | Condition | Description |
|--------|-----------|-------------|
| `crossover` | Bullish or bearish crossover | Active signal |
| `trending_up` | Histogram increasing, MACD > Signal | Bullish trend |
| `trending_down` | Histogram decreasing, MACD < Signal | Bearish trend |
| `consolidating` | Small histogram (< 0.5 std) | No clear direction |
| `ranging` | Other conditions | Sideways movement |

---

## Implementation

### Strategy Class

```python
from strategies import MACDStrategy

# Initialize with default parameters
strategy = MACDStrategy(name="my-macd-strategy")

# Or with custom parameters
strategy = MACDStrategy(
    name="custom-macd",
    fast_period=8,
    slow_period=21,
    signal_period=5,
)

# Generate signal
signal = strategy.generate_signal(data, timestamp)
```

### Signal Output

```python
signal.signal_type     # ENTRY, EXIT, or HOLD
signal.direction       # LONG or NEUTRAL
signal.probability     # 0.5 - 0.8
signal.score           # -1.0 to 1.0
signal.regime          # crossover/trending_up/down/consolidating/ranging
signal.metadata        # Contains period values, current price
signal.feature_contributions  # MACD values, crossover flags, etc.
```

---

## Best Use Cases

- **Trending markets** - MACD excels in directional moves
- **Momentum-driven assets** - High-beta stocks, growth sectors
- **Medium-term trading** - Swing trades lasting days to weeks
- **Trend confirmation** - Use with other indicators

## Risk Considerations

| Risk | Description | Mitigation |
|------|-------------|------------|
| Lagging Indicator | Signals come after move begins | Use histogram for early warning |
| Whipsaws | False signals in ranging markets | Use regime detection |
| Missed Tops/Bottoms | Won't catch exact reversals | Accept partial moves |
| Parameter Sensitivity | Different periods give different signals | Backtest parameters |

---

## Feature Contributions

The signal includes these feature contributions for explainability:

| Feature | Description | Typical Range |
|---------|-------------|---------------|
| `macd` | Current MACD line value | Varies with price |
| `signal` | Current signal line value | Varies with price |
| `histogram` | MACD - Signal difference | Varies |
| `histogram_strength` | Histogram normalized by price | 0 - 0.05 |
| `macd_above_zero` | MACD line position | 0 or 1 |
| `bullish_crossover` | Just crossed bullish | 0 or 1 |
| `bearish_crossover` | Just crossed bearish | 0 or 1 |
| `histogram_increasing` | Histogram momentum positive | 0 or 1 |

---

## Combining with Other Indicators

MACD works well in combination with:

1. **RSI** - Confirm overbought/oversold with momentum
2. **Bollinger Bands** - Mean reversion + momentum confirmation
3. **Volume** - Confirm breakouts with volume expansion
4. **Moving Averages** - Use 50/200 SMA for trend direction

Example combination strategy:
```
BUY when:
  - MACD bullish crossover
  - RSI between 30-70 (not extreme)
  - Price above 50-day SMA (uptrend)
  - Volume above average
```

---

## Testing

Tests are located in `tests/test_strategies/test_macd.py`

Run tests:
```bash
pytest tests/test_strategies/test_macd.py -v
```

Test coverage includes:
- Strategy initialization and configuration
- Parameter handling and validation
- Signal generation with various market conditions
- Crossover detection (bullish and bearish)
- Zero-line confirmation logic
- Regime detection
- Feature contributions

---

## References

- Gerald Appel, "The Moving Average Convergence-Divergence Trading Method" (1979)
- Technical Analysis of the Financial Markets, John J. Murphy
- Gerald Appel, "Technical Analysis: Power Tools for Active Investors" (2005)
