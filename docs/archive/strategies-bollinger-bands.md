# Bollinger Bands Strategy

## Overview

The Bollinger Bands strategy is a mean reversion approach that trades oversold and overbought conditions using price position relative to dynamic volatility bands.

---

## Strategy Specification

```yaml
strategy:
  # Identification
  id: "STRAT-BB-001"
  name: "Bollinger Bands Mean Reversion"
  version: "1.0.0"
  author: "Intelligent Investor System"
  created: "2024-12-01"
  last_updated: "2024-12-01"
  status: "development"

  # Classification
  category: "mean_reversion"
  style: "swing"
  instruments: ["equities", "etfs", "forex"]

  # Description
  description: |
    Trades mean reversion opportunities using Bollinger Bands indicator.
    Enters long positions when price touches or crosses below the lower band
    (oversold condition), exits when price touches or crosses above the upper
    band (overbought condition).

  hypothesis: |
    Price tends to revert to its mean over time. When price deviates significantly
    from the moving average (beyond 2 standard deviations), there is a statistical
    tendency for it to move back toward the mean. This creates systematic trading
    opportunities in range-bound markets.

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
      - oversold: |
          close <= lower_band
          OR (prev_close > prev_lower_band AND close < lower_band)  # Crossing below
      - volatility_filter: |
          band_width > min_band_width  # Avoid low volatility squeezes
    logic: "oversold AND volatility_filter"
    confirmation:
      - volume_required: false
      - price_confirmation: true
    timing:
      - allowed_hours: "09:30-16:00"
      - avoid_days: ["FOMC", "NFP", "Earnings"]

  # Exit Rules
  exit:
    profit_target:
      method: "indicator"
      condition: "close >= upper_band"
    stop_loss:
      method: "fixed_pct"
      value: 0.05
    trailing_stop:
      enabled: false
    time_stop:
      enabled: true
      max_hold_days: 20

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
    lookback_period: 40  # bb_period + 20
    indicators:
      - name: "Bollinger Bands"
        params: {period: 20, std: 2.0}
    fundamental_data: false
    news_data: false
    options_data: false

  # Evaluation Metrics
  evaluation:
    primary_metric: "sharpe_ratio"
    minimum_thresholds:
      sharpe: 1.0
      win_rate: 0.45
      profit_factor: 1.3
      max_drawdown: 0.15
    required_sample_size: 100
```

---

## Technical Details

### Bollinger Bands Calculation

```
Middle Band = SMA(close, period)
Upper Band = Middle Band + (std_dev * standard_deviations)
Lower Band = Middle Band - (std_dev * standard_deviations)
```

Default parameters:
- **Period**: 20 bars
- **Standard Deviations**: 2.0
- **Minimum Band Width**: 2% (0.02)

### Key Metrics

| Metric | Description | Usage |
|--------|-------------|-------|
| %B | Position within bands (0=lower, 1=upper) | Signal strength |
| Band Width | (Upper - Lower) / Middle | Volatility filter |
| Price vs Middle | Distance from mean | Reversion potential |

### Signal Generation Logic

1. **BUY Signal (ENTRY/LONG)**
   - Price touches or crosses below lower band
   - Band width exceeds minimum threshold (not low volatility)
   - Stronger signal when price is further below the band

2. **SELL Signal (EXIT)**
   - Price touches or crosses above upper band
   - Indicates overbought condition
   - Exit existing long positions

3. **HOLD Signal**
   - Price between bands
   - Score reflects distance from middle band

### Signal Strength Scoring

| Condition | Score Range | Probability |
|-----------|-------------|-------------|
| Below lower band (strong) | 0.5 - 1.0 | 65% - 80% |
| At lower band | 0.5 | 60% |
| Between bands (hold) | -0.3 to 0.3 | 50% |
| At upper band | -0.5 | 60% |
| Above upper band (strong) | -0.5 to -1.0 | 65% - 80% |

---

## Regime Detection

The strategy automatically detects market volatility regimes:

| Regime | Band Width | Strategy Behavior |
|--------|------------|-------------------|
| `low_volatility` | < 3% | Reduced signal generation |
| `moderate_volatility` | 3% - 8% | Normal operation |
| `high_volatility` | > 8% | Standard signals |

---

## Implementation

### Strategy Class

```python
from strategies import BollingerBandsStrategy

# Initialize with default parameters
strategy = BollingerBandsStrategy(name="my-bb-strategy")

# Or with custom parameters
strategy = BollingerBandsStrategy(
    name="custom-bb",
    bb_period=25,
    bb_std=2.5,
    min_band_width=0.03,
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
signal.regime          # low/moderate/high_volatility
signal.metadata        # Contains band values
signal.feature_contributions  # %B, band_width, etc.
```

---

## Best Use Cases

- **Range-bound markets** - Performs best when price oscillates
- **Mean-reverting assets** - Stocks with stable fundamentals
- **Counter-trend trading** - Captures oversold bounces
- **Moderate volatility** - Needs movement but not extreme trends

## Risk Considerations

| Risk | Description | Mitigation |
|------|-------------|------------|
| Band Walking | Price can ride bands in strong trends | Use with trend filter |
| Drawdowns | Counter-trend can catch falling knives | Use stop losses |
| Low Volatility | Compressed bands give false signals | min_band_width filter |
| Breakouts | Position against breakout direction | Time stops, trend confirmation |

---

## Feature Contributions

The signal includes these feature contributions for explainability:

| Feature | Description | Range |
|---------|-------------|-------|
| `percent_b` | Position within bands | 0.0 - 1.0 (can exceed) |
| `band_width` | Normalized band width | > 0.02 |
| `price_vs_middle` | Distance from mean | Percentage |
| `touching_lower` | At or below lower band | 0 or 1 |
| `touching_upper` | At or above upper band | 0 or 1 |
| `crossing_lower` | Just crossed lower band | 0 or 1 |
| `crossing_upper` | Just crossed upper band | 0 or 1 |
| `low_volatility` | Volatility too low | 0 or 1 |

---

## Testing

Tests are located in `tests/test_strategies/test_bollinger_bands.py`

Run tests:
```bash
pytest tests/test_strategies/test_bollinger_bands.py -v
```

Test coverage includes:
- Strategy initialization and configuration
- Parameter handling and validation
- Signal generation with various market conditions
- Edge cases (constant prices, missing values, extreme movements)
- Regime detection
- Feature contributions

---

## References

- John Bollinger, "Bollinger on Bollinger Bands" (2001)
- Technical Analysis of the Financial Markets, John J. Murphy
