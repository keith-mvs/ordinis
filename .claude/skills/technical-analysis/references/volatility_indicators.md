# Volatility Indicators Reference

## Average True Range (ATR)

### Purpose
Measures market volatility by decomposing the entire range of asset price movement. Essential for position sizing, stop-loss placement, and volatility regime assessment.

### Calculation

**True Range (TR)** is the maximum of:
1. Current High - Current Low
2. |Current High - Previous Close|
3. |Current Low - Previous Close|

**ATR** = N-period moving average of True Range

Standard period: 14

### Python Implementation

```python
def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr
```

### Interpretation

**Absolute value**: ATR measures volatility magnitude, not direction. Higher ATR = higher volatility; lower ATR = lower volatility.

**Relative assessment**:
- Compare current ATR to historical ATR range for the security
- ATR expanding: Volatility increasing
- ATR contracting: Volatility decreasing
- ATR at historical highs: Extreme volatility, potential exhaustion
- ATR at historical lows: Compression, potential breakout ahead

**Percentage ATR** for cross-asset comparison:
```
ATR% = (ATR / Close) × 100
```

### Trading Applications

**Position sizing**: Scale position size inversely with ATR. Higher volatility = smaller position to maintain consistent risk.

Example:
```
Position Size = Account Risk / (ATR × ATR Multiplier)
```

**Stop-loss placement**:
- Conservative: 2× ATR from entry
- Standard: 1.5× ATR from entry
- Aggressive: 1× ATR from entry

Place stops beyond normal price fluctuation range to avoid premature exit.

**Profit targets**: Set targets at 2-3× ATR to achieve favorable risk/reward ratios.

**Volatility breakouts**: ATR compression (lowest 20% of range) often precedes significant price moves. Monitor for expansion confirmation.

**Trailing stops**: Use ATR-based trailing stops:
```
Long: Stop = Highest High - (ATR × Multiplier)
Short: Stop = Lowest Low + (ATR × Multiplier)
```

### ATR Limitations

**Non-directional**: ATR does not indicate price direction, only volatility magnitude.

**Lagging**: ATR smooths data over N periods, lagging current volatility changes.

**Absolute values**: ATR values are not directly comparable across securities with different price levels. Use ATR% for comparison.

---

## Bollinger Bands

### Purpose
Dynamic volatility bands that expand and contract with market volatility. Identifies overbought/oversold conditions, potential reversals, and breakout opportunities.

### Components

**Middle Band**: 20-period Simple Moving Average
**Upper Band**: Middle Band + (2 × 20-period Standard Deviation)
**Lower Band**: Middle Band - (2 × 20-period Standard Deviation)

Standard settings: 20-period SMA, 2 standard deviations

### Calculation

```python
def calculate_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    middle_band = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)

    # Calculate %B (position within bands)
    percent_b = (close - lower_band) / (upper_band - lower_band)

    # Calculate bandwidth (band width as % of middle band)
    bandwidth = (upper_band - lower_band) / middle_band

    return pd.DataFrame({
        'Upper': upper_band,
        'Middle': middle_band,
        'Lower': lower_band,
        '%B': percent_b,
        'Bandwidth': bandwidth
    })
```

### Interpretation

**Band positioning**:
- Price at upper band: Overbought, potential reversal or trend strength
- Price at lower band: Oversold, potential reversal or trend weakness
- Price at middle band: Fair value, neutral

**Bollinger Squeeze**: Bands contract to narrow range (bandwidth near historical low)
- Indicates low volatility compression
- Often precedes significant price movement
- Trade the breakout direction

**Bollinger Expansion**: Bands widen significantly
- Indicates high volatility
- Often follows squeeze or major news
- Suggests trend development or climax

**%B indicator**:
- %B > 1.0: Price above upper band
- %B = 0.5: Price at middle band
- %B < 0.0: Price below lower band
- %B can identify divergences (price makes new high but %B does not)

**Walking the bands**:
- In strong uptrend: Price consistently touches/exceeds upper band
- In strong downtrend: Price consistently touches/exceeds lower band
- Indicates persistent directional pressure

### Trading Applications

**Mean reversion in range-bound markets**:
- Buy when price touches lower band and reversal candlestick forms
- Sell when price touches upper band and reversal candlestick forms
- Exit at middle band or opposite band

**Trend following in directional markets**:
- In uptrend, use pullbacks to middle band as buy opportunities
- In downtrend, use rallies to middle band as sell opportunities
- Do not fade bands in strong trends (avoid selling upper band in uptrend)

**Squeeze breakouts**:
1. Identify squeeze: Bandwidth at historical low (lowest 20% of 6-month range)
2. Wait for directional break: Price closes outside bands with increased volume
3. Enter in breakout direction
4. Set stop below/above opposite band

**Bollinger Band %B divergence**:
- Price makes new high but %B makes lower high: Bearish divergence
- Price makes new low but %B makes higher low: Bullish divergence
- Trade divergences at major support/resistance levels

### Advanced Techniques

**Multiple time frames**:
- Daily bands for swing trade context
- 4-hour bands for intermediate entries
- 1-hour bands for precise timing

**Adaptive standard deviations**:
- Volatile markets: Use 2.5 or 3 standard deviations to avoid premature signals
- Stable markets: Use 1.5 standard deviations for tighter signals

**Keltner Channel comparison**: Bollinger Bands volatility-based; Keltner Channels ATR-based. Use both for confirmation:
- Price outside Bollinger but inside Keltner: Extended but sustainable
- Price outside both: Extreme extension, potential exhaustion

### Bollinger Band Limitations

**Not standalone**: Requires confirmation from momentum indicators, volume, and price action.

**Whipsaws in choppy markets**: Generates many false signals during low-volatility consolidation.

**Parameter sensitivity**: Default 20-period, 2 std dev may not suit all securities. Consider optimization.

---

## Volatility Analysis Framework

### Volatility regime classification

**Low volatility regime** (ATR in lowest 20% of range, narrow Bollinger Bands):
- Characteristics: Range-bound price action, low participation
- Strategies: Prepare for breakout, reduce position sizes
- Risk: Potential whipsaw if false breakout

**Normal volatility regime** (ATR in middle 60% of range):
- Characteristics: Typical price movement, moderate trends
- Strategies: Standard trend-following and mean-reversion
- Risk: Regime change toward extremes

**High volatility regime** (ATR in highest 20% of range, wide Bollinger Bands):
- Characteristics: Large price swings, emotional trading
- Strategies: Reduce position size, widen stops, avoid counter-trend
- Risk: Sudden reversals, gap risk

### Volatility mean reversion

Volatility tends to revert to mean over time:
- Extended low volatility → Expect increase
- Extended high volatility → Expect decrease

Monitor volatility duration:
- Low volatility lasting >30 days: High breakout probability
- High volatility lasting >30 days: Potential exhaustion

### ATR-based position sizing example

Account: $100,000
Risk per trade: 1% = $1,000
ATR: $2.50
ATR multiplier for stop: 2× = $5.00

```
Position Size = $1,000 / $5.00 = 200 shares
```

If ATR increases to $3.00:
```
New Position Size = $1,000 / $6.00 = 167 shares
```

Maintain consistent dollar risk despite volatility changes.

### Volatility expansion trading

1. **Identify compression**: ATR and Bollinger Bandwidth at lows
2. **Monitor for catalyst**: News, earnings, technical breakout
3. **Confirm expansion**: ATR begins rising, bands widening
4. **Enter with trend**: Trade in direction of expansion
5. **Manage with volatility**: Use ATR-based stops and targets

### Volatility contraction trading

1. **Identify extreme volatility**: ATR and Bollinger Bandwidth at highs
2. **Observe exhaustion signals**: Slowing momentum, reversal patterns
3. **Anticipate contraction**: ATR peaks, bands stop expanding
4. **Reduce exposure**: Take profits, tighten stops
5. **Wait for normalization**: Let volatility return to normal before re-entry

---

## Practical Integration

### Combining ATR and Bollinger Bands

**Confirming breakouts**:
- Price breaks Bollinger Band
- ATR expanding confirms genuine move
- Volume increasing validates breakout

**Identifying false breakouts**:
- Price breaks Bollinger Band
- ATR not expanding or declining
- Low volume suggests false breakout

### Volatility-adjusted trading rules

**High volatility (ATR >75th percentile)**:
- Reduce position size by 30-50%
- Widen stops to 2.5-3× ATR
- Set conservative profit targets
- Avoid counter-trend trades

**Low volatility (ATR <25th percentile)**:
- Anticipate breakout, prepare entry orders
- Tighten stops to 1-1.5× ATR during consolidation
- Expand stops once breakout confirmed
- Consider straddle strategies (options)

### Multi-time frame volatility analysis

**Daily ATR**: Primary volatility gauge for swing trades
**4-hour ATR**: Intermediate volatility for day trades
**1-hour ATR**: Short-term volatility for scalping

Ensure time frame consistency: Use daily ATR for daily charts, hourly ATR for hourly charts.

### Volatility metrics dashboard

Monitor these metrics for comprehensive volatility assessment:

1. **Current ATR vs 50-day average ATR**
2. **ATR percentile rank (0-100) over 6 months**
3. **Bollinger Bandwidth percentile rank**
4. **Days since ATR peak/trough**
5. **Volatility regime classification**

Create alerts for:
- ATR reaching historical extremes (>90th or <10th percentile)
- Bollinger Bandwidth compression (lowest 10%)
- Volatility regime changes

---

## References

Bollinger, John (2001). "Bollinger on Bollinger Bands". McGraw-Hill.

Wilder, J. Welles (1978). "New Concepts in Technical Trading Systems". Trend Research.

Perry J. Kaufman (2013). "Trading Systems and Methods". Wiley Trading.
