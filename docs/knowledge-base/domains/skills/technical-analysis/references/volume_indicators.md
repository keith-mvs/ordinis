# Volume Indicators Reference

## On-Balance Volume (OBV)

### Purpose
Cumulative volume indicator that relates volume flow to price movement. Confirms trends, identifies divergences, and anticipates price reversals through volume accumulation/distribution patterns.

### Calculation

```
If Close > Close_previous: OBV = OBV_previous + Volume
If Close < Close_previous: OBV = OBV_previous - Volume
If Close = Close_previous: OBV = OBV_previous
```

Starting value typically set to 0 or first period's volume.

### Python Implementation

```python
def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]

    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]

    return obv
```

### Interpretation

**Trend confirmation**:
- OBV rising with price: Confirms uptrend (volume supporting price advance)
- OBV falling with price: Confirms downtrend (volume supporting price decline)
- Strong trend shows OBV and price moving in tandem

**Divergence signals**:
- Bullish divergence: Price makes lower low, OBV makes higher low (accumulation)
- Bearish divergence: Price makes higher high, OBV makes lower high (distribution)
- Divergences suggest potential trend reversal

**OBV trend**:
- OBV in uptrend (making higher highs and higher lows): Accumulation phase
- OBV in downtrend (making lower highs and lower lows): Distribution phase
- OBV breaking trendline may precede price trendline break

**Breakout confirmation**:
- Price breaks resistance with OBV confirming new high: Valid breakout
- Price breaks resistance but OBV does not confirm: Weak breakout, potential failure

### Trading Applications

**Confirming breakouts**:
1. Identify key resistance level
2. Price breaks above resistance
3. Check OBV: Must make new high to confirm
4. Enter long if OBV confirms, avoid if OBV lags

**Divergence trading**:
1. Identify price extreme (new high or low)
2. Compare with OBV action at same price level
3. If divergence present, wait for price reversal confirmation
4. Enter in direction indicated by divergence

**Trend validation**:
- In uptrend, OBV should make new highs with price
- If OBV fails to confirm new price high, trend strength questionable
- Consider taking profits or tightening stops

**Early warning system**:
- OBV often leads price
- OBV breaking trend before price suggests impending price trend change
- Monitor OBV trendlines for early signals

### OBV Moving Average

Apply moving average to OBV for smoother signals:

```python
def obv_with_ma(close: pd.Series, volume: pd.Series, ma_period: int = 20) -> pd.DataFrame:
    """Calculate OBV with moving average."""
    obv = calculate_obv(close, volume)
    obv_ma = obv.rolling(window=ma_period).mean()

    return pd.DataFrame({
        'OBV': obv,
        'OBV_MA': obv_ma
    })
```

**Signals**:
- OBV crosses above OBV MA: Bullish
- OBV crosses below OBV MA: Bearish

### Advanced OBV Analysis

**Volume analysis by regime**:
- Accumulation: OBV rising during price consolidation or minor pullback
- Distribution: OBV falling during price consolidation or minor rally
- These patterns suggest smart money positioning before major moves

**OBV failure swings**:
- Price makes new high, OBV makes new high, then OBV quickly reverses: Distribution signal
- Price makes new low, OBV makes new low, then OBV quickly reverses: Accumulation signal

**Multiple time frame OBV**:
- Daily OBV: Primary volume trend
- Weekly OBV: Longer-term institutional volume flow
- Alignment across time frames strengthens signal

### Limitations

**Absolute values meaningless**: OBV is relative indicator. Compare current OBV to historical OBV for same security, not across different securities.

**All volume equal**: OBV treats all volume the same regardless of trade size. Large institutional blocks and small retail trades weighted equally.

**Intraday gaps**: Overnight gaps treated as single-period moves. May distort OBV in gap-prone securities.

**No price magnitude consideration**: $1 price move on 1M volume treated same as $10 price move on 1M volume.

---

## Volume Analysis Principles

### Volume significance

**Volume validates price action**:
- High volume + price breakout = Strong signal
- Low volume + price breakout = Weak signal, potential failure
- High volume + price reversal = Significant turning point
- Low volume + price reversal = Minor countertrend move

### Volume patterns

**Volume spike**: Single period with significantly higher volume (>2× average)
- At resistance: Potential reversal or breakout
- At support: Potential reversal or breakdown
- Mid-trend: Possible continuation or exhaustion

**Diminishing volume**: Progressively lower volume on each price move
- In uptrend: Trend losing momentum, potential reversal
- In downtrend: Selling pressure exhausting, potential reversal

**Volume climax**: Extreme volume spike marking trend endpoint
- Buying climax: Massive volume at top, then price reversal
- Selling climax: Massive volume at bottom, then price reversal

### Volume confirmation checklist

Before entering trade, verify:
1. **Trend confirmation**: Volume expanding in trend direction
2. **Breakout validation**: Volume surge on breakout day
3. **Pullback character**: Volume declining on pullback (healthy)
4. **Divergence check**: No bearish volume divergence developing

### Volume-price relationships

**Price up, volume up**: Healthy uptrend, buyers in control
**Price up, volume down**: Weak rally, potential reversal
**Price down, volume up**: Healthy downtrend, sellers in control
**Price down, volume down**: Weak decline, potential reversal

---

## Relative Volume (RVOL)

### Purpose
Compares current volume to average volume to identify unusual activity.

### Calculation

```
RVOL = Current Volume / Average Volume (past N days)
```

Standard period: 20 or 50 days

```python
def calculate_relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Relative Volume."""
    avg_volume = volume.rolling(window=period).mean()
    rvol = volume / avg_volume
    return rvol
```

### Interpretation

**RVOL = 1.0**: Volume at average
**RVOL > 1.5**: Above-average volume, increased interest
**RVOL > 2.0**: High volume, significant activity
**RVOL < 0.5**: Below-average volume, low interest

### Applications

**Breakout screening**: Look for RVOL >2.0 on breakout day to confirm validity.

**Climax detection**: RVOL >3.0 often marks exhaustion points.

**Low activity warning**: RVOL <0.5 suggests lack of conviction, avoid entry.

---

## Volume Weighted Average Price (VWAP)

### Purpose
Represents average price weighted by volume. Institutional benchmark for execution quality.

### Calculation

```
VWAP = Σ(Price × Volume) / Σ(Volume)
```

Typically calculated from market open and reset daily.

```python
def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap
```

### Interpretation

**Price vs VWAP**:
- Price > VWAP: Stock trading above average, bullish sentiment
- Price < VWAP: Stock trading below average, bearish sentiment

**Institutional reference**: Institutions use VWAP as performance benchmark. Buying below VWAP considered good execution.

**Support/resistance**: VWAP acts as dynamic support in uptrends, resistance in downtrends.

### Trading applications

**Mean reversion**: In trending market, buy pullbacks to VWAP in uptrend, sell rallies to VWAP in downtrend.

**Breakout trades**: VWAP break in direction of trend with volume confirmation provides entry signal.

**Institutional flow**: Multiple tests of VWAP followed by break suggests institutional accumulation/distribution.

---

## Accumulation/Distribution Line (A/D Line)

### Purpose
Similar to OBV but considers price position within daily range, providing more nuanced volume flow analysis.

### Calculation

```
Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
Money Flow Volume = Money Flow Multiplier × Volume
A/D Line = Previous A/D Line + Money Flow Volume
```

### Python Implementation

```python
def calculate_accumulation_distribution(high: pd.Series, low: pd.Series,
                                       close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Accumulation/Distribution Line."""
    mfm = ((close - low) - (high - close)) / (high - low)
    mfv = mfm * volume
    ad_line = mfv.cumsum()
    return ad_line
```

### Interpretation

Similar to OBV but weighted by close position in range:
- Close near high → Money Flow Multiplier near +1 (accumulation)
- Close near low → Money Flow Multiplier near -1 (distribution)
- Close at midpoint → Money Flow Multiplier near 0 (neutral)

### Applications

More sensitive than OBV to intraday price action. Use for same applications as OBV but with finer granularity in volume flow assessment.

---

## Practical Integration

### Volume + Price Action

**High-confidence breakout**:
1. Price breaks key resistance
2. Volume >2× average (RVOL >2.0)
3. OBV confirms with new high
4. Close in top 25% of day's range

**False breakout warning**:
1. Price breaks resistance
2. Volume below average (RVOL <1.0)
3. OBV not confirming
4. Close in lower half of day's range

### Volume Divergence Trading

1. **Identify divergence**: Price and OBV moving opposite directions
2. **Confirm with other indicators**: Check RSI, MACD for supporting divergence
3. **Wait for price confirmation**: Reversal candlestick or support/resistance test
4. **Enter with volume**: Enter when volume validates reversal move

### Volume-Based Position Sizing

**High volume conviction**: Increase position size when:
- Breakout with volume >2× average
- OBV strongly confirming
- Multiple volume indicators aligned

**Low volume caution**: Reduce position size when:
- Breakout with volume <1× average
- OBV diverging from price
- Conflicting volume signals

---

## References

Granville, Joseph E. (1963). "Granville's New Key to Stock Market Profits". Prentice-Hall.

Williams, Larry (1999). "Long-Term Secrets to Short-Term Trading". Wiley.

Buff Dormeier (2011). "Investing with Volume Analysis". FT Press.
