# Trend Indicators Reference

## Average Directional Index (ADX)

### Purpose
Quantifies trend strength without indicating direction. Values above 25 indicate strong trending conditions; values below 20 suggest weak or absent trends.

### Components

**ADX Line**: Measures trend strength (0-100 scale)
**+DI (Positive Directional Indicator)**: Measures upward price movement
**-DI (Negative Directional Indicator)**: Measures downward price movement

### Calculation

1. Calculate True Range (TR):
   ```
   TR = max[(High - Low), |High - Close_prev|, |Low - Close_prev|]
   ```

2. Calculate Directional Movement:
   ```
   +DM = High - High_prev (if positive, else 0)
   -DM = Low_prev - Low (if positive, else 0)
   ```

3. Smooth TR, +DM, -DM over N periods (typically 14)

4. Calculate Directional Indicators:
   ```
   +DI = 100 × (Smoothed +DM / Smoothed TR)
   -DI = 100 × (Smoothed -DM / Smoothed TR)
   ```

5. Calculate Directional Index:
   ```
   DX = 100 × |+DI - -DI| / (+DI + -DI)
   ```

6. Calculate ADX:
   ```
   ADX = Smoothed DX over N periods
   ```

### Interpretation

**ADX > 25**: Strong trend present
- +DI > -DI: Uptrend
- -DI > +DI: Downtrend
- Follow trend-following strategies

**ADX < 20**: Weak or no trend
- Market likely range-bound
- Consider mean-reversion strategies
- Avoid trend-following systems

**ADX rising**: Trend strengthening
**ADX falling**: Trend weakening or consolidating

### Python Implementation

```python
def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """Calculate ADX and directional indicators."""
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # Smooth components
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    # ADX calculation
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    return pd.DataFrame({
        'ADX': adx,
        '+DI': plus_di,
        '-DI': minus_di
    })
```

### Trading Applications

**Trend confirmation**: Use ADX to validate signals from other indicators. Only take trend-following trades when ADX > 25.

**Trend exhaustion**: Extremely high ADX (>50) may indicate overextended trend vulnerable to reversal.

**Directional bias**: Combine ADX with +DI/-DI crossovers for entry timing in established trends.

---

## Ichimoku Cloud

### Purpose
Comprehensive trend indicator providing support/resistance levels, trend direction, and momentum assessment in a single system.

### Components

**Tenkan-sen (Conversion Line)**: (9-period high + 9-period low) / 2
**Kijun-sen (Base Line)**: (26-period high + 26-period low) / 2
**Senkou Span A (Leading Span A)**: (Tenkan-sen + Kijun-sen) / 2, plotted 26 periods ahead
**Senkou Span B (Leading Span B)**: (52-period high + 52-period low) / 2, plotted 26 periods ahead
**Chikou Span (Lagging Span)**: Current close plotted 26 periods behind
**Kumo (Cloud)**: Area between Senkou Span A and Senkou Span B

### Calculation

```python
def calculate_ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Calculate Ichimoku Cloud components."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = high.rolling(window=9).max()
    period9_low = low.rolling(window=9).min()
    tenkan_sen = (period9_high + period9_low) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = high.rolling(window=26).max()
    period26_low = low.rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = high.rolling(window=52).max()
    period52_low = low.rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    # Chikou Span (Lagging Span): Current close shifted back 26 periods
    chikou_span = close.shift(-26)

    return pd.DataFrame({
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    })
```

### Interpretation

**Bullish signals**:
- Price above cloud (Kumo)
- Tenkan-sen crosses above Kijun-sen
- Chikou Span above price from 26 periods ago
- Senkou Span A above Senkou Span B (green cloud)

**Bearish signals**:
- Price below cloud
- Tenkan-sen crosses below Kijun-sen
- Chikou Span below price from 26 periods ago
- Senkou Span A below Senkou Span B (red cloud)

**Cloud characteristics**:
- Thick cloud: Strong support/resistance
- Thin cloud: Weak support/resistance, easier to penetrate
- Price in cloud: Consolidation, unclear trend

### Trading Applications

**Trend following**: Take long positions when all components align bullish; short when all align bearish.

**Support/resistance**: Cloud acts as dynamic support in uptrends, resistance in downtrends.

**Breakout confirmation**: Cloud breakouts with volume confirmation signal strong trend initiations.

---

## Moving Averages (MA)

### Purpose
Smooth price data to identify trend direction and filter market noise. Foundation for many technical analysis strategies.

### Types

**Simple Moving Average (SMA)**: Arithmetic mean of prices over N periods
**Exponential Moving Average (EMA)**: Weighted average giving more weight to recent prices
**Weighted Moving Average (WMA)**: Linear weighting of prices, most recent weighted highest

### Calculation

**SMA**:
```
SMA = (P1 + P2 + ... + Pn) / n
```

**EMA**:
```
Multiplier = 2 / (n + 1)
EMA_today = (Price_today × Multiplier) + (EMA_yesterday × (1 - Multiplier))
```

**WMA**:
```
WMA = (P1×n + P2×(n-1) + ... + Pn×1) / (n + (n-1) + ... + 1)
```

### Python Implementation

```python
def calculate_moving_averages(close: pd.Series, periods: list = [20, 50, 200]) -> pd.DataFrame:
    """Calculate SMA and EMA for multiple periods."""
    result = pd.DataFrame(index=close.index)

    for period in periods:
        result[f'SMA_{period}'] = close.rolling(window=period).mean()
        result[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()

    return result
```

### Interpretation

**Price position**:
- Price > MA: Bullish, uptrend
- Price < MA: Bearish, downtrend
- Price at MA: Potential support/resistance

**MA slope**:
- Rising MA: Uptrend
- Falling MA: Downtrend
- Flat MA: Consolidation

**Crossovers**:
- Golden Cross: Short-term MA crosses above long-term MA (bullish)
- Death Cross: Short-term MA crosses below long-term MA (bearish)

### Common Periods

**Short-term**: 10, 20-day (weeks)
**Medium-term**: 50-day (2.5 months)
**Long-term**: 100, 200-day (major trend)

### Trading Applications

**Trend identification**: Use 200-day MA as primary trend indicator. Price above = bull market; below = bear market.

**Dynamic support/resistance**: 20-day and 50-day MAs often act as support in uptrends, resistance in downtrends.

**Entry timing**: Buy pullbacks to rising MA in uptrend; sell rallies to falling MA in downtrend.

**Multiple MA systems**: Use 20/50/200 MA confluence for high-probability setups.

---

## Parabolic SAR

### Purpose
Identify trend direction and potential reversal points. Provides trailing stop levels for position management.

### Calculation

**SAR (Stop and Reverse)**:

Initial SAR = Most recent extreme point (EP)

For long position:
```
SAR_tomorrow = SAR_today + AF × (EP - SAR_today)
```

For short position:
```
SAR_tomorrow = SAR_today - AF × (SAR_today - EP)
```

Where:
- AF (Acceleration Factor): Starts at 0.02, increases by 0.02 each time EP updates, max 0.20
- EP (Extreme Point): Highest high for long, lowest low for short

Reversal occurs when price crosses SAR.

### Python Implementation

```python
def calculate_parabolic_sar(high: pd.Series, low: pd.Series,
                           af_start: float = 0.02, af_increment: float = 0.02,
                           af_max: float = 0.20) -> pd.Series:
    """Calculate Parabolic SAR."""
    length = len(high)
    sar = np.zeros(length)
    ep = np.zeros(length)
    af = np.zeros(length)
    trend = np.zeros(length, dtype=int)

    # Initialize
    sar[0] = low[0]
    ep[0] = high[0]
    af[0] = af_start
    trend[0] = 1  # 1 for uptrend, -1 for downtrend

    for i in range(1, length):
        # Calculate SAR
        sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])

        # Check for reversal
        if trend[i-1] == 1:  # Uptrend
            if low[i] < sar[i]:
                # Reverse to downtrend
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = af_start
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # Downtrend
            if high[i] > sar[i]:
                # Reverse to uptrend
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = af_start
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]

    return pd.Series(sar, index=high.index)
```

### Interpretation

**SAR position**:
- SAR below price: Uptrend, long position
- SAR above price: Downtrend, short position

**SAR flip**: Reversal signal
- SAR moves from below to above price: Sell signal
- SAR moves from above to below price: Buy signal

**SAR distance**: Increasing distance from price indicates accelerating trend

### Trading Applications

**Trailing stops**: Use SAR as dynamic stop-loss level. Exit long when price touches SAR from above; exit short when price touches SAR from below.

**Trend following**: Enter long on SAR flip to bullish; enter short on SAR flip to bearish.

**Acceleration tracking**: Monitor AF increases to gauge trend momentum. Higher AF suggests stronger trend.

### Limitations

**Choppy markets**: Frequent whipsaws during consolidation or range-bound periods.

**Early signals**: Can reverse prematurely in strong trends during minor pullbacks.

**Lagging nature**: SAR follows price action and may provide late signals in fast-moving markets.

---

## Best Practices

**Combine indicators**: Use trend indicators together for confirmation. Example: ADX > 25 + price above 200-day MA + bullish Ichimoku cloud alignment.

**Time frame consistency**: Apply indicators on same time frame for consistent analysis.

**Validate with price action**: Confirm indicator signals with candlestick patterns and support/resistance levels.

**Regular review**: Monitor indicator effectiveness across market regimes. Adjust parameters or methods as conditions change.
