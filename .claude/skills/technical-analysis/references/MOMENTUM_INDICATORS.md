# Momentum Indicators Reference

## Commodity Channel Index (CCI)

### Purpose
Measures deviation from average price to identify overbought/oversold conditions and cyclical turns. Effective in both trending and range-bound markets.

### Calculation

```
CCI = (Typical Price - SMA of Typical Price) / (0.015 × Mean Deviation)
```

Where:
```
Typical Price = (High + Low + Close) / 3
Mean Deviation = Average of |Typical Price - SMA of Typical Price| over N periods
```

Standard period: 20

### Python Implementation

```python
def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=period).mean()
    mean_deviation = (typical_price - sma_tp).abs().rolling(window=period).mean()
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
    return cci
```

### Interpretation

**Overbought/Oversold levels**:
- CCI > +100: Overbought, potential reversal or trend strength
- CCI < -100: Oversold, potential reversal or trend weakness
- CCI between -100 and +100: Normal range

**Directional signals**:
- CCI crosses above +100: Strong upward momentum
- CCI crosses below -100: Strong downward momentum
- CCI crosses above 0: Bullish momentum shift
- CCI crosses below 0: Bearish momentum shift

**Divergences**:
- Price makes new high, CCI does not: Bearish divergence
- Price makes new low, CCI does not: Bullish divergence

### Trading Applications

**Trend confirmation**: In uptrend, look for CCI to remain above 0 with excursions above +100. In downtrend, CCI stays below 0 with drops below -100.

**Overbought/oversold trades**: In range-bound markets, sell when CCI exceeds +100 and buy when CCI drops below -100.

**Divergence signals**: Trade divergences in combination with support/resistance levels for reversal trades.

---

## Moving Average Convergence Divergence (MACD)

### Purpose
Identifies momentum shifts, trend direction changes, and potential entry/exit points through moving average relationships.

### Components

**MACD Line**: 12-period EMA - 26-period EMA
**Signal Line**: 9-period EMA of MACD Line
**Histogram**: MACD Line - Signal Line

### Calculation

```python
def calculate_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD components."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    })
```

### Interpretation

**Crossovers**:
- MACD crosses above Signal: Bullish signal
- MACD crosses below Signal: Bearish signal

**Zero-line crossovers**:
- MACD crosses above 0: Bullish momentum, 12-EMA > 26-EMA
- MACD crosses below 0: Bearish momentum, 12-EMA < 26-EMA

**Histogram**:
- Expanding histogram: Momentum accelerating
- Contracting histogram: Momentum decelerating
- Histogram crosses zero: Same as MACD/Signal crossover

**Divergences**:
- Bullish divergence: Price makes lower low, MACD makes higher low
- Bearish divergence: Price makes higher high, MACD makes lower high

### Trading Applications

**Entry timing**: Enter long on MACD bullish crossover above signal line with histogram turning positive. Enter short on bearish crossover.

**Trend confirmation**: Use zero-line position to confirm trend. MACD above zero supports long bias; below zero supports short bias.

**Divergence trades**: Trade divergences at major support/resistance levels for high-probability reversals.

**Momentum assessment**: Monitor histogram slope. Steep positive slope = strong bullish momentum; steep negative slope = strong bearish momentum.

### Limitations

**Lagging indicator**: MACD follows price action. Signals may arrive after significant move has occurred.

**Whipsaws**: Frequent false signals in choppy, range-bound markets.

**Parameter sensitivity**: Standard 12/26/9 settings may not suit all securities or time frames. Consider optimization for specific instruments.

---

## Relative Strength Index (RSI)

### Purpose
Measures momentum by comparing magnitude of recent gains to recent losses. Identifies overbought/oversold conditions and potential reversals.

### Calculation

```
RS = Average Gain over N periods / Average Loss over N periods
RSI = 100 - [100 / (1 + RS)]
```

Standard period: 14

First calculation uses simple average. Subsequent values use exponential smoothing:
```
Average Gain = [(Previous Avg Gain × 13) + Current Gain] / 14
Average Loss = [(Previous Avg Loss × 13) + Current Loss] / 14
```

### Python Implementation

```python
def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```

### Interpretation

**Overbought/Oversold**:
- RSI > 70: Overbought, potential reversal or strong trend
- RSI < 30: Oversold, potential reversal or weak trend
- RSI = 50: Neutral momentum

**Trend interpretation**:
- Strong uptrend: RSI ranges 40-90, rarely drops below 40
- Strong downtrend: RSI ranges 10-60, rarely rises above 60
- Adjust overbought/oversold levels in strong trends (80/40 for uptrend, 60/20 for downtrend)

**Failure swings**:
- Top failure swing: RSI peaks above 70, declines, rallies but fails to exceed prior peak, then breaks below recent trough (bearish)
- Bottom failure swing: RSI troughs below 30, rallies, declines but holds above prior trough, then breaks above recent peak (bullish)

**Divergences**:
- Bullish divergence: Price makes lower low, RSI makes higher low (reversal signal)
- Bearish divergence: Price makes higher high, RSI makes lower high (reversal signal)

### Trading Applications

**Mean reversion**: In range-bound markets, sell when RSI exceeds 70 and buy when RSI drops below 30.

**Trend following**: In trending markets, use RSI pullbacks to trend support (40 in uptrend, 60 in downtrend) as entry opportunities.

**Divergence trading**: Combine RSI divergences with support/resistance levels for reversal trades. Wait for price confirmation.

**Failure swings**: Trade failure swing patterns as high-probability reversal signals, especially at major price levels.

### Limitations

**Overbought can remain overbought**: Strong trends can keep RSI in overbought/oversold territory for extended periods. Avoid counter-trend trades in strong trends.

**False signals**: RSI generates many signals. Filter with trend indicators and price action confirmation.

---

## Stochastic Oscillator

### Purpose
Compares closing price to price range over given period. Measures momentum and identifies overbought/oversold conditions, particularly effective for short-term trading.

### Components

**%K (Fast Stochastic)**: Raw stochastic value
**%D (Slow Stochastic)**: 3-period SMA of %K

### Calculation

```
%K = 100 × (Close - Lowest Low) / (Highest High - Lowest Low)
%D = 3-period SMA of %K
```

Standard period: 14

### Python Implementation

```python
def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                        k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    percent_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    percent_d = percent_k.rolling(window=d_period).mean()

    return pd.DataFrame({
        '%K': percent_k,
        '%D': percent_d
    })
```

### Interpretation

**Overbought/Oversold**:
- %K > 80 and %D > 80: Overbought
- %K < 20 and %D < 20: Oversold

**Crossovers**:
- %K crosses above %D in oversold zone: Bullish signal
- %K crosses below %D in overbought zone: Bearish signal

**Divergences**:
- Bullish divergence: Price makes lower low, Stochastic makes higher low
- Bearish divergence: Price makes higher high, Stochastic makes lower high

**Bull/Bear setup**:
- Bull setup: %D below 30, three consecutive lower lows in %D, followed by rise
- Bear setup: %D above 70, three consecutive higher highs in %D, followed by decline

### Trading Applications

**Short-term reversals**: Trade %K/%D crossovers in overbought/oversold zones for quick scalps.

**Trend confirmation**: In uptrend, buy %K/%D crossovers above 50. In downtrend, sell crossovers below 50.

**Divergence signals**: Trade divergences at support/resistance for reversal entries.

**Multiple time frames**: Use longer time frame Stochastic for trend, shorter time frame for entry timing.

### Stochastic Settings

**Fast Stochastic (5,3,3)**: Highly sensitive, many signals
**Slow Stochastic (14,3,3)**: Standard, balanced sensitivity
**Full Stochastic (14,3,1)**: Most responsive variation

### Limitations

**Choppy markets**: Generates excessive signals in range-bound conditions with frequent whipsaws.

**Overbought in uptrends**: Can remain overbought during strong uptrends. Avoid counter-trend shorts.

**False divergences**: Divergences occur frequently and require price action confirmation.

---

## Indicator Comparison

### Response Speed

**Fastest**: Stochastic Oscillator, CCI
**Moderate**: RSI, MACD
**Slower**: MACD histogram turns

### Best Time Frames

**Intraday/Short-term**: Stochastic (5-minute to daily)
**Swing trading**: RSI, CCI (daily to weekly)
**Position trading**: MACD (daily to monthly)

### Market Conditions

**Trending markets**: MACD, RSI with adjusted levels
**Range-bound markets**: Stochastic, CCI with traditional overbought/oversold
**Volatile markets**: CCI (wider range accommodates volatility)

---

## Practical Integration

### Multi-indicator confirmation

**Strong bullish setup**:
- MACD bullish crossover
- RSI rising from oversold (<30) toward 50
- Stochastic %K crosses above %D in oversold zone
- CCI crosses above -100

**Strong bearish setup**:
- MACD bearish crossover
- RSI falling from overbought (>70) toward 50
- Stochastic %K crosses below %D in overbought zone
- CCI crosses below +100

### Divergence validation

Confirm divergences across multiple momentum indicators:
1. Identify divergence on RSI
2. Check for same divergence on MACD histogram
3. Verify with Stochastic pattern
4. Wait for price action confirmation (candlestick reversal, support/resistance test)

### Parameter optimization

Test multiple period settings for specific securities:
- Volatile stocks: Longer periods (RSI 21, Stochastic 21)
- Stable stocks: Standard periods (RSI 14, Stochastic 14)
- Commodities: Shorter periods (RSI 9, Stochastic 9)

Avoid overfitting. Select periods that perform across multiple market cycles.
