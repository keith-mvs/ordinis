# Technical Analysis Methods - Knowledge Base

## Purpose

This section provides technical analysis methods as **modular building blocks** that can be combined into algorithmic entry/exit rules. Each method includes:
- What it measures
- Standard parameters
- Rule-based expressions (if/then logic)
- Common pitfalls and false signal scenarios
- Academic validation status

---

## 1. Trend Identification

### 1.1 Moving Averages

**What it measures**: Smoothed price trend direction and momentum.

**Types**:
| Type | Formula | Characteristics |
|------|---------|-----------------|
| SMA | Sum(Close, N) / N | Equal weight, lagging |
| EMA | Close × k + EMA_prev × (1-k), k=2/(N+1) | Recent weight, responsive |
| WMA | Weighted sum / Sum of weights | Linear weight decay |
| HMA | WMA(2×WMA(n/2) - WMA(n), sqrt(n)) | Reduced lag, smooth |

**Standard Parameters**:
- Short-term: 9, 10, 20 periods
- Medium-term: 50 periods
- Long-term: 100, 200 periods

**Rule Templates**:
```python
# Trend direction
UPTREND = price > MA(50)
DOWNTREND = price < MA(50)

# MA crossover (Golden Cross / Death Cross)
GOLDEN_CROSS = MA(50) crosses_above MA(200)
DEATH_CROSS = MA(50) crosses_below MA(200)

# Price-MA relationship
BULLISH_PULLBACK = price > MA(200) AND price touches MA(20) from above
BEARISH_RALLY = price < MA(200) AND price touches MA(20) from below

# Trend strength via MA slope
STRONG_UPTREND = MA(20) > MA(20)[5] AND MA(50) > MA(50)[5]  # Rising MAs
```

**Pitfalls**:
- Whipsaws in ranging/choppy markets
- Lag causes late entries in fast moves
- Crossovers fail in mean-reverting regimes

**Academic Notes**:
- Mixed evidence on MA profitability after costs (Bessembinder & Chan, 1998)
- May work better as filters than primary signals

---

### 1.2 Higher Highs / Higher Lows (HH/HL)

**What it measures**: Structural trend via swing points.

**Definitions**:
- **Swing High**: Bar where high > high of N bars before AND after
- **Swing Low**: Bar where low < low of N bars before AND after
- **Uptrend**: Series of HH and HL
- **Downtrend**: Series of LL and LH

**Parameters**:
- Lookback for swing detection: 3-5 bars typical

**Rule Templates**:
```python
# Swing point detection
SWING_HIGH = high > highest(high[1:N]) AND high > highest(high[-N:-1])
SWING_LOW = low < lowest(low[1:N]) AND low < lowest(low[-N:-1])

# Trend structure
UPTREND_STRUCTURE = last_swing_high > prev_swing_high AND last_swing_low > prev_swing_low
DOWNTREND_STRUCTURE = last_swing_low < prev_swing_low AND last_swing_high < prev_swing_high

# Trend break
UPTREND_BREAK = price < last_swing_low  # Structure violation
```

**Pitfalls**:
- Requires lookback, so delayed detection
- Subjective swing identification without strict rules
- Fails in choppy, overlapping price action

---

### 1.3 Trendlines & Regression Channels

**What it measures**: Dynamic support/resistance along trend.

**Methods**:
- **Linear Regression**: Least squares fit to price data
- **Standard Error Channels**: Regression ± N standard errors
- **Anchored trendlines**: Manual or algorithmic anchor points

**Parameters**:
- Regression period: 20, 50, 100 bars
- Channel width: 1.5, 2.0, 2.5 standard errors

**Rule Templates**:
```python
# Regression trend
REG_SLOPE = linear_regression_slope(close, 50)
UPTREND = REG_SLOPE > 0
DOWNTREND = REG_SLOPE < 0
STRONG_TREND = abs(REG_SLOPE) > threshold

# Channel position
CHANNEL_UPPER = regression_line + 2 * std_error
CHANNEL_LOWER = regression_line - 2 * std_error
OVERBOUGHT = close > CHANNEL_UPPER
OVERSOLD = close < CHANNEL_LOWER
```

**Pitfalls**:
- Regression is backward-looking, can miss trend changes
- Standard error channels assume normal distribution

---

## 2. Momentum Indicators

### 2.1 Relative Strength Index (RSI)

**What it measures**: Speed and magnitude of price changes; overbought/oversold conditions.

**Formula**:
```
RS = Average Gain / Average Loss (over N periods)
RSI = 100 - (100 / (1 + RS))
```

**Standard Parameters**:
- Period: 14 (Wilder's original), also 7, 21, 28
- Overbought: 70 (traditional), 80 (strong trend)
- Oversold: 30 (traditional), 20 (strong trend)

**Rule Templates**:
```python
# Overbought/Oversold
OVERBOUGHT = RSI(14) > 70
OVERSOLD = RSI(14) < 30

# Extreme readings
EXTREME_OB = RSI(14) > 80
EXTREME_OS = RSI(14) < 20

# Centerline crossover (trend confirmation)
BULLISH_MOMENTUM = RSI(14) crosses_above 50
BEARISH_MOMENTUM = RSI(14) crosses_below 50

# Divergence detection
BULLISH_DIVERGENCE = price makes lower_low AND RSI makes higher_low
BEARISH_DIVERGENCE = price makes higher_high AND RSI makes lower_high

# Range-bound mean reversion
MEAN_REVERT_LONG = RSI(14) < 30 AND RSI(14) > RSI(14)[1]  # Oversold + turning up
MEAN_REVERT_SHORT = RSI(14) > 70 AND RSI(14) < RSI(14)[1]  # Overbought + turning down
```

**Pitfalls**:
- RSI can stay overbought/oversold for extended periods in strong trends
- Divergences can persist without price reversal
- False signals in trending markets when used for mean reversion

**Academic Notes**:
- Some evidence of short-term predictability (Chong & Ng, 2008)
- Works better with filters (trend, volume confirmation)

---

### 2.2 MACD (Moving Average Convergence Divergence)

**What it measures**: Trend direction, momentum, and potential reversals via MA relationships.

**Formula**:
```
MACD Line = EMA(12) - EMA(26)
Signal Line = EMA(9) of MACD Line
Histogram = MACD Line - Signal Line
```

**Standard Parameters**:
- Fast EMA: 12
- Slow EMA: 26
- Signal EMA: 9
- Variations: 8/17/9, 5/35/5

**Rule Templates**:
```python
# Signal line crossover
BULLISH_CROSS = MACD crosses_above Signal
BEARISH_CROSS = MACD crosses_below Signal

# Zero line crossover (trend confirmation)
BULLISH_TREND = MACD crosses_above 0
BEARISH_TREND = MACD crosses_below 0

# Histogram momentum
INCREASING_MOMENTUM = histogram > histogram[1] > histogram[2]
DECREASING_MOMENTUM = histogram < histogram[1] < histogram[2]

# Divergence
BULLISH_DIV = price makes lower_low AND MACD makes higher_low
BEARISH_DIV = price makes higher_high AND MACD makes lower_high

# Combined signal
STRONG_BUY = MACD crosses_above Signal AND MACD > 0
STRONG_SELL = MACD crosses_below Signal AND MACD < 0
```

**Pitfalls**:
- Lagging indicator, late signals
- Whipsaws in choppy markets
- Multiple parameters increase overfitting risk

---

### 2.3 Stochastic Oscillator

**What it measures**: Current price relative to price range over N periods.

**Formula**:
```
%K = 100 × (Close - Lowest Low) / (Highest High - Lowest Low)
%D = SMA(3) of %K
```

**Standard Parameters**:
- Lookback: 14 periods
- %K smoothing: 3
- %D smoothing: 3
- Overbought: 80, Oversold: 20

**Rule Templates**:
```python
# Overbought/Oversold
OVERBOUGHT = %K > 80
OVERSOLD = %K < 20

# Stochastic crossover
BUY_SIGNAL = %K crosses_above %D AND %K < 20  # Oversold cross
SELL_SIGNAL = %K crosses_below %D AND %K > 80  # Overbought cross

# Trend filter combination
FILTERED_BUY = %K crosses_above %D AND %K < 30 AND price > MA(50)
```

**Pitfalls**:
- Very sensitive, many false signals
- Can remain extreme in strong trends
- Requires additional filters for reliability

---

### 2.4 Rate of Change (ROC) / Momentum

**What it measures**: Percentage price change over N periods.

**Formula**:
```
ROC = ((Close - Close[N]) / Close[N]) × 100
Momentum = Close - Close[N]  # Absolute version
```

**Standard Parameters**:
- Short-term: 10, 12 periods
- Medium-term: 20, 25 periods
- Long-term: 50, 100 periods

**Rule Templates**:
```python
# Momentum direction
POSITIVE_MOMENTUM = ROC(10) > 0
NEGATIVE_MOMENTUM = ROC(10) < 0

# Momentum acceleration
ACCELERATING = ROC(10) > ROC(10)[1] > ROC(10)[2]
DECELERATING = ROC(10) < ROC(10)[1] < ROC(10)[2]

# Zero-line cross
BULLISH = ROC(10) crosses_above 0
BEARISH = ROC(10) crosses_below 0

# Extreme readings (percentile-based)
EXTREME_HIGH = ROC(10) > percentile(ROC(10), 90, lookback=252)
EXTREME_LOW = ROC(10) < percentile(ROC(10), 10, lookback=252)
```

---

## 3. Volatility Measures

### 3.1 Average True Range (ATR)

**What it measures**: Volatility as average of true ranges.

**Formula**:
```
True Range = max(High - Low, |High - Close[1]|, |Low - Close[1]|)
ATR = SMA or EMA of True Range over N periods
```

**Standard Parameters**:
- Period: 14 (Wilder's), also 10, 20
- Multipliers for stops: 1.5, 2.0, 2.5, 3.0

**Rule Templates**:
```python
# Volatility state
HIGH_VOLATILITY = ATR(14) > ATR(14).rolling(50).mean() * 1.5
LOW_VOLATILITY = ATR(14) < ATR(14).rolling(50).mean() * 0.75

# ATR-based stops
STOP_LOSS = entry_price - 2 * ATR(14)  # Long position
STOP_LOSS = entry_price + 2 * ATR(14)  # Short position

# Position sizing (volatility-adjusted)
POSITION_SIZE = risk_amount / (ATR(14) * atr_multiplier)

# Volatility breakout
BREAKOUT = close > close[1] + 1.5 * ATR(14)
BREAKDOWN = close < close[1] - 1.5 * ATR(14)
```

**Usage Notes**:
- ATR is non-directional (measures volatility, not direction)
- Excellent for position sizing and stop placement
- Normalizes across different price levels

---

### 3.2 Bollinger Bands

**What it measures**: Price deviation from moving average; volatility expansion/contraction.

**Formula**:
```
Middle Band = SMA(20)
Upper Band = SMA(20) + 2 × StdDev(20)
Lower Band = SMA(20) - 2 × StdDev(20)
%B = (Close - Lower) / (Upper - Lower)
Bandwidth = (Upper - Lower) / Middle
```

**Standard Parameters**:
- Period: 20
- Standard deviations: 2.0 (also 1.5, 2.5)

**Rule Templates**:
```python
# Band position
AT_UPPER = close >= upper_band
AT_LOWER = close <= lower_band
IN_RANGE = close > lower_band AND close < upper_band

# %B readings
OVERBOUGHT = percent_b > 1.0  # Above upper band
OVERSOLD = percent_b < 0.0  # Below lower band
NEUTRAL = 0.2 < percent_b < 0.8

# Volatility state (squeeze)
SQUEEZE = bandwidth < percentile(bandwidth, 10, lookback=126)
EXPANSION = bandwidth > percentile(bandwidth, 90, lookback=126)

# Mean reversion signals
MEAN_REVERT_LONG = close < lower_band AND close > close[1]  # Touch lower, turn up
MEAN_REVERT_SHORT = close > upper_band AND close < close[1]  # Touch upper, turn down

# Breakout signals (after squeeze)
BREAKOUT_LONG = SQUEEZE[5] AND close > upper_band AND volume > avg_volume * 1.5
```

**Pitfalls**:
- Bands can ride along price in strong trends
- Mean reversion fails in trending markets
- Squeeze breakouts have false signals

---

### 3.3 Implied vs Realized Volatility

**What it measures**: Market's expected volatility vs actual historical volatility.

**Definitions**:
- **Implied Volatility (IV)**: Forward-looking, derived from option prices
- **Realized Volatility (RV/HV)**: Historical, calculated from price returns
- **IV Rank**: Current IV percentile over past year
- **IV Percentile**: % of days IV was lower than current

**Rule Templates**:
```python
# IV Rank thresholds
HIGH_IV = IV_rank > 50
LOW_IV = IV_rank < 30
ELEVATED_IV = IV_rank > 70

# Volatility premium/discount
VOL_PREMIUM = IV > HV(20)  # IV above realized
VOL_DISCOUNT = IV < HV(20)  # IV below realized

# Options strategy selection
SELL_PREMIUM = IV_rank > 50 AND IV > HV(20) * 1.1  # High IV, sell options
BUY_PREMIUM = IV_rank < 30 AND IV < HV(20) * 0.9  # Low IV, buy options

# Volatility mean reversion
IV_ELEVATED = IV_rank > 80  # Expect IV to decline
IV_DEPRESSED = IV_rank < 20  # Expect IV to rise
```

---

## 4. Support/Resistance & Patterns

### 4.1 Horizontal Support/Resistance

**What it measures**: Price levels where buying/selling pressure has historically emerged.

**Detection Methods**:
- Swing high/low clustering
- Volume profile peaks
- Round numbers (psychological)
- Previous day/week/month highs and lows

**Rule Templates**:
```python
# Key level identification
RESISTANCE = highest(high, 20)
SUPPORT = lowest(low, 20)

# Previous session levels
PREV_DAY_HIGH = high of previous session
PREV_DAY_LOW = low of previous session
PREV_DAY_CLOSE = close of previous session

# Level tests
TESTING_RESISTANCE = high > RESISTANCE * 0.995 AND close < RESISTANCE
TESTING_SUPPORT = low < SUPPORT * 1.005 AND close > SUPPORT

# Breakout/Breakdown
BREAKOUT = close > RESISTANCE AND volume > avg_volume * 1.5
BREAKDOWN = close < SUPPORT AND volume > avg_volume * 1.5

# Failed breakout (bull/bear trap)
FAILED_BREAKOUT = close[1] > RESISTANCE AND close < RESISTANCE
FAILED_BREAKDOWN = close[1] < SUPPORT AND close > SUPPORT
```

---

### 4.2 Breakout Patterns

**What it measures**: Price emerging from consolidation/range.

**Types**:
- Range breakout (horizontal consolidation)
- Triangle breakout (converging trendlines)
- Channel breakout
- Volatility breakout (ATR-based)

**Rule Templates**:
```python
# Range definition
RANGE_HIGH = highest(high, 20)
RANGE_LOW = lowest(low, 20)
RANGE_WIDTH = RANGE_HIGH - RANGE_LOW
IN_RANGE = RANGE_WIDTH < ATR(20) * 3  # Tight range

# Breakout conditions
UPSIDE_BREAKOUT = close > RANGE_HIGH AND volume > avg_volume * 1.5
DOWNSIDE_BREAKOUT = close < RANGE_LOW AND volume > avg_volume * 1.5

# Volatility squeeze breakout
SQUEEZE = bandwidth(20) < percentile(bandwidth, 20, 126)
SQUEEZE_BREAKOUT = SQUEEZE[1:5].any() AND (close > upper_band OR close < lower_band)

# Breakout validation
VALID_BREAKOUT = breakout_bar AND next_bar confirms direction
```

**Pitfalls**:
- Many breakouts fail (false breakouts)
- Volume confirmation critical
- Need follow-through on subsequent bars

---

### 4.3 Pullback Entries

**What it measures**: Retracement in trending markets for better entry.

**Methods**:
- Moving average pullback
- Fibonacci retracement levels
- Previous support/resistance retest

**Rule Templates**:
```python
# MA pullback in uptrend
UPTREND = MA(20) > MA(50) AND price > MA(50)
PULLBACK_TO_MA = low < MA(20) AND close > MA(20)
PULLBACK_BUY = UPTREND AND PULLBACK_TO_MA AND RSI(14) < 50

# Fibonacci retracement
SWING_RANGE = recent_swing_high - recent_swing_low
FIB_382 = recent_swing_high - (SWING_RANGE * 0.382)
FIB_500 = recent_swing_high - (SWING_RANGE * 0.500)
FIB_618 = recent_swing_high - (SWING_RANGE * 0.618)

PULLBACK_ZONE = low < FIB_500 AND close > FIB_618  # Pullback to 50-61.8%

# Previous resistance becomes support
BREAKOUT_RETEST = price broke above level AND returns to test level as support
```

---

### 4.4 Mean Reversion Setups

**What it measures**: Overextension from average, expecting return to mean.

**Methods**:
- Bollinger Band extremes
- RSI extremes
- Distance from moving average
- Z-score of returns

**Rule Templates**:
```python
# Distance from MA
DEVIATION = (close - MA(20)) / MA(20) * 100
OVEREXTENDED_UP = DEVIATION > 5  # 5% above MA
OVEREXTENDED_DOWN = DEVIATION < -5  # 5% below MA

# Z-score based
ZSCORE = (close - MA(20)) / StdDev(close, 20)
EXTREME_UP = ZSCORE > 2
EXTREME_DOWN = ZSCORE < -2

# Mean reversion entry (with confirmation)
MEAN_REVERT_LONG = ZSCORE < -2 AND ZSCORE > ZSCORE[1]  # Extreme + turning
MEAN_REVERT_SHORT = ZSCORE > 2 AND ZSCORE < ZSCORE[1]

# Exit on mean return
EXIT_LONG = close > MA(20)  # Returned to mean
EXIT_SHORT = close < MA(20)
```

**Pitfalls**:
- "Markets can stay irrational longer than you can stay solvent"
- Trends can extend further than expected
- Requires tight risk management

---

## 5. Trend Strength

### 5.1 ADX (Average Directional Index)

**What it measures**: Trend strength regardless of direction.

**Formula**:
```
+DI = 100 × Smoothed(+DM) / ATR
-DI = 100 × Smoothed(-DM) / ATR
DX = 100 × |+DI - -DI| / (+DI + -DI)
ADX = Smoothed DX over 14 periods
```

**Standard Parameters**:
- Period: 14
- Trend threshold: 25 (ADX > 25 = trending)
- Strong trend: ADX > 40

**Rule Templates**:
```python
# Trend strength
TRENDING = ADX(14) > 25
STRONG_TREND = ADX(14) > 40
WEAK_TREND = ADX(14) < 20  # Ranging/choppy

# Trend direction
BULLISH_TREND = +DI > -DI AND ADX > 25
BEARISH_TREND = -DI > +DI AND ADX > 25

# DI crossover
BULLISH_CROSS = +DI crosses_above -DI
BEARISH_CROSS = -DI crosses_above +DI

# Strategy selection based on ADX
USE_TREND_STRATEGY = ADX > 25
USE_MEAN_REVERT_STRATEGY = ADX < 20
```

**Pitfalls**:
- ADX lags, slow to react
- High ADX can occur near trend exhaustion
- DI crossovers produce many false signals alone

---

## 6. Candlestick Patterns (Algorithmic)

### Statistical Validity Note

Most candlestick patterns have weak standalone predictive power in academic studies. Use as **confirmation only**, combined with context (trend, support/resistance, volume).

### Pattern Definitions (Machine-Detectable)

```python
# Single candle patterns
DOJI = abs(open - close) < (high - low) * 0.1
HAMMER = lower_wick > body * 2 AND upper_wick < body * 0.5 AND downtrend
SHOOTING_STAR = upper_wick > body * 2 AND lower_wick < body * 0.5 AND uptrend

# Two candle patterns
BULLISH_ENGULFING = close[1] < open[1] AND close > open AND open < close[1] AND close > open[1]
BEARISH_ENGULFING = close[1] > open[1] AND close < open AND open > close[1] AND close < open[1]

# Context requirements
VALID_HAMMER = HAMMER AND at_support AND volume > avg_volume
VALID_ENGULFING = BULLISH_ENGULFING AND RSI < 40 AND near_support
```

**Usage Guidance**:
- Always require context (trend, level, volume)
- Treat as secondary confirmation, not primary signal
- Backtest specific patterns in your universe before relying on them

---

## Academic References

1. **Aronson, D.R. (2006)**: "Evidence-Based Technical Analysis" - Rigorous statistical testing of TA methods
2. **Bessembinder, H. & Chan, K. (1998)**: "Market Efficiency and the Returns to Technical Analysis" - Mixed results for TA profitability
3. **Lo, A., Mamaysky, H., & Wang, J. (2000)**: "Foundations of Technical Analysis" - Pattern recognition with statistical methods
4. **Neftci, S. (1991)**: "Naive Trading Rules in Financial Markets" - Early academic analysis of TA
5. **Park, C.H. & Irwin, S.H. (2007)**: "What Do We Know About the Profitability of Technical Analysis?" - Survey of TA research

---

## Implementation Notes

1. **Combine indicators**: No single indicator is reliable alone
2. **Use multiple timeframes**: Align signals across timeframes
3. **Require volume confirmation**: Especially for breakouts
4. **Backtest rigorously**: Validate before deployment
5. **Account for costs**: Transaction costs and slippage matter
6. **Avoid overfitting**: Fewer parameters = more robust
7. **Context matters**: Trend vs range requires different approaches
