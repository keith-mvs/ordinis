# Technical Indicators: Examples and Case Studies

## Example 1: Trend Following with ADX and Moving Averages

### Market Condition
- Asset: SPY (S&P 500 ETF)
- Period: Strong uptrend
- ADX > 25 (strong trend)

### Analysis
```python
from scripts.indicator_calculator import TechnicalIndicators
import yfinance as yf

# Load data
data = yf.Ticker("SPY").history(period="6mo")
ti = TechnicalIndicators(data)

# Calculate indicators
adx = ti.calculate_adx()
ma = ti.calculate_moving_averages()

# Check conditions
current_adx = adx['ADX'].iloc[-1]
plus_di = adx['+DI'].iloc[-1]
minus_di = adx['-DI'].iloc[-1]
price = data['Close'].iloc[-1]
sma_50 = ma['SMA_slow'].iloc[-1]

print(f"ADX: {current_adx:.2f}")
print(f"+DI: {plus_di:.2f}, -DI: {minus_di:.2f}")
print(f"Price: {price:.2f}, 50-day MA: {sma_50:.2f}")

# Signal
if current_adx > 25 and plus_di > minus_di and price > sma_50:
    print("SIGNAL: STRONG UPTREND - Hold long positions")
```

### Result
Strong uptrend confirmed by multiple indicators. Appropriate for trend-following strategies.

---

## Example 2: Oversold Bounce with RSI and Stochastic

### Market Condition
- Asset: Individual stock
- RSI < 30 (oversold)
- Stochastic < 20 (oversold)

### Analysis
```python
# Calculate momentum indicators
rsi = ti.calculate_rsi()
stoch = ti.calculate_stochastic()

current_rsi = rsi.iloc[-1]
current_k = stoch['%K'].iloc[-1]
current_d = stoch['%D'].iloc[-1]

print(f"RSI: {current_rsi:.2f}")
print(f"Stochastic %K: {current_k:.2f}, %D: {current_d:.2f}")

# Oversold reversal setup
if current_rsi < 30 and current_k < 20:
    # Wait for turn
    if rsi.iloc[-1] > rsi.iloc[-2] and current_k > current_d:
        print("SIGNAL: OVERSOLD REVERSAL - Consider long entry")
```

### Trading Plan
- Entry: When both RSI and Stochastic turn up from oversold
- Stop: Below recent low
- Target: RSI 50 or resistance level

---

## Example 3: Volatility Breakout with Bollinger Bands

### Market Condition
- Bollinger Band width at multi-month low
- Price consolidating
- Setup for volatility expansion

### Analysis
```python
# Calculate Bollinger Bands
bb = ti.calculate_bollinger_bands()
atr = ti.calculate_atr()

bandwidth = bb['bandwidth'].iloc[-1]
avg_bandwidth = bb['bandwidth'].rolling(60).mean().iloc[-1]
price = data['Close'].iloc[-1]
upper = bb['upper'].iloc[-1]
lower = bb['lower'].iloc[-1]

print(f"Current Bandwidth: {bandwidth:.4f}")
print(f"Average Bandwidth: {avg_bandwidth:.4f}")
print(f"Price: {price:.2f} (Range: {lower:.2f} - {upper:.2f})")

# Squeeze detection
if bandwidth < avg_bandwidth * 0.7:
    print("SETUP: Bollinger Band Squeeze - Breakout imminent")
    print(f"Buy above: {upper:.2f}")
    print(f"Sell below: {lower:.2f}")
```

### Strategy
- Wait for squeeze (narrow bands)
- Enter on breakout direction
- Use ATR for stop placement
- High probability of significant move

---

## Example 4: Divergence Detection with MACD

### Market Condition
- Price making new highs
- MACD making lower highs
- Bearish divergence

### Analysis
```python
# Calculate MACD
macd = ti.calculate_macd()

# Check for divergence
price_highs = data['Close'].rolling(20).max()
macd_highs = macd['MACD'].rolling(20).max()

# Compare recent vs previous highs
current_price_high = price_highs.iloc[-1]
prev_price_high = price_highs.iloc[-20]
current_macd_high = macd_highs.iloc[-1]
prev_macd_high = macd_highs.iloc[-20]

if current_price_high > prev_price_high and current_macd_high < prev_macd_high:
    print("WARNING: Bearish divergence detected")
    print("Price making higher highs, MACD making lower highs")
    print("Potential trend reversal")
```

### Trading Decision
- Existing longs: Consider taking profits
- New positions: Wait for confirmation
- Short traders: Setup forming, wait for breakdown

---

## Example 5: Volume Confirmation with OBV

### Market Condition
- Price breaking resistance
- Need volume confirmation

### Analysis
```python
# Calculate OBV
obv = ti.calculate_obv()

# Check trend
price_breakout = data['Close'].iloc[-1] > data['Close'].rolling(20).max().iloc[-2]
obv_trend = obv.iloc[-1] > obv.rolling(20).max().iloc[-2]
volume_surge = data['Volume'].iloc[-1] > data['Volume'].rolling(20).mean().iloc[-1] * 1.5

print(f"Price Breakout: {price_breakout}")
print(f"OBV New High: {obv_trend}")
print(f"Volume Surge: {volume_surge}")

if price_breakout and obv_trend and volume_surge:
    print("SIGNAL: CONFIRMED BREAKOUT - Strong buying pressure")
else:
    print("WARNING: Breakout lacks volume confirmation")
```

### Result
Volume and OBV confirm price breakout, suggesting genuine accumulation rather than false breakout.

---

## Example 6: Multi-Timeframe Confluence

### Market Condition
- Analyzing multiple timeframes for confirmation
- Daily and weekly alignment

### Analysis
```python
# Daily analysis
daily_data = yf.Ticker("SPY").history(period="1y", interval="1d")
daily_ti = TechnicalIndicators(daily_data)
daily_signals = daily_ti.generate_signals()

# Weekly analysis  
weekly_data = yf.Ticker("SPY").history(period="2y", interval="1wk")
weekly_ti = TechnicalIndicators(weekly_data)
weekly_signals = weekly_ti.generate_signals()

print("Daily Signals:", daily_signals)
print("Weekly Signals:", weekly_signals)

# Check alignment
if daily_signals['macd'] == 'BULLISH' and weekly_signals['macd'] == 'BULLISH':
    if daily_signals['rsi'] == 'BULLISH' and weekly_signals['rsi'] == 'BULLISH':
        print("HIGH CONFIDENCE: Multi-timeframe bullish alignment")
```

### Trading Plan
- Higher probability when multiple timeframes align
- Use longer timeframe for trend direction
- Use shorter timeframe for entry timing

---

## Example 7: Fibonacci Retracement Entry

### Market Condition
- Strong uptrend
- Pullback in progress
- Looking for entry at Fibonacci level

### Analysis
```python
# Calculate Fibonacci levels
fib = ti.calculate_fibonacci_levels(lookback=60)

price = data['Close'].iloc[-1]
levels = fib['levels']

print(f"Current Price: {price:.2f}")
print("\nFibonacci Levels:")
for level, value in levels.items():
    distance = abs(price - value)
    pct = (distance / price) * 100
    print(f"{level}: {value:.2f} (Distance: {pct:.1f}%)")

# Find nearest level
nearest = min(levels.items(), key=lambda x: abs(x[1] - price))
print(f"\nNearest Level: {nearest[0]} at {nearest[1]:.2f}")

# Entry decision
if 0.5 < abs(price - nearest[1]) / price * 100 < 2.0:
    print(f"SETUP: Price near {nearest[0]} Fibonacci level")
    print("Wait for reversal candlestick pattern")
```

---

## Example 8: Regime-Adaptive Strategy Selection

### Concept
Select appropriate indicators based on detected market regime.

### Implementation
```python
def detect_market_regime(data):
    """Classify market regime."""
    ti = TechnicalIndicators(data)
    
    # Calculate indicators
    adx = ti.calculate_adx()
    atr = ti.calculate_atr()
    bb = ti.calculate_bollinger_bands()
    
    # Get current values
    current_adx = adx['ADX'].iloc[-1]
    current_atr = atr.iloc[-1]
    avg_atr = atr.rolling(20).mean().iloc[-1]
    bandwidth = bb['bandwidth'].iloc[-1]
    avg_bandwidth = bb['bandwidth'].rolling(60).mean().iloc[-1]
    
    # Classify regime
    if current_adx > 25 and bandwidth > avg_bandwidth:
        return 'TRENDING'
    elif current_atr > avg_atr * 1.5:
        return 'VOLATILE'
    elif bandwidth < avg_bandwidth * 0.7:
        return 'CONSOLIDATING'
    else:
        return 'RANGING'

regime = detect_market_regime(data)
print(f"Market Regime: {regime}")

# Select appropriate indicators
if regime == 'TRENDING':
    print("Use: ADX, Moving Averages, Parabolic SAR")
    print("Strategy: Trend following")
elif regime == 'RANGING':
    print("Use: RSI, Stochastic, Bollinger Bands")
    print("Strategy: Mean reversion")
elif regime == 'VOLATILE':
    print("Use: ATR, Volatility-based stops")
    print("Strategy: Reduce position size")
else:
    print("Use: Wait for clearer regime")
    print("Strategy: Stay in cash or reduce exposure")
```

---

## Best Practices from Case Studies

### 1. Never Rely on Single Indicator
Always use confluence of multiple indicators for higher probability setups.

### 2. Context Matters
Same indicator reading means different things in different market regimes.

### 3. Volume Confirms Price
Always check volume and OBV for confirmation of price movements.

### 4. Multi-Timeframe Analysis
Align longer timeframe trend with shorter timeframe entry timing.

### 5. Risk Management
Use ATR for position sizing and stop placement, not arbitrary levels.

### 6. Wait for Confirmation
Indicators provide setup; wait for price action confirmation before entry.

### 7. Adapt to Conditions
Use trend indicators in trends, oscillators in ranges, volatility measures in choppy markets.
