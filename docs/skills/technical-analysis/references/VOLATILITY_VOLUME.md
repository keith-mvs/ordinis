# Volatility and Volume Indicators Reference

## Volatility Indicators

### Average True Range (ATR)

**Purpose**: Measures market volatility by calculating average range of price movement.

**Calculation**:
```
True Range = max[(High - Low), abs(High - PrevClose), abs(Low - PrevClose)]
ATR = Smoothed average of True Range (typically 14 periods)
```

**Python Implementation**:
```python
def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range."""
    import pandas as pd

    # True Range components
    high_low = high - low
    high_close = abs(high - close.shift(1))
    low_close = abs(low - close.shift(1))

    # True Range
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # ATR using Wilder's smoothing
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    return atr
```

**Interpretation**:
- High ATR: High volatility, wider stops needed
- Low ATR: Low volatility, tighter stops appropriate
- Rising ATR: Increasing volatility
- Falling ATR: Decreasing volatility

**Applications**:
- Position sizing: Adjust size inversely to ATR
- Stop-loss placement: Set stops at N × ATR from entry
- Breakout confirmation: High ATR confirms genuine breakout
- Volatility filtering: Trade only when ATR in acceptable range

**Common Parameters**: 14 periods (standard)

---

### Bollinger Bands

**Purpose**: Volatility bands around moving average that expand/contract with market volatility.

**Components**:
```
Middle Band = 20-period SMA
Upper Band = Middle Band + (2 × Standard Deviation)
Lower Band = Middle Band - (2 × Standard Deviation)
```

**Python Implementation**:
```python
def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    import pandas as pd

    # Middle band (SMA)
    middle = close.rolling(period).mean()

    # Standard deviation
    std = close.rolling(period).std()

    # Upper and lower bands
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    # Bandwidth
    bandwidth = (upper - lower) / middle

    # %B (position within bands)
    percent_b = (close - lower) / (upper - lower)

    return {
        'upper': upper,
        'middle': middle,
        'lower': lower,
        'bandwidth': bandwidth,
        '%B': percent_b
    }
```

**Interpretation**:

*Band Position*:
- Price at upper band: Overbought
- Price at lower band: Oversold
- Price at middle band: Fair value

*Band Width*:
- Narrow bands: Low volatility (squeeze)
- Wide bands: High volatility (expansion)
- Contracting bands: Breakout likely

*%B Indicator*:
- %B > 1: Price above upper band
- %B = 0.5: Price at middle band
- %B < 0: Price below lower band

**Trading Strategies**:

*Bollinger Bounce*:
```
Entry Long: Price touches lower band
Exit: Price reaches middle or upper band

Entry Short: Price touches upper band
Exit: Price reaches middle or lower band
```

*Bollinger Squeeze*:
```
Setup: Bandwidth at multi-month low
Entry: Price breaks out of narrow bands
Direction: Follow breakout direction
Stop: Opposite band
```

**Common Parameters**:
- Standard: 20, 2 (period, std dev)
- Aggressive: 20, 1.5
- Conservative: 20, 2.5

---

## Volume Indicators

### On-Balance Volume (OBV)

**Purpose**: Cumulative volume indicator that tracks buying/selling pressure.

**Calculation**:
```
If Close > Previous Close: OBV = Previous OBV + Volume
If Close < Previous Close: OBV = Previous OBV - Volume
If Close = Previous Close: OBV = Previous OBV
```

**Python Implementation**:
```python
def calculate_obv(close, volume):
    """Calculate On-Balance Volume."""
    import pandas as pd
    import numpy as np

    # Determine price direction
    direction = np.sign(close.diff())

    # Calculate OBV
    obv = (direction * volume).cumsum()

    return obv
```

**Interpretation**:

*OBV Trend*:
- Rising OBV: Accumulation (buying pressure)
- Falling OBV: Distribution (selling pressure)
- Flat OBV: Equilibrium

*OBV vs Price*:
- Both rising: Strong uptrend
- Both falling: Strong downtrend
- OBV rising, price flat: Accumulation before breakout
- OBV falling, price flat: Distribution before breakdown

*Divergence*:
- Price rising, OBV falling: Bearish divergence
- Price falling, OBV rising: Bullish divergence

**Trading Strategy**:
```
Confirmation:
- Price breakout confirmed by OBV breakout
- Both making new highs/lows together

Divergence:
- Wait for OBV to confirm price direction
- Enter when both align
```

**Advantages**:
- Simple calculation
- Clear trend indication
- Effective divergence signals

**Limitations**:
- Cumulative (never resets)
- No absolute values (relative only)
- Requires volume data

---

## Indicator Integration

### Volatility-Adjusted Position Sizing

```python
def calculate_position_size(capital, atr, risk_per_trade=0.02):
    """
    Calculate position size based on ATR volatility.

    Parameters:
    -----------
    capital : float
        Available capital
    atr : float
        Current ATR value
    risk_per_trade : float
        Risk as fraction of capital (default: 2%)

    Returns:
    --------
    Number of shares/contracts
    """
    risk_amount = capital * risk_per_trade
    stop_distance = 2 * atr  # 2 ATR stop loss
    position_size = risk_amount / stop_distance

    return int(position_size)
```

### Volume-Confirmed Breakouts

```python
def confirm_breakout(price, resistance, obv, avg_volume, volume):
    """
    Confirm breakout using price, volume, and OBV.

    Returns True if breakout is confirmed.
    """
    price_breakout = price > resistance
    volume_surge = volume > avg_volume * 1.5
    obv_confirm = obv > obv.rolling(20).max()

    return price_breakout and volume_surge and obv_confirm
```

### Volatility Regime Filter

```python
def get_volatility_regime(atr, period=20):
    """
    Classify current volatility regime.

    Returns: LOW, NORMAL, or HIGH
    """
    atr_ma = atr.rolling(period).mean()
    atr_std = atr.rolling(period).std()

    if atr.iloc[-1] < atr_ma.iloc[-1] - atr_std.iloc[-1]:
        return 'LOW'
    elif atr.iloc[-1] > atr_ma.iloc[-1] + atr_std.iloc[-1]:
        return 'HIGH'
    else:
        return 'NORMAL'
```

---

## Validation Tests

```bash
python scripts/validate_volatility_volume.py --test-all
```

Tests include:
- ATR calculation accuracy
- Bollinger Band width statistics
- OBV divergence detection
- Volume spike identification
- Volatility regime classification

---

## Further Reading

- Wilder, J.W. (1978). "New Concepts in Technical Trading Systems" (ATR)
- Bollinger, J. (2002). "Bollinger on Bollinger Bands"
- Granville, J. (1963). "Granville's New Key to Stock Market Profits" (OBV)
- CMT Association: Volume Analysis curriculum
