# Volume, Liquidity & Order Flow - Knowledge Base

## Purpose

Volume and liquidity analysis provides **confirmation signals** and **eligibility filters** for automated trading. This section documents how to use volume systematically in rule-based logic.

---

## 1. Volume Fundamentals

### 1.1 Absolute Volume

**Definition**: Total shares/contracts traded in a period.

**Usage**:
- Identify unusual activity
- Validate price moves
- Assess institutional participation

**Rule Templates**:
```python
# Volume thresholds
HIGH_VOLUME = volume > 1_000_000  # Absolute threshold
LOW_VOLUME = volume < 100_000

# Tradability filter
SUFFICIENT_LIQUIDITY = avg_volume(20) > 500_000  # Minimum ADV for trading
```

---

### 1.2 Relative Volume (RVOL)

**Definition**: Current volume compared to historical average.

**Formula**:
```
RVOL = Current Volume / Average Volume (same time period)
```

**Types**:
- Intraday RVOL: Current bar vs same time of day average
- Daily RVOL: Today's volume vs N-day average

**Rule Templates**:
```python
# Relative volume calculation
RVOL = volume / SMA(volume, 20)

# RVOL thresholds
ELEVATED_VOLUME = RVOL > 1.5
HIGH_VOLUME = RVOL > 2.0
VOLUME_SPIKE = RVOL > 3.0
LOW_VOLUME = RVOL < 0.5

# Intraday RVOL (time-adjusted)
INTRADAY_RVOL = current_bar_volume / avg_volume_this_time_of_day(20)
```

**Usage Notes**:
- RVOL normalizes across different stocks
- Time-of-day adjustment important for intraday
- Opening 30 min typically has highest volume

---

### 1.3 Volume Spikes

**Definition**: Sudden, significant increase in volume.

**Detection Methods**:
```python
# Simple spike detection
VOLUME_SPIKE = volume > SMA(volume, 20) * 2.5

# Standard deviation based
VOL_ZSCORE = (volume - SMA(volume, 20)) / StdDev(volume, 20)
VOLUME_SPIKE = VOL_ZSCORE > 2.0

# Percentile based
VOLUME_SPIKE = volume > percentile(volume, 95, lookback=50)
```

**Interpretation**:
- Spike + breakout = confirmation of move
- Spike at support/resistance = potential reversal
- Spike without price move = absorption (potential reversal)

---

### 1.4 Volume Dry-ups

**Definition**: Unusually low volume, often preceding moves.

**Detection**:
```python
# Volume contraction
VOLUME_DRYUP = volume < SMA(volume, 20) * 0.5
VOLUME_DRYUP_PERIOD = all(volume < avg_volume * 0.6 for last 5 bars)

# Lowest volume in range
LOWEST_VOLUME = volume == min(volume, lookback=20)
```

**Significance**:
- Often precedes breakouts (calm before storm)
- In trend, may signal consolidation before continuation
- Combined with price contraction = setup for move

---

## 2. Volume Confirmation Rules

### 2.1 Breakout Volume Confirmation

**Principle**: Valid breakouts should occur on above-average volume.

**Rule Templates**:
```python
# Basic breakout confirmation
VALID_BREAKOUT = (
    close > resistance AND
    volume > SMA(volume, 20) * 1.5
)

# Strict breakout confirmation
STRONG_BREAKOUT = (
    close > resistance AND
    volume > SMA(volume, 20) * 2.0 AND
    close > high[1]  # Close above previous high
)

# Weak/suspect breakout
SUSPECT_BREAKOUT = (
    close > resistance AND
    volume < SMA(volume, 20)  # Below average volume
)

# Multi-day breakout
CONFIRMED_BREAKOUT = (
    VALID_BREAKOUT[1] AND  # Breakout yesterday
    close > resistance AND  # Still above today
    volume > SMA(volume, 20) * 1.2  # Follow-through volume
)
```

---

### 2.2 Trend Confirmation with Volume

**Principle**: Healthy trends have volume in direction of trend.

**Rule Templates**:
```python
# Uptrend volume pattern
HEALTHY_UPTREND = (
    avg_volume_on_up_days(10) > avg_volume_on_down_days(10)
)

# Volume expansion on moves
UP_MOVE_WITH_VOLUME = (
    close > close[1] AND
    volume > volume[1] AND
    volume > SMA(volume, 10)
)

# Declining volume warning
TREND_WEAKENING = (
    price_making_new_high AND
    volume < SMA(volume, 10)  # Lower volume on new high
)

# Volume divergence (bearish)
VOLUME_DIVERGENCE = (
    close > close[5] AND  # Price higher
    volume < SMA(volume, 20) * 0.8  # Volume lower
)
```

---

### 2.3 Reversal Volume Patterns

**Principle**: Reversals often show volume climax or volume dry-up.

**Rule Templates**:
```python
# Climax top (blowoff)
CLIMAX_TOP = (
    close near highest(high, 20) AND
    volume > SMA(volume, 20) * 3.0 AND  # Extreme volume
    close < open  # Bearish candle
)

# Climax bottom (capitulation)
CAPITULATION = (
    close near lowest(low, 20) AND
    volume > SMA(volume, 20) * 3.0 AND
    close > open  # Bullish candle on extreme volume
)

# Drying up before reversal
DRYUP_REVERSAL_SETUP = (
    downtrend AND
    volume < SMA(volume, 20) * 0.5 for 3+ bars AND
    RSI < 30
)
```

---

## 3. Liquidity Assessment

### 3.1 Average Daily Volume (ADV)

**Definition**: Average shares traded per day over N periods.

**Rule Templates**:
```python
# ADV calculation
ADV_20 = SMA(volume, 20)
ADV_50 = SMA(volume, 50)

# Liquidity tiers
HIGHLY_LIQUID = ADV_20 > 5_000_000
LIQUID = ADV_20 > 1_000_000
MODERATELY_LIQUID = ADV_20 > 500_000
ILLIQUID = ADV_20 < 100_000

# Trading eligibility filter
TRADEABLE = ADV_20 > minimum_adv_threshold
```

---

### 3.2 Bid-Ask Spread

**Definition**: Difference between best bid and best ask.

**Metrics**:
```python
# Absolute spread
SPREAD = ask - bid

# Relative spread (percentage)
SPREAD_PCT = (ask - bid) / midpoint * 100

# Spread thresholds
TIGHT_SPREAD = SPREAD_PCT < 0.05  # <5 bps
NORMAL_SPREAD = SPREAD_PCT < 0.10  # <10 bps
WIDE_SPREAD = SPREAD_PCT > 0.20  # >20 bps
```

**Rule Templates**:
```python
# Spread filter for entry
ACCEPTABLE_SPREAD = SPREAD_PCT < max_spread_threshold

# Adjusted for cost
EFFECTIVE_COST = SPREAD_PCT / 2  # Half-spread as execution cost

# Skip trade if spread too wide
DO_NOT_TRADE = SPREAD_PCT > 0.50  # >50 bps spread
```

---

### 3.3 Market Impact Estimation

**Definition**: Expected price impact of order size.

**Simple Model**:
```python
# Impact as function of order size vs ADV
ORDER_SIZE_PCT = order_shares / ADV_20

# Simple impact estimate
ESTIMATED_IMPACT_BPS = ORDER_SIZE_PCT * impact_coefficient

# Size limit based on ADV
MAX_POSITION_SHARES = ADV_20 * max_adv_percentage  # e.g., 1% of ADV
```

**Rule Templates**:
```python
# Position size limits
MAX_SHARES = min(
    intended_shares,
    ADV_20 * 0.01,  # Max 1% of ADV
    max_dollar_position / price
)

# Warn if position too large
LIQUIDITY_WARNING = intended_shares > ADV_20 * 0.005  # >0.5% of ADV
```

---

## 4. Order Flow Proxies

### 4.1 On-Balance Volume (OBV)

**Definition**: Cumulative volume based on price direction.

**Formula**:
```
If close > close[1]: OBV = OBV[1] + volume
If close < close[1]: OBV = OBV[1] - volume
If close = close[1]: OBV = OBV[1]
```

**Rule Templates**:
```python
# OBV trend
OBV_UPTREND = OBV > SMA(OBV, 20)
OBV_DOWNTREND = OBV < SMA(OBV, 20)

# OBV divergence
BULLISH_OBV_DIV = price makes lower_low AND OBV makes higher_low
BEARISH_OBV_DIV = price makes higher_high AND OBV makes lower_high

# OBV breakout (leading indicator)
OBV_BREAKOUT = OBV > highest(OBV, 20)  # Before price breakout
```

---

### 4.2 Accumulation/Distribution (A/D)

**Definition**: Volume-weighted price position within range.

**Formula**:
```
CLV = ((close - low) - (high - close)) / (high - low)
A/D = A/D[1] + CLV × volume
```

**Rule Templates**:
```python
# A/D trend
AD_UPTREND = AD > SMA(AD, 20)
AD_DOWNTREND = AD < SMA(AD, 20)

# A/D divergence
BULLISH_AD_DIV = price downtrend AND AD uptrend
BEARISH_AD_DIV = price uptrend AND AD downtrend
```

---

### 4.3 Money Flow Index (MFI)

**Definition**: Volume-weighted RSI.

**Formula**:
```
Typical Price = (High + Low + Close) / 3
Raw Money Flow = Typical Price × Volume
MFI = 100 - (100 / (1 + Positive MF / Negative MF))
```

**Rule Templates**:
```python
# Overbought/Oversold
MFI_OVERBOUGHT = MFI(14) > 80
MFI_OVERSOLD = MFI(14) < 20

# Divergence
BULLISH_MFI_DIV = price makes lower_low AND MFI makes higher_low
```

---

### 4.4 VWAP (Volume Weighted Average Price)

**Definition**: Average price weighted by volume, typically intraday.

**Formula**:
```
VWAP = Cumulative(Price × Volume) / Cumulative(Volume)
```

**Rule Templates**:
```python
# VWAP as support/resistance
ABOVE_VWAP = close > VWAP  # Bullish intraday bias
BELOW_VWAP = close < VWAP  # Bearish intraday bias

# VWAP pullback entry
VWAP_PULLBACK_LONG = uptrend AND low < VWAP AND close > VWAP
VWAP_PULLBACK_SHORT = downtrend AND high > VWAP AND close < VWAP

# Distance from VWAP
EXTENDED_FROM_VWAP = abs(close - VWAP) / VWAP > 0.02  # >2% from VWAP
```

**Usage Notes**:
- VWAP resets each session (typically)
- Institutional benchmark for execution quality
- Acts as intraday magnet

---

## 5. Volume-Based Filters

### 5.1 Eligibility Filters

**Purpose**: Ensure sufficient liquidity before considering trade.

```python
# Master liquidity filter
TRADEABLE = (
    ADV_20 > 500_000 AND            # Minimum average volume
    SPREAD_PCT < 0.10 AND            # Maximum spread
    price > 5.00 AND                 # Avoid penny stocks
    market_cap > 300_000_000         # Minimum market cap
)

# Intraday specific
INTRADAY_ELIGIBLE = (
    TRADEABLE AND
    current_volume > 10_000 AND      # Some activity today
    RVOL > 0.8                       # Not abnormally quiet
)
```

---

### 5.2 Confirmation Filters

**Purpose**: Require volume confirmation for signals.

```python
# Breakout confirmation required
CONFIRMED_BREAKOUT = (
    price_breakout AND
    volume > SMA(volume, 20) * 1.5
)

# Trend entry confirmation
CONFIRMED_TREND_ENTRY = (
    trend_signal AND
    RVOL > 1.0  # At least average volume
)

# Reject low-volume signals
REJECT_IF = volume < SMA(volume, 20) * 0.5
```

---

### 5.3 Warning Filters

**Purpose**: Generate warnings for suspicious conditions.

```python
# Volume warnings
LIQUIDITY_WARNING = RVOL < 0.3  # Very low relative volume
SPREAD_WARNING = SPREAD_PCT > avg_spread * 2  # Spread widening
IMPACT_WARNING = order_size > ADV_20 * 0.01  # Large relative to ADV

# Action: reduce size or skip trade
IF LIQUIDITY_WARNING: reduce_position_size()
IF SPREAD_WARNING: use_limit_order_only()
IF IMPACT_WARNING: split_order() or skip()
```

---

## 6. Volume Patterns

### 6.1 Volume Profile Concepts

**Definition**: Distribution of volume across price levels.

**Key Levels**:
- **Point of Control (POC)**: Price with highest volume
- **Value Area High (VAH)**: Upper bound of 70% volume
- **Value Area Low (VAL)**: Lower bound of 70% volume

**Rule Templates**:
```python
# Value area positioning
ABOVE_VALUE = close > VAH  # Price accepted above value area
BELOW_VALUE = close < VAL  # Price accepted below value area
IN_VALUE = VAL < close < VAH  # Price in value area

# POC as support/resistance
AT_POC = abs(close - POC) < ATR * 0.5

# Volume node trade
HIGH_VOLUME_NODE = local_volume > avg_profile_volume * 1.5
LOW_VOLUME_NODE = local_volume < avg_profile_volume * 0.5  # Potential gap
```

---

### 6.2 Session Volume Patterns

**Intraday Volume Distribution** (US Equities):
- Opening 30 min: ~15-20% of daily volume
- Midday: Lower volume, choppier
- Closing 30 min: ~10-15% of daily volume

**Rule Templates**:
```python
# Time-based volume expectations
OPENING_RANGE = time between 09:30 and 10:00
LUNCH_DOLDRUMS = time between 12:00 and 14:00
POWER_HOUR = time between 15:00 and 16:00

# Adjust expectations by session
IF LUNCH_DOLDRUMS:
    lower_volume_threshold()
    widen_filters()
```

---

## 7. Implementation Examples

### Example 1: Volume-Confirmed Breakout Strategy Filter

```python
def volume_confirmed_breakout(data, lookback=20, vol_mult=1.5):
    """
    Confirm breakout with volume.
    """
    resistance = data['high'].rolling(lookback).max()
    avg_volume = data['volume'].rolling(lookback).mean()

    price_breakout = data['close'] > resistance.shift(1)
    volume_confirm = data['volume'] > avg_volume * vol_mult

    return price_breakout & volume_confirm
```

### Example 2: Liquidity Filter

```python
def liquidity_filter(data, min_adv=500000, max_spread_pct=0.10):
    """
    Filter for sufficient liquidity.
    """
    adv_20 = data['volume'].rolling(20).mean()
    spread_pct = (data['ask'] - data['bid']) / data['mid'] * 100

    liquid = adv_20 > min_adv
    tight_spread = spread_pct < max_spread_pct

    return liquid & tight_spread
```

### Example 3: RVOL Signal Enhancement

```python
def enhanced_signal_with_rvol(signal, volume, lookback=20, min_rvol=1.0):
    """
    Only accept signals with sufficient relative volume.
    """
    avg_volume = volume.rolling(lookback).mean()
    rvol = volume / avg_volume

    return signal & (rvol > min_rvol)
```

---

## Academic References

1. **Harris, L. (2003)**: "Trading and Exchanges: Market Microstructure for Practitioners"
2. **Easley, D. & O'Hara, M. (1987)**: "Price, Trade Size, and Information in Securities Markets"
3. **Blume, L., Easley, D., & O'Hara, M. (1994)**: "Market Statistics and Technical Analysis: The Role of Volume"
4. **Karpoff, J.M. (1987)**: "The Relation Between Price Changes and Trading Volume: A Survey"
5. **Campbell, J., Grossman, S., & Wang, J. (1993)**: "Trading Volume and Serial Correlation in Stock Returns"

---

## Key Takeaways

1. **Volume confirms price**: Price moves on low volume are suspect
2. **Liquidity gates**: Always filter for tradeable stocks first
3. **RVOL normalizes**: Use relative volume to compare across securities
4. **Spikes and dryups**: Both carry information about coming moves
5. **Cost matters**: Spread and impact affect real returns
6. **Time of day**: Volume patterns differ intraday
