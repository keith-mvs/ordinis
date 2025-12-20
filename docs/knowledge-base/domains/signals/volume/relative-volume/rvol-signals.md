# Relative Volume (RVOL) Signals

## Overview

Relative Volume (RVOL) normalizes current volume against historical averages, enabling cross-security comparison and unusual activity detection. RVOL signals are critical for **breakout confirmation**, **momentum validation**, and **institutional flow detection**.

---

## 1. RVOL Calculation Methods

### 1.1 Simple RVOL

**Signal Logic**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy import stats


@dataclass
class RVOLConfig:
    """Configuration for RVOL calculations."""

    # Lookback periods
    lookback_short: int = 10
    lookback_medium: int = 20
    lookback_long: int = 50

    # Thresholds
    very_high: float = 3.0
    high: float = 2.0
    elevated: float = 1.5
    normal_upper: float = 1.2
    normal_lower: float = 0.8
    low: float = 0.5
    very_low: float = 0.3


class RVOLCalculator:
    """Calculate various RVOL metrics."""

    def __init__(self, config: RVOLConfig = None):
        self.config = config or RVOLConfig()

    def simple_rvol(
        self,
        volume: pd.Series,
        lookback: int = None
    ) -> pd.Series:
        """
        Calculate simple RVOL as volume / SMA(volume).

        Args:
            volume: Volume series
            lookback: Lookback period for average

        Returns:
            RVOL series
        """
        lookback = lookback or self.config.lookback_medium
        avg_volume = volume.rolling(lookback, min_periods=1).mean()
        return volume / avg_volume.replace(0, np.nan)

    def ewma_rvol(
        self,
        volume: pd.Series,
        span: int = None
    ) -> pd.Series:
        """
        Calculate RVOL using exponentially weighted average.
        More responsive to recent volume changes.
        """
        span = span or self.config.lookback_medium
        ewm_volume = volume.ewm(span=span, min_periods=1).mean()
        return volume / ewm_volume.replace(0, np.nan)

    def median_rvol(
        self,
        volume: pd.Series,
        lookback: int = None
    ) -> pd.Series:
        """
        Calculate RVOL using median (more robust to outliers).
        """
        lookback = lookback or self.config.lookback_medium
        median_volume = volume.rolling(lookback, min_periods=1).median()
        return volume / median_volume.replace(0, np.nan)

    def z_score_rvol(
        self,
        volume: pd.Series,
        lookback: int = None
    ) -> pd.Series:
        """
        Calculate volume z-score for statistical significance.

        Returns:
            Z-score (standard deviations from mean)
        """
        lookback = lookback or self.config.lookback_medium
        vol_mean = volume.rolling(lookback, min_periods=10).mean()
        vol_std = volume.rolling(lookback, min_periods=10).std()
        return (volume - vol_mean) / vol_std.replace(0, np.nan)


class IntradayRVOL:
    """Time-of-day adjusted RVOL for intraday analysis."""

    def __init__(self, lookback_days: int = 20):
        self.lookback_days = lookback_days

    def calculate_time_adjusted_rvol(
        self,
        volume: pd.Series,
        timestamp: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Calculate RVOL adjusted for time-of-day volume patterns.

        Args:
            volume: Intraday volume series
            timestamp: Timestamps for each bar

        Returns:
            Time-adjusted RVOL
        """
        df = pd.DataFrame({'volume': volume, 'time': timestamp.time})

        # Calculate average volume for each time bucket
        # Group by time of day
        df['time_bucket'] = df['time'].apply(
            lambda t: f"{t.hour:02d}:{(t.minute // 5) * 5:02d}"
        )

        # Rolling average by time bucket
        time_avg = df.groupby('time_bucket')['volume'].transform(
            lambda x: x.rolling(self.lookback_days, min_periods=5).mean()
        )

        # Time-adjusted RVOL
        df['tod_rvol'] = volume / time_avg.replace(0, np.nan)

        return df['tod_rvol']

    def cumulative_intraday_rvol(
        self,
        volume: pd.Series,
        timestamp: pd.DatetimeIndex,
        expected_daily_volume: pd.Series
    ) -> pd.Series:
        """
        Calculate cumulative RVOL vs expected pace.

        Useful for detecting if stock is running hot/cold vs typical pace.
        """
        df = pd.DataFrame({
            'volume': volume,
            'timestamp': timestamp,
            'expected': expected_daily_volume
        })

        df['date'] = df['timestamp'].dt.date
        df['cumulative_volume'] = df.groupby('date')['volume'].cumsum()

        # Estimate expected cumulative based on time of day
        # Simplified: linear assumption
        total_minutes = 390  # 6.5 hours trading day
        df['minute_of_day'] = (
            df['timestamp'].dt.hour * 60 +
            df['timestamp'].dt.minute -
            9 * 60 - 30  # Minutes since 9:30 AM
        )
        df['expected_cumulative'] = df['expected'] * (df['minute_of_day'] / total_minutes)

        df['cumulative_rvol'] = df['cumulative_volume'] / df['expected_cumulative'].replace(0, np.nan)

        return df['cumulative_rvol']
```

---

## 2. RVOL Signal Generation

### 2.1 Threshold-Based Signals

**Signal Logic**:
```python
class RVOLSignalGenerator:
    """Generate trading signals from RVOL metrics."""

    def __init__(self, config: RVOLConfig = None):
        self.config = config or RVOLConfig()
        self.calculator = RVOLCalculator(config)

    def generate_rvol_signals(
        self,
        volume: pd.Series
    ) -> pd.DataFrame:
        """
        Generate comprehensive RVOL signals.
        """
        signals = pd.DataFrame(index=volume.index)

        # Calculate RVOL metrics
        rvol_short = self.calculator.simple_rvol(volume, self.config.lookback_short)
        rvol_medium = self.calculator.simple_rvol(volume, self.config.lookback_medium)
        rvol_long = self.calculator.simple_rvol(volume, self.config.lookback_long)
        z_score = self.calculator.z_score_rvol(volume, self.config.lookback_medium)

        signals['rvol_short'] = rvol_short
        signals['rvol_medium'] = rvol_medium
        signals['rvol_long'] = rvol_long
        signals['volume_z_score'] = z_score

        # Threshold signals
        signals['very_high_volume'] = rvol_medium > self.config.very_high
        signals['high_volume'] = (
            (rvol_medium > self.config.high) &
            (rvol_medium <= self.config.very_high)
        )
        signals['elevated_volume'] = (
            (rvol_medium > self.config.elevated) &
            (rvol_medium <= self.config.high)
        )
        signals['normal_volume'] = (
            (rvol_medium >= self.config.normal_lower) &
            (rvol_medium <= self.config.normal_upper)
        )
        signals['low_volume'] = (
            (rvol_medium >= self.config.very_low) &
            (rvol_medium < self.config.low)
        )
        signals['very_low_volume'] = rvol_medium < self.config.very_low

        # Statistical significance
        signals['statistically_significant'] = z_score.abs() > 2.0
        signals['highly_significant'] = z_score.abs() > 3.0

        # Volume acceleration (short vs long)
        signals['volume_accelerating'] = rvol_short > rvol_long * 1.2
        signals['volume_decelerating'] = rvol_short < rvol_long * 0.8

        return signals

    def volume_spike_detection(
        self,
        volume: pd.Series,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Detect volume spikes using multiple methods.
        """
        signals = pd.DataFrame(index=volume.index)

        # Method 1: Multiple of average
        avg_volume = volume.rolling(lookback).mean()
        signals['spike_2x'] = volume > avg_volume * 2
        signals['spike_3x'] = volume > avg_volume * 3
        signals['spike_5x'] = volume > avg_volume * 5

        # Method 2: Percentile based
        pctl_95 = volume.rolling(lookback).quantile(0.95)
        pctl_99 = volume.rolling(lookback).quantile(0.99)
        signals['above_95th_pctl'] = volume > pctl_95
        signals['above_99th_pctl'] = volume > pctl_99

        # Method 3: Standard deviation based
        vol_std = volume.rolling(lookback).std()
        vol_mean = volume.rolling(lookback).mean()
        signals['spike_2std'] = volume > (vol_mean + 2 * vol_std)
        signals['spike_3std'] = volume > (vol_mean + 3 * vol_std)

        # Consensus spike (all methods agree)
        signals['confirmed_spike'] = (
            signals['spike_2x'] &
            signals['above_95th_pctl'] &
            signals['spike_2std']
        )

        return signals

    def volume_dryup_detection(
        self,
        volume: pd.Series,
        lookback: int = 20,
        dryup_periods: int = 3
    ) -> pd.DataFrame:
        """
        Detect volume dry-ups (often precede breakouts).
        """
        signals = pd.DataFrame(index=volume.index)

        rvol = self.calculator.simple_rvol(volume, lookback)

        # Single bar dry-up
        signals['volume_dryup'] = rvol < 0.5

        # Sustained dry-up (multiple bars)
        signals['sustained_dryup'] = (
            (rvol < 0.6).rolling(dryup_periods).sum() == dryup_periods
        )

        # Lowest volume in range
        min_vol = volume.rolling(lookback).min()
        signals['volume_at_low'] = volume == min_vol

        # Volume contraction pattern (VCP)
        # Progressively lower volume over time
        vol_ma_5 = volume.rolling(5).mean()
        vol_ma_10 = volume.rolling(10).mean()
        vol_ma_20 = volume.rolling(20).mean()
        signals['volume_contraction'] = (
            (vol_ma_5 < vol_ma_10) &
            (vol_ma_10 < vol_ma_20)
        )

        return signals
```

---

## 3. Price-Volume Relationship Signals

### 3.1 Volume Confirmation

**Signal Logic**:
```python
class PriceVolumeSignals:
    """Analyze price-volume relationships for signal generation."""

    def __init__(self, config: RVOLConfig = None):
        self.config = config or RVOLConfig()

    def breakout_confirmation(
        self,
        close: pd.Series,
        high: pd.Series,
        volume: pd.Series,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Confirm price breakouts with volume.
        """
        signals = pd.DataFrame(index=close.index)

        # Price breakout
        resistance = high.rolling(lookback).max()
        price_breakout = close > resistance.shift(1)

        # Volume metrics
        avg_volume = volume.rolling(lookback).mean()
        rvol = volume / avg_volume

        signals['price_breakout'] = price_breakout
        signals['rvol'] = rvol

        # Confirmation levels
        signals['weak_breakout'] = price_breakout & (rvol < 1.0)
        signals['normal_breakout'] = price_breakout & (rvol >= 1.0) & (rvol < 1.5)
        signals['strong_breakout'] = price_breakout & (rvol >= 1.5) & (rvol < 2.5)
        signals['powerful_breakout'] = price_breakout & (rvol >= 2.5)

        # Breakout quality score
        signals['breakout_quality'] = np.where(
            price_breakout,
            np.clip(rvol, 0, 5) * 20,  # Scale to 0-100
            0
        )

        # Failed breakout warning
        signals['suspect_breakout'] = price_breakout & (rvol < 0.8)

        return signals

    def trend_volume_confirmation(
        self,
        close: pd.Series,
        volume: pd.Series,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Analyze volume behavior within trends.
        """
        signals = pd.DataFrame(index=close.index)

        # Identify up/down days
        up_day = close > close.shift(1)
        down_day = close < close.shift(1)

        # Average volume on up vs down days
        def rolling_avg_up(v, u, window):
            vol_up = v.where(u, np.nan)
            return vol_up.rolling(window, min_periods=5).mean()

        def rolling_avg_down(v, d, window):
            vol_down = v.where(d, np.nan)
            return vol_down.rolling(window, min_periods=5).mean()

        avg_vol_up = rolling_avg_up(volume, up_day, lookback)
        avg_vol_down = rolling_avg_down(volume, down_day, lookback)

        signals['avg_volume_up_days'] = avg_vol_up
        signals['avg_volume_down_days'] = avg_vol_down

        # Up/Down volume ratio
        ud_ratio = avg_vol_up / avg_vol_down.replace(0, np.nan)
        signals['up_down_volume_ratio'] = ud_ratio

        # Healthy uptrend: more volume on up days
        signals['healthy_uptrend_volume'] = ud_ratio > 1.2

        # Healthy downtrend: more volume on down days
        signals['healthy_downtrend_volume'] = ud_ratio < 0.8

        # Volume divergence warnings
        price_higher = close > close.shift(lookback)
        volume_lower = volume.rolling(5).mean() < volume.rolling(lookback).mean()

        signals['bearish_volume_divergence'] = price_higher & volume_lower
        signals['bullish_volume_divergence'] = ~price_higher & ~volume_lower

        return signals

    def reversal_volume_signals(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Detect potential reversals from volume patterns.
        """
        signals = pd.DataFrame(index=close.index)

        avg_volume = volume.rolling(lookback).mean()
        rvol = volume / avg_volume

        # Climax patterns
        near_high = close >= high.rolling(lookback).max() * 0.98
        near_low = close <= low.rolling(lookback).min() * 1.02
        extreme_volume = rvol > 3.0

        # Climax top (blow-off)
        bearish_candle = close < (high + low) / 2
        signals['climax_top'] = near_high & extreme_volume & bearish_candle

        # Climax bottom (capitulation)
        bullish_candle = close > (high + low) / 2
        signals['capitulation'] = near_low & extreme_volume & bullish_candle

        # Exhaustion patterns
        # High volume but price barely moves
        price_range = (high - low) / close
        avg_range = price_range.rolling(lookback).mean()
        small_range = price_range < avg_range * 0.5

        signals['absorption'] = extreme_volume & small_range

        # Buying/selling pressure
        # True Range with volume weighting
        buying_pressure = (close - low) / (high - low + 0.0001)
        signals['strong_buying'] = (buying_pressure > 0.7) & (rvol > 1.5)
        signals['strong_selling'] = (buying_pressure < 0.3) & (rvol > 1.5)

        return signals
```

---

## 4. Multi-Timeframe RVOL

### 4.1 Timeframe Integration

**Signal Logic**:
```python
class MultiTimeframeRVOL:
    """Analyze RVOL across multiple timeframes."""

    def __init__(self):
        self.calculator = RVOLCalculator()

    def mtf_rvol_signals(
        self,
        volume_1m: pd.Series,
        volume_5m: pd.Series,
        volume_hourly: pd.Series,
        volume_daily: pd.Series
    ) -> pd.DataFrame:
        """
        Generate multi-timeframe RVOL signals.
        """
        signals = pd.DataFrame()

        # Calculate RVOL at each timeframe
        rvol_1m = self.calculator.simple_rvol(volume_1m, 20)
        rvol_5m = self.calculator.simple_rvol(volume_5m, 20)
        rvol_hourly = self.calculator.simple_rvol(volume_hourly, 20)
        rvol_daily = self.calculator.simple_rvol(volume_daily, 20)

        # Latest values
        latest_1m = rvol_1m.iloc[-1] if len(rvol_1m) > 0 else 1.0
        latest_5m = rvol_5m.iloc[-1] if len(rvol_5m) > 0 else 1.0
        latest_hourly = rvol_hourly.iloc[-1] if len(rvol_hourly) > 0 else 1.0
        latest_daily = rvol_daily.iloc[-1] if len(rvol_daily) > 0 else 1.0

        # Composite signal
        signals['rvol_1m'] = latest_1m
        signals['rvol_5m'] = latest_5m
        signals['rvol_hourly'] = latest_hourly
        signals['rvol_daily'] = latest_daily

        # Alignment score
        elevated_count = sum([
            latest_1m > 1.5,
            latest_5m > 1.5,
            latest_hourly > 1.5,
            latest_daily > 1.5
        ])
        signals['mtf_elevated_count'] = elevated_count

        # All timeframes elevated = strong signal
        signals['mtf_confirmed_high_volume'] = elevated_count >= 3

        # Volume building (lower to higher TF confirmation)
        signals['volume_building'] = (
            (latest_1m > 1.5) &
            (latest_5m > 1.3) &
            (latest_daily < 1.2)  # Still early in day
        )

        return signals

    def intraday_volume_pace(
        self,
        cumulative_volume: pd.Series,
        expected_daily_volume: float,
        current_time: pd.Timestamp
    ) -> Dict:
        """
        Calculate intraday volume pace metrics.
        """
        # Market hours: 9:30 AM to 4:00 PM ET
        market_open = current_time.replace(hour=9, minute=30, second=0)
        market_close = current_time.replace(hour=16, minute=0, second=0)

        total_minutes = (market_close - market_open).seconds / 60
        elapsed_minutes = (current_time - market_open).seconds / 60

        if elapsed_minutes <= 0:
            return {'pace': 0, 'projected': 0, 'rvol_pace': 0}

        # Expected volume at this time (linear assumption)
        expected_at_time = expected_daily_volume * (elapsed_minutes / total_minutes)

        # Actual cumulative
        actual = cumulative_volume.iloc[-1]

        # Pace metrics
        pace = actual / expected_at_time if expected_at_time > 0 else 0
        projected_daily = actual * (total_minutes / elapsed_minutes)
        rvol_projected = projected_daily / expected_daily_volume

        return {
            'pace': pace,
            'projected_volume': projected_daily,
            'rvol_pace': rvol_projected,
            'on_track': pace >= 0.9,
            'running_hot': pace >= 1.5,
            'running_cold': pace < 0.7
        }
```

---

## 5. Composite RVOL Score

### 5.1 Integrated RVOL Engine

```python
class RVOLEngine:
    """
    Production RVOL signal engine.
    """

    def __init__(self, config: RVOLConfig = None):
        self.config = config or RVOLConfig()
        self.signal_gen = RVOLSignalGenerator(config)
        self.pv_signals = PriceVolumeSignals(config)

    def generate_all_signals(
        self,
        ohlcv: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate comprehensive RVOL signals.

        Args:
            ohlcv: DataFrame with open, high, low, close, volume columns

        Returns:
            DataFrame with all RVOL signals
        """
        signals = pd.DataFrame(index=ohlcv.index)

        # Basic RVOL signals
        rvol_signals = self.signal_gen.generate_rvol_signals(ohlcv['volume'])
        signals = pd.concat([signals, rvol_signals], axis=1)

        # Spike detection
        spike_signals = self.signal_gen.volume_spike_detection(ohlcv['volume'])
        signals = pd.concat([signals, spike_signals], axis=1)

        # Dryup detection
        dryup_signals = self.signal_gen.volume_dryup_detection(ohlcv['volume'])
        signals = pd.concat([signals, dryup_signals], axis=1)

        # Breakout confirmation
        breakout_signals = self.pv_signals.breakout_confirmation(
            ohlcv['close'], ohlcv['high'], ohlcv['volume']
        )
        signals = pd.concat([signals, breakout_signals], axis=1)

        # Trend confirmation
        trend_signals = self.pv_signals.trend_volume_confirmation(
            ohlcv['close'], ohlcv['volume']
        )
        signals = pd.concat([signals, trend_signals], axis=1)

        # Reversal signals
        reversal_signals = self.pv_signals.reversal_volume_signals(
            ohlcv['close'], ohlcv['high'], ohlcv['low'], ohlcv['volume']
        )
        signals = pd.concat([signals, reversal_signals], axis=1)

        # Composite score
        signals['rvol_composite_score'] = self._calculate_composite_score(signals)

        return signals

    def _calculate_composite_score(
        self,
        signals: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate composite RVOL signal strength (0-100).
        """
        score = pd.Series(50.0, index=signals.index)  # Neutral base

        # RVOL level component
        if 'rvol_medium' in signals.columns:
            rvol = signals['rvol_medium'].clip(0, 5)
            score += (rvol - 1) * 15  # +/- 15 points per RVOL unit

        # Spike bonus
        if 'confirmed_spike' in signals.columns:
            score += signals['confirmed_spike'].astype(float) * 20

        # Breakout quality
        if 'breakout_quality' in signals.columns:
            score += signals['breakout_quality'] * 0.2

        # Trend confirmation bonus
        if 'healthy_uptrend_volume' in signals.columns:
            score += signals['healthy_uptrend_volume'].astype(float) * 10

        # Reversal warnings (negative)
        if 'bearish_volume_divergence' in signals.columns:
            score -= signals['bearish_volume_divergence'].astype(float) * 15

        return score.clip(0, 100)

    def get_rvol_filter(
        self,
        signals: pd.DataFrame,
        min_rvol: float = 1.0
    ) -> pd.Series:
        """
        Generate filter for minimum RVOL requirement.
        """
        if 'rvol_medium' in signals.columns:
            return signals['rvol_medium'] >= min_rvol
        return pd.Series(True, index=signals.index)
```

---

## Signal Usage Guidelines

### RVOL Trading Matrix

| RVOL Level | Interpretation | Action |
|------------|----------------|--------|
| > 3.0 | Extreme activity | Major event, use caution |
| 2.0 - 3.0 | High activity | Strong confirmation |
| 1.5 - 2.0 | Elevated | Good confirmation |
| 0.8 - 1.2 | Normal | Neutral signal |
| 0.5 - 0.8 | Low | Weak confirmation |
| < 0.5 | Very low | Setup forming / avoid |

### Integration with Ordinis

```python
# RVOL filtering in signal pipeline
rvol_engine = RVOLEngine()
signals = rvol_engine.generate_all_signals(ohlcv_data)

# Require elevated RVOL for breakout trades
breakout_filter = (
    signals['price_breakout'] &
    signals['rvol_medium'] > 1.5
)

# Volume quality score for ranking
ranking_scores = signals['rvol_composite_score']
```

---

## Academic References

1. **Karpoff (1987)**: "The Relation Between Price Changes and Trading Volume"
2. **Blume et al. (1994)**: "Market Statistics and Technical Analysis: The Role of Volume"
3. **Lee & Swaminathan (2000)**: "Price Momentum and Trading Volume"
4. **Gervais et al. (2001)**: "The High-Volume Return Premium"
5. **Llorente et al. (2002)**: "Dynamic Volume-Return Relation"
