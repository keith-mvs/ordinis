# Order Flow Signals

## Overview

Order flow analysis examines the buying and selling pressure behind price movements. These signals provide **institutional detection**, **momentum confirmation**, and **reversal warnings** through volume distribution analysis.

---

## 1. On-Balance Volume (OBV)

### 1.1 OBV Calculation and Signals

**Signal Logic**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class OrderFlowConfig:
    """Configuration for order flow signals."""

    # OBV settings
    obv_ma_short: int = 10
    obv_ma_long: int = 30

    # A/D settings
    ad_smoothing: int = 14

    # MFI settings
    mfi_period: int = 14
    mfi_overbought: float = 80.0
    mfi_oversold: float = 20.0

    # CMF settings
    cmf_period: int = 20
    cmf_strong: float = 0.25
    cmf_weak: float = -0.25


class OBVSignals:
    """On-Balance Volume signal generator."""

    def __init__(self, config: OrderFlowConfig = None):
        self.config = config or OrderFlowConfig()

    def calculate_obv(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate On-Balance Volume.

        OBV adds volume on up days, subtracts on down days.
        """
        direction = np.where(
            close > close.shift(1), 1,
            np.where(close < close.shift(1), -1, 0)
        )
        obv = (volume * direction).cumsum()
        return pd.Series(obv, index=close.index)

    def obv_signals(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """
        Generate OBV-based trading signals.
        """
        signals = pd.DataFrame(index=close.index)

        obv = self.calculate_obv(close, volume)
        signals['obv'] = obv

        # OBV trend
        obv_ma_short = obv.rolling(self.config.obv_ma_short).mean()
        obv_ma_long = obv.rolling(self.config.obv_ma_long).mean()

        signals['obv_ma_short'] = obv_ma_short
        signals['obv_ma_long'] = obv_ma_long

        # Trend signals
        signals['obv_uptrend'] = obv > obv_ma_short
        signals['obv_strong_uptrend'] = obv_ma_short > obv_ma_long
        signals['obv_downtrend'] = obv < obv_ma_short
        signals['obv_strong_downtrend'] = obv_ma_short < obv_ma_long

        # OBV breakout (leading indicator)
        obv_high = obv.rolling(20).max()
        obv_low = obv.rolling(20).min()
        signals['obv_breakout_up'] = obv > obv_high.shift(1)
        signals['obv_breakout_down'] = obv < obv_low.shift(1)

        # OBV momentum
        obv_roc = (obv - obv.shift(10)) / obv.shift(10).abs().replace(0, 1) * 100
        signals['obv_momentum'] = obv_roc
        signals['obv_momentum_strong'] = obv_roc > 10
        signals['obv_momentum_weak'] = obv_roc < -10

        return signals

    def obv_divergence_signals(
        self,
        close: pd.Series,
        obv: pd.Series,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Detect OBV divergences with price.
        """
        signals = pd.DataFrame(index=close.index)

        # Find local highs and lows
        price_high = close.rolling(lookback, center=True).max() == close
        price_low = close.rolling(lookback, center=True).min() == close

        obv_high = obv.rolling(lookback, center=True).max() == obv
        obv_low = obv.rolling(lookback, center=True).min() == obv

        # Bearish divergence: price higher high, OBV lower high
        def detect_bearish_div(price, indicator, window=lookback):
            div = pd.Series(False, index=price.index)
            for i in range(window * 2, len(price)):
                price_window = price.iloc[i-window:i+1]
                ind_window = indicator.iloc[i-window:i+1]

                # Current price at high
                if price.iloc[i] >= price_window.max() * 0.98:
                    # Check if indicator is making lower high
                    recent_ind_max = ind_window.max()
                    prior_ind_max = indicator.iloc[max(0, i-window*2):i-window].max()
                    if recent_ind_max < prior_ind_max * 0.95:
                        div.iloc[i] = True
            return div

        signals['bearish_divergence'] = detect_bearish_div(close, obv)

        # Bullish divergence: price lower low, OBV higher low
        def detect_bullish_div(price, indicator, window=lookback):
            div = pd.Series(False, index=price.index)
            for i in range(window * 2, len(price)):
                price_window = price.iloc[i-window:i+1]
                ind_window = indicator.iloc[i-window:i+1]

                # Current price at low
                if price.iloc[i] <= price_window.min() * 1.02:
                    # Check if indicator is making higher low
                    recent_ind_min = ind_window.min()
                    prior_ind_min = indicator.iloc[max(0, i-window*2):i-window].min()
                    if recent_ind_min > prior_ind_min * 1.05:
                        div.iloc[i] = True
            return div

        signals['bullish_divergence'] = detect_bullish_div(close, obv)

        return signals
```

---

## 2. Accumulation/Distribution (A/D)

### 2.1 A/D Line Signals

**Signal Logic**:
```python
class AccumulationDistributionSignals:
    """Accumulation/Distribution line signal generator."""

    def __init__(self, config: OrderFlowConfig = None):
        self.config = config or OrderFlowConfig()

    def calculate_ad_line(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line.

        CLV = [(Close - Low) - (High - Close)] / (High - Low)
        A/D = Previous A/D + CLV * Volume
        """
        high_low_range = high - low
        clv = np.where(
            high_low_range > 0,
            ((close - low) - (high - close)) / high_low_range,
            0
        )
        ad = (clv * volume).cumsum()
        return pd.Series(ad, index=close.index)

    def ad_signals(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """
        Generate A/D line signals.
        """
        signals = pd.DataFrame(index=close.index)

        ad = self.calculate_ad_line(high, low, close, volume)
        signals['ad_line'] = ad

        # A/D trend
        ad_ma = ad.rolling(self.config.ad_smoothing).mean()
        signals['ad_ma'] = ad_ma

        signals['accumulation'] = ad > ad_ma
        signals['distribution'] = ad < ad_ma

        # A/D momentum
        ad_change = ad - ad.shift(10)
        signals['ad_momentum'] = ad_change
        signals['strong_accumulation'] = ad_change > ad_change.rolling(20).std() * 2
        signals['strong_distribution'] = ad_change < -ad_change.rolling(20).std() * 2

        # A/D vs price divergence
        price_trend = close > close.rolling(20).mean()
        ad_trend = ad > ad.rolling(20).mean()

        signals['ad_price_aligned'] = price_trend == ad_trend
        signals['ad_price_diverging'] = price_trend != ad_trend

        # Bullish divergence: price down, A/D up
        signals['bullish_ad_divergence'] = ~price_trend & ad_trend

        # Bearish divergence: price up, A/D down
        signals['bearish_ad_divergence'] = price_trend & ~ad_trend

        return signals

    def chaikin_ad_oscillator(
        self,
        ad_line: pd.Series,
        fast_period: int = 3,
        slow_period: int = 10
    ) -> pd.DataFrame:
        """
        Calculate Chaikin A/D Oscillator.

        ADOSC = EMA(A/D, fast) - EMA(A/D, slow)
        """
        signals = pd.DataFrame(index=ad_line.index)

        ema_fast = ad_line.ewm(span=fast_period).mean()
        ema_slow = ad_line.ewm(span=slow_period).mean()

        adosc = ema_fast - ema_slow
        signals['chaikin_oscillator'] = adosc

        # Oscillator signals
        signals['chaikin_positive'] = adosc > 0
        signals['chaikin_negative'] = adosc < 0
        signals['chaikin_rising'] = adosc > adosc.shift(1)
        signals['chaikin_falling'] = adosc < adosc.shift(1)

        # Zero line crossovers
        signals['chaikin_bullish_cross'] = (adosc > 0) & (adosc.shift(1) <= 0)
        signals['chaikin_bearish_cross'] = (adosc < 0) & (adosc.shift(1) >= 0)

        return signals
```

---

## 3. Money Flow Index (MFI)

### 3.1 MFI Signals

**Signal Logic**:
```python
class MoneyFlowIndexSignals:
    """Money Flow Index signal generator."""

    def __init__(self, config: OrderFlowConfig = None):
        self.config = config or OrderFlowConfig()

    def calculate_mfi(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = None
    ) -> pd.Series:
        """
        Calculate Money Flow Index.

        MFI is volume-weighted RSI.
        """
        period = period or self.config.mfi_period

        # Typical price
        typical_price = (high + low + close) / 3

        # Raw money flow
        raw_mf = typical_price * volume

        # Positive/negative flow
        positive_flow = np.where(
            typical_price > typical_price.shift(1),
            raw_mf, 0
        )
        negative_flow = np.where(
            typical_price < typical_price.shift(1),
            raw_mf, 0
        )

        positive_flow = pd.Series(positive_flow, index=close.index)
        negative_flow = pd.Series(negative_flow, index=close.index)

        # Sum over period
        positive_sum = positive_flow.rolling(period).sum()
        negative_sum = negative_flow.rolling(period).sum()

        # Money flow ratio
        mf_ratio = positive_sum / negative_sum.replace(0, np.nan)

        # MFI
        mfi = 100 - (100 / (1 + mf_ratio))

        return mfi

    def mfi_signals(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """
        Generate MFI trading signals.
        """
        signals = pd.DataFrame(index=close.index)

        mfi = self.calculate_mfi(high, low, close, volume)
        signals['mfi'] = mfi

        # Overbought/oversold
        signals['mfi_overbought'] = mfi > self.config.mfi_overbought
        signals['mfi_oversold'] = mfi < self.config.mfi_oversold
        signals['mfi_neutral'] = (
            (mfi >= self.config.mfi_oversold) &
            (mfi <= self.config.mfi_overbought)
        )

        # Extreme levels
        signals['mfi_extreme_overbought'] = mfi > 90
        signals['mfi_extreme_oversold'] = mfi < 10

        # MFI direction
        signals['mfi_rising'] = mfi > mfi.shift(1)
        signals['mfi_falling'] = mfi < mfi.shift(1)

        # MFI reversal signals
        signals['mfi_bullish_reversal'] = (
            (mfi > self.config.mfi_oversold) &
            (mfi.shift(1) <= self.config.mfi_oversold) &
            signals['mfi_rising']
        )
        signals['mfi_bearish_reversal'] = (
            (mfi < self.config.mfi_overbought) &
            (mfi.shift(1) >= self.config.mfi_overbought) &
            signals['mfi_falling']
        )

        # MFI divergence
        price_higher = close > close.shift(10)
        mfi_lower = mfi < mfi.shift(10)

        signals['mfi_bearish_divergence'] = price_higher & mfi_lower
        signals['mfi_bullish_divergence'] = ~price_higher & ~mfi_lower

        return signals
```

---

## 4. Chaikin Money Flow (CMF)

### 4.1 CMF Signals

**Signal Logic**:
```python
class ChaikinMoneyFlowSignals:
    """Chaikin Money Flow signal generator."""

    def __init__(self, config: OrderFlowConfig = None):
        self.config = config or OrderFlowConfig()

    def calculate_cmf(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = None
    ) -> pd.Series:
        """
        Calculate Chaikin Money Flow.

        CMF = Sum(CLV * Volume, n) / Sum(Volume, n)
        """
        period = period or self.config.cmf_period

        high_low_range = high - low
        clv = np.where(
            high_low_range > 0,
            ((close - low) - (high - close)) / high_low_range,
            0
        )
        clv = pd.Series(clv, index=close.index)

        mf_volume = clv * volume

        cmf = mf_volume.rolling(period).sum() / volume.rolling(period).sum()

        return cmf

    def cmf_signals(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """
        Generate CMF trading signals.
        """
        signals = pd.DataFrame(index=close.index)

        cmf = self.calculate_cmf(high, low, close, volume)
        signals['cmf'] = cmf

        # Level-based signals
        signals['cmf_positive'] = cmf > 0
        signals['cmf_negative'] = cmf < 0
        signals['cmf_strong_buying'] = cmf > self.config.cmf_strong
        signals['cmf_strong_selling'] = cmf < self.config.cmf_weak

        # CMF zero line cross
        signals['cmf_bullish_cross'] = (cmf > 0) & (cmf.shift(1) <= 0)
        signals['cmf_bearish_cross'] = (cmf < 0) & (cmf.shift(1) >= 0)

        # CMF trend
        cmf_ma = cmf.rolling(10).mean()
        signals['cmf_trending_up'] = cmf > cmf_ma
        signals['cmf_trending_down'] = cmf < cmf_ma

        # CMF vs price divergence
        price_uptrend = close > close.rolling(20).mean()

        signals['cmf_confirms_uptrend'] = price_uptrend & (cmf > 0)
        signals['cmf_confirms_downtrend'] = ~price_uptrend & (cmf < 0)
        signals['cmf_diverges_from_price'] = (
            (price_uptrend & (cmf < 0)) |
            (~price_uptrend & (cmf > 0))
        )

        return signals
```

---

## 5. Force Index

### 5.1 Force Index Signals

**Signal Logic**:
```python
class ForceIndexSignals:
    """Force Index signal generator."""

    def __init__(self, short_period: int = 2, long_period: int = 13):
        self.short_period = short_period
        self.long_period = long_period

    def calculate_force_index(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate Force Index.

        Force Index = (Close - Previous Close) * Volume
        """
        force = (close - close.shift(1)) * volume
        return force

    def force_index_signals(
        self,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.DataFrame:
        """
        Generate Force Index trading signals.
        """
        signals = pd.DataFrame(index=close.index)

        force = self.calculate_force_index(close, volume)
        signals['force_index'] = force

        # Smoothed versions
        force_short = force.ewm(span=self.short_period).mean()
        force_long = force.ewm(span=self.long_period).mean()

        signals['force_short'] = force_short
        signals['force_long'] = force_long

        # Direction signals
        signals['buyers_in_control'] = force_long > 0
        signals['sellers_in_control'] = force_long < 0

        # Short-term momentum
        signals['short_term_buying'] = force_short > 0
        signals['short_term_selling'] = force_short < 0

        # Crossover signals
        signals['force_bullish_cross'] = (
            (force_short > 0) & (force_short.shift(1) <= 0)
        )
        signals['force_bearish_cross'] = (
            (force_short < 0) & (force_short.shift(1) >= 0)
        )

        # Strength of move
        force_ma = force_long.rolling(20).mean()
        force_std = force_long.rolling(20).std()

        signals['strong_buying_pressure'] = force_long > (force_ma + 2 * force_std)
        signals['strong_selling_pressure'] = force_long < (force_ma - 2 * force_std)

        # Pullback entry signals (short-term dip in long-term uptrend)
        signals['pullback_buy'] = (force_long > 0) & (force_short < 0)
        signals['pullback_sell'] = (force_long < 0) & (force_short > 0)

        return signals
```

---

## 6. Composite Order Flow Engine

### 6.1 Integrated Signal Generation

```python
class OrderFlowEngine:
    """
    Production order flow signal engine.
    """

    def __init__(self, config: OrderFlowConfig = None):
        self.config = config or OrderFlowConfig()
        self.obv = OBVSignals(config)
        self.ad = AccumulationDistributionSignals(config)
        self.mfi = MoneyFlowIndexSignals(config)
        self.cmf = ChaikinMoneyFlowSignals(config)
        self.force = ForceIndexSignals()

    def generate_all_signals(
        self,
        ohlcv: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate comprehensive order flow signals.
        """
        signals = pd.DataFrame(index=ohlcv.index)

        # OBV signals
        obv_sigs = self.obv.obv_signals(ohlcv['close'], ohlcv['volume'])
        signals = pd.concat([signals, obv_sigs], axis=1)

        # A/D signals
        ad_sigs = self.ad.ad_signals(
            ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume']
        )
        signals = pd.concat([signals, ad_sigs], axis=1)

        # MFI signals
        mfi_sigs = self.mfi.mfi_signals(
            ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume']
        )
        signals = pd.concat([signals, mfi_sigs], axis=1)

        # CMF signals
        cmf_sigs = self.cmf.cmf_signals(
            ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume']
        )
        signals = pd.concat([signals, cmf_sigs], axis=1)

        # Force Index signals
        force_sigs = self.force.force_index_signals(
            ohlcv['close'], ohlcv['volume']
        )
        signals = pd.concat([signals, force_sigs], axis=1)

        # Composite score
        signals['order_flow_score'] = self._calculate_composite_score(signals)

        return signals

    def _calculate_composite_score(
        self,
        signals: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate composite order flow score (-100 to +100).

        Positive = buying pressure dominant
        Negative = selling pressure dominant
        """
        score = pd.Series(0.0, index=signals.index)

        # OBV component
        if 'obv_strong_uptrend' in signals.columns:
            score += signals['obv_strong_uptrend'].astype(float) * 20
            score -= signals['obv_strong_downtrend'].astype(float) * 20

        # A/D component
        if 'accumulation' in signals.columns:
            score += signals['accumulation'].astype(float) * 15
            score -= signals['distribution'].astype(float) * 15

        # MFI component
        if 'mfi' in signals.columns:
            mfi_contrib = (signals['mfi'] - 50) / 50 * 20
            score += mfi_contrib

        # CMF component
        if 'cmf' in signals.columns:
            cmf_contrib = signals['cmf'] * 50  # CMF already -1 to +1
            score += cmf_contrib

        # Force Index component
        if 'buyers_in_control' in signals.columns:
            score += signals['buyers_in_control'].astype(float) * 15
            score -= signals['sellers_in_control'].astype(float) * 15

        return score.clip(-100, 100)

    def get_order_flow_bias(
        self,
        signals: pd.DataFrame
    ) -> str:
        """
        Get current order flow bias interpretation.
        """
        if 'order_flow_score' not in signals.columns:
            return "neutral"

        score = signals['order_flow_score'].iloc[-1]

        if score > 50:
            return "strong_buying"
        elif score > 20:
            return "moderate_buying"
        elif score > -20:
            return "neutral"
        elif score > -50:
            return "moderate_selling"
        else:
            return "strong_selling"
```

---

## Signal Usage Guidelines

### Order Flow Indicator Summary

| Indicator | Best Use | Timeframe |
|-----------|----------|-----------|
| OBV | Trend confirmation, divergence | Daily/Weekly |
| A/D Line | Accumulation detection | Daily |
| MFI | Overbought/oversold with volume | Intraday/Daily |
| CMF | Money flow direction | Daily |
| Force Index | Momentum with volume weight | Daily |

### Integration with Ordinis

```python
# Order flow integration
of_engine = OrderFlowEngine()
signals = of_engine.generate_all_signals(ohlcv_data)

# Get bias
bias = of_engine.get_order_flow_bias(signals)

# Use in entry logic
if bias == "strong_buying" and technical_signal:
    enter_long()
elif bias == "strong_selling" and technical_signal:
    enter_short()
```

---

## Academic References

1. **Granville (1963)**: "New Key to Stock Market Profits" (OBV)
2. **Chaikin (1966)**: "Accumulation/Distribution Line"
3. **Quong & Soudack (1989)**: "Money Flow Index"
4. **Elder (1993)**: "Trading for a Living" (Force Index)
5. **Arms (1983)**: "The Arms Index" (TRIN, volume analysis)
