# Ichimoku Cloud Signals

## Overview

The Ichimoku Kinko Hyo ("equilibrium chart at a glance") is a comprehensive indicator system providing trend direction, support/resistance, and momentum in a single view. These signals enable **trend identification**, **entry timing**, and **risk management** through multiple confirming components.

---

## 1. Ichimoku Components

### 1.1 Component Calculations

**Signal Logic**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum


class IchimokuSignal(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class IchimokuConfig:
    """Configuration for Ichimoku analysis."""

    # Standard parameters (9, 26, 52)
    tenkan_period: int = 9
    kijun_period: int = 26
    senkou_b_period: int = 52
    displacement: int = 26

    # Alternative parameters for different timeframes
    # Crypto: (10, 30, 60, 30)
    # Weekly: (9, 26, 52, 26)


class IchimokuCalculator:
    """Calculate Ichimoku components."""

    def __init__(self, config: IchimokuConfig = None):
        self.config = config or IchimokuConfig()

    def calculate_all_components(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate all Ichimoku components.
        """
        components = pd.DataFrame(index=close.index)

        # Tenkan-sen (Conversion Line)
        # (9-period high + 9-period low) / 2
        tenkan_high = high.rolling(self.config.tenkan_period).max()
        tenkan_low = low.rolling(self.config.tenkan_period).min()
        components['tenkan_sen'] = (tenkan_high + tenkan_low) / 2

        # Kijun-sen (Base Line)
        # (26-period high + 26-period low) / 2
        kijun_high = high.rolling(self.config.kijun_period).max()
        kijun_low = low.rolling(self.config.kijun_period).min()
        components['kijun_sen'] = (kijun_high + kijun_low) / 2

        # Senkou Span A (Leading Span A)
        # (Tenkan-sen + Kijun-sen) / 2, displaced 26 periods forward
        components['senkou_span_a'] = (
            (components['tenkan_sen'] + components['kijun_sen']) / 2
        ).shift(self.config.displacement)

        # Senkou Span B (Leading Span B)
        # (52-period high + 52-period low) / 2, displaced 26 periods forward
        senkou_b_high = high.rolling(self.config.senkou_b_period).max()
        senkou_b_low = low.rolling(self.config.senkou_b_period).min()
        components['senkou_span_b'] = (
            (senkou_b_high + senkou_b_low) / 2
        ).shift(self.config.displacement)

        # Chikou Span (Lagging Span)
        # Close price displaced 26 periods backward
        components['chikou_span'] = close.shift(-self.config.displacement)

        # Cloud boundaries
        components['cloud_top'] = components[['senkou_span_a', 'senkou_span_b']].max(axis=1)
        components['cloud_bottom'] = components[['senkou_span_a', 'senkou_span_b']].min(axis=1)

        # Cloud color (bullish when A > B)
        components['cloud_bullish'] = components['senkou_span_a'] > components['senkou_span_b']

        return components


class IchimokuSignalGenerator:
    """Generate trading signals from Ichimoku analysis."""

    def __init__(self, config: IchimokuConfig = None):
        self.config = config or IchimokuConfig()
        self.calculator = IchimokuCalculator(config)

    def generate_signals(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Generate comprehensive Ichimoku signals.
        """
        # Calculate components
        components = self.calculator.calculate_all_components(high, low, close)

        signals = pd.DataFrame(index=close.index)

        # Include components
        for col in components.columns:
            signals[col] = components[col]

        # === TK Cross Signals ===
        # Tenkan-Kijun crossover
        signals['tk_cross_bullish'] = (
            (components['tenkan_sen'] > components['kijun_sen']) &
            (components['tenkan_sen'].shift(1) <= components['kijun_sen'].shift(1))
        )
        signals['tk_cross_bearish'] = (
            (components['tenkan_sen'] < components['kijun_sen']) &
            (components['tenkan_sen'].shift(1) >= components['kijun_sen'].shift(1))
        )
        signals['tenkan_above_kijun'] = components['tenkan_sen'] > components['kijun_sen']

        # === Cloud Signals ===
        # Price vs Cloud
        signals['price_above_cloud'] = close > components['cloud_top']
        signals['price_below_cloud'] = close < components['cloud_bottom']
        signals['price_in_cloud'] = (
            (close >= components['cloud_bottom']) &
            (close <= components['cloud_top'])
        )

        # Cloud breakout
        signals['cloud_breakout_up'] = (
            (close > components['cloud_top']) &
            (close.shift(1) <= components['cloud_top'].shift(1))
        )
        signals['cloud_breakout_down'] = (
            (close < components['cloud_bottom']) &
            (close.shift(1) >= components['cloud_bottom'].shift(1))
        )

        # Cloud thickness (momentum)
        signals['cloud_thickness'] = (
            components['cloud_top'] - components['cloud_bottom']
        ).abs() / close
        signals['thick_cloud'] = signals['cloud_thickness'] > 0.03
        signals['thin_cloud'] = signals['cloud_thickness'] < 0.01

        # Future cloud color (26 periods ahead)
        # Using non-displaced values
        tenkan_high = high.rolling(self.config.tenkan_period).max()
        tenkan_low = low.rolling(self.config.tenkan_period).min()
        current_tenkan = (tenkan_high + tenkan_low) / 2

        kijun_high = high.rolling(self.config.kijun_period).max()
        kijun_low = low.rolling(self.config.kijun_period).min()
        current_kijun = (kijun_high + kijun_low) / 2

        future_span_a = (current_tenkan + current_kijun) / 2

        senkou_b_high = high.rolling(self.config.senkou_b_period).max()
        senkou_b_low = low.rolling(self.config.senkou_b_period).min()
        future_span_b = (senkou_b_high + senkou_b_low) / 2

        signals['future_cloud_bullish'] = future_span_a > future_span_b

        # === Chikou Signals ===
        # Compare current Chikou (close) to price 26 periods ago
        price_26_ago = close.shift(self.config.displacement)
        signals['chikou_above_price'] = close > price_26_ago
        signals['chikou_below_price'] = close < price_26_ago

        # Chikou vs cloud (26 periods ago)
        cloud_top_26_ago = components['cloud_top'].shift(self.config.displacement)
        cloud_bottom_26_ago = components['cloud_bottom'].shift(self.config.displacement)

        signals['chikou_above_cloud'] = close > cloud_top_26_ago
        signals['chikou_below_cloud'] = close < cloud_bottom_26_ago

        # === Kijun Support/Resistance ===
        signals['near_kijun'] = (
            (close - components['kijun_sen']).abs() / close < 0.01
        )
        signals['bounced_off_kijun'] = (
            signals['near_kijun'].shift(1) &
            (close > close.shift(1)) &
            (components['kijun_sen'] < close)
        )

        return signals

    def classify_signal_strength(
        self,
        signals: pd.DataFrame,
        close: pd.Series
    ) -> pd.Series:
        """
        Classify overall Ichimoku signal strength.
        """
        strength = pd.Series(IchimokuSignal.NEUTRAL.value, index=close.index)

        # Count bullish/bearish conditions
        bullish_conditions = (
            signals['tenkan_above_kijun'].astype(int) +
            signals['price_above_cloud'].astype(int) +
            signals['chikou_above_price'].astype(int) +
            signals['cloud_bullish'].astype(int) +
            signals['future_cloud_bullish'].astype(int)
        )

        # Classify
        strength = np.where(
            bullish_conditions >= 5, IchimokuSignal.STRONG_BULLISH.value,
            np.where(
                bullish_conditions >= 4, IchimokuSignal.BULLISH.value,
                np.where(
                    bullish_conditions <= 1, IchimokuSignal.STRONG_BEARISH.value,
                    np.where(
                        bullish_conditions <= 2, IchimokuSignal.BEARISH.value,
                        IchimokuSignal.NEUTRAL.value
                    )
                )
            )
        )

        return pd.Series(strength, index=close.index)
```

---

## 2. Entry Signal Strategies

### 2.1 TK Cross Strategy

**Signal Logic**:
```python
class TKCrossStrategy:
    """Tenkan-Kijun cross trading signals."""

    def __init__(self, config: IchimokuConfig = None):
        self.config = config or IchimokuConfig()

    def generate_tk_signals(
        self,
        signals: pd.DataFrame,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Generate TK cross entry signals with filters.
        """
        tk_signals = pd.DataFrame(index=close.index)

        # Basic TK cross
        tk_signals['tk_cross_long'] = signals['tk_cross_bullish']
        tk_signals['tk_cross_short'] = signals['tk_cross_bearish']

        # Filtered signals (cross location matters)

        # Strong bullish: TK cross above cloud
        tk_signals['strong_tk_long'] = (
            signals['tk_cross_bullish'] &
            signals['price_above_cloud'] &
            signals['cloud_bullish']
        )

        # Medium bullish: TK cross inside cloud
        tk_signals['medium_tk_long'] = (
            signals['tk_cross_bullish'] &
            signals['price_in_cloud']
        )

        # Weak bullish: TK cross below cloud (counter-trend)
        tk_signals['weak_tk_long'] = (
            signals['tk_cross_bullish'] &
            signals['price_below_cloud']
        )

        # Strong bearish: TK cross below cloud
        tk_signals['strong_tk_short'] = (
            signals['tk_cross_bearish'] &
            signals['price_below_cloud'] &
            ~signals['cloud_bullish']
        )

        # Cross with Chikou confirmation
        tk_signals['confirmed_tk_long'] = (
            signals['tk_cross_bullish'] &
            signals['chikou_above_price']
        )
        tk_signals['confirmed_tk_short'] = (
            signals['tk_cross_bearish'] &
            signals['chikou_below_price']
        )

        return tk_signals


class CloudBreakoutStrategy:
    """Cloud breakout trading signals."""

    def __init__(self):
        pass

    def generate_breakout_signals(
        self,
        signals: pd.DataFrame,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Generate cloud breakout entry signals.
        """
        breakout = pd.DataFrame(index=close.index)

        # Basic cloud breakout
        breakout['cloud_long'] = signals['cloud_breakout_up']
        breakout['cloud_short'] = signals['cloud_breakout_down']

        # Confirmed breakout (multiple conditions)
        breakout['confirmed_cloud_long'] = (
            signals['cloud_breakout_up'] &
            signals['tenkan_above_kijun'] &
            signals['chikou_above_price'] &
            signals['future_cloud_bullish']
        )

        breakout['confirmed_cloud_short'] = (
            signals['cloud_breakout_down'] &
            ~signals['tenkan_above_kijun'] &
            signals['chikou_below_price'] &
            ~signals['future_cloud_bullish']
        )

        # Breakout quality based on cloud thickness
        breakout['strong_breakout_up'] = (
            signals['cloud_breakout_up'] &
            signals['thick_cloud']  # Breaking thick cloud = stronger move
        )

        breakout['weak_breakout_up'] = (
            signals['cloud_breakout_up'] &
            signals['thin_cloud']  # Breaking thin cloud = less conviction
        )

        return breakout


class KijunBounceStrategy:
    """Kijun-sen support/resistance trading signals."""

    def __init__(self, config: IchimokuConfig = None):
        self.config = config or IchimokuConfig()

    def generate_kijun_signals(
        self,
        signals: pd.DataFrame,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Generate Kijun bounce entry signals.
        """
        kijun_signals = pd.DataFrame(index=close.index)

        kijun = signals['kijun_sen']

        # Price approaching Kijun from above (potential support)
        approaching_from_above = (
            (low < kijun * 1.01) &
            (close > kijun) &
            signals['price_above_cloud']
        )
        kijun_signals['kijun_support_test'] = approaching_from_above

        # Kijun bounce long (touch and bounce)
        kijun_signals['kijun_bounce_long'] = (
            approaching_from_above.shift(1) &
            (close > close.shift(1)) &
            (close > kijun)
        )

        # Price approaching Kijun from below (potential resistance)
        approaching_from_below = (
            (high > kijun * 0.99) &
            (close < kijun) &
            signals['price_below_cloud']
        )
        kijun_signals['kijun_resistance_test'] = approaching_from_below

        # Kijun bounce short
        kijun_signals['kijun_bounce_short'] = (
            approaching_from_below.shift(1) &
            (close < close.shift(1)) &
            (close < kijun)
        )

        # Kijun breakout
        kijun_signals['kijun_break_up'] = (
            (close > kijun) &
            (close.shift(1) <= kijun.shift(1))
        )
        kijun_signals['kijun_break_down'] = (
            (close < kijun) &
            (close.shift(1) >= kijun.shift(1))
        )

        return kijun_signals
```

---

## 3. Exit and Stop Signals

### 3.1 Ichimoku Exits

**Signal Logic**:
```python
class IchimokuExitSignals:
    """Generate exit signals from Ichimoku analysis."""

    def __init__(self):
        pass

    def generate_exit_signals(
        self,
        signals: pd.DataFrame,
        close: pd.Series,
        position_direction: str = 'long'
    ) -> pd.DataFrame:
        """
        Generate exit signals for existing positions.
        """
        exits = pd.DataFrame(index=close.index)

        if position_direction == 'long':
            # Exit long signals

            # TK cross bearish
            exits['tk_exit'] = signals['tk_cross_bearish']

            # Price closes below cloud
            exits['cloud_exit'] = signals['cloud_breakout_down']

            # Chikou crosses below price
            exits['chikou_exit'] = signals['chikou_below_price'] & signals['chikou_above_price'].shift(1)

            # Price closes below Kijun
            exits['kijun_exit'] = (
                (close < signals['kijun_sen']) &
                (close.shift(1) >= signals['kijun_sen'].shift(1))
            )

            # Combined exit trigger
            exits['exit_signal'] = (
                exits['tk_exit'] |
                exits['cloud_exit'] |
                exits['kijun_exit']
            )

        else:  # short position
            # Exit short signals
            exits['tk_exit'] = signals['tk_cross_bullish']
            exits['cloud_exit'] = signals['cloud_breakout_up']
            exits['chikou_exit'] = signals['chikou_above_price'] & signals['chikou_below_price'].shift(1)
            exits['kijun_exit'] = (
                (close > signals['kijun_sen']) &
                (close.shift(1) <= signals['kijun_sen'].shift(1))
            )

            exits['exit_signal'] = (
                exits['tk_exit'] |
                exits['cloud_exit'] |
                exits['kijun_exit']
            )

        return exits

    def get_stop_levels(
        self,
        signals: pd.DataFrame,
        close: pd.Series,
        position_direction: str = 'long'
    ) -> pd.DataFrame:
        """
        Get suggested stop levels from Ichimoku.
        """
        stops = pd.DataFrame(index=close.index)

        if position_direction == 'long':
            # Possible stop levels for long
            stops['kijun_stop'] = signals['kijun_sen']
            stops['cloud_bottom_stop'] = signals['cloud_bottom']
            stops['tenkan_stop'] = signals['tenkan_sen']

            # Recommended: Kijun or cloud bottom, whichever is closer
            stops['recommended_stop'] = signals[['kijun_sen', 'cloud_bottom']].max(axis=1)

        else:
            # Possible stop levels for short
            stops['kijun_stop'] = signals['kijun_sen']
            stops['cloud_top_stop'] = signals['cloud_top']
            stops['tenkan_stop'] = signals['tenkan_sen']

            stops['recommended_stop'] = signals[['kijun_sen', 'cloud_top']].min(axis=1)

        return stops
```

---

## 4. Composite Ichimoku Engine

### 4.1 Production Signal Engine

```python
class IchimokuEngine:
    """
    Production Ichimoku signal engine.
    """

    def __init__(self, config: IchimokuConfig = None):
        self.config = config or IchimokuConfig()
        self.signal_gen = IchimokuSignalGenerator(config)
        self.tk_strategy = TKCrossStrategy(config)
        self.cloud_strategy = CloudBreakoutStrategy()
        self.kijun_strategy = KijunBounceStrategy(config)
        self.exit_signals = IchimokuExitSignals()

    def generate_all_signals(
        self,
        ohlcv: pd.DataFrame
    ) -> Dict:
        """
        Generate comprehensive Ichimoku signals.
        """
        results = {}

        # Core signals
        core_signals = self.signal_gen.generate_signals(
            ohlcv['high'], ohlcv['low'], ohlcv['close']
        )
        results['core'] = core_signals

        # Signal strength
        strength = self.signal_gen.classify_signal_strength(
            core_signals, ohlcv['close']
        )
        results['strength'] = strength

        # Strategy signals
        results['tk_signals'] = self.tk_strategy.generate_tk_signals(
            core_signals, ohlcv['close']
        )
        results['cloud_signals'] = self.cloud_strategy.generate_breakout_signals(
            core_signals, ohlcv['close']
        )
        results['kijun_signals'] = self.kijun_strategy.generate_kijun_signals(
            core_signals, ohlcv['high'], ohlcv['low'], ohlcv['close']
        )

        # Key levels
        results['key_levels'] = {
            'tenkan': core_signals['tenkan_sen'].iloc[-1],
            'kijun': core_signals['kijun_sen'].iloc[-1],
            'cloud_top': core_signals['cloud_top'].iloc[-1],
            'cloud_bottom': core_signals['cloud_bottom'].iloc[-1]
        }

        # Trading recommendation
        results['recommendation'] = self._generate_recommendation(results)

        return results

    def _generate_recommendation(
        self,
        results: Dict
    ) -> Dict:
        """
        Generate trading recommendation from all signals.
        """
        strength = results['strength'].iloc[-1]
        tk = results['tk_signals']
        cloud = results['cloud_signals']

        # Latest entry signals
        has_tk_long = tk['strong_tk_long'].iloc[-1] if 'strong_tk_long' in tk.columns else False
        has_cloud_long = cloud['confirmed_cloud_long'].iloc[-1] if 'confirmed_cloud_long' in cloud.columns else False
        has_tk_short = tk['strong_tk_short'].iloc[-1] if 'strong_tk_short' in tk.columns else False
        has_cloud_short = cloud['confirmed_cloud_short'].iloc[-1] if 'confirmed_cloud_short' in cloud.columns else False

        recommendation = {
            'bias': strength,
            'entry_long': has_tk_long or has_cloud_long,
            'entry_short': has_tk_short or has_cloud_short,
            'stop_long': results['key_levels']['kijun'],
            'stop_short': results['key_levels']['kijun']
        }

        return recommendation
```

---

## Signal Usage Guidelines

### Ichimoku Signal Interpretation

| Component | Bullish | Bearish |
|-----------|---------|---------|
| Tenkan vs Kijun | Tenkan > Kijun | Tenkan < Kijun |
| Price vs Cloud | Above cloud | Below cloud |
| Cloud Color | Span A > Span B | Span A < Span B |
| Chikou | Above price 26 ago | Below price 26 ago |
| Future Cloud | Bullish (A > B) | Bearish (A < B) |

### Signal Strength

- **5/5 conditions bullish**: Strong uptrend, aggressive longs
- **4/5 conditions bullish**: Uptrend, normal longs
- **3/5 conditions**: Neutral, avoid or reduce
- **2/5 conditions bullish**: Downtrend, normal shorts
- **1/5 or less bullish**: Strong downtrend, aggressive shorts

### Integration with Ordinis

```python
# Ichimoku in trend following strategy
ichi_engine = IchimokuEngine()
signals = ichi_engine.generate_all_signals(ohlcv_data)

# Check signal strength
if signals['strength'].iloc[-1] == 'strong_bullish':
    if signals['tk_signals']['confirmed_tk_long'].iloc[-1]:
        stop = signals['key_levels']['kijun']
        enter_long(stop_loss=stop)

# Exit check
if position == 'long':
    exits = ichi_engine.exit_signals.generate_exit_signals(
        signals['core'], ohlcv['close'], 'long'
    )
    if exits['exit_signal'].iloc[-1]:
        close_position()
```

---

## Academic References

1. **Hosoda, Goichi (1969)**: "Ichimoku Kinko Hyo" (Original work)
2. **Elliott, Nicole (2007)**: "Ichimoku Charts"
3. **Patel, Manesh (2010)**: "Trading with Ichimoku Clouds"
4. **Shannon, Dave (2011)**: "Ichimoku Charting & Technical Analysis"
5. **Cloud Charts (2015)**: "Ichimoku Traders Guide"
