# Market Breadth Signals

## Overview

Market breadth measures the participation of individual stocks in market moves, revealing the health and sustainability of trends. These signals provide **trend confirmation**, **divergence warnings**, and **market regime** classification unavailable from index prices alone.

---

## 1. Advance-Decline Indicators

### 1.1 A/D Line and Ratio

**Signal Logic**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class MarketBreadthRegime(Enum):
    STRONG_PARTICIPATION = "strong_participation"
    HEALTHY = "healthy"
    NARROWING = "narrowing"
    WEAK = "weak"
    DIVERGING = "diverging"


@dataclass
class BreadthConfig:
    """Configuration for breadth analysis."""

    # Moving average periods
    ma_short: int = 10
    ma_medium: int = 20
    ma_long: int = 50

    # Threshold percentages
    strong_threshold: float = 0.65  # 65% advancing
    healthy_threshold: float = 0.55
    weak_threshold: float = 0.45
    very_weak_threshold: float = 0.35


class AdvanceDeclineSignals:
    """Advance/Decline breadth indicators."""

    def __init__(self, config: BreadthConfig = None):
        self.config = config or BreadthConfig()

    def calculate_ad_line(
        self,
        advances: pd.Series,
        declines: pd.Series,
        unchanged: pd.Series = None
    ) -> pd.DataFrame:
        """
        Calculate Advance-Decline Line and related metrics.
        """
        signals = pd.DataFrame(index=advances.index)

        # Net advances
        net_advances = advances - declines
        signals['net_advances'] = net_advances

        # Cumulative A/D Line
        ad_line = net_advances.cumsum()
        signals['ad_line'] = ad_line

        # A/D Line moving averages
        signals['ad_line_ma_short'] = ad_line.rolling(self.config.ma_short).mean()
        signals['ad_line_ma_long'] = ad_line.rolling(self.config.ma_long).mean()

        # A/D Line trend
        signals['ad_uptrend'] = ad_line > signals['ad_line_ma_short']
        signals['ad_strong_uptrend'] = signals['ad_line_ma_short'] > signals['ad_line_ma_long']
        signals['ad_downtrend'] = ad_line < signals['ad_line_ma_short']
        signals['ad_strong_downtrend'] = signals['ad_line_ma_short'] < signals['ad_line_ma_long']

        # A/D Line new highs/lows
        signals['ad_new_high'] = ad_line >= ad_line.rolling(52).max()
        signals['ad_new_low'] = ad_line <= ad_line.rolling(52).min()

        # A/D Ratio
        total = advances + declines
        if unchanged is not None:
            total += unchanged
        signals['ad_ratio'] = advances / declines.replace(0, 1)
        signals['ad_percent'] = advances / total.replace(0, 1)

        # Breadth thrust
        ad_ratio_ma = signals['ad_ratio'].rolling(10).mean()
        signals['breadth_thrust'] = signals['ad_ratio'] > ad_ratio_ma * 2

        return signals

    def ad_divergence_signals(
        self,
        ad_line: pd.Series,
        index_price: pd.Series,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Detect divergences between A/D Line and index.
        """
        signals = pd.DataFrame(index=ad_line.index)

        # Price making new highs but A/D not
        price_high = index_price >= index_price.rolling(lookback).max()
        ad_high = ad_line >= ad_line.rolling(lookback).max()

        signals['bearish_divergence'] = price_high & ~ad_high

        # Price making new lows but A/D not
        price_low = index_price <= index_price.rolling(lookback).min()
        ad_low = ad_line <= ad_line.rolling(lookback).min()

        signals['bullish_divergence'] = price_low & ~ad_low

        # Sustained divergence (multiple days)
        signals['sustained_bearish_div'] = signals['bearish_divergence'].rolling(5).sum() >= 3
        signals['sustained_bullish_div'] = signals['bullish_divergence'].rolling(5).sum() >= 3

        return signals


class AdvanceDeclineRatioSignals:
    """A/D Ratio specific signals."""

    def __init__(self, config: BreadthConfig = None):
        self.config = config or BreadthConfig()

    def generate_ratio_signals(
        self,
        advances: pd.Series,
        declines: pd.Series
    ) -> pd.DataFrame:
        """
        Generate signals from A/D ratio analysis.
        """
        signals = pd.DataFrame(index=advances.index)

        total = advances + declines
        ad_percent = advances / total.replace(0, 1)

        signals['ad_percent'] = ad_percent

        # Participation levels
        signals['strong_participation'] = ad_percent > self.config.strong_threshold
        signals['healthy_participation'] = (
            (ad_percent > self.config.healthy_threshold) &
            (ad_percent <= self.config.strong_threshold)
        )
        signals['weak_participation'] = (
            (ad_percent > self.config.weak_threshold) &
            (ad_percent <= self.config.healthy_threshold)
        )
        signals['very_weak'] = ad_percent <= self.config.very_weak_threshold

        # Moving averages
        ad_percent_ma = ad_percent.rolling(10).mean()
        signals['participation_improving'] = ad_percent > ad_percent_ma
        signals['participation_deteriorating'] = ad_percent < ad_percent_ma

        # Extreme readings
        signals['breadth_overbought'] = ad_percent > 0.80
        signals['breadth_oversold'] = ad_percent < 0.20

        # McClellan style ratio
        ratio = advances / declines.replace(0, 1)
        signals['ad_ratio'] = ratio
        signals['ratio_extreme_bullish'] = ratio > 3.0
        signals['ratio_extreme_bearish'] = ratio < 0.33

        return signals
```

---

## 2. McClellan Indicators

### 2.1 McClellan Oscillator and Summation Index

**Signal Logic**:
```python
class McClellanIndicators:
    """McClellan Oscillator and Summation Index."""

    def __init__(
        self,
        ema_fast: int = 19,
        ema_slow: int = 39
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    def calculate_oscillator(
        self,
        advances: pd.Series,
        declines: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate McClellan Oscillator.

        Oscillator = EMA(19) of net advances - EMA(39) of net advances
        """
        signals = pd.DataFrame(index=advances.index)

        net_advances = advances - declines

        # EMAs of net advances
        ema_fast = net_advances.ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = net_advances.ewm(span=self.ema_slow, adjust=False).mean()

        # McClellan Oscillator
        oscillator = ema_fast - ema_slow
        signals['mcclellan_osc'] = oscillator

        # Zero line signals
        signals['osc_positive'] = oscillator > 0
        signals['osc_negative'] = oscillator < 0
        signals['osc_cross_above_zero'] = (oscillator > 0) & (oscillator.shift(1) <= 0)
        signals['osc_cross_below_zero'] = (oscillator < 0) & (oscillator.shift(1) >= 0)

        # Extreme levels
        signals['osc_extreme_positive'] = oscillator > 100
        signals['osc_extreme_negative'] = oscillator < -100
        signals['osc_very_extreme_positive'] = oscillator > 150
        signals['osc_very_extreme_negative'] = oscillator < -150

        # Oscillator direction
        signals['osc_rising'] = oscillator > oscillator.shift(1)
        signals['osc_falling'] = oscillator < oscillator.shift(1)

        # Breadth thrust
        signals['breadth_thrust_up'] = oscillator > 100
        signals['breadth_thrust_down'] = oscillator < -100

        return signals

    def calculate_summation_index(
        self,
        oscillator: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate McClellan Summation Index.

        Summation Index = Cumulative sum of McClellan Oscillator
        """
        signals = pd.DataFrame(index=oscillator.index)

        summation = oscillator.cumsum()
        signals['summation_index'] = summation

        # Trend
        signals['summation_uptrend'] = summation > summation.shift(1)
        signals['summation_downtrend'] = summation < summation.shift(1)

        # Key levels
        signals['summation_positive'] = summation > 0
        signals['summation_negative'] = summation < 0

        # Moving average
        summation_ma = summation.rolling(20).mean()
        signals['summation_above_ma'] = summation > summation_ma
        signals['summation_below_ma'] = summation < summation_ma

        # Extreme readings (historically)
        # These vary by market, but approximate levels
        signals['summation_high'] = summation > 1500
        signals['summation_low'] = summation < -1500
        signals['summation_very_high'] = summation > 2500
        signals['summation_very_low'] = summation < -2500

        return signals
```

---

## 3. New Highs/Lows Indicators

### 3.1 New Highs-Lows Analysis

**Signal Logic**:
```python
class NewHighsLowsSignals:
    """New 52-week highs and lows analysis."""

    def __init__(self, config: BreadthConfig = None):
        self.config = config or BreadthConfig()

    def calculate_nh_nl_signals(
        self,
        new_highs: pd.Series,
        new_lows: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate new highs minus new lows indicators.
        """
        signals = pd.DataFrame(index=new_highs.index)

        # Net new highs
        net_nh = new_highs - new_lows
        signals['net_new_highs'] = net_nh

        # Cumulative new highs-lows
        signals['cumulative_nh_nl'] = net_nh.cumsum()

        # Moving averages
        signals['nh_nl_ma_10'] = net_nh.rolling(10).mean()
        signals['nh_nl_ma_20'] = net_nh.rolling(20).mean()

        # Trend signals
        signals['nh_nl_positive'] = net_nh > 0
        signals['nh_nl_expanding'] = signals['nh_nl_ma_10'] > signals['nh_nl_ma_20']

        # Extreme readings
        total_issues = new_highs + new_lows + 100  # Approximate total
        nh_percent = new_highs / total_issues
        nl_percent = new_lows / total_issues

        signals['many_new_highs'] = nh_percent > 0.05  # >5% making new highs
        signals['many_new_lows'] = nl_percent > 0.05   # >5% making new lows

        # New lows expansion (warning)
        signals['new_lows_expanding'] = (
            new_lows > new_lows.rolling(10).mean() * 1.5
        )

        # High-Low Logic Index
        # Smoothed new highs / (new highs + new lows)
        hl_ratio = new_highs / (new_highs + new_lows).replace(0, 1)
        hl_ratio_smooth = hl_ratio.ewm(span=10).mean()
        signals['hl_logic_index'] = hl_ratio_smooth

        signals['hl_bullish'] = hl_ratio_smooth > 0.6
        signals['hl_bearish'] = hl_ratio_smooth < 0.4

        return signals

    def nh_nl_divergence(
        self,
        net_new_highs: pd.Series,
        index_price: pd.Series,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Detect divergences in new highs/lows.
        """
        signals = pd.DataFrame(index=net_new_highs.index)

        # Index at highs but net new highs declining
        price_high = index_price >= index_price.rolling(lookback).max() * 0.98
        nh_declining = net_new_highs < net_new_highs.rolling(lookback).max() * 0.5

        signals['nh_bearish_divergence'] = price_high & nh_declining

        # Index at lows but net new lows improving
        price_low = index_price <= index_price.rolling(lookback).min() * 1.02
        nh_improving = net_new_highs > net_new_highs.rolling(lookback).min() * 0.5

        signals['nh_bullish_divergence'] = price_low & nh_improving

        return signals
```

---

## 4. Percent of Stocks Indicators

### 4.1 Percentage Above Moving Average

**Signal Logic**:
```python
class PercentAboveMASignals:
    """Percentage of stocks above moving averages."""

    def __init__(self, config: BreadthConfig = None):
        self.config = config or BreadthConfig()

    def generate_percent_signals(
        self,
        pct_above_50ma: pd.Series,
        pct_above_200ma: pd.Series
    ) -> pd.DataFrame:
        """
        Generate signals from percent above MA indicators.
        """
        signals = pd.DataFrame(index=pct_above_50ma.index)

        signals['pct_above_50ma'] = pct_above_50ma
        signals['pct_above_200ma'] = pct_above_200ma

        # Market condition based on 200MA
        signals['broad_uptrend'] = pct_above_200ma > 60
        signals['broad_downtrend'] = pct_above_200ma < 40
        signals['mixed_market'] = (
            (pct_above_200ma >= 40) & (pct_above_200ma <= 60)
        )

        # Short-term condition (50MA)
        signals['short_term_strong'] = pct_above_50ma > 70
        signals['short_term_weak'] = pct_above_50ma < 30

        # Overbought/oversold extremes
        signals['breadth_overbought'] = pct_above_50ma > 80
        signals['breadth_oversold'] = pct_above_50ma < 20

        # Trend alignment
        signals['trend_aligned_up'] = (
            (pct_above_50ma > 50) & (pct_above_200ma > 50)
        )
        signals['trend_aligned_down'] = (
            (pct_above_50ma < 50) & (pct_above_200ma < 50)
        )
        signals['trend_conflicted'] = (
            ((pct_above_50ma > 50) & (pct_above_200ma < 50)) |
            ((pct_above_50ma < 50) & (pct_above_200ma > 50))
        )

        # Crossovers
        signals['short_cross_above_50'] = (
            (pct_above_50ma > 50) & (pct_above_50ma.shift(1) <= 50)
        )
        signals['short_cross_below_50'] = (
            (pct_above_50ma < 50) & (pct_above_50ma.shift(1) >= 50)
        )

        # Rate of change
        pct_change = pct_above_50ma - pct_above_50ma.shift(5)
        signals['participation_surge'] = pct_change > 20
        signals['participation_collapse'] = pct_change < -20

        return signals

    def multi_timeframe_breadth(
        self,
        pct_above_20ma: pd.Series,
        pct_above_50ma: pd.Series,
        pct_above_200ma: pd.Series
    ) -> pd.DataFrame:
        """
        Multi-timeframe breadth analysis.
        """
        signals = pd.DataFrame(index=pct_above_20ma.index)

        signals['pct_20ma'] = pct_above_20ma
        signals['pct_50ma'] = pct_above_50ma
        signals['pct_200ma'] = pct_above_200ma

        # All timeframes aligned bullish
        signals['all_tf_bullish'] = (
            (pct_above_20ma > 50) &
            (pct_above_50ma > 50) &
            (pct_above_200ma > 50)
        )

        # All timeframes aligned bearish
        signals['all_tf_bearish'] = (
            (pct_above_20ma < 50) &
            (pct_above_50ma < 50) &
            (pct_above_200ma < 50)
        )

        # Strong bull (all >60%)
        signals['strong_bull_breadth'] = (
            (pct_above_20ma > 60) &
            (pct_above_50ma > 60) &
            (pct_above_200ma > 60)
        )

        # Strong bear (all <40%)
        signals['strong_bear_breadth'] = (
            (pct_above_20ma < 40) &
            (pct_above_50ma < 40) &
            (pct_above_200ma < 40)
        )

        # Breadth improving (short > medium > long)
        signals['breadth_improving'] = (
            (pct_above_20ma > pct_above_50ma) &
            (pct_above_50ma > pct_above_200ma)
        )

        # Breadth deteriorating
        signals['breadth_deteriorating'] = (
            (pct_above_20ma < pct_above_50ma) &
            (pct_above_50ma < pct_above_200ma)
        )

        return signals
```

---

## 5. Composite Breadth Engine

### 5.1 Production Signal Engine

```python
class MarketBreadthEngine:
    """
    Production market breadth signal engine.
    """

    def __init__(self, config: BreadthConfig = None):
        self.config = config or BreadthConfig()
        self.ad_signals = AdvanceDeclineSignals(config)
        self.mcclellan = McClellanIndicators()
        self.nh_nl = NewHighsLowsSignals(config)
        self.pct_above = PercentAboveMASignals(config)

    def generate_all_signals(
        self,
        breadth_data: pd.DataFrame
    ) -> Dict:
        """
        Generate comprehensive market breadth signals.

        Expected columns in breadth_data:
        - advances, declines, unchanged
        - new_highs, new_lows
        - pct_above_50ma, pct_above_200ma
        - index_price (for divergence analysis)
        """
        results = {}

        # A/D signals
        if all(c in breadth_data.columns for c in ['advances', 'declines']):
            ad = self.ad_signals.calculate_ad_line(
                breadth_data['advances'],
                breadth_data['declines'],
                breadth_data.get('unchanged')
            )
            results['ad_line'] = ad

            # McClellan
            mccl_osc = self.mcclellan.calculate_oscillator(
                breadth_data['advances'],
                breadth_data['declines']
            )
            results['mcclellan_oscillator'] = mccl_osc

            mccl_sum = self.mcclellan.calculate_summation_index(
                mccl_osc['mcclellan_osc']
            )
            results['summation_index'] = mccl_sum

            # Divergence
            if 'index_price' in breadth_data.columns:
                div = self.ad_signals.ad_divergence_signals(
                    ad['ad_line'],
                    breadth_data['index_price']
                )
                results['divergences'] = div

        # New highs/lows
        if all(c in breadth_data.columns for c in ['new_highs', 'new_lows']):
            nh_nl = self.nh_nl.calculate_nh_nl_signals(
                breadth_data['new_highs'],
                breadth_data['new_lows']
            )
            results['new_highs_lows'] = nh_nl

        # Percent above MA
        if all(c in breadth_data.columns for c in ['pct_above_50ma', 'pct_above_200ma']):
            pct = self.pct_above.generate_percent_signals(
                breadth_data['pct_above_50ma'],
                breadth_data['pct_above_200ma']
            )
            results['percent_above_ma'] = pct

        # Composite breadth score
        results['composite_score'] = self._calculate_composite(results)

        # Market regime
        results['regime'] = self._classify_regime(results)

        return results

    def _calculate_composite(
        self,
        results: Dict
    ) -> pd.Series:
        """
        Calculate composite breadth score (-100 to +100).
        """
        score = pd.Series(0.0, index=results.get('ad_line', pd.DataFrame()).index)

        # A/D component
        if 'ad_line' in results:
            ad = results['ad_line']
            score += np.where(ad['ad_uptrend'], 20, -20)

        # McClellan component
        if 'mcclellan_oscillator' in results:
            osc = results['mcclellan_oscillator']
            score += np.where(osc['osc_positive'], 20, -20)
            score += np.where(osc['osc_extreme_positive'], 10, 0)
            score += np.where(osc['osc_extreme_negative'], -10, 0)

        # New highs/lows component
        if 'new_highs_lows' in results:
            nh = results['new_highs_lows']
            score += np.where(nh['nh_nl_positive'], 15, -15)

        # Percent above MA component
        if 'percent_above_ma' in results:
            pct = results['percent_above_ma']
            score += np.where(pct['broad_uptrend'], 15, 0)
            score += np.where(pct['broad_downtrend'], -15, 0)

        return score.clip(-100, 100)

    def _classify_regime(
        self,
        results: Dict
    ) -> pd.Series:
        """
        Classify market breadth regime.
        """
        score = results.get('composite_score', pd.Series(0))

        regime = np.where(
            score > 60, MarketBreadthRegime.STRONG_PARTICIPATION.value,
            np.where(
                score > 20, MarketBreadthRegime.HEALTHY.value,
                np.where(
                    score > -20, MarketBreadthRegime.NARROWING.value,
                    np.where(
                        score > -60, MarketBreadthRegime.WEAK.value,
                        MarketBreadthRegime.DIVERGING.value
                    )
                )
            )
        )

        return pd.Series(regime, index=score.index)
```

---

## Signal Usage Guidelines

### Breadth Interpretation Matrix

| Signal | Bullish | Bearish |
|--------|---------|---------|
| A/D Line | New highs, uptrend | New lows, downtrend |
| McClellan Osc | >0, rising | <0, falling |
| % Above 200MA | >60% | <40% |
| New Highs-Lows | Net positive | Net negative |
| Divergence | Bullish | Bearish (WARNING) |

### Integration with Ordinis

```python
# Market breadth for portfolio decisions
breadth_engine = MarketBreadthEngine()
signals = breadth_engine.generate_all_signals(market_breadth_data)

# Check regime
regime = signals['regime'].iloc[-1]

if regime == 'strong_participation':
    increase_exposure()
elif regime == 'diverging':
    # Warning: market may be topping
    reduce_exposure()
    tighten_stops()

# Check for divergences
if signals['divergences']['sustained_bearish_div'].iloc[-1]:
    warning("Bearish breadth divergence detected")
```

---

## Data Sources

| Indicator | Source | Update Frequency |
|-----------|--------|------------------|
| Advances/Declines | NYSE, NASDAQ | Daily |
| New Highs/Lows | NYSE, NASDAQ | Daily |
| % Above MA | Custom calculation | Daily |
| McClellan | Calculated from A/D | Daily |

---

## Academic References

1. **McClellan (1969)**: "Patterns for Profit" (McClellan Oscillator)
2. **Zweig (1986)**: "Winning on Wall Street" (Breadth thrust)
3. **Fosback (1991)**: "Stock Market Logic"
4. **Dorsey (2007)**: "Point and Figure Charting" (Bullish %)
5. **Achelis (2001)**: "Technical Analysis from A to Z"
