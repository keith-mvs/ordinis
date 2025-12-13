# Volume Profile Signals

## Overview

Volume Profile analyzes the distribution of volume across price levels, revealing **value areas**, **support/resistance zones**, and **institutional activity levels**. Unlike time-based volume, Volume Profile shows where trading interest concentrates.

---

## 1. Volume Profile Fundamentals

### 1.1 Key Concepts and Calculations

**Signal Logic**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from scipy import stats


@dataclass
class VolumeProfileConfig:
    """Configuration for volume profile analysis."""

    # Profile construction
    price_buckets: int = 50           # Number of price levels
    value_area_pct: float = 0.70      # 70% volume for value area

    # Session types
    session_daily: bool = True
    session_weekly: bool = True
    session_monthly: bool = True

    # Signal thresholds
    hvn_threshold: float = 1.5        # High volume node multiplier
    lvn_threshold: float = 0.5        # Low volume node multiplier


class VolumeProfileCalculator:
    """Calculate volume profile metrics."""

    def __init__(self, config: VolumeProfileConfig = None):
        self.config = config or VolumeProfileConfig()

    def calculate_volume_profile(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> Dict:
        """
        Calculate volume profile for given data.

        Returns:
            Dictionary with profile data:
            - profile: volume at each price level
            - poc: Point of Control price
            - vah: Value Area High
            - val: Value Area Low
        """
        # Define price range
        price_low = low.min()
        price_high = high.max()
        price_range = price_high - price_low

        # Create price buckets
        bucket_size = price_range / self.config.price_buckets
        price_levels = np.linspace(price_low, price_high, self.config.price_buckets + 1)

        # Distribute volume across price levels
        volume_profile = np.zeros(self.config.price_buckets)

        for i in range(len(close)):
            bar_high = high.iloc[i]
            bar_low = low.iloc[i]
            bar_volume = volume.iloc[i]
            bar_close = close.iloc[i]

            # Distribute volume based on where price traded
            # Simplified: weight towards close
            for j in range(self.config.price_buckets):
                level_low = price_levels[j]
                level_high = price_levels[j + 1]

                # Check if bar overlaps this level
                if bar_low <= level_high and bar_high >= level_low:
                    # Calculate overlap
                    overlap_low = max(bar_low, level_low)
                    overlap_high = min(bar_high, level_high)
                    overlap_pct = (overlap_high - overlap_low) / (bar_high - bar_low + 0.0001)

                    # Weight more volume near close
                    mid_level = (level_low + level_high) / 2
                    close_weight = 1.0 + 0.5 * (1 - abs(bar_close - mid_level) / bucket_size)

                    volume_profile[j] += bar_volume * overlap_pct * close_weight

        # Normalize
        total_volume = volume_profile.sum()
        if total_volume > 0:
            volume_profile_pct = volume_profile / total_volume

        # Point of Control (POC)
        poc_idx = np.argmax(volume_profile)
        poc_price = (price_levels[poc_idx] + price_levels[poc_idx + 1]) / 2

        # Value Area
        vah, val = self._calculate_value_area(
            volume_profile, price_levels, poc_idx
        )

        # Price at each level
        level_prices = (price_levels[:-1] + price_levels[1:]) / 2

        return {
            'profile': pd.Series(volume_profile, index=level_prices),
            'profile_pct': pd.Series(volume_profile_pct, index=level_prices),
            'poc': poc_price,
            'vah': vah,
            'val': val,
            'total_volume': total_volume
        }

    def _calculate_value_area(
        self,
        profile: np.ndarray,
        price_levels: np.ndarray,
        poc_idx: int
    ) -> Tuple[float, float]:
        """
        Calculate Value Area High and Low.

        Value Area contains 70% of volume, centered on POC.
        """
        total_volume = profile.sum()
        target_volume = total_volume * self.config.value_area_pct

        # Start from POC and expand outward
        accumulated = profile[poc_idx]
        upper_idx = poc_idx
        lower_idx = poc_idx

        while accumulated < target_volume:
            # Check which direction adds more volume
            upper_vol = profile[upper_idx + 1] if upper_idx < len(profile) - 1 else 0
            lower_vol = profile[lower_idx - 1] if lower_idx > 0 else 0

            if upper_vol >= lower_vol and upper_idx < len(profile) - 1:
                upper_idx += 1
                accumulated += upper_vol
            elif lower_idx > 0:
                lower_idx -= 1
                accumulated += lower_vol
            else:
                break

        vah = (price_levels[upper_idx] + price_levels[upper_idx + 1]) / 2
        val = (price_levels[lower_idx] + price_levels[lower_idx + 1]) / 2

        return vah, val


class VolumeNodeAnalysis:
    """Analyze high and low volume nodes."""

    def __init__(self, config: VolumeProfileConfig = None):
        self.config = config or VolumeProfileConfig()

    def identify_volume_nodes(
        self,
        profile: pd.Series
    ) -> Dict:
        """
        Identify High Volume Nodes (HVN) and Low Volume Nodes (LVN).

        HVN = Support/resistance areas
        LVN = Potential gap/fast move areas
        """
        avg_volume = profile.mean()
        std_volume = profile.std()

        # High Volume Nodes
        hvn_threshold = avg_volume * self.config.hvn_threshold
        hvn_mask = profile > hvn_threshold
        hvn_levels = profile[hvn_mask]

        # Low Volume Nodes
        lvn_threshold = avg_volume * self.config.lvn_threshold
        lvn_mask = profile < lvn_threshold
        lvn_levels = profile[lvn_mask]

        # Find node clusters
        hvn_clusters = self._cluster_nodes(hvn_levels.index.tolist())
        lvn_clusters = self._cluster_nodes(lvn_levels.index.tolist())

        return {
            'hvn_levels': hvn_levels,
            'lvn_levels': lvn_levels,
            'hvn_clusters': hvn_clusters,
            'lvn_clusters': lvn_clusters
        }

    def _cluster_nodes(
        self,
        levels: List[float],
        min_gap_pct: float = 0.01
    ) -> List[Tuple[float, float]]:
        """
        Cluster nearby price levels into zones.
        """
        if not levels:
            return []

        levels = sorted(levels)
        clusters = []
        cluster_start = levels[0]
        cluster_end = levels[0]

        for level in levels[1:]:
            gap = (level - cluster_end) / cluster_end

            if gap < min_gap_pct:
                cluster_end = level
            else:
                clusters.append((cluster_start, cluster_end))
                cluster_start = level
                cluster_end = level

        clusters.append((cluster_start, cluster_end))
        return clusters
```

---

## 2. Volume Profile Signals

### 2.1 POC and Value Area Signals

**Signal Logic**:
```python
class VolumeProfileSignals:
    """Generate trading signals from volume profile."""

    def __init__(self, config: VolumeProfileConfig = None):
        self.config = config or VolumeProfileConfig()
        self.calculator = VolumeProfileCalculator(config)
        self.node_analyzer = VolumeNodeAnalysis(config)

    def generate_profile_signals(
        self,
        ohlcv: pd.DataFrame,
        lookback: int = 20
    ) -> pd.DataFrame:
        """
        Generate signals from volume profile analysis.
        """
        signals = pd.DataFrame(index=ohlcv.index)

        # Calculate rolling profile
        for i in range(lookback, len(ohlcv)):
            window = ohlcv.iloc[i-lookback:i]

            profile = self.calculator.calculate_volume_profile(
                window['high'],
                window['low'],
                window['close'],
                window['volume']
            )

            current_price = ohlcv['close'].iloc[i]

            # POC signals
            signals.loc[ohlcv.index[i], 'poc'] = profile['poc']
            signals.loc[ohlcv.index[i], 'vah'] = profile['vah']
            signals.loc[ohlcv.index[i], 'val'] = profile['val']

            # Position relative to value area
            signals.loc[ohlcv.index[i], 'above_value_area'] = current_price > profile['vah']
            signals.loc[ohlcv.index[i], 'below_value_area'] = current_price < profile['val']
            signals.loc[ohlcv.index[i], 'in_value_area'] = (
                current_price >= profile['val'] and
                current_price <= profile['vah']
            )

            # Distance from POC
            poc_distance = (current_price - profile['poc']) / profile['poc']
            signals.loc[ohlcv.index[i], 'poc_distance'] = poc_distance
            signals.loc[ohlcv.index[i], 'near_poc'] = abs(poc_distance) < 0.01

            # Value area width (volatility proxy)
            va_width = (profile['vah'] - profile['val']) / profile['poc']
            signals.loc[ohlcv.index[i], 'value_area_width'] = va_width

        return signals

    def value_area_trading_signals(
        self,
        close: pd.Series,
        poc: pd.Series,
        vah: pd.Series,
        val: pd.Series
    ) -> pd.DataFrame:
        """
        Generate specific trading signals from value area.
        """
        signals = pd.DataFrame(index=close.index)

        # Break above value area
        signals['break_above_va'] = (close > vah) & (close.shift(1) <= vah.shift(1))

        # Break below value area
        signals['break_below_va'] = (close < val) & (close.shift(1) >= val.shift(1))

        # Return to value area (mean reversion)
        signals['return_to_va_from_above'] = (
            (close <= vah) &
            (close.shift(1) > vah.shift(1)) &
            (close > val)
        )
        signals['return_to_va_from_below'] = (
            (close >= val) &
            (close.shift(1) < val.shift(1)) &
            (close < vah)
        )

        # POC rejection
        poc_distance = (close - poc).abs() / poc
        close_to_poc = poc_distance < 0.005  # Within 0.5%

        signals['poc_support'] = (
            close_to_poc &
            (close > close.shift(1)) &
            (close.shift(1) < poc.shift(1))
        )
        signals['poc_resistance'] = (
            close_to_poc &
            (close < close.shift(1)) &
            (close.shift(1) > poc.shift(1))
        )

        # Naked POC (POC not tested from prior session)
        signals['naked_poc_above'] = (close < poc) & (close.rolling(5).max() < poc)
        signals['naked_poc_below'] = (close > poc) & (close.rolling(5).min() > poc)

        return signals

    def volume_node_signals(
        self,
        close: pd.Series,
        profile: pd.Series
    ) -> pd.DataFrame:
        """
        Generate signals from high/low volume nodes.
        """
        signals = pd.DataFrame(index=close.index)

        nodes = self.node_analyzer.identify_volume_nodes(profile)

        # Current price relative to nodes
        hvn_levels = nodes['hvn_levels'].index.tolist() if len(nodes['hvn_levels']) > 0 else []
        lvn_levels = nodes['lvn_levels'].index.tolist() if len(nodes['lvn_levels']) > 0 else []

        latest_price = close.iloc[-1]

        # Find nearest HVN
        if hvn_levels:
            hvn_above = [l for l in hvn_levels if l > latest_price]
            hvn_below = [l for l in hvn_levels if l < latest_price]

            signals['nearest_hvn_above'] = min(hvn_above) if hvn_above else np.nan
            signals['nearest_hvn_below'] = max(hvn_below) if hvn_below else np.nan

            # Near HVN signals
            for hvn in hvn_levels:
                if abs(latest_price - hvn) / hvn < 0.01:
                    signals['at_hvn'] = True
                    break

        # Find nearest LVN
        if lvn_levels:
            lvn_above = [l for l in lvn_levels if l > latest_price]
            lvn_below = [l for l in lvn_levels if l < latest_price]

            signals['nearest_lvn_above'] = min(lvn_above) if lvn_above else np.nan
            signals['nearest_lvn_below'] = max(lvn_below) if lvn_below else np.nan

            # Approaching LVN (potential fast move zone)
            for lvn in lvn_levels:
                if abs(latest_price - lvn) / lvn < 0.02:
                    signals['approaching_lvn'] = True
                    break

        return signals
```

---

## 3. Session-Based Profile

### 3.1 Multi-Session Analysis

**Signal Logic**:
```python
class SessionProfileAnalysis:
    """Analyze volume profiles across different sessions."""

    def __init__(self, config: VolumeProfileConfig = None):
        self.config = config or VolumeProfileConfig()
        self.calculator = VolumeProfileCalculator(config)

    def calculate_session_profiles(
        self,
        ohlcv: pd.DataFrame
    ) -> Dict:
        """
        Calculate profiles for different session types.
        """
        profiles = {}

        # Daily profile (current day)
        today = ohlcv.index[-1].date()
        today_data = ohlcv[ohlcv.index.date == today]
        if len(today_data) > 0:
            profiles['daily'] = self.calculator.calculate_volume_profile(
                today_data['high'],
                today_data['low'],
                today_data['close'],
                today_data['volume']
            )

        # Weekly profile
        week_start = ohlcv.index[-1] - pd.Timedelta(days=7)
        week_data = ohlcv[ohlcv.index >= week_start]
        if len(week_data) > 0:
            profiles['weekly'] = self.calculator.calculate_volume_profile(
                week_data['high'],
                week_data['low'],
                week_data['close'],
                week_data['volume']
            )

        # Monthly profile
        month_start = ohlcv.index[-1] - pd.Timedelta(days=30)
        month_data = ohlcv[ohlcv.index >= month_start]
        if len(month_data) > 0:
            profiles['monthly'] = self.calculator.calculate_volume_profile(
                month_data['high'],
                month_data['low'],
                month_data['close'],
                month_data['volume']
            )

        return profiles

    def composite_profile_signals(
        self,
        current_price: float,
        profiles: Dict
    ) -> pd.DataFrame:
        """
        Generate signals from multiple session profiles.
        """
        signals = {}

        for session, profile in profiles.items():
            prefix = f"{session}_"

            # Position relative to each profile
            signals[f"{prefix}above_va"] = current_price > profile['vah']
            signals[f"{prefix}below_va"] = current_price < profile['val']
            signals[f"{prefix}in_va"] = (
                current_price >= profile['val'] and
                current_price <= profile['vah']
            )
            signals[f"{prefix}poc"] = profile['poc']
            signals[f"{prefix}vah"] = profile['vah']
            signals[f"{prefix}val"] = profile['val']

        # Multi-timeframe confluence
        if len(profiles) >= 2:
            poc_values = [p['poc'] for p in profiles.values()]
            poc_range = max(poc_values) - min(poc_values)
            avg_poc = sum(poc_values) / len(poc_values)

            # POCs aligned = strong level
            signals['pocs_aligned'] = (poc_range / avg_poc) < 0.02

            # Value areas overlapping
            vah_values = [p['vah'] for p in profiles.values()]
            val_values = [p['val'] for p in profiles.values()]

            # Overlap region
            overlap_high = min(vah_values)
            overlap_low = max(val_values)

            signals['value_area_overlap'] = overlap_high > overlap_low
            if signals['value_area_overlap']:
                signals['confluence_zone_high'] = overlap_high
                signals['confluence_zone_low'] = overlap_low

        return pd.DataFrame([signals])
```

---

## 4. Composite Volume Profile Engine

### 4.1 Production Signal Engine

```python
class VolumeProfileEngine:
    """
    Production volume profile signal engine.
    """

    def __init__(self, config: VolumeProfileConfig = None):
        self.config = config or VolumeProfileConfig()
        self.profile_signals = VolumeProfileSignals(config)
        self.session_analysis = SessionProfileAnalysis(config)

    def generate_all_signals(
        self,
        ohlcv: pd.DataFrame,
        lookback: int = 20
    ) -> Dict:
        """
        Generate comprehensive volume profile signals.
        """
        results = {}

        # Calculate current profile
        profile_data = self.profile_signals.calculator.calculate_volume_profile(
            ohlcv['high'].iloc[-lookback:],
            ohlcv['low'].iloc[-lookback:],
            ohlcv['close'].iloc[-lookback:],
            ohlcv['volume'].iloc[-lookback:]
        )
        results['current_profile'] = profile_data

        # Generate profile signals
        profile_signals = self.profile_signals.generate_profile_signals(
            ohlcv, lookback
        )
        results['profile_signals'] = profile_signals

        # Value area trading signals
        if all(c in profile_signals.columns for c in ['poc', 'vah', 'val']):
            va_signals = self.profile_signals.value_area_trading_signals(
                ohlcv['close'],
                profile_signals['poc'],
                profile_signals['vah'],
                profile_signals['val']
            )
            results['value_area_signals'] = va_signals

        # Volume node analysis
        node_signals = self.profile_signals.volume_node_signals(
            ohlcv['close'],
            profile_data['profile']
        )
        results['node_signals'] = node_signals

        # Session profiles
        session_profiles = self.session_analysis.calculate_session_profiles(ohlcv)
        results['session_profiles'] = session_profiles

        # Key levels summary
        results['key_levels'] = self._extract_key_levels(profile_data, node_signals)

        return results

    def _extract_key_levels(
        self,
        profile: Dict,
        nodes: pd.DataFrame
    ) -> Dict:
        """
        Extract key price levels from profile analysis.
        """
        levels = {
            'poc': profile['poc'],
            'vah': profile['vah'],
            'val': profile['val']
        }

        # Add nearest HVN/LVN
        if 'nearest_hvn_above' in nodes.columns:
            hvn_above = nodes['nearest_hvn_above'].iloc[-1] if len(nodes) > 0 else None
            if pd.notna(hvn_above):
                levels['hvn_above'] = hvn_above

        if 'nearest_hvn_below' in nodes.columns:
            hvn_below = nodes['nearest_hvn_below'].iloc[-1] if len(nodes) > 0 else None
            if pd.notna(hvn_below):
                levels['hvn_below'] = hvn_below

        return levels

    def get_profile_bias(
        self,
        current_price: float,
        profile: Dict
    ) -> str:
        """
        Determine bias based on price relative to profile.
        """
        poc = profile['poc']
        vah = profile['vah']
        val = profile['val']

        if current_price > vah:
            return "bullish_breakout"
        elif current_price > poc:
            return "bullish_in_value"
        elif current_price > val:
            return "neutral_in_value"
        elif current_price > val * 0.99:
            return "bearish_in_value"
        else:
            return "bearish_breakdown"
```

---

## Signal Usage Guidelines

### Volume Profile Trading Rules

| Position | Signal | Action |
|----------|--------|--------|
| Above VAH | Break above VA | Trend continuation long |
| At VAH | Rejection | Mean reversion short |
| In VA | Near POC | Wait for breakout |
| At VAL | Rejection | Mean reversion long |
| Below VAL | Break below VA | Trend continuation short |

### Key Level Interpretations

- **POC**: Highest volume price - strong S/R, fair value
- **VAH/VAL**: Value area boundaries - breakout/rejection zones
- **HVN**: High volume nodes - S/R clusters
- **LVN**: Low volume nodes - fast move potential

### Integration with Ordinis

```python
# Volume profile in trade planning
vp_engine = VolumeProfileEngine()
signals = vp_engine.generate_all_signals(ohlcv_data)

# Get key levels for stop/target
key_levels = signals['key_levels']

# Set stops at HVN levels
stop_price = key_levels.get('hvn_below', key_levels['val'])

# Target at POC or opposite VA boundary
target_price = key_levels['vah'] if long else key_levels['val']
```

---

## Academic References

1. **Steidlmayer & Jones (1988)**: "Markets and Market Logic" (Market Profile)
2. **Dalton (1990)**: "Mind Over Markets" (Value Area trading)
3. **Jones (1991)**: "Value-Based Power Trading"
4. **Dalton (2007)**: "Markets in Profile"
5. **Wyckoff (1931)**: "The Richard D. Wyckoff Method" (Volume analysis)
