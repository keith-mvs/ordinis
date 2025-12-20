# Fibonacci Analysis Signals

## Overview

Fibonacci analysis applies mathematical ratios derived from the Fibonacci sequence to identify potential support, resistance, and price targets. These signals provide **retracement levels**, **extension targets**, and **time projections** for systematic trading.

---

## 1. Fibonacci Retracement

### 1.1 Retracement Calculation

**Signal Logic**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class FibLevel(Enum):
    """Standard Fibonacci levels."""
    LEVEL_0 = 0.0
    LEVEL_236 = 0.236
    LEVEL_382 = 0.382
    LEVEL_500 = 0.500
    LEVEL_618 = 0.618
    LEVEL_786 = 0.786
    LEVEL_100 = 1.0


@dataclass
class FibonacciConfig:
    """Configuration for Fibonacci analysis."""

    # Retracement levels
    retracement_levels: List[float] = None

    # Extension levels
    extension_levels: List[float] = None

    # Zone tolerance
    zone_tolerance_pct: float = 0.01  # 1% around level

    # Minimum swing size
    min_swing_pct: float = 0.05  # 5% minimum move

    def __post_init__(self):
        if self.retracement_levels is None:
            self.retracement_levels = [0.236, 0.382, 0.500, 0.618, 0.786]
        if self.extension_levels is None:
            self.extension_levels = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]


class FibonacciCalculator:
    """Calculate Fibonacci levels and signals."""

    def __init__(self, config: FibonacciConfig = None):
        self.config = config or FibonacciConfig()

    def find_swing_points(
        self,
        high: pd.Series,
        low: pd.Series,
        lookback: int = 5
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Identify swing highs and swing lows.
        """
        swing_highs = pd.Series(False, index=high.index)
        swing_lows = pd.Series(False, index=low.index)

        for i in range(lookback, len(high) - lookback):
            # Swing high: highest in window
            window_highs = high.iloc[i-lookback:i+lookback+1]
            if high.iloc[i] == window_highs.max():
                swing_highs.iloc[i] = True

            # Swing low: lowest in window
            window_lows = low.iloc[i-lookback:i+lookback+1]
            if low.iloc[i] == window_lows.min():
                swing_lows.iloc[i] = True

        return swing_highs, swing_lows

    def calculate_retracement_levels(
        self,
        swing_low: float,
        swing_high: float,
        direction: str = 'up'
    ) -> Dict[float, float]:
        """
        Calculate Fibonacci retracement levels.

        Args:
            swing_low: Lower price point
            swing_high: Higher price point
            direction: 'up' for uptrend (retrace down), 'down' for downtrend (retrace up)

        Returns:
            Dictionary of {level: price}
        """
        price_range = swing_high - swing_low
        levels = {}

        if direction == 'up':
            # Uptrend: measure retracement from high
            for level in self.config.retracement_levels:
                levels[level] = swing_high - (price_range * level)
        else:
            # Downtrend: measure retracement from low
            for level in self.config.retracement_levels:
                levels[level] = swing_low + (price_range * level)

        return levels

    def calculate_extension_levels(
        self,
        swing_1_start: float,
        swing_1_end: float,
        swing_2_start: float,
        direction: str = 'up'
    ) -> Dict[float, float]:
        """
        Calculate Fibonacci extension levels.

        Uses three points: start of initial move, end of initial move,
        and start of extension move (after retracement).
        """
        initial_range = abs(swing_1_end - swing_1_start)
        levels = {}

        if direction == 'up':
            for level in self.config.extension_levels:
                levels[level] = swing_2_start + (initial_range * level)
        else:
            for level in self.config.extension_levels:
                levels[level] = swing_2_start - (initial_range * level)

        return levels


class FibonacciSignals:
    """Generate trading signals from Fibonacci analysis."""

    def __init__(self, config: FibonacciConfig = None):
        self.config = config or FibonacciConfig()
        self.calculator = FibonacciCalculator(config)

    def generate_retracement_signals(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lookback: int = 50
    ) -> pd.DataFrame:
        """
        Generate signals based on Fibonacci retracement levels.
        """
        signals = pd.DataFrame(index=close.index)

        # Find recent swing points
        swing_highs, swing_lows = self.calculator.find_swing_points(high, low)

        # Find most recent significant swing
        recent_swing_high_idx = swing_highs[swing_highs].last_valid_index()
        recent_swing_low_idx = swing_lows[swing_lows].last_valid_index()

        if recent_swing_high_idx is None or recent_swing_low_idx is None:
            return signals

        # Determine trend direction
        if recent_swing_high_idx > recent_swing_low_idx:
            # Recent high after low = uptrend, look for retracement down
            direction = 'up'
            swing_low_price = low[recent_swing_low_idx]
            swing_high_price = high[recent_swing_high_idx]
        else:
            # Recent low after high = downtrend, look for retracement up
            direction = 'down'
            swing_low_price = low[recent_swing_low_idx]
            swing_high_price = high[recent_swing_high_idx]

        # Calculate retracement levels
        retracement_levels = self.calculator.calculate_retracement_levels(
            swing_low_price, swing_high_price, direction
        )

        # Store levels
        for level, price in retracement_levels.items():
            signals[f'fib_{int(level*1000)}'] = price

        # Generate proximity signals
        tolerance = self.config.zone_tolerance_pct
        current_price = close.iloc[-1]

        for level, price in retracement_levels.items():
            level_name = f'fib_{int(level*1000)}'

            # At level
            price_diff_pct = abs(current_price - price) / price
            signals[f'at_{level_name}'] = price_diff_pct < tolerance

            # Above/below level
            signals[f'above_{level_name}'] = current_price > price * (1 + tolerance)
            signals[f'below_{level_name}'] = current_price < price * (1 - tolerance)

        # Key level signals
        fib_382 = retracement_levels.get(0.382, np.nan)
        fib_618 = retracement_levels.get(0.618, np.nan)

        signals['golden_zone'] = (
            (current_price >= fib_618 * 0.99) &
            (current_price <= fib_382 * 1.01)
        ) if pd.notna(fib_382) and pd.notna(fib_618) else False

        signals['trend_direction'] = direction
        signals['swing_low'] = swing_low_price
        signals['swing_high'] = swing_high_price

        return signals

    def level_test_signals(
        self,
        close: pd.Series,
        fib_levels: Dict[float, float]
    ) -> pd.DataFrame:
        """
        Generate signals when price tests Fibonacci levels.
        """
        signals = pd.DataFrame(index=close.index)
        tolerance = self.config.zone_tolerance_pct

        for level, price in fib_levels.items():
            level_name = f'fib_{int(level*1000)}'

            # Touch detection (price within tolerance)
            near_level = (close - price).abs() / price < tolerance
            signals[f'{level_name}_touch'] = near_level

            # Rejection (touch then move away)
            touched_recently = near_level.rolling(3).max() > 0
            moved_away = (close - price).abs() / price > tolerance * 2

            signals[f'{level_name}_rejection'] = touched_recently.shift(1) & moved_away

            # Break through (close beyond level)
            signals[f'{level_name}_break_up'] = (close > price) & (close.shift(1) <= price)
            signals[f'{level_name}_break_down'] = (close < price) & (close.shift(1) >= price)

        return signals
```

---

## 2. Fibonacci Extensions

### 2.1 Extension Targets

**Signal Logic**:
```python
class FibonacciExtensionSignals:
    """Generate extension target signals."""

    def __init__(self, config: FibonacciConfig = None):
        self.config = config or FibonacciConfig()
        self.calculator = FibonacciCalculator(config)

    def calculate_extension_targets(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate extension targets from completed retracement.
        """
        signals = pd.DataFrame(index=close.index)

        # Find three-wave pattern
        swing_highs, swing_lows = self.calculator.find_swing_points(high, low)

        # Need at least 3 swing points
        high_indices = swing_highs[swing_highs].index.tolist()
        low_indices = swing_lows[swing_lows].index.tolist()

        if len(high_indices) < 2 or len(low_indices) < 2:
            return signals

        # Determine pattern type
        # Bullish: low -> high -> higher low -> extension up
        # Bearish: high -> low -> lower high -> extension down

        # Example: Bullish extension
        recent_highs = high_indices[-2:]
        recent_lows = low_indices[-2:]

        # Simple case: last swing was low (potential bullish extension)
        if low_indices[-1] > high_indices[-1]:
            # Wave 1: prior low to prior high
            wave1_start = low[low_indices[-2]] if len(low_indices) >= 2 else low.min()
            wave1_end = high[high_indices[-1]]
            # Wave 2 end (retracement): most recent low
            wave2_end = low[low_indices[-1]]

            # Calculate extensions
            extensions = self.calculator.calculate_extension_levels(
                wave1_start, wave1_end, wave2_end, direction='up'
            )

            for level, target in extensions.items():
                signals[f'ext_{int(level*1000)}_target'] = target

            # Distance to targets
            current_price = close.iloc[-1]
            for level, target in extensions.items():
                distance_pct = (target - current_price) / current_price * 100
                signals[f'ext_{int(level*1000)}_distance_pct'] = distance_pct

        return signals

    def projection_signals(
        self,
        close: pd.Series,
        extension_targets: Dict[float, float]
    ) -> pd.DataFrame:
        """
        Generate signals as price approaches extension targets.
        """
        signals = pd.DataFrame(index=close.index)

        current_price = close.iloc[-1]

        for level, target in extension_targets.items():
            level_name = f'ext_{int(level*1000)}'

            # Distance to target
            distance = target - current_price
            distance_pct = distance / current_price * 100

            signals[f'{level_name}_target'] = target
            signals[f'{level_name}_distance'] = distance
            signals[f'{level_name}_distance_pct'] = distance_pct

            # Approaching target (within 2%)
            signals[f'{level_name}_approaching'] = (
                (distance_pct > 0) & (distance_pct < 2)
            )

            # At target (within 0.5%)
            signals[f'{level_name}_at_target'] = abs(distance_pct) < 0.5

            # Beyond target
            signals[f'{level_name}_exceeded'] = distance_pct < 0

        # Best next target
        upcoming_targets = {k: v for k, v in extension_targets.items()
                          if v > current_price}
        if upcoming_targets:
            nearest_level = min(upcoming_targets.keys())
            signals['next_target_level'] = nearest_level
            signals['next_target_price'] = upcoming_targets[nearest_level]

        return signals
```

---

## 3. Fibonacci Time Analysis

### 3.1 Time Projections

**Signal Logic**:
```python
class FibonacciTimeSignals:
    """Generate time-based Fibonacci signals."""

    def __init__(self, config: FibonacciConfig = None):
        self.config = config or FibonacciConfig()
        self.time_ratios = [0.382, 0.500, 0.618, 1.0, 1.618, 2.618]

    def calculate_time_projections(
        self,
        swing_1_bars: int,
        current_bar: int
    ) -> Dict[float, int]:
        """
        Project future turning points based on time ratios.

        Args:
            swing_1_bars: Number of bars in reference move
            current_bar: Current bar number from reference start

        Returns:
            Dictionary of {ratio: projected_bar}
        """
        projections = {}
        for ratio in self.time_ratios:
            projected_bar = int(swing_1_bars * ratio)
            projections[ratio] = projected_bar
        return projections

    def time_zone_signals(
        self,
        close: pd.Series,
        swing_start_idx: int,
        swing_end_idx: int
    ) -> pd.DataFrame:
        """
        Generate signals for Fibonacci time zones.
        """
        signals = pd.DataFrame(index=close.index)

        reference_bars = swing_end_idx - swing_start_idx

        if reference_bars <= 0:
            return signals

        # Calculate time projections from swing end
        for i, idx in enumerate(close.index):
            if idx <= close.index[swing_end_idx]:
                continue

            bars_since_swing = close.index.get_loc(idx) - swing_end_idx

            # Check if at Fibonacci time zone
            for ratio in self.time_ratios:
                target_bars = int(reference_bars * ratio)
                tolerance = max(1, int(target_bars * 0.05))  # 5% tolerance

                if abs(bars_since_swing - target_bars) <= tolerance:
                    signals.loc[idx, f'time_zone_{int(ratio*1000)}'] = True

        return signals
```

---

## 4. Fibonacci Clusters

### 4.1 Confluence Detection

**Signal Logic**:
```python
class FibonacciClusterSignals:
    """Detect Fibonacci level clusters (confluence zones)."""

    def __init__(self, config: FibonacciConfig = None):
        self.config = config or FibonacciConfig()
        self.calculator = FibonacciCalculator(config)

    def find_cluster_zones(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lookback_periods: List[int] = None
    ) -> pd.DataFrame:
        """
        Find zones where multiple Fibonacci levels cluster.
        """
        if lookback_periods is None:
            lookback_periods = [20, 50, 100, 200]

        signals = pd.DataFrame(index=close.index)

        # Collect all Fibonacci levels from different swings
        all_levels = []

        for period in lookback_periods:
            if period > len(close):
                continue

            # Get swing high/low for this period
            period_high = high.iloc[-period:].max()
            period_low = low.iloc[-period:].min()
            period_high_idx = high.iloc[-period:].idxmax()
            period_low_idx = low.iloc[-period:].idxmin()

            # Determine direction
            if period_high_idx > period_low_idx:
                direction = 'up'
            else:
                direction = 'down'

            # Calculate retracement levels
            retracement = self.calculator.calculate_retracement_levels(
                period_low, period_high, direction
            )

            for level, price in retracement.items():
                all_levels.append({
                    'period': period,
                    'level': level,
                    'price': price,
                    'direction': direction
                })

        # Find clusters
        if not all_levels:
            return signals

        prices = [l['price'] for l in all_levels]
        price_range = max(prices) - min(prices) if prices else 0

        # Group nearby levels
        cluster_tolerance = price_range * 0.02  # 2% of total range

        clusters = []
        used = set()

        for i, level in enumerate(all_levels):
            if i in used:
                continue

            cluster = [level]
            used.add(i)

            for j, other in enumerate(all_levels):
                if j in used:
                    continue

                if abs(level['price'] - other['price']) < cluster_tolerance:
                    cluster.append(other)
                    used.add(j)

            if len(cluster) >= 2:  # At least 2 levels = cluster
                cluster_center = np.mean([l['price'] for l in cluster])
                clusters.append({
                    'price': cluster_center,
                    'count': len(cluster),
                    'levels': cluster
                })

        # Store cluster zones
        for i, cluster in enumerate(sorted(clusters, key=lambda x: -x['count'])):
            signals[f'cluster_{i}_price'] = cluster['price']
            signals[f'cluster_{i}_strength'] = cluster['count']

        # Current price relative to clusters
        current_price = close.iloc[-1]
        for i, cluster in enumerate(clusters):
            distance_pct = (current_price - cluster['price']) / cluster['price'] * 100
            signals[f'cluster_{i}_distance_pct'] = distance_pct

            # At cluster zone
            signals[f'at_cluster_{i}'] = abs(distance_pct) < self.config.zone_tolerance_pct * 100

        # Strongest cluster
        if clusters:
            strongest = max(clusters, key=lambda x: x['count'])
            signals['strongest_cluster_price'] = strongest['price']
            signals['strongest_cluster_count'] = strongest['count']

        return signals
```

---

## 5. Composite Fibonacci Engine

### 5.1 Integrated Signal Generation

```python
class FibonacciEngine:
    """
    Production Fibonacci signal engine.
    """

    def __init__(self, config: FibonacciConfig = None):
        self.config = config or FibonacciConfig()
        self.retracement = FibonacciSignals(config)
        self.extension = FibonacciExtensionSignals(config)
        self.time = FibonacciTimeSignals(config)
        self.cluster = FibonacciClusterSignals(config)

    def generate_all_signals(
        self,
        ohlcv: pd.DataFrame
    ) -> Dict:
        """
        Generate comprehensive Fibonacci signals.
        """
        results = {}

        # Retracement signals
        retr_signals = self.retracement.generate_retracement_signals(
            ohlcv['high'], ohlcv['low'], ohlcv['close']
        )
        results['retracement'] = retr_signals

        # Extension signals
        ext_signals = self.extension.calculate_extension_targets(
            ohlcv['high'], ohlcv['low'], ohlcv['close']
        )
        results['extension'] = ext_signals

        # Cluster analysis
        cluster_signals = self.cluster.find_cluster_zones(
            ohlcv['high'], ohlcv['low'], ohlcv['close']
        )
        results['clusters'] = cluster_signals

        # Key levels summary
        results['key_levels'] = self._extract_key_levels(
            retr_signals, ext_signals, cluster_signals
        )

        # Trading signals
        results['trading_signals'] = self._generate_trading_signals(results)

        return results

    def _extract_key_levels(
        self,
        retracement: pd.DataFrame,
        extension: pd.DataFrame,
        clusters: pd.DataFrame
    ) -> Dict:
        """
        Extract key price levels from all Fibonacci analysis.
        """
        levels = {}

        # Retracement levels
        for col in retracement.columns:
            if col.startswith('fib_') and not col.startswith('fib_') + '_':
                if col in retracement.columns:
                    val = retracement[col].iloc[-1] if len(retracement) > 0 else None
                    if pd.notna(val):
                        levels[col] = val

        # Extension targets
        for col in extension.columns:
            if '_target' in col:
                val = extension[col].iloc[-1] if len(extension) > 0 else None
                if pd.notna(val):
                    levels[col] = val

        # Cluster zones
        for col in clusters.columns:
            if '_price' in col and 'cluster' in col:
                val = clusters[col].iloc[-1] if len(clusters) > 0 else None
                if pd.notna(val):
                    levels[col] = val

        return levels

    def _generate_trading_signals(
        self,
        results: Dict
    ) -> pd.DataFrame:
        """
        Generate actionable trading signals.
        """
        signals = pd.DataFrame()

        retr = results.get('retracement', pd.DataFrame())
        clusters = results.get('clusters', pd.DataFrame())

        if retr.empty:
            return signals

        # Golden zone entry signal
        if 'golden_zone' in retr.columns:
            signals['fib_golden_zone_entry'] = retr['golden_zone']

        # At cluster zone signal
        cluster_cols = [c for c in clusters.columns if c.startswith('at_cluster_')]
        if cluster_cols:
            signals['at_fib_cluster'] = clusters[cluster_cols].any(axis=1)

        # Direction bias
        if 'trend_direction' in retr.columns:
            signals['fib_trend_direction'] = retr['trend_direction']

        return signals

    def get_nearest_levels(
        self,
        current_price: float,
        key_levels: Dict
    ) -> Dict:
        """
        Get nearest support and resistance from Fibonacci levels.
        """
        supports = {k: v for k, v in key_levels.items() if v < current_price}
        resistances = {k: v for k, v in key_levels.items() if v > current_price}

        nearest_support = max(supports.values()) if supports else None
        nearest_resistance = min(resistances.values()) if resistances else None

        return {
            'nearest_support': nearest_support,
            'nearest_resistance': nearest_resistance,
            'support_distance_pct': (current_price - nearest_support) / current_price * 100 if nearest_support else None,
            'resistance_distance_pct': (nearest_resistance - current_price) / current_price * 100 if nearest_resistance else None
        }
```

---

## Signal Usage Guidelines

### Key Fibonacci Levels

| Level | Significance | Use |
|-------|--------------|-----|
| 0.236 | Minor retracement | Shallow pullback |
| 0.382 | Golden zone start | Strong support |
| 0.500 | Half retracement | Psychological level |
| 0.618 | Golden ratio | Primary support/resistance |
| 0.786 | Deep retracement | Last chance support |
| 1.272 | Extension | First target |
| 1.618 | Extension | Primary target |

### Integration with Ordinis

```python
# Fibonacci analysis in trade planning
fib_engine = FibonacciEngine()
signals = fib_engine.generate_all_signals(ohlcv_data)

# Get key levels for stops and targets
key_levels = signals['key_levels']
nearest = fib_engine.get_nearest_levels(current_price, key_levels)

# Entry in golden zone
if signals['trading_signals'].get('fib_golden_zone_entry', False).iloc[-1]:
    stop = nearest['nearest_support'] * 0.99
    target = nearest['nearest_resistance']
    enter_trade(stop=stop, target=target)
```

---

## Academic References

1. **Fischer & Fischer (2003)**: "The New Fibonacci Trader"
2. **Miner (2008)**: "High Probability Trading Strategies"
3. **Brown (2008)**: "Fibonacci Analysis"
4. **Boroden (2008)**: "Fibonacci Trading"
5. **Carney (2010)**: "Harmonic Trading Volume One"
