# Sector Rotation Signals

## Overview

Sector rotation signals identify optimal sector allocations based on economic cycles, relative valuations, and momentum. These signals drive **tactical allocation**, **factor tilts**, and **defensive positioning**.

---

## 1. Economic Cycle Rotation

### 1.1 Business Cycle Phase Detection

**Signal Logic**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from scipy import stats


class EconomicPhase(Enum):
    EARLY_EXPANSION = "early_expansion"
    MID_EXPANSION = "mid_expansion"
    LATE_EXPANSION = "late_expansion"
    EARLY_CONTRACTION = "early_contraction"
    LATE_CONTRACTION = "late_contraction"
    RECOVERY = "recovery"


@dataclass
class SectorCycleConfig:
    """Configuration for cycle-based sector rotation."""

    # Sector preferences by cycle phase
    sector_preferences: Dict[EconomicPhase, List[str]] = None

    def __post_init__(self):
        if self.sector_preferences is None:
            self.sector_preferences = {
                EconomicPhase.EARLY_EXPANSION: [
                    "Financials", "Industrials", "Materials", "Consumer Discretionary"
                ],
                EconomicPhase.MID_EXPANSION: [
                    "Information Technology", "Communication Services", "Industrials"
                ],
                EconomicPhase.LATE_EXPANSION: [
                    "Energy", "Materials", "Consumer Staples", "Health Care"
                ],
                EconomicPhase.EARLY_CONTRACTION: [
                    "Consumer Staples", "Health Care", "Utilities"
                ],
                EconomicPhase.LATE_CONTRACTION: [
                    "Utilities", "Consumer Staples", "Health Care"
                ],
                EconomicPhase.RECOVERY: [
                    "Consumer Discretionary", "Technology", "Financials"
                ]
            }


class EconomicCycleDetector:
    """Detect economic cycle phase from macro indicators."""

    def __init__(self, config: SectorCycleConfig = None):
        self.config = config or SectorCycleConfig()

    def detect_cycle_phase(
        self,
        gdp_growth: pd.Series,
        pmi: pd.Series,
        unemployment: pd.Series,
        yield_curve: pd.Series,
        credit_spreads: pd.Series
    ) -> pd.DataFrame:
        """
        Classify economic cycle phase from macro indicators.

        Returns:
            DataFrame with phase classification and confidence
        """
        signals = pd.DataFrame(index=gdp_growth.index)

        # GDP growth regime
        gdp_expanding = gdp_growth > 0
        gdp_accelerating = gdp_growth > gdp_growth.shift(1)

        # PMI regime
        pmi_expansion = pmi > 50
        pmi_accelerating = pmi > pmi.shift(1)

        # Yield curve regime
        curve_positive = yield_curve > 0
        curve_steepening = yield_curve > yield_curve.shift(1)

        # Credit spread regime
        spreads_tight = credit_spreads < credit_spreads.rolling(252).median()
        spreads_tightening = credit_spreads < credit_spreads.shift(21)

        # Unemployment regime
        unemployment_falling = unemployment < unemployment.shift(1)
        unemployment_low = unemployment < unemployment.rolling(252).median()

        # Phase classification logic
        signals['phase'] = EconomicPhase.MID_EXPANSION.value  # Default

        # Early expansion: GDP accelerating, PMI rising, unemployment falling
        early_exp = (
            gdp_expanding & gdp_accelerating &
            pmi_accelerating & unemployment_falling &
            ~unemployment_low
        )
        signals.loc[early_exp, 'phase'] = EconomicPhase.EARLY_EXPANSION.value

        # Mid expansion: steady growth, low unemployment
        mid_exp = (
            gdp_expanding & pmi_expansion &
            unemployment_low & spreads_tight
        )
        signals.loc[mid_exp, 'phase'] = EconomicPhase.MID_EXPANSION.value

        # Late expansion: growth slowing, tight conditions
        late_exp = (
            gdp_expanding & ~gdp_accelerating &
            pmi_expansion & ~pmi_accelerating &
            curve_positive & ~curve_steepening
        )
        signals.loc[late_exp, 'phase'] = EconomicPhase.LATE_EXPANSION.value

        # Early contraction: PMI falling, curve inverting
        early_cont = (
            pmi_expansion & ~pmi_accelerating &
            ~curve_positive
        )
        signals.loc[early_cont, 'phase'] = EconomicPhase.EARLY_CONTRACTION.value

        # Late contraction: GDP negative, PMI contracting
        late_cont = (
            ~gdp_expanding & ~pmi_expansion
        )
        signals.loc[late_cont, 'phase'] = EconomicPhase.LATE_CONTRACTION.value

        # Recovery: GDP turning, PMI improving
        recovery = (
            gdp_accelerating & pmi_accelerating &
            ~pmi_expansion & curve_steepening
        )
        signals.loc[recovery, 'phase'] = EconomicPhase.RECOVERY.value

        # Phase persistence
        signals['phase_duration'] = signals.groupby(
            (signals['phase'] != signals['phase'].shift()).cumsum()
        ).cumcount() + 1

        return signals

    def get_sector_weights(
        self,
        phase: EconomicPhase
    ) -> Dict[str, float]:
        """
        Get recommended sector weights for cycle phase.
        """
        preferred = self.config.sector_preferences.get(phase, [])
        all_sectors = [
            "Information Technology", "Health Care", "Financials",
            "Consumer Discretionary", "Communication Services",
            "Industrials", "Consumer Staples", "Energy",
            "Utilities", "Real Estate", "Materials"
        ]

        weights = {}
        preferred_weight = 0.12  # Overweight
        neutral_weight = 0.09   # Market weight
        underweight = 0.06     # Underweight

        for sector in all_sectors:
            if sector in preferred:
                weights[sector] = preferred_weight
            else:
                weights[sector] = neutral_weight

        # Normalize to sum to 1
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
```

---

## 2. Relative Strength Rotation

### 2.1 Sector Momentum Signals

**Signal Logic**:
```python
class SectorMomentumSignals:
    """Generate sector rotation signals from relative strength."""

    def __init__(
        self,
        lookback_short: int = 21,
        lookback_medium: int = 63,
        lookback_long: int = 252
    ):
        self.lookback_short = lookback_short
        self.lookback_medium = lookback_medium
        self.lookback_long = lookback_long

    def calculate_sector_momentum(
        self,
        sector_prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate multi-horizon sector momentum.

        Args:
            sector_prices: DataFrame with sector ETF/index prices

        Returns:
            DataFrame with momentum signals per sector
        """
        signals = {}

        for sector in sector_prices.columns:
            prices = sector_prices[sector]

            sector_signals = pd.DataFrame(index=prices.index)

            # Multi-horizon returns
            sector_signals['return_1m'] = prices.pct_change(self.lookback_short)
            sector_signals['return_3m'] = prices.pct_change(self.lookback_medium)
            sector_signals['return_12m'] = prices.pct_change(self.lookback_long)

            # 12-1 momentum (skip most recent month)
            sector_signals['mom_12_1'] = (
                prices.shift(self.lookback_short) /
                prices.shift(self.lookback_long) - 1
            )

            # Moving average trend
            ma_50 = prices.rolling(50).mean()
            ma_200 = prices.rolling(200).mean()
            sector_signals['above_50ma'] = prices > ma_50
            sector_signals['above_200ma'] = prices > ma_200
            sector_signals['golden_cross'] = ma_50 > ma_200

            signals[sector] = sector_signals

        return pd.concat(signals, axis=1)

    def rank_sectors(
        self,
        sector_returns: pd.DataFrame,
        horizon: str = 'return_3m'
    ) -> pd.DataFrame:
        """
        Rank sectors by momentum.
        """
        # Extract specified horizon returns
        if isinstance(sector_returns.columns, pd.MultiIndex):
            returns = sector_returns.xs(horizon, level=1, axis=1)
        else:
            returns = sector_returns

        rankings = pd.DataFrame(index=returns.index)

        # Cross-sectional rank (1 = highest momentum)
        rankings['rank'] = returns.rank(axis=1, ascending=False)

        # Relative strength score (percentile)
        rankings['rs_score'] = returns.rank(axis=1, pct=True) * 100

        # Top/bottom classification
        n_sectors = returns.shape[1]
        top_n = max(2, n_sectors // 4)
        bottom_n = max(2, n_sectors // 4)

        rankings['top_quartile'] = rankings['rank'] <= top_n
        rankings['bottom_quartile'] = rankings['rank'] > (n_sectors - bottom_n)

        return rankings

    def rotation_signals(
        self,
        sector_momentum: pd.DataFrame,
        current_weights: pd.Series
    ) -> pd.DataFrame:
        """
        Generate rotation signals (overweight/underweight).
        """
        signals = pd.DataFrame(index=sector_momentum.index)

        # Extract 3-month momentum for ranking
        mom_3m = sector_momentum.xs('return_3m', level=1, axis=1).iloc[-1]
        mom_12_1 = sector_momentum.xs('mom_12_1', level=1, axis=1).iloc[-1]

        # Combined momentum score
        combined_mom = (mom_3m.rank(pct=True) + mom_12_1.rank(pct=True)) / 2

        # Target weights based on momentum
        momentum_weights = combined_mom / combined_mom.sum()

        # Rotation signal
        weight_diff = momentum_weights - current_weights

        signals['target_weight'] = momentum_weights
        signals['current_weight'] = current_weights
        signals['weight_change'] = weight_diff

        signals['overweight'] = weight_diff > 0.02  # Add >2%
        signals['underweight'] = weight_diff < -0.02  # Reduce >2%
        signals['hold'] = weight_diff.abs() <= 0.02

        return signals
```

---

## 3. Valuation-Based Rotation

### 3.1 Sector Valuation Signals

**Signal Logic**:
```python
class SectorValuationSignals:
    """Generate rotation signals from sector valuations."""

    def __init__(
        self,
        discount_threshold: float = 0.15,
        premium_threshold: float = 0.15
    ):
        self.discount_threshold = discount_threshold
        self.premium_threshold = premium_threshold

    def relative_valuation_signals(
        self,
        sector_pe: pd.DataFrame,
        sector_pe_history: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate signals from sector relative valuations.

        Args:
            sector_pe: Current sector P/E ratios
            sector_pe_history: Historical P/E for percentile calc
        """
        signals = pd.DataFrame(index=sector_pe.index)

        # Market median P/E
        market_median = sector_pe.median(axis=1)

        for sector in sector_pe.columns:
            pe = sector_pe[sector]

            # Relative to market
            relative_pe = pe / market_median
            signals[f'{sector}_relative_pe'] = relative_pe
            signals[f'{sector}_discount'] = relative_pe < (1 - self.discount_threshold)
            signals[f'{sector}_premium'] = relative_pe > (1 + self.premium_threshold)

            # Historical percentile
            if sector_pe_history is not None and sector in sector_pe_history.columns:
                hist_pe = sector_pe_history[sector]
                pctl = hist_pe.rolling(252 * 5).apply(
                    lambda x: stats.percentileofscore(x[:-1], x[-1])
                    if len(x) > 1 else 50
                )
                signals[f'{sector}_pe_percentile'] = pctl
                signals[f'{sector}_historically_cheap'] = pctl < 25
                signals[f'{sector}_historically_expensive'] = pctl > 75

        return signals

    def value_momentum_combined(
        self,
        sector_valuations: pd.DataFrame,
        sector_momentum: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine value and momentum for rotation signals.
        """
        signals = pd.DataFrame()

        # Value score (lower PE = higher score)
        value_rank = sector_valuations.rank(axis=1, ascending=True)
        value_score = value_rank / value_rank.max(axis=1).values.reshape(-1, 1) * 100

        # Momentum score (higher return = higher score)
        mom_rank = sector_momentum.rank(axis=1, ascending=False)
        mom_score = mom_rank / mom_rank.max(axis=1).values.reshape(-1, 1) * 100

        # Combined score (equal weight)
        combined = (value_score + mom_score) / 2

        signals['value_score'] = value_score
        signals['momentum_score'] = mom_score
        signals['combined_score'] = combined

        # Quintile classification
        for sector in combined.columns:
            quintile = pd.qcut(combined[sector], q=5, labels=[1, 2, 3, 4, 5])
            signals[f'{sector}_quintile'] = quintile
            signals[f'{sector}_attractive'] = quintile >= 4
            signals[f'{sector}_avoid'] = quintile <= 2

        return signals
```

---

## 4. Defensive Rotation

### 4.1 Risk-Off Signals

**Signal Logic**:
```python
class DefensiveRotationSignals:
    """Generate defensive sector rotation signals."""

    def __init__(self):
        self.defensive_sectors = ["Utilities", "Consumer Staples", "Health Care"]
        self.cyclical_sectors = [
            "Consumer Discretionary", "Industrials", "Materials", "Financials"
        ]
        self.growth_sectors = ["Information Technology", "Communication Services"]

    def risk_regime_rotation(
        self,
        vix: pd.Series,
        credit_spreads: pd.Series,
        market_trend: pd.Series
    ) -> pd.DataFrame:
        """
        Generate rotation signals based on risk regime.
        """
        signals = pd.DataFrame(index=vix.index)

        # VIX regime
        vix_ma = vix.rolling(20).mean()
        signals['vix_elevated'] = vix > 25
        signals['vix_spike'] = vix > vix_ma * 1.3
        signals['vix_low'] = vix < 15

        # Credit stress
        spread_ma = credit_spreads.rolling(60).mean()
        spread_std = credit_spreads.rolling(60).std()
        signals['credit_stress'] = credit_spreads > (spread_ma + 2 * spread_std)
        signals['credit_calm'] = credit_spreads < spread_ma

        # Market trend
        signals['market_uptrend'] = market_trend > 0
        signals['market_downtrend'] = market_trend < 0

        # Risk regime classification
        signals['risk_off'] = (
            signals['vix_elevated'] |
            signals['credit_stress'] |
            signals['market_downtrend']
        )
        signals['risk_on'] = (
            signals['vix_low'] &
            signals['credit_calm'] &
            signals['market_uptrend']
        )

        # Sector recommendations
        signals['favor_defensive'] = signals['risk_off']
        signals['favor_cyclical'] = signals['risk_on']

        return signals

    def get_defensive_allocation(
        self,
        risk_signals: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Get sector weights for defensive positioning.
        """
        base_weight = 1 / 11  # Equal weight

        if risk_signals['risk_off'].iloc[-1]:
            weights = {
                "Utilities": 0.15,
                "Consumer Staples": 0.15,
                "Health Care": 0.15,
                "Real Estate": 0.05,
                "Information Technology": 0.10,
                "Communication Services": 0.08,
                "Financials": 0.08,
                "Consumer Discretionary": 0.06,
                "Industrials": 0.06,
                "Materials": 0.06,
                "Energy": 0.06
            }
        elif risk_signals['risk_on'].iloc[-1]:
            weights = {
                "Information Technology": 0.15,
                "Consumer Discretionary": 0.12,
                "Financials": 0.12,
                "Industrials": 0.10,
                "Communication Services": 0.10,
                "Materials": 0.08,
                "Energy": 0.08,
                "Health Care": 0.08,
                "Consumer Staples": 0.06,
                "Utilities": 0.06,
                "Real Estate": 0.05
            }
        else:
            # Neutral
            weights = {sector: base_weight for sector in [
                "Information Technology", "Health Care", "Financials",
                "Consumer Discretionary", "Communication Services",
                "Industrials", "Consumer Staples", "Energy",
                "Utilities", "Real Estate", "Materials"
            ]}

        return weights

    def drawdown_rotation(
        self,
        sector_drawdowns: pd.DataFrame,
        max_drawdown_threshold: float = -0.15
    ) -> pd.DataFrame:
        """
        Rotate out of sectors in significant drawdown.
        """
        signals = pd.DataFrame(index=sector_drawdowns.index)

        for sector in sector_drawdowns.columns:
            dd = sector_drawdowns[sector]

            signals[f'{sector}_drawdown'] = dd
            signals[f'{sector}_severe_dd'] = dd < max_drawdown_threshold
            signals[f'{sector}_recovering'] = (dd > dd.shift(5)) & (dd < -0.05)

        # Rotation recommendation
        severe_dd_count = sector_drawdowns.apply(
            lambda x: (x < max_drawdown_threshold).sum()
        )
        signals['broad_stress'] = severe_dd_count > 3

        return signals
```

---

## 5. Sector Signal Integration

### 5.1 Unified Rotation Engine

```python
class SectorRotationEngine:
    """
    Production sector rotation signal engine.
    """

    def __init__(self):
        self.cycle_detector = EconomicCycleDetector()
        self.momentum_signals = SectorMomentumSignals()
        self.valuation_signals = SectorValuationSignals()
        self.defensive_signals = DefensiveRotationSignals()

    def generate_rotation_signals(
        self,
        sector_prices: pd.DataFrame,
        sector_valuations: pd.DataFrame,
        macro_data: pd.DataFrame,
        current_weights: pd.Series
    ) -> Dict:
        """
        Generate comprehensive sector rotation signals.

        Returns:
            Dictionary with signals and recommendations
        """
        results = {}

        # Economic cycle analysis
        if all(c in macro_data.columns for c in
               ['gdp_growth', 'pmi', 'unemployment', 'yield_curve', 'credit_spreads']):
            cycle_signals = self.cycle_detector.detect_cycle_phase(
                macro_data['gdp_growth'],
                macro_data['pmi'],
                macro_data['unemployment'],
                macro_data['yield_curve'],
                macro_data['credit_spreads']
            )
            results['cycle_phase'] = cycle_signals['phase'].iloc[-1]
            results['cycle_weights'] = self.cycle_detector.get_sector_weights(
                EconomicPhase(results['cycle_phase'])
            )

        # Momentum signals
        momentum = self.momentum_signals.calculate_sector_momentum(sector_prices)
        rotation = self.momentum_signals.rotation_signals(momentum, current_weights)
        results['momentum_signals'] = momentum
        results['rotation_recommendations'] = rotation

        # Valuation signals
        valuation_sigs = self.valuation_signals.relative_valuation_signals(
            sector_valuations
        )
        results['valuation_signals'] = valuation_sigs

        # Risk regime
        if all(c in macro_data.columns for c in ['vix', 'credit_spreads', 'market_trend']):
            risk_signals = self.defensive_signals.risk_regime_rotation(
                macro_data['vix'],
                macro_data['credit_spreads'],
                macro_data['market_trend']
            )
            results['risk_signals'] = risk_signals
            results['defensive_weights'] = self.defensive_signals.get_defensive_allocation(
                risk_signals
            )

        # Final recommendation
        results['final_weights'] = self._blend_recommendations(results)

        return results

    def _blend_recommendations(
        self,
        results: Dict,
        cycle_weight: float = 0.3,
        momentum_weight: float = 0.4,
        risk_weight: float = 0.3
    ) -> pd.Series:
        """
        Blend multiple rotation signals into final weights.
        """
        sectors = [
            "Information Technology", "Health Care", "Financials",
            "Consumer Discretionary", "Communication Services",
            "Industrials", "Consumer Staples", "Energy",
            "Utilities", "Real Estate", "Materials"
        ]

        final = pd.Series(0.0, index=sectors)

        # Cycle-based weights
        if 'cycle_weights' in results:
            cycle = pd.Series(results['cycle_weights'])
            for s in sectors:
                final[s] += cycle.get(s, 1/11) * cycle_weight

        # Momentum-based weights
        if 'rotation_recommendations' in results:
            mom_weights = results['rotation_recommendations']['target_weight']
            for s in sectors:
                if s in mom_weights.index:
                    final[s] += mom_weights[s] * momentum_weight

        # Risk-adjusted weights
        if 'defensive_weights' in results:
            risk = pd.Series(results['defensive_weights'])
            for s in sectors:
                final[s] += risk.get(s, 1/11) * risk_weight

        # Normalize
        return final / final.sum()
```

---

## Signal Usage Guidelines

### Sector Rotation Matrix

| Cycle Phase | Overweight | Underweight |
|-------------|------------|-------------|
| Early Expansion | Financials, Industrials | Utilities, Staples |
| Mid Expansion | Technology, Comm Services | Utilities |
| Late Expansion | Energy, Materials, Health | Tech, Discretionary |
| Contraction | Utilities, Staples, Health | Cyclicals |
| Recovery | Discretionary, Tech | Defensives |

### Integration with Ordinis

```python
# Sector rotation in portfolio
rotation_engine = SectorRotationEngine()
signals = rotation_engine.generate_rotation_signals(
    sector_prices, sector_pe, macro_data, current_weights
)

# Apply sector tilts
target_weights = signals['final_weights']

# Generate rebalancing trades
for sector, target in target_weights.items():
    current = current_weights.get(sector, 0)
    if abs(target - current) > 0.02:  # 2% threshold
        trade_size = target - current
        # Generate trade signal
```

---

## Academic References

1. **Fama & French (1988)**: "Business Conditions and Expected Returns"
2. **Moskowitz & Grinblatt (1999)**: "Do Industries Explain Momentum?"
3. **Hong et al. (2007)**: "Industry Information Diffusion and the Lead-lag Effect"
4. **Conover et al. (2008)**: "Sector Rotation and Monetary Conditions"
5. **Beber et al. (2011)**: "What Does Equity Sector Orderflow Tell Us?"
