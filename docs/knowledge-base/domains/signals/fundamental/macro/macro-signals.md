# Macroeconomic Trading Signals

## Overview

Macro signals translate economic data into systematic trading rules. These signals provide **regime classification**, **risk management**, and **allocation adjustments** for portfolio-level decisions.

---

## 1. Interest Rate Signals

### 1.1 Yield Curve Analysis

**Signal Logic**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from scipy import interpolate


class YieldCurveRegime(Enum):
    STEEP_NORMAL = "steep_normal"
    FLAT = "flat"
    INVERTED = "inverted"
    BEAR_STEEPENING = "bear_steepening"
    BULL_STEEPENING = "bull_steepening"
    BEAR_FLATTENING = "bear_flattening"
    BULL_FLATTENING = "bull_flattening"


@dataclass
class YieldCurveConfig:
    """Configuration for yield curve signals."""

    # Spread thresholds (in percentage points)
    steep_threshold: float = 1.50
    flat_threshold: float = 0.25
    inversion_threshold: float = -0.10

    # Rate change thresholds
    rate_change_threshold: float = 0.25  # 25 bps

    # Lookback for dynamics
    dynamics_lookback: int = 63  # ~3 months


class YieldCurveSignals:
    """Generate trading signals from yield curve analysis."""

    def __init__(self, config: YieldCurveConfig = None):
        self.config = config or YieldCurveConfig()

    def curve_shape_signals(
        self,
        yield_2y: pd.Series,
        yield_10y: pd.Series,
        yield_3m: pd.Series = None
    ) -> pd.DataFrame:
        """
        Analyze yield curve shape and generate signals.
        """
        signals = pd.DataFrame(index=yield_2y.index)

        # Primary spread (10Y - 2Y)
        spread_10_2 = yield_10y - yield_2y
        signals['spread_10_2'] = spread_10_2

        # Near-term spread (if 3M available)
        if yield_3m is not None:
            spread_10_3m = yield_10y - yield_3m
            spread_2_3m = yield_2y - yield_3m
            signals['spread_10_3m'] = spread_10_3m
            signals['spread_2_3m'] = spread_2_3m

        # Shape classification
        signals['steep'] = spread_10_2 > self.config.steep_threshold
        signals['normal'] = (
            (spread_10_2 > self.config.flat_threshold) &
            (spread_10_2 <= self.config.steep_threshold)
        )
        signals['flat'] = (
            (spread_10_2 >= self.config.inversion_threshold) &
            (spread_10_2 <= self.config.flat_threshold)
        )
        signals['inverted'] = spread_10_2 < self.config.inversion_threshold

        # Spread dynamics
        spread_change = spread_10_2 - spread_10_2.shift(self.config.dynamics_lookback)
        signals['steepening'] = spread_change > 0.20
        signals['flattening'] = spread_change < -0.20

        # Rate level dynamics
        rate_change_2y = yield_2y - yield_2y.shift(self.config.dynamics_lookback)
        rate_change_10y = yield_10y - yield_10y.shift(self.config.dynamics_lookback)

        signals['rates_rising'] = (rate_change_2y + rate_change_10y) / 2 > self.config.rate_change_threshold
        signals['rates_falling'] = (rate_change_2y + rate_change_10y) / 2 < -self.config.rate_change_threshold

        # Combined dynamics regime
        signals['bear_steepening'] = signals['rates_rising'] & signals['steepening']
        signals['bull_steepening'] = signals['rates_falling'] & signals['steepening']
        signals['bear_flattening'] = signals['rates_rising'] & signals['flattening']
        signals['bull_flattening'] = signals['rates_falling'] & signals['flattening']

        return signals

    def recession_probability_signal(
        self,
        yield_curve_spread: pd.Series,
        lookforward: int = 252  # 1 year
    ) -> pd.DataFrame:
        """
        Estimate recession probability from yield curve.

        Based on NY Fed model: P(recession) = f(spread)
        """
        signals = pd.DataFrame(index=yield_curve_spread.index)

        # Simplified probit approximation
        # Negative spread correlates with higher recession probability
        def probit_recession_prob(spread):
            # Coefficients approximated from NY Fed model
            z = -0.5 - 0.7 * spread
            from scipy.stats import norm
            return norm.cdf(z)

        signals['recession_prob'] = yield_curve_spread.apply(probit_recession_prob)

        # Classification
        signals['recession_warning'] = signals['recession_prob'] > 0.30
        signals['recession_likely'] = signals['recession_prob'] > 0.50
        signals['recession_imminent'] = signals['recession_prob'] > 0.70

        # Change in probability
        signals['prob_rising'] = signals['recession_prob'] > signals['recession_prob'].shift(21)

        return signals

    def rate_environment_signals(
        self,
        fed_funds: pd.Series,
        yield_10y: pd.Series
    ) -> pd.DataFrame:
        """
        Classify the overall rate environment.
        """
        signals = pd.DataFrame(index=fed_funds.index)

        # Rate level regime
        ff_ma_200 = fed_funds.rolling(200).mean()
        signals['rates_high'] = fed_funds > ff_ma_200 * 1.2
        signals['rates_low'] = fed_funds < ff_ma_200 * 0.8
        signals['rates_neutral'] = ~signals['rates_high'] & ~signals['rates_low']

        # Fed policy direction
        ff_change_3m = fed_funds - fed_funds.shift(63)
        signals['fed_tightening'] = ff_change_3m > 0.25
        signals['fed_easing'] = ff_change_3m < -0.25
        signals['fed_on_hold'] = ff_change_3m.abs() <= 0.25

        # Real rate estimate (simplified)
        # Assumes ~2% inflation for this example
        implied_inflation = 2.0
        real_rate = yield_10y - implied_inflation
        signals['real_rate'] = real_rate
        signals['positive_real_rate'] = real_rate > 0.5
        signals['negative_real_rate'] = real_rate < 0

        return signals
```

---

## 2. Inflation Signals

### 2.1 Inflation Regime Detection

**Signal Logic**:
```python
class InflationRegime(Enum):
    DEFLATION = "deflation"
    LOW_STABLE = "low_stable"
    MODERATE = "moderate"
    HIGH = "high"
    HYPERINFLATION = "hyperinflation"
    RISING = "rising"
    FALLING = "falling"


class InflationSignals:
    """Generate signals from inflation data."""

    def __init__(
        self,
        low_threshold: float = 0.02,
        moderate_threshold: float = 0.03,
        high_threshold: float = 0.05
    ):
        self.low_threshold = low_threshold
        self.moderate_threshold = moderate_threshold
        self.high_threshold = high_threshold

    def inflation_regime_signals(
        self,
        cpi_yoy: pd.Series,
        core_cpi_yoy: pd.Series = None
    ) -> pd.DataFrame:
        """
        Classify inflation regime and generate signals.
        """
        signals = pd.DataFrame(index=cpi_yoy.index)

        # Use core if available, else headline
        inflation = core_cpi_yoy if core_cpi_yoy is not None else cpi_yoy

        signals['inflation_rate'] = inflation

        # Level classification
        signals['deflation'] = inflation < 0
        signals['low_inflation'] = (inflation >= 0) & (inflation < self.low_threshold)
        signals['moderate_inflation'] = (
            (inflation >= self.low_threshold) &
            (inflation < self.moderate_threshold)
        )
        signals['elevated_inflation'] = (
            (inflation >= self.moderate_threshold) &
            (inflation < self.high_threshold)
        )
        signals['high_inflation'] = inflation >= self.high_threshold

        # Direction
        inflation_ma = inflation.rolling(3).mean()
        inflation_change = inflation_ma - inflation_ma.shift(3)
        signals['inflation_rising'] = inflation_change > 0.005
        signals['inflation_falling'] = inflation_change < -0.005
        signals['inflation_stable'] = inflation_change.abs() <= 0.005

        # Surprise (vs expectations if available)
        # Simplified: momentum relative to trend
        trend = inflation.rolling(12).mean()
        signals['above_trend'] = inflation > trend
        signals['below_trend'] = inflation < trend

        return signals

    def breakeven_signals(
        self,
        tips_breakeven_5y: pd.Series,
        tips_breakeven_10y: pd.Series
    ) -> pd.DataFrame:
        """
        Analyze inflation expectations from TIPS breakevens.
        """
        signals = pd.DataFrame(index=tips_breakeven_5y.index)

        signals['breakeven_5y'] = tips_breakeven_5y
        signals['breakeven_10y'] = tips_breakeven_10y

        # Expectations level
        signals['expectations_anchored'] = (
            (tips_breakeven_5y > 0.015) &
            (tips_breakeven_5y < 0.03) &
            (tips_breakeven_10y > 0.018) &
            (tips_breakeven_10y < 0.028)
        )

        signals['expectations_elevated'] = tips_breakeven_5y > 0.03
        signals['deflation_expectations'] = tips_breakeven_5y < 0.01

        # Term structure of expectations
        be_slope = tips_breakeven_10y - tips_breakeven_5y
        signals['inflation_expectations_rising'] = be_slope < 0  # Near > far
        signals['inflation_expectations_falling'] = be_slope > 0.005

        return signals

    def inflation_hedging_signals(
        self,
        inflation_signals: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Generate asset allocation signals for inflation hedging.
        """
        latest = inflation_signals.iloc[-1]

        if latest.get('high_inflation', False):
            return {
                'commodities': 0.15,
                'tips': 0.10,
                'real_estate': 0.10,
                'gold': 0.05,
                'short_duration_bonds': 0.20,
                'equities': 0.40  # Reduced
            }
        elif latest.get('deflation', False):
            return {
                'long_duration_bonds': 0.30,
                'cash': 0.15,
                'defensive_equities': 0.35,
                'gold': 0.10,
                'tips': 0.10
            }
        else:
            return {
                'equities': 0.60,
                'bonds': 0.30,
                'alternatives': 0.10
            }
```

---

## 3. Growth Indicators

### 3.1 PMI and Leading Indicators

**Signal Logic**:
```python
class EconomicGrowthSignals:
    """Generate signals from economic growth indicators."""

    def __init__(
        self,
        pmi_expansion: float = 50.0,
        pmi_strong: float = 55.0,
        pmi_weak: float = 45.0
    ):
        self.pmi_expansion = pmi_expansion
        self.pmi_strong = pmi_strong
        self.pmi_weak = pmi_weak

    def pmi_signals(
        self,
        manufacturing_pmi: pd.Series,
        services_pmi: pd.Series = None
    ) -> pd.DataFrame:
        """
        Generate signals from PMI data.
        """
        signals = pd.DataFrame(index=manufacturing_pmi.index)

        # Manufacturing PMI signals
        signals['mfg_pmi'] = manufacturing_pmi
        signals['mfg_expansion'] = manufacturing_pmi > self.pmi_expansion
        signals['mfg_strong'] = manufacturing_pmi > self.pmi_strong
        signals['mfg_contraction'] = manufacturing_pmi < self.pmi_expansion
        signals['mfg_deep_contraction'] = manufacturing_pmi < self.pmi_weak

        # PMI momentum
        pmi_ma = manufacturing_pmi.rolling(3).mean()
        signals['pmi_improving'] = pmi_ma > pmi_ma.shift(1)
        signals['pmi_deteriorating'] = pmi_ma < pmi_ma.shift(1)

        # Services PMI (if available)
        if services_pmi is not None:
            signals['svc_pmi'] = services_pmi
            signals['svc_expansion'] = services_pmi > self.pmi_expansion

            # Composite
            composite_pmi = (manufacturing_pmi * 0.3 + services_pmi * 0.7)
            signals['composite_pmi'] = composite_pmi
            signals['economy_expanding'] = composite_pmi > self.pmi_expansion

        # Inflection points
        signals['pmi_turning_up'] = (
            (manufacturing_pmi > manufacturing_pmi.shift(1)) &
            (manufacturing_pmi.shift(1) < manufacturing_pmi.shift(2))
        )
        signals['pmi_turning_down'] = (
            (manufacturing_pmi < manufacturing_pmi.shift(1)) &
            (manufacturing_pmi.shift(1) > manufacturing_pmi.shift(2))
        )

        return signals

    def gdp_signals(
        self,
        gdp_growth: pd.Series,
        gdp_nowcast: pd.Series = None
    ) -> pd.DataFrame:
        """
        Generate signals from GDP data.
        """
        signals = pd.DataFrame(index=gdp_growth.index)

        signals['gdp_growth'] = gdp_growth

        # Growth classification
        signals['recession'] = gdp_growth < 0
        signals['slow_growth'] = (gdp_growth >= 0) & (gdp_growth < 0.02)
        signals['trend_growth'] = (gdp_growth >= 0.02) & (gdp_growth < 0.035)
        signals['above_trend'] = gdp_growth >= 0.035

        # Direction
        signals['accelerating'] = gdp_growth > gdp_growth.shift(1)
        signals['decelerating'] = gdp_growth < gdp_growth.shift(1)

        # Nowcast signals (if available)
        if gdp_nowcast is not None:
            signals['nowcast'] = gdp_nowcast
            signals['nowcast_surprise'] = gdp_nowcast - gdp_growth.shift(1)
            signals['positive_surprise'] = signals['nowcast_surprise'] > 0.005
            signals['negative_surprise'] = signals['nowcast_surprise'] < -0.005

        return signals

    def employment_signals(
        self,
        unemployment_rate: pd.Series,
        nonfarm_payrolls_change: pd.Series,
        initial_claims: pd.Series = None
    ) -> pd.DataFrame:
        """
        Generate signals from employment data.
        """
        signals = pd.DataFrame(index=unemployment_rate.index)

        # Unemployment rate signals
        signals['unemployment'] = unemployment_rate
        ur_ma = unemployment_rate.rolling(3).mean()

        signals['unemployment_low'] = unemployment_rate < 0.04
        signals['unemployment_high'] = unemployment_rate > 0.06
        signals['unemployment_rising'] = ur_ma > ur_ma.shift(3)
        signals['unemployment_falling'] = ur_ma < ur_ma.shift(3)

        # Sahm Rule: recession indicator
        # Recession when 3-month moving average rises 0.5% from 12-month low
        ur_3m_avg = unemployment_rate.rolling(3).mean()
        ur_12m_low = ur_3m_avg.rolling(12).min()
        sahm_indicator = ur_3m_avg - ur_12m_low
        signals['sahm_indicator'] = sahm_indicator
        signals['sahm_recession_signal'] = sahm_indicator > 0.005  # 0.5%

        # Payrolls signals
        signals['payrolls_change'] = nonfarm_payrolls_change
        signals['strong_job_growth'] = nonfarm_payrolls_change > 200_000
        signals['moderate_job_growth'] = (
            (nonfarm_payrolls_change > 100_000) &
            (nonfarm_payrolls_change <= 200_000)
        )
        signals['weak_job_growth'] = (
            (nonfarm_payrolls_change > 0) &
            (nonfarm_payrolls_change <= 100_000)
        )
        signals['job_losses'] = nonfarm_payrolls_change < 0

        # Initial claims (if available)
        if initial_claims is not None:
            signals['initial_claims'] = initial_claims
            claims_ma = initial_claims.rolling(4).mean()
            signals['claims_elevated'] = claims_ma > 300_000
            signals['claims_spike'] = initial_claims > claims_ma * 1.2

        return signals
```

---

## 4. Financial Conditions

### 4.1 Financial Conditions Index

**Signal Logic**:
```python
class FinancialConditionsSignals:
    """Generate signals from financial conditions indicators."""

    def __init__(self):
        self.tight_threshold = 0.5
        self.loose_threshold = -0.5

    def conditions_index_signals(
        self,
        fci: pd.Series  # Financial Conditions Index (e.g., GS FCI)
    ) -> pd.DataFrame:
        """
        Generate signals from Financial Conditions Index.

        Note: Higher FCI typically = tighter conditions
        """
        signals = pd.DataFrame(index=fci.index)

        signals['fci'] = fci
        fci_ma = fci.rolling(20).mean()

        # Level signals
        signals['conditions_tight'] = fci > self.tight_threshold
        signals['conditions_loose'] = fci < self.loose_threshold
        signals['conditions_neutral'] = (
            (fci >= self.loose_threshold) &
            (fci <= self.tight_threshold)
        )

        # Direction
        signals['tightening'] = fci > fci_ma
        signals['easing'] = fci < fci_ma

        # Rate of change
        fci_change = fci - fci.shift(21)
        signals['rapid_tightening'] = fci_change > 0.5
        signals['rapid_easing'] = fci_change < -0.5

        return signals

    def credit_spread_signals(
        self,
        high_yield_spread: pd.Series,
        investment_grade_spread: pd.Series
    ) -> pd.DataFrame:
        """
        Generate signals from credit spreads.
        """
        signals = pd.DataFrame(index=high_yield_spread.index)

        signals['hy_spread'] = high_yield_spread
        signals['ig_spread'] = investment_grade_spread

        # Historical percentiles
        hy_pctl = high_yield_spread.rolling(252 * 3).apply(
            lambda x: (x[-1] - x.min()) / (x.max() - x.min()) * 100
            if x.max() != x.min() else 50
        )
        signals['hy_spread_percentile'] = hy_pctl

        # Spread levels
        signals['credit_calm'] = hy_pctl < 30
        signals['credit_normal'] = (hy_pctl >= 30) & (hy_pctl <= 70)
        signals['credit_stressed'] = hy_pctl > 70
        signals['credit_crisis'] = hy_pctl > 90

        # Spread dynamics
        hy_change = high_yield_spread - high_yield_spread.shift(21)
        signals['spreads_widening'] = hy_change > 0.50
        signals['spreads_tightening'] = hy_change < -0.50

        # Quality spread (HY - IG)
        quality_spread = high_yield_spread - investment_grade_spread
        signals['quality_spread'] = quality_spread
        signals['flight_to_quality'] = quality_spread > quality_spread.rolling(60).mean() * 1.2

        return signals

    def liquidity_signals(
        self,
        ted_spread: pd.Series,
        libor_ois_spread: pd.Series = None
    ) -> pd.DataFrame:
        """
        Generate signals from money market liquidity indicators.
        """
        signals = pd.DataFrame(index=ted_spread.index)

        signals['ted_spread'] = ted_spread

        # TED spread signals
        ted_ma = ted_spread.rolling(20).mean()
        signals['liquidity_normal'] = ted_spread < 0.50
        signals['liquidity_tight'] = ted_spread > 0.50
        signals['liquidity_stress'] = ted_spread > 1.00
        signals['liquidity_crisis'] = ted_spread > 2.00

        # LIBOR-OIS (if available)
        if libor_ois_spread is not None:
            signals['libor_ois'] = libor_ois_spread
            signals['funding_stress'] = libor_ois_spread > 0.35

        return signals
```

---

## 5. Composite Macro Regime

### 5.1 Integrated Macro Signal Engine

```python
class MacroRegime(Enum):
    GOLDILOCKS = "goldilocks"        # Growth + low inflation
    REFLATION = "reflation"           # Rising growth + inflation
    STAGFLATION = "stagflation"       # Slow growth + inflation
    DEFLATION = "deflation"           # Falling growth + prices
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"


class MacroSignalEngine:
    """
    Production macro signal engine integrating multiple indicators.
    """

    def __init__(self):
        self.yield_curve = YieldCurveSignals()
        self.inflation = InflationSignals()
        self.growth = EconomicGrowthSignals()
        self.conditions = FinancialConditionsSignals()

    def classify_macro_regime(
        self,
        macro_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Classify overall macro regime.
        """
        signals = pd.DataFrame(index=macro_data.index)

        # Growth component
        growth_positive = macro_data.get('gdp_growth', 0) > 0.015
        growth_accelerating = macro_data.get('pmi', 50) > 50

        # Inflation component
        inflation_elevated = macro_data.get('cpi_yoy', 0.02) > 0.03
        inflation_low = macro_data.get('cpi_yoy', 0.02) < 0.02

        # Regime classification
        signals['goldilocks'] = growth_positive & inflation_low
        signals['reflation'] = growth_accelerating & inflation_elevated
        signals['stagflation'] = ~growth_accelerating & inflation_elevated
        signals['deflation'] = ~growth_positive & inflation_low

        # Risk regime
        vix_low = macro_data.get('vix', 20) < 20
        spreads_tight = macro_data.get('hy_spread', 4) < 5

        signals['risk_on'] = vix_low & spreads_tight & growth_positive
        signals['risk_off'] = ~vix_low | ~spreads_tight | ~growth_positive

        return signals

    def generate_macro_signals(
        self,
        macro_data: pd.DataFrame
    ) -> Dict:
        """
        Generate comprehensive macro signals.
        """
        results = {}

        # Yield curve
        if all(c in macro_data.columns for c in ['yield_2y', 'yield_10y']):
            yc_signals = self.yield_curve.curve_shape_signals(
                macro_data['yield_2y'],
                macro_data['yield_10y']
            )
            results['yield_curve'] = yc_signals

        # Inflation
        if 'cpi_yoy' in macro_data.columns:
            inf_signals = self.inflation.inflation_regime_signals(
                macro_data['cpi_yoy'],
                macro_data.get('core_cpi_yoy')
            )
            results['inflation'] = inf_signals

        # Growth
        if 'pmi' in macro_data.columns:
            growth_signals = self.growth.pmi_signals(macro_data['pmi'])
            results['growth'] = growth_signals

        # Financial conditions
        if 'hy_spread' in macro_data.columns:
            cond_signals = self.conditions.credit_spread_signals(
                macro_data['hy_spread'],
                macro_data.get('ig_spread', macro_data['hy_spread'] * 0.3)
            )
            results['conditions'] = cond_signals

        # Regime classification
        results['regime'] = self.classify_macro_regime(macro_data)

        # Portfolio implications
        results['allocation'] = self._derive_allocation(results)

        return results

    def _derive_allocation(
        self,
        signals: Dict
    ) -> Dict[str, float]:
        """
        Derive portfolio allocation from macro signals.
        """
        regime = signals.get('regime', pd.DataFrame())
        if regime.empty:
            return {'equities': 0.60, 'bonds': 0.30, 'cash': 0.10}

        latest = regime.iloc[-1]

        if latest.get('goldilocks', False):
            return {
                'equities': 0.70,
                'corporate_bonds': 0.15,
                'treasuries': 0.10,
                'cash': 0.05
            }
        elif latest.get('stagflation', False):
            return {
                'equities': 0.40,
                'commodities': 0.15,
                'tips': 0.15,
                'short_duration': 0.20,
                'cash': 0.10
            }
        elif latest.get('risk_off', False):
            return {
                'equities': 0.40,
                'treasuries': 0.30,
                'gold': 0.10,
                'cash': 0.20
            }
        else:
            return {
                'equities': 0.55,
                'bonds': 0.30,
                'alternatives': 0.10,
                'cash': 0.05
            }
```

---

## Signal Usage Guidelines

### Macro Signal Matrix

| Regime | Equities | Bonds | Commodities | Cash |
|--------|----------|-------|-------------|------|
| Goldilocks | OW | N | N | UW |
| Reflation | OW | UW | OW | UW |
| Stagflation | UW | UW | OW | OW |
| Deflation | UW | OW | UW | N |
| Risk-Off | UW | OW | Mixed | OW |

### Integration with Ordinis

```python
# Macro signal integration
macro_engine = MacroSignalEngine()
signals = macro_engine.generate_macro_signals(macro_data)

# Get regime-based allocation
allocation = signals['allocation']

# Adjust strategy weights
if signals['regime'].iloc[-1].get('risk_off'):
    reduce_equity_exposure()
    increase_hedges()
```

---

## Data Sources

| Indicator | Source | Frequency |
|-----------|--------|-----------|
| Treasury Yields | FRED | Daily |
| CPI/PPI | BLS | Monthly |
| PMI | ISM | Monthly |
| GDP | BEA | Quarterly |
| Employment | BLS | Monthly |
| Credit Spreads | FRED/Bloomberg | Daily |

---

## Academic References

1. **Estrella & Mishkin (1998)**: "Predicting U.S. Recessions: Financial Variables"
2. **Ang & Piazzesi (2003)**: "A No-Arbitrage Vector Autoregression"
3. **Sahm (2019)**: "Direct Stimulus Payments to Individuals" (Sahm Rule)
4. **Bernanke & Blinder (1992)**: "The Federal Funds Rate and Monetary Policy"
5. **Stock & Watson (2003)**: "Forecasting Output and Inflation"
