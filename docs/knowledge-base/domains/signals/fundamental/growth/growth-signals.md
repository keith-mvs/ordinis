# Growth-Based Trading Signals

## Overview

Growth signals identify companies with accelerating business momentum. These signals provide **growth screening**, **momentum confirmation**, and **GARP (Growth at Reasonable Price)** filtering for systematic strategies.

---

## 1. Revenue Growth Signals

### 1.1 Revenue Growth Analysis

**Signal Logic**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy import stats


@dataclass
class RevenueGrowthConfig:
    """Configuration for revenue growth signals."""

    # Growth thresholds
    hyper_growth: float = 0.40    # >40% YoY
    high_growth: float = 0.20     # >20% YoY
    moderate_growth: float = 0.10 # >10% YoY
    slow_growth: float = 0.05     # >5% YoY

    # Acceleration thresholds
    acceleration_threshold: float = 0.05  # 5% acceleration

    # Consistency
    min_positive_quarters: int = 4


class RevenueGrowthSignals:
    """Generate signals from revenue growth patterns."""

    def __init__(self, config: RevenueGrowthConfig = None):
        self.config = config or RevenueGrowthConfig()

    def calculate_growth_metrics(
        self,
        revenue: pd.Series,
        quarterly: bool = True
    ) -> pd.DataFrame:
        """
        Calculate comprehensive revenue growth metrics.

        Args:
            revenue: Revenue time series
            quarterly: True if quarterly data, False if annual

        Returns:
            DataFrame with growth metrics
        """
        signals = pd.DataFrame(index=revenue.index)

        # YoY growth (4 quarters ago for quarterly data)
        lag = 4 if quarterly else 1
        yoy_growth = (revenue - revenue.shift(lag)) / revenue.shift(lag).abs()
        signals['revenue_yoy'] = yoy_growth

        # QoQ growth (sequential)
        qoq_growth = (revenue - revenue.shift(1)) / revenue.shift(1).abs()
        signals['revenue_qoq'] = qoq_growth

        # 2-year CAGR
        lag_2y = 8 if quarterly else 2
        cagr_2y = (revenue / revenue.shift(lag_2y)) ** (1 / 2) - 1
        signals['revenue_cagr_2y'] = cagr_2y

        # 3-year CAGR
        lag_3y = 12 if quarterly else 3
        cagr_3y = (revenue / revenue.shift(lag_3y)) ** (1 / 3) - 1
        signals['revenue_cagr_3y'] = cagr_3y

        # Growth classification
        signals['hyper_growth'] = yoy_growth > self.config.hyper_growth
        signals['high_growth'] = (
            (yoy_growth > self.config.high_growth) &
            (yoy_growth <= self.config.hyper_growth)
        )
        signals['moderate_growth'] = (
            (yoy_growth > self.config.moderate_growth) &
            (yoy_growth <= self.config.high_growth)
        )
        signals['slow_growth'] = (
            (yoy_growth > 0) &
            (yoy_growth <= self.config.moderate_growth)
        )
        signals['declining'] = yoy_growth < 0

        return signals

    def growth_acceleration_signals(
        self,
        revenue: pd.Series,
        quarterly: bool = True
    ) -> pd.DataFrame:
        """
        Detect revenue growth acceleration/deceleration.
        """
        lag = 4 if quarterly else 1
        signals = pd.DataFrame(index=revenue.index)

        # Current and prior period YoY growth
        yoy_growth = (revenue - revenue.shift(lag)) / revenue.shift(lag).abs()
        prior_yoy_growth = yoy_growth.shift(lag)

        # Acceleration = current growth - prior growth
        acceleration = yoy_growth - prior_yoy_growth
        signals['growth_acceleration'] = acceleration

        # Classification
        signals['accelerating'] = acceleration > self.config.acceleration_threshold
        signals['stable_growth'] = acceleration.abs() <= self.config.acceleration_threshold
        signals['decelerating'] = acceleration < -self.config.acceleration_threshold

        # Consecutive acceleration
        signals['multi_quarter_acceleration'] = (
            signals['accelerating'].rolling(2).sum() == 2
        )

        # Growth inflection points
        signals['growth_inflection_up'] = (
            (acceleration > 0) & (acceleration.shift(1) < 0)
        )
        signals['growth_inflection_down'] = (
            (acceleration < 0) & (acceleration.shift(1) > 0)
        )

        return signals

    def growth_consistency_signals(
        self,
        revenue: pd.Series,
        quarterly: bool = True
    ) -> pd.DataFrame:
        """
        Analyze growth consistency and quality.
        """
        lag = 4 if quarterly else 1
        signals = pd.DataFrame(index=revenue.index)

        yoy_growth = (revenue - revenue.shift(lag)) / revenue.shift(lag).abs()

        # Positive growth streak
        positive_growth = (yoy_growth > 0).astype(int)
        signals['positive_growth_streak'] = positive_growth.rolling(
            self.config.min_positive_quarters
        ).sum()
        signals['consistent_grower'] = (
            signals['positive_growth_streak'] == self.config.min_positive_quarters
        )

        # Growth volatility
        growth_std = yoy_growth.rolling(8).std()
        growth_mean = yoy_growth.rolling(8).mean()
        signals['growth_volatility'] = growth_std / growth_mean.abs()
        signals['stable_high_growth'] = (
            (growth_mean > 0.10) & (signals['growth_volatility'] < 0.5)
        )

        # Beat prior quarter sequentially
        sequential_increase = (revenue > revenue.shift(1)).astype(int)
        signals['sequential_beats'] = sequential_increase.rolling(4).sum()

        return signals
```

---

## 2. Earnings Growth Signals

### 2.1 EPS Growth Analysis

**Signal Logic**:
```python
class EPSGrowthSignals:
    """Generate signals from earnings per share growth."""

    def __init__(
        self,
        high_growth_threshold: float = 0.25,
        moderate_growth_threshold: float = 0.10
    ):
        self.high_growth_threshold = high_growth_threshold
        self.moderate_growth_threshold = moderate_growth_threshold

    def eps_growth_signals(
        self,
        eps: pd.Series,
        quarterly: bool = True
    ) -> pd.DataFrame:
        """
        Generate EPS growth signals.
        """
        lag = 4 if quarterly else 1
        signals = pd.DataFrame(index=eps.index)

        # Filter for positive EPS comparisons
        eps_positive = eps.where(eps > 0)
        eps_prior_positive = eps.shift(lag).where(eps.shift(lag) > 0)

        # YoY EPS growth
        eps_growth = (eps_positive - eps_prior_positive) / eps_prior_positive.abs()
        signals['eps_yoy'] = eps_growth

        # Growth classification
        signals['eps_high_growth'] = eps_growth > self.high_growth_threshold
        signals['eps_moderate_growth'] = (
            (eps_growth > self.moderate_growth_threshold) &
            (eps_growth <= self.high_growth_threshold)
        )
        signals['eps_positive'] = (eps_growth > 0) & (eps_growth <= self.moderate_growth_threshold)
        signals['eps_declining'] = eps_growth < 0

        # Turn around (loss to profit)
        signals['turnaround'] = (eps > 0) & (eps.shift(lag) < 0)
        signals['deteriorating'] = (eps < 0) & (eps.shift(lag) > 0)

        # EPS acceleration
        eps_growth_change = eps_growth - eps_growth.shift(lag)
        signals['eps_accelerating'] = eps_growth_change > 0.05
        signals['eps_decelerating'] = eps_growth_change < -0.05

        return signals

    def earnings_surprise_signals(
        self,
        actual_eps: pd.Series,
        estimated_eps: pd.Series
    ) -> pd.DataFrame:
        """
        Generate signals from earnings surprises.
        """
        signals = pd.DataFrame(index=actual_eps.index)

        # Surprise calculation
        surprise = actual_eps - estimated_eps
        surprise_pct = surprise / estimated_eps.abs()

        signals['earnings_surprise'] = surprise
        signals['surprise_pct'] = surprise_pct

        # Surprise classification
        signals['big_beat'] = surprise_pct > 0.10
        signals['beat'] = (surprise_pct > 0.02) & (surprise_pct <= 0.10)
        signals['inline'] = surprise_pct.abs() <= 0.02
        signals['miss'] = (surprise_pct < -0.02) & (surprise_pct >= -0.10)
        signals['big_miss'] = surprise_pct < -0.10

        # Surprise streak
        beat_streak = (surprise_pct > 0).astype(int).rolling(4).sum()
        signals['beat_streak'] = beat_streak
        signals['consistent_beater'] = beat_streak == 4

        miss_streak = (surprise_pct < 0).astype(int).rolling(4).sum()
        signals['miss_streak'] = miss_streak
        signals['serial_misser'] = miss_streak == 4

        return signals
```

---

## 3. Operating Metrics Growth

### 3.1 Margin Expansion Signals

**Signal Logic**:
```python
class MarginGrowthSignals:
    """Signals based on margin trends and expansion."""

    def __init__(
        self,
        expansion_threshold: float = 0.02,  # 200 bps
        compression_threshold: float = -0.02
    ):
        self.expansion_threshold = expansion_threshold
        self.compression_threshold = compression_threshold

    def margin_signals(
        self,
        revenue: pd.Series,
        gross_profit: pd.Series,
        operating_income: pd.Series,
        net_income: pd.Series
    ) -> pd.DataFrame:
        """
        Generate margin expansion/compression signals.
        """
        signals = pd.DataFrame(index=revenue.index)

        # Calculate margins
        gross_margin = gross_profit / revenue.replace(0, np.nan)
        operating_margin = operating_income / revenue.replace(0, np.nan)
        net_margin = net_income / revenue.replace(0, np.nan)

        signals['gross_margin'] = gross_margin
        signals['operating_margin'] = operating_margin
        signals['net_margin'] = net_margin

        # YoY margin changes
        gm_change = gross_margin - gross_margin.shift(4)
        om_change = operating_margin - operating_margin.shift(4)
        nm_change = net_margin - net_margin.shift(4)

        signals['gm_change'] = gm_change
        signals['om_change'] = om_change
        signals['nm_change'] = nm_change

        # Expansion signals
        signals['gross_margin_expanding'] = gm_change > self.expansion_threshold
        signals['operating_margin_expanding'] = om_change > self.expansion_threshold
        signals['net_margin_expanding'] = nm_change > self.expansion_threshold

        # Compression signals
        signals['gross_margin_compressing'] = gm_change < self.compression_threshold
        signals['operating_margin_compressing'] = om_change < self.compression_threshold
        signals['net_margin_compressing'] = nm_change < self.compression_threshold

        # Operating leverage signal
        revenue_growth = revenue.pct_change(4)
        op_income_growth = operating_income.pct_change(4)

        # Positive leverage: operating income growing faster than revenue
        signals['positive_leverage'] = (
            (op_income_growth > revenue_growth) &
            (revenue_growth > 0) &
            (operating_income > 0)
        )

        # Negative leverage: costs growing faster
        signals['negative_leverage'] = (
            (op_income_growth < revenue_growth) &
            (revenue_growth > 0)
        )

        return signals

    def profitability_transition_signals(
        self,
        operating_income: pd.Series,
        net_income: pd.Series
    ) -> pd.DataFrame:
        """
        Detect profitability transitions.
        """
        signals = pd.DataFrame(index=operating_income.index)

        # Operating profitability
        signals['op_profitable'] = operating_income > 0
        signals['op_turning_profitable'] = (
            (operating_income > 0) & (operating_income.shift(4) < 0)
        )
        signals['op_turning_unprofitable'] = (
            (operating_income < 0) & (operating_income.shift(4) > 0)
        )

        # Net profitability
        signals['net_profitable'] = net_income > 0
        signals['net_turning_profitable'] = (
            (net_income > 0) & (net_income.shift(4) < 0)
        )

        # Sustained profitability
        signals['sustained_profitability'] = (
            (operating_income > 0).rolling(4).sum() == 4
        )

        return signals
```

---

## 4. Growth Quality Assessment

### 4.1 Sustainable Growth Analysis

**Signal Logic**:
```python
class SustainableGrowthSignals:
    """Assess quality and sustainability of growth."""

    def __init__(self):
        self.sgr_threshold = 0.15  # Sustainable growth rate threshold

    def sustainable_growth_rate(
        self,
        roe: pd.Series,
        retention_ratio: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate sustainable growth rate and related signals.

        SGR = ROE * Retention Ratio
        """
        signals = pd.DataFrame(index=roe.index)

        # Sustainable growth rate
        sgr = roe * retention_ratio
        signals['sustainable_growth_rate'] = sgr

        # Compare actual growth to SGR
        signals['sgr_high'] = sgr > self.sgr_threshold
        signals['sgr_moderate'] = (sgr > 0.08) & (sgr <= self.sgr_threshold)

        return signals

    def growth_efficiency_signals(
        self,
        revenue_growth: pd.Series,
        capex: pd.Series,
        revenue: pd.Series,
        employee_count: pd.Series = None
    ) -> pd.DataFrame:
        """
        Measure efficiency of growth investments.
        """
        signals = pd.DataFrame(index=revenue.index)

        # Capital efficiency: revenue growth per dollar of capex
        capex_intensity = capex.abs() / revenue
        signals['capex_intensity'] = capex_intensity

        # Efficient growth: high revenue growth with low capex
        signals['capital_efficient_growth'] = (
            (revenue_growth > 0.15) & (capex_intensity < 0.10)
        )

        # Capital intensive growth
        signals['capital_intensive_growth'] = (
            (revenue_growth > 0.10) & (capex_intensity > 0.20)
        )

        # Revenue per employee growth (if available)
        if employee_count is not None:
            rev_per_employee = revenue / employee_count
            rpe_growth = rev_per_employee.pct_change(4)
            signals['productivity_growth'] = rpe_growth
            signals['improving_productivity'] = rpe_growth > 0.05

        return signals

    def organic_vs_acquisition_growth(
        self,
        total_revenue: pd.Series,
        organic_revenue: pd.Series,
        acquisition_revenue: pd.Series = None
    ) -> pd.DataFrame:
        """
        Distinguish organic growth from acquisition-driven growth.
        """
        signals = pd.DataFrame(index=total_revenue.index)

        total_growth = total_revenue.pct_change(4)
        organic_growth = organic_revenue.pct_change(4)

        signals['total_growth'] = total_growth
        signals['organic_growth'] = organic_growth

        # Organic growth quality
        organic_ratio = organic_growth / total_growth.replace(0, np.nan)
        signals['organic_contribution'] = organic_ratio

        signals['primarily_organic'] = organic_ratio > 0.70
        signals['acquisition_driven'] = organic_ratio < 0.30

        # High quality: strong organic growth
        signals['high_quality_growth'] = (
            (organic_growth > 0.10) & signals['primarily_organic']
        )

        return signals
```

---

## 5. Growth Valuation Signals

### 5.1 GARP (Growth at Reasonable Price)

**Signal Logic**:
```python
class GARPSignals:
    """Growth at Reasonable Price signal generation."""

    def __init__(
        self,
        peg_attractive: float = 1.0,
        peg_fair: float = 2.0,
        max_pe: float = 40
    ):
        self.peg_attractive = peg_attractive
        self.peg_fair = peg_fair
        self.max_pe = max_pe

    def peg_ratio_signals(
        self,
        pe_ratio: pd.Series,
        eps_growth: pd.Series
    ) -> pd.DataFrame:
        """
        Generate PEG ratio signals.

        PEG = P/E / EPS Growth Rate (%)
        """
        signals = pd.DataFrame(index=pe_ratio.index)

        # Convert growth to percentage
        eps_growth_pct = eps_growth * 100

        # Filter valid data
        valid = (pe_ratio > 0) & (eps_growth_pct > 0)

        peg = pe_ratio / eps_growth_pct.replace(0, np.nan)
        peg = peg.where(valid)

        signals['peg_ratio'] = peg

        # PEG classification
        signals['peg_very_attractive'] = (peg < 0.5) & valid
        signals['peg_attractive'] = (peg >= 0.5) & (peg < self.peg_attractive) & valid
        signals['peg_fair'] = (peg >= self.peg_attractive) & (peg < self.peg_fair) & valid
        signals['peg_expensive'] = (peg >= self.peg_fair) & valid

        # Combined GARP screen
        signals['garp_candidate'] = (
            (peg < self.peg_fair) &
            (pe_ratio < self.max_pe) &
            (eps_growth > 0.10) &
            valid
        )

        return signals

    def growth_adjusted_value_signals(
        self,
        pe_ratio: pd.Series,
        revenue_growth: pd.Series,
        eps_growth: pd.Series,
        roe: pd.Series
    ) -> pd.DataFrame:
        """
        Multi-factor growth-adjusted valuation.
        """
        signals = pd.DataFrame(index=pe_ratio.index)

        # Growth score (0-100)
        rev_score = revenue_growth.rank(pct=True) * 100
        eps_score = eps_growth.rank(pct=True) * 100
        growth_score = (rev_score + eps_score) / 2

        # Value score (lower PE = higher score)
        value_score = (1 - pe_ratio.rank(pct=True)) * 100

        # Quality adjustment
        quality_score = roe.rank(pct=True) * 100

        # GARP composite (40% value, 40% growth, 20% quality)
        garp_score = value_score * 0.40 + growth_score * 0.40 + quality_score * 0.20

        signals['growth_score'] = growth_score
        signals['value_score'] = value_score
        signals['garp_score'] = garp_score

        # Quintile signals
        signals['garp_quintile'] = pd.qcut(
            garp_score, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
        )
        signals['garp_top_quintile'] = signals['garp_quintile'] == 5

        return signals
```

---

## 6. Composite Growth Scoring

### 6.1 Integrated Growth Signal Engine

```python
class GrowthSignalEngine:
    """
    Production-ready growth signal generation.
    """

    def __init__(self):
        self.revenue_signals = RevenueGrowthSignals()
        self.eps_signals = EPSGrowthSignals()
        self.margin_signals = MarginGrowthSignals()
        self.sustainable_signals = SustainableGrowthSignals()
        self.garp_signals = GARPSignals()

    def generate_growth_signals(
        self,
        financial_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate comprehensive growth signals.

        Args:
            financial_data: DataFrame with financial metrics

        Returns:
            DataFrame with all growth signals
        """
        all_signals = pd.DataFrame(index=financial_data.index)

        # Revenue growth signals
        if 'revenue' in financial_data.columns:
            rev_sigs = self.revenue_signals.calculate_growth_metrics(
                financial_data['revenue']
            )
            all_signals = pd.concat([all_signals, rev_sigs], axis=1)

            accel_sigs = self.revenue_signals.growth_acceleration_signals(
                financial_data['revenue']
            )
            all_signals = pd.concat([all_signals, accel_sigs], axis=1)

        # EPS growth signals
        if 'eps' in financial_data.columns:
            eps_sigs = self.eps_signals.eps_growth_signals(
                financial_data['eps']
            )
            all_signals = pd.concat([all_signals, eps_sigs], axis=1)

        # Margin signals
        margin_cols = ['revenue', 'gross_profit', 'operating_income', 'net_income']
        if all(c in financial_data.columns for c in margin_cols):
            margin_sigs = self.margin_signals.margin_signals(
                financial_data['revenue'],
                financial_data['gross_profit'],
                financial_data['operating_income'],
                financial_data['net_income']
            )
            all_signals = pd.concat([all_signals, margin_sigs], axis=1)

        # GARP signals
        if all(c in financial_data.columns for c in ['pe_ratio', 'eps_growth']):
            garp_sigs = self.garp_signals.peg_ratio_signals(
                financial_data['pe_ratio'],
                financial_data['eps_growth']
            )
            all_signals = pd.concat([all_signals, garp_sigs], axis=1)

        # Composite growth score
        all_signals['growth_composite'] = self._calculate_composite_score(all_signals)

        return all_signals

    def _calculate_composite_score(
        self,
        signals: pd.DataFrame
    ) -> pd.Series:
        """
        Calculate composite growth score.
        """
        score = pd.Series(50.0, index=signals.index)  # Base score

        # Revenue growth component
        if 'revenue_yoy' in signals.columns:
            rev_growth = signals['revenue_yoy'].clip(-0.5, 0.5)
            score += rev_growth * 50  # Up to +/- 25 points

        # EPS growth component
        if 'eps_yoy' in signals.columns:
            eps_growth = signals['eps_yoy'].clip(-0.5, 0.5)
            score += eps_growth * 40  # Up to +/- 20 points

        # Acceleration bonus
        if 'accelerating' in signals.columns:
            score += signals['accelerating'].astype(float) * 10

        # Margin expansion bonus
        if 'operating_margin_expanding' in signals.columns:
            score += signals['operating_margin_expanding'].astype(float) * 10

        # Consistency bonus
        if 'consistent_grower' in signals.columns:
            score += signals['consistent_grower'].astype(float) * 10

        return score.clip(0, 100)

    def get_growth_universe(
        self,
        signals: pd.DataFrame,
        min_score: float = 60
    ) -> pd.Index:
        """
        Get universe of growth stocks.
        """
        if 'growth_composite' in signals.columns:
            return signals[signals['growth_composite'] >= min_score].index
        elif 'high_growth' in signals.columns:
            return signals[signals['high_growth']].index
        return signals.index
```

---

## Signal Usage Guidelines

### Key Growth Signal Combinations

| Strategy | Signals | Use Case |
|----------|---------|----------|
| Momentum Growth | `accelerating` + `high_growth` | Aggressive growth plays |
| GARP | `garp_candidate` + `quality_value` | Balanced approach |
| Turnaround | `turnaround` + `improving_productivity` | Special situations |
| Defensive Growth | `consistent_grower` + `stable_high_growth` | Lower volatility |

### Integration with Ordinis

```python
# Growth universe construction
growth_engine = GrowthSignalEngine()
signals = growth_engine.generate_growth_signals(fundamentals)

# Filter to growth universe
growth_stocks = growth_engine.get_growth_universe(signals, min_score=65)

# Combine with value screens for GARP
garp_universe = signals[
    signals['garp_candidate'] & signals['high_quality_earnings']
].index
```

---

## Academic References

1. **Jegadeesh & Titman (1993)**: "Returns to Buying Winners and Selling Losers"
2. **Chan, Jegadeesh & Lakonishok (1996)**: "Momentum Strategies"
3. **Asness (1997)**: "The Interaction of Value and Momentum Strategies"
4. **Fama & French (2008)**: "Dissecting Anomalies"
5. **Novy-Marx (2013)**: "The Other Side of Value: The Gross Profitability Premium"
