# Earnings Quality Signals

## Overview

Earnings quality signals detect the sustainability and reliability of reported earnings. High-quality earnings translate to cash flow, persist over time, and are free from manipulation. These signals identify **quality filters**, **red flags**, and **alpha opportunities**.

---

## 1. Accruals Analysis

### 1.1 Total Accruals Signal

**Theory**: High accruals (earnings significantly above cash flow) predict future underperformance.

**Signal Logic**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum


class AccrualQuality(Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    WARNING = "warning"


@dataclass
class AccrualSignalConfig:
    """Configuration for accrual-based signals."""

    # Total accruals thresholds (as % of assets)
    high_quality_threshold: float = -0.05  # Negative accruals = high quality
    low_quality_threshold: float = 0.10    # >10% accruals = warning
    extreme_threshold: float = 0.20        # >20% = red flag

    # Discretionary accruals
    jones_model_threshold: float = 0.10    # Abnormal accruals cutoff


class AccrualSignalGenerator:
    """Generate earnings quality signals from accruals analysis."""

    def __init__(self, config: AccrualSignalConfig = None):
        self.config = config or AccrualSignalConfig()

    def calculate_total_accruals(
        self,
        net_income: pd.Series,
        operating_cash_flow: pd.Series,
        total_assets: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate total accruals and generate signals.

        Total Accruals = Net Income - Operating Cash Flow
        Scaled by average total assets for comparability.
        """
        # Raw accruals
        accruals = net_income - operating_cash_flow

        # Scale by average assets
        avg_assets = (total_assets + total_assets.shift(1)) / 2
        scaled_accruals = accruals / avg_assets.replace(0, np.nan)

        signals = pd.DataFrame(index=net_income.index)

        signals['total_accruals'] = scaled_accruals
        signals['accruals_raw'] = accruals

        # Quality classification
        signals['high_quality'] = scaled_accruals < self.config.high_quality_threshold
        signals['moderate_quality'] = (
            (scaled_accruals >= self.config.high_quality_threshold) &
            (scaled_accruals < self.config.low_quality_threshold)
        )
        signals['low_quality'] = (
            (scaled_accruals >= self.config.low_quality_threshold) &
            (scaled_accruals < self.config.extreme_threshold)
        )
        signals['accruals_warning'] = scaled_accruals >= self.config.extreme_threshold

        # Cash conversion ratio
        cash_conversion = operating_cash_flow / net_income.replace(0, np.nan)
        signals['cash_conversion'] = cash_conversion
        signals['strong_cash_conversion'] = cash_conversion > 1.0
        signals['weak_cash_conversion'] = (cash_conversion > 0) & (cash_conversion < 0.5)
        signals['negative_cash_conversion'] = cash_conversion < 0

        return signals

    def calculate_discretionary_accruals(
        self,
        total_accruals: pd.Series,
        delta_revenue: pd.Series,
        delta_receivables: pd.Series,
        ppe: pd.Series,
        total_assets: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate discretionary accruals using Modified Jones Model.

        Non-discretionary Accruals = a1*(1/Assets) + a2*(DRev-DRec)/Assets + a3*PPE/Assets
        Discretionary Accruals = Total Accruals - Non-discretionary Accruals
        """
        avg_assets = (total_assets + total_assets.shift(1)) / 2

        # Scaled variables
        one_over_assets = 1 / avg_assets.replace(0, np.nan)
        adj_revenue_change = (delta_revenue - delta_receivables) / avg_assets
        ppe_scaled = ppe / avg_assets
        scaled_accruals = total_accruals / avg_assets

        # Create feature matrix for regression
        features = pd.DataFrame({
            'intercept': one_over_assets,
            'adj_revenue': adj_revenue_change,
            'ppe': ppe_scaled
        })

        signals = pd.DataFrame(index=total_accruals.index)

        # In production, estimate coefficients from cross-section
        # Here we use simplified residual approach
        from scipy import stats

        # Rolling estimation
        lookback = 20  # quarters

        def estimate_residuals(window_data):
            if len(window_data) < 10:
                return np.nan
            X = window_data[['intercept', 'adj_revenue', 'ppe']].values
            y = window_data['accruals'].values
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
                predicted = X @ coeffs
                return y[-1] - predicted[-1]
            except:
                return np.nan

        features['accruals'] = scaled_accruals

        signals['discretionary_accruals'] = scaled_accruals  # Simplified
        signals['abnormal_accruals_warning'] = (
            signals['discretionary_accruals'].abs() > self.config.jones_model_threshold
        )

        return signals


class BalanceSheetAccruals:
    """Analyze balance sheet based accrual components."""

    def calculate_component_accruals(
        self,
        delta_receivables: pd.Series,
        delta_inventory: pd.Series,
        delta_prepaid: pd.Series,
        delta_payables: pd.Series,
        delta_accrued_expenses: pd.Series,
        depreciation: pd.Series,
        total_assets: pd.Series
    ) -> pd.DataFrame:
        """
        Decompose accruals into working capital and non-current components.
        """
        avg_assets = (total_assets + total_assets.shift(1)) / 2

        signals = pd.DataFrame(index=total_assets.index)

        # Working capital accruals
        wc_accruals = (
            delta_receivables + delta_inventory + delta_prepaid
            - delta_payables - delta_accrued_expenses
        )
        signals['wc_accruals'] = wc_accruals / avg_assets

        # Non-current accruals
        signals['depreciation_scaled'] = depreciation / avg_assets

        # Component warnings
        signals['receivables_buildup'] = (
            delta_receivables / avg_assets > 0.05
        )
        signals['inventory_buildup'] = (
            delta_inventory / avg_assets > 0.05
        )

        # Combined working capital quality
        signals['wc_quality_concern'] = (
            signals['receivables_buildup'] | signals['inventory_buildup']
        )

        return signals
```

---

## 2. Cash Flow Quality

### 2.1 Operating Cash Flow Analysis

**Signal Logic**:
```python
class CashFlowQualitySignals:
    """Signals based on cash flow quality metrics."""

    def __init__(
        self,
        sustained_periods: int = 4,  # quarters
        ocf_ni_threshold: float = 0.8
    ):
        self.sustained_periods = sustained_periods
        self.ocf_ni_threshold = ocf_ni_threshold

    def ocf_quality_signals(
        self,
        operating_cash_flow: pd.Series,
        net_income: pd.Series,
        capital_expenditure: pd.Series,
        dividends: pd.Series
    ) -> pd.DataFrame:
        """
        Generate cash flow quality signals.
        """
        signals = pd.DataFrame(index=operating_cash_flow.index)

        # OCF vs Net Income
        ocf_ni_ratio = operating_cash_flow / net_income.replace(0, np.nan)
        signals['ocf_ni_ratio'] = ocf_ni_ratio

        # Cash earnings quality
        signals['cash_backs_earnings'] = (
            (operating_cash_flow > 0) & (operating_cash_flow >= net_income * self.ocf_ni_threshold)
        )
        signals['earnings_not_cash'] = (
            (net_income > 0) & (operating_cash_flow < net_income * 0.5)
        )
        signals['cash_exceeds_earnings'] = ocf_ni_ratio > 1.2

        # Free cash flow
        fcf = operating_cash_flow - capital_expenditure.abs()
        signals['fcf'] = fcf
        signals['fcf_positive'] = fcf > 0

        # FCF coverage
        if dividends is not None:
            fcf_div_coverage = fcf / dividends.replace(0, np.nan)
            signals['dividend_covered'] = fcf > dividends
            signals['dividend_coverage_ratio'] = fcf_div_coverage

        # Sustained cash generation
        signals['sustained_positive_ocf'] = (
            operating_cash_flow.rolling(self.sustained_periods).min() > 0
        )
        signals['sustained_positive_fcf'] = (
            fcf.rolling(self.sustained_periods).min() > 0
        )

        # Cash flow trend
        ocf_growth = operating_cash_flow.pct_change(4)  # YoY for quarterly
        signals['ocf_growing'] = ocf_growth > 0
        signals['ocf_growth_rate'] = ocf_growth

        return signals

    def fcf_yield_signals(
        self,
        free_cash_flow: pd.Series,
        market_cap: pd.Series,
        enterprise_value: pd.Series
    ) -> pd.DataFrame:
        """
        Generate FCF yield signals.
        """
        signals = pd.DataFrame(index=free_cash_flow.index)

        # FCF yield to market cap
        fcf_yield_mc = free_cash_flow / market_cap.replace(0, np.nan)
        signals['fcf_yield_mc'] = fcf_yield_mc
        signals['high_fcf_yield'] = fcf_yield_mc > 0.08
        signals['attractive_fcf_yield'] = fcf_yield_mc > 0.05
        signals['low_fcf_yield'] = (fcf_yield_mc > 0) & (fcf_yield_mc < 0.03)

        # FCF yield to EV
        fcf_yield_ev = free_cash_flow / enterprise_value.replace(0, np.nan)
        signals['fcf_yield_ev'] = fcf_yield_ev

        # Relative FCF yield (requires cross-sectional comparison)
        signals['fcf_yield_percentile'] = fcf_yield_mc.rank(pct=True)

        return signals
```

---

## 3. Revenue Quality

### 3.1 Revenue Recognition Signals

**Signal Logic**:
```python
class RevenueQualitySignals:
    """Detect revenue quality issues and manipulation risks."""

    def __init__(
        self,
        receivables_growth_threshold: float = 1.5,
        deferred_revenue_importance: float = 0.10
    ):
        self.receivables_growth_threshold = receivables_growth_threshold
        self.deferred_revenue_importance = deferred_revenue_importance

    def revenue_quality_signals(
        self,
        revenue: pd.Series,
        accounts_receivable: pd.Series,
        deferred_revenue: pd.Series = None
    ) -> pd.DataFrame:
        """
        Analyze revenue quality and recognition patterns.
        """
        signals = pd.DataFrame(index=revenue.index)

        # Revenue growth
        revenue_growth = revenue.pct_change(4)  # YoY
        signals['revenue_growth'] = revenue_growth

        # Receivables growth vs revenue growth
        ar_growth = accounts_receivable.pct_change(4)
        signals['ar_growth'] = ar_growth

        # Warning: receivables growing faster than revenue
        signals['ar_outpacing_revenue'] = (
            (ar_growth > revenue_growth * self.receivables_growth_threshold) &
            (revenue_growth > 0)
        )

        # Days Sales Outstanding
        dso = (accounts_receivable / revenue) * 365
        signals['dso'] = dso
        dso_change = dso.pct_change(4)
        signals['dso_increasing'] = dso_change > 0.10
        signals['dso_decreasing'] = dso_change < -0.10

        # Deferred revenue analysis
        if deferred_revenue is not None:
            dr_ratio = deferred_revenue / revenue
            signals['deferred_revenue_ratio'] = dr_ratio

            dr_growth = deferred_revenue.pct_change(4)
            signals['deferred_growing'] = dr_growth > revenue_growth

            # Positive: deferred revenue growing faster = future revenue secured
            signals['strong_backlog'] = (
                (dr_growth > revenue_growth) & (dr_ratio > self.deferred_revenue_importance)
            )

        return signals

    def channel_stuffing_detection(
        self,
        quarterly_revenue: pd.Series,
        quarterly_receivables: pd.Series,
        prior_year_revenue: pd.Series
    ) -> pd.DataFrame:
        """
        Detect potential channel stuffing patterns.

        Warning signs:
        - Q4 revenue spikes relative to prior quarters
        - Receivables spike at quarter end
        - Large sales to related parties
        """
        signals = pd.DataFrame(index=quarterly_revenue.index)

        # Seasonality-adjusted revenue
        yoy_growth = (quarterly_revenue / prior_year_revenue) - 1
        signals['yoy_growth'] = yoy_growth

        # Quarter-end receivables spike
        ar_to_revenue = quarterly_receivables / quarterly_revenue
        ar_ma = ar_to_revenue.rolling(4).mean()
        signals['ar_spike'] = ar_to_revenue > ar_ma * 1.3

        # Suspicious Q4 pattern
        signals['q4_revenue_spike'] = False  # Requires quarter identification logic

        # Combined warning
        signals['channel_stuffing_risk'] = (
            signals['ar_spike'] & (yoy_growth > 0.20)
        )

        return signals
```

---

## 4. Earnings Persistence

### 4.1 Persistence Analysis

**Signal Logic**:
```python
class EarningsPersistenceSignals:
    """Measure how persistent/sustainable earnings are."""

    def __init__(
        self,
        persistence_threshold: float = 0.6,
        volatility_threshold: float = 0.3
    ):
        self.persistence_threshold = persistence_threshold
        self.volatility_threshold = volatility_threshold

    def earnings_persistence_signals(
        self,
        eps: pd.Series,
        operating_income: pd.Series,
        one_time_items: pd.Series = None
    ) -> pd.DataFrame:
        """
        Analyze earnings persistence and predictability.
        """
        signals = pd.DataFrame(index=eps.index)

        # EPS autocorrelation (persistence measure)
        eps_autocorr = eps.rolling(20).apply(
            lambda x: x.autocorr(lag=4) if len(x) > 4 else np.nan
        )
        signals['eps_persistence'] = eps_autocorr
        signals['high_persistence'] = eps_autocorr > self.persistence_threshold
        signals['low_persistence'] = eps_autocorr < 0.3

        # Earnings volatility
        eps_std = eps.rolling(8).std()
        eps_mean = eps.rolling(8).mean().abs()
        eps_cv = eps_std / eps_mean.replace(0, np.nan)
        signals['earnings_volatility'] = eps_cv
        signals['stable_earnings'] = eps_cv < self.volatility_threshold
        signals['volatile_earnings'] = eps_cv > 0.5

        # Core vs non-core
        if one_time_items is not None:
            one_time_ratio = one_time_items.abs() / operating_income.abs()
            signals['one_time_impact'] = one_time_ratio
            signals['clean_earnings'] = one_time_ratio < 0.10
            signals['noisy_earnings'] = one_time_ratio > 0.30

        # Earnings streak
        positive_streak = (eps > 0).rolling(8).sum()
        signals['positive_streak'] = positive_streak
        signals['consistent_profitability'] = positive_streak == 8

        return signals

    def gaap_vs_adjusted_signals(
        self,
        gaap_eps: pd.Series,
        adjusted_eps: pd.Series
    ) -> pd.DataFrame:
        """
        Compare GAAP earnings to company-adjusted figures.

        Large gaps may indicate aggressive adjustments.
        """
        signals = pd.DataFrame(index=gaap_eps.index)

        # Gap analysis
        gap = adjusted_eps - gaap_eps
        gap_ratio = gap / gaap_eps.abs().replace(0, np.nan)

        signals['adj_gaap_gap'] = gap
        signals['adj_gaap_gap_ratio'] = gap_ratio

        # Warning levels
        signals['minor_adjustment'] = gap_ratio.abs() < 0.10
        signals['moderate_adjustment'] = (
            (gap_ratio.abs() >= 0.10) & (gap_ratio.abs() < 0.25)
        )
        signals['large_adjustment'] = gap_ratio.abs() >= 0.25

        # Persistent large gaps are suspicious
        signals['persistent_large_gap'] = (
            signals['large_adjustment'].rolling(4).sum() >= 3
        )

        # Adjusted always higher = aggressive
        signals['consistently_adjusted_higher'] = (
            (adjusted_eps > gaap_eps).rolling(4).sum() == 4
        )

        return signals
```

---

## 5. Composite Quality Score

### 5.1 Integrated Quality Scoring

```python
class EarningsQualityScorer:
    """
    Composite earnings quality scoring system.
    """

    def __init__(self):
        self.weights = {
            'accruals_score': 0.25,
            'cash_flow_score': 0.25,
            'revenue_score': 0.20,
            'persistence_score': 0.20,
            'adjustment_score': 0.10
        }

    def calculate_quality_score(
        self,
        accrual_signals: pd.DataFrame,
        cash_signals: pd.DataFrame,
        revenue_signals: pd.DataFrame,
        persistence_signals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate composite earnings quality score (0-100).

        Higher score = higher quality earnings.
        """
        scores = pd.DataFrame(index=accrual_signals.index)

        # Accruals score (0-100)
        accruals_score = np.where(
            accrual_signals['high_quality'], 100,
            np.where(accrual_signals['moderate_quality'], 70,
            np.where(accrual_signals['low_quality'], 40, 10))
        )
        scores['accruals_score'] = accruals_score

        # Cash flow score
        cf_score = np.where(
            cash_signals['cash_exceeds_earnings'] & cash_signals['sustained_positive_fcf'], 100,
            np.where(cash_signals['cash_backs_earnings'], 75,
            np.where(cash_signals['fcf_positive'], 50, 25))
        )
        scores['cash_flow_score'] = cf_score

        # Revenue score
        rev_score = np.where(
            revenue_signals.get('ar_outpacing_revenue', False), 40,
            np.where(revenue_signals.get('dso_increasing', False), 60, 85)
        )
        scores['revenue_score'] = rev_score

        # Persistence score
        pers_score = np.where(
            persistence_signals['high_persistence'] & persistence_signals['stable_earnings'], 100,
            np.where(persistence_signals['consistent_profitability'], 75, 50)
        )
        scores['persistence_score'] = pers_score

        # Composite weighted score
        composite = (
            scores['accruals_score'] * self.weights['accruals_score'] +
            scores['cash_flow_score'] * self.weights['cash_flow_score'] +
            scores['revenue_score'] * self.weights['revenue_score'] +
            scores['persistence_score'] * self.weights['persistence_score']
        )

        scores['composite_quality_score'] = composite

        # Quintile ranking
        scores['quality_quintile'] = pd.qcut(
            composite, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop'
        )

        # Simple quality flags
        scores['high_quality_earnings'] = composite >= 75
        scores['quality_concern'] = composite < 50
        scores['quality_warning'] = composite < 30

        return scores

    def generate_quality_signals(
        self,
        financial_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        End-to-end quality signal generation.
        """
        accrual_gen = AccrualSignalGenerator()
        cf_gen = CashFlowQualitySignals()
        rev_gen = RevenueQualitySignals()
        pers_gen = EarningsPersistenceSignals()

        # Generate component signals
        accrual_signals = accrual_gen.calculate_total_accruals(
            financial_data['net_income'],
            financial_data['operating_cash_flow'],
            financial_data['total_assets']
        )

        cf_signals = cf_gen.ocf_quality_signals(
            financial_data['operating_cash_flow'],
            financial_data['net_income'],
            financial_data['capex'],
            financial_data.get('dividends')
        )

        rev_signals = rev_gen.revenue_quality_signals(
            financial_data['revenue'],
            financial_data['accounts_receivable'],
            financial_data.get('deferred_revenue')
        )

        pers_signals = pers_gen.earnings_persistence_signals(
            financial_data['eps'],
            financial_data['operating_income']
        )

        # Calculate composite score
        quality_scores = self.calculate_quality_score(
            accrual_signals,
            cf_signals,
            rev_signals,
            pers_signals
        )

        # Combine all signals
        all_signals = pd.concat([
            accrual_signals,
            cf_signals,
            rev_signals,
            pers_signals,
            quality_scores
        ], axis=1)

        return all_signals
```

---

## Signal Usage Guidelines

### Red Flags to Monitor

| Signal | Severity | Action |
|--------|----------|--------|
| High accruals (>10% of assets) | Warning | Reduce position weight |
| OCF << Net Income | Warning | Investigate cash conversion |
| AR growing faster than revenue | Caution | Monitor DSO trends |
| Persistent GAAP vs adjusted gap | Caution | Review one-time items |
| Volatile earnings | Info | Adjust volatility estimates |

### Integration with Ordinis

```python
# Quality filter for strategy universe
quality_signals = EarningsQualityScorer().generate_quality_signals(fundamentals)

# Only trade high-quality names
quality_universe = quality_signals[
    quality_signals['composite_quality_score'] >= 60
].index

# Avoid quality concerns
excluded = quality_signals[
    quality_signals['quality_warning']
].index
```

---

## Academic References

1. **Sloan (1996)**: "Do Stock Prices Fully Reflect Information in Accruals?"
2. **Dechow, Sloan & Sweeney (1995)**: "Detecting Earnings Management"
3. **Richardson et al. (2005)**: "Accrual Reliability, Earnings Persistence and Stock Prices"
4. **Beneish (1999)**: "The Detection of Earnings Manipulation"
5. **Penman & Zhang (2002)**: "Accounting Conservatism, the Quality of Earnings"
