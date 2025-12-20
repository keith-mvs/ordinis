# Value-Based Trading Signals

## Overview

Value signals identify securities trading below intrinsic worth based on fundamental metrics. These signals provide **universe filtering**, **ranking systems**, and **mean-reversion opportunities** for systematic strategies.

---

## 1. Price-to-Earnings Signals

### 1.1 Trailing P/E Analysis

**Signal Logic**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from scipy import stats


@dataclass
class PESignalConfig:
    """Configuration for P/E based signals."""

    # Absolute thresholds
    deep_value_threshold: float = 10.0
    value_threshold: float = 15.0
    fair_value_upper: float = 25.0
    growth_premium_threshold: float = 35.0

    # Relative thresholds (vs sector)
    relative_discount: float = 0.20  # 20% below sector median
    relative_premium: float = 0.20   # 20% above sector median

    # Historical percentile
    historical_lookback: int = 252 * 5  # 5 years
    low_percentile: float = 20.0
    high_percentile: float = 80.0


class PESignalGenerator:
    """Generate trading signals from P/E ratio analysis."""

    def __init__(self, config: PESignalConfig = None):
        self.config = config or PESignalConfig()

    def calculate_pe_signals(
        self,
        price: pd.Series,
        eps_ttm: pd.Series,
        sector_median_pe: pd.Series
    ) -> pd.DataFrame:
        """
        Generate P/E based signals.

        Args:
            price: Stock price series
            eps_ttm: Trailing 12-month EPS
            sector_median_pe: Sector median P/E ratio

        Returns:
            DataFrame with signal columns
        """
        pe_ratio = price / eps_ttm.replace(0, np.nan)

        signals = pd.DataFrame(index=price.index)

        # Absolute value signals
        signals['deep_value'] = pe_ratio < self.config.deep_value_threshold
        signals['value'] = pe_ratio < self.config.value_threshold
        signals['fair_value'] = (
            (pe_ratio >= self.config.value_threshold) &
            (pe_ratio <= self.config.fair_value_upper)
        )
        signals['growth_premium'] = pe_ratio > self.config.growth_premium_threshold

        # Relative value signals
        relative_pe = pe_ratio / sector_median_pe
        signals['sector_discount'] = relative_pe < (1 - self.config.relative_discount)
        signals['sector_premium'] = relative_pe > (1 + self.config.relative_premium)

        # Historical percentile signals
        pe_pctl = pe_ratio.rolling(
            self.config.historical_lookback, min_periods=252
        ).apply(lambda x: stats.percentileofscore(x[:-1], x[-1]) if len(x) > 1 else 50)

        signals['historically_cheap'] = pe_pctl < self.config.low_percentile
        signals['historically_expensive'] = pe_pctl > self.config.high_percentile

        # Composite value score
        signals['pe_value_score'] = (
            signals['deep_value'].astype(int) * 3 +
            signals['value'].astype(int) * 2 +
            signals['sector_discount'].astype(int) * 2 +
            signals['historically_cheap'].astype(int) * 1
        )

        return signals

    def pe_mean_reversion_signal(
        self,
        pe_ratio: pd.Series,
        lookback: int = 252
    ) -> pd.Series:
        """
        Generate mean-reversion signal when P/E deviates from historical average.

        Returns:
            Signal strength: negative = cheap (buy), positive = expensive (sell)
        """
        pe_ma = pe_ratio.rolling(lookback, min_periods=60).mean()
        pe_std = pe_ratio.rolling(lookback, min_periods=60).std()

        z_score = (pe_ratio - pe_ma) / pe_std

        # Normalize to -1 to 1 range
        signal = np.clip(z_score / 2, -1, 1)

        return -signal  # Negative z-score = cheap = positive signal
```

### 1.2 Forward P/E Signals

**Signal Logic**:
```python
class ForwardPESignals:
    """Signals based on forward earnings estimates."""

    def __init__(
        self,
        estimate_discount_factor: float = 0.90,
        revision_threshold: float = 0.05
    ):
        self.estimate_discount_factor = estimate_discount_factor
        self.revision_threshold = revision_threshold

    def forward_pe_value_signal(
        self,
        price: pd.Series,
        forward_eps: pd.Series,
        trailing_eps: pd.Series
    ) -> pd.DataFrame:
        """
        Generate forward P/E signals with conservatism adjustment.
        """
        # Apply haircut to analyst estimates (historically optimistic)
        conservative_eps = forward_eps * self.estimate_discount_factor
        forward_pe = price / conservative_eps.replace(0, np.nan)
        trailing_pe = price / trailing_eps.replace(0, np.nan)

        signals = pd.DataFrame(index=price.index)

        # Forward value signal
        signals['forward_value'] = forward_pe < 15

        # Implied growth signal
        implied_growth = (trailing_pe / forward_pe) - 1
        signals['strong_growth_expectation'] = implied_growth > 0.15
        signals['declining_expectation'] = implied_growth < 0

        # PEG ratio signal
        # Note: growth should be annualized percentage
        eps_growth = (forward_eps / trailing_eps - 1) * 100
        peg_ratio = trailing_pe / eps_growth.replace(0, np.nan)

        signals['peg_attractive'] = (peg_ratio > 0) & (peg_ratio < 1.0)
        signals['peg_fair'] = (peg_ratio >= 1.0) & (peg_ratio <= 2.0)
        signals['peg_expensive'] = peg_ratio > 2.0

        return signals

    def estimate_revision_signal(
        self,
        current_estimate: pd.Series,
        prior_estimate: pd.Series
    ) -> pd.DataFrame:
        """
        Generate signals from earnings estimate revisions.
        """
        revision_pct = (current_estimate - prior_estimate) / prior_estimate.abs()

        signals = pd.DataFrame(index=current_estimate.index)

        signals['positive_revision'] = revision_pct > self.revision_threshold
        signals['negative_revision'] = revision_pct < -self.revision_threshold
        signals['revision_magnitude'] = revision_pct

        # Revision momentum
        signals['revision_accelerating'] = revision_pct.diff() > 0

        return signals
```

---

## 2. Enterprise Value Signals

### 2.1 EV/EBITDA Analysis

**Signal Logic**:
```python
@dataclass
class EVSignalConfig:
    """Configuration for EV-based signals."""

    ev_ebitda_deep_value: float = 6.0
    ev_ebitda_value: float = 8.0
    ev_ebitda_fair: float = 12.0
    ev_ebitda_expensive: float = 15.0

    ev_sales_value: float = 1.0
    ev_sales_expensive: float = 5.0

    ev_fcf_value: float = 10.0
    ev_fcf_expensive: float = 20.0


class EVSignalGenerator:
    """Generate signals from Enterprise Value metrics."""

    def __init__(self, config: EVSignalConfig = None):
        self.config = config or EVSignalConfig()

    def calculate_enterprise_value(
        self,
        market_cap: pd.Series,
        total_debt: pd.Series,
        cash: pd.Series,
        minority_interest: pd.Series = None
    ) -> pd.Series:
        """Calculate Enterprise Value."""
        ev = market_cap + total_debt - cash
        if minority_interest is not None:
            ev += minority_interest
        return ev

    def ev_ebitda_signals(
        self,
        ev: pd.Series,
        ebitda: pd.Series,
        sector_median_ev_ebitda: pd.Series
    ) -> pd.DataFrame:
        """
        Generate EV/EBITDA based signals.
        """
        ev_ebitda = ev / ebitda.replace(0, np.nan)

        signals = pd.DataFrame(index=ev.index)

        # Absolute signals
        signals['ev_deep_value'] = ev_ebitda < self.config.ev_ebitda_deep_value
        signals['ev_value'] = ev_ebitda < self.config.ev_ebitda_value
        signals['ev_fair'] = (
            (ev_ebitda >= self.config.ev_ebitda_value) &
            (ev_ebitda <= self.config.ev_ebitda_fair)
        )
        signals['ev_expensive'] = ev_ebitda > self.config.ev_ebitda_expensive

        # Relative to sector
        relative_ev = ev_ebitda / sector_median_ev_ebitda
        signals['ev_sector_discount'] = relative_ev < 0.75
        signals['ev_sector_premium'] = relative_ev > 1.25

        # M&A target screen (cheap with good fundamentals)
        signals['potential_target'] = (
            (ev_ebitda < self.config.ev_ebitda_value) &
            (ebitda > 0)
        )

        return signals

    def ev_fcf_signals(
        self,
        ev: pd.Series,
        free_cash_flow: pd.Series
    ) -> pd.DataFrame:
        """
        Generate EV/FCF yield signals.
        """
        ev_fcf = ev / free_cash_flow.replace(0, np.nan)
        fcf_yield = free_cash_flow / ev

        signals = pd.DataFrame(index=ev.index)

        signals['high_fcf_yield'] = fcf_yield > 0.10  # >10% yield
        signals['moderate_fcf_yield'] = (fcf_yield > 0.05) & (fcf_yield <= 0.10)
        signals['low_fcf_yield'] = (fcf_yield > 0) & (fcf_yield <= 0.05)
        signals['negative_fcf'] = fcf_yield < 0

        signals['ev_fcf_value'] = ev_fcf < self.config.ev_fcf_value
        signals['ev_fcf_expensive'] = ev_fcf > self.config.ev_fcf_expensive

        return signals
```

---

## 3. Book Value Signals

### 3.1 Price-to-Book Analysis

**Signal Logic**:
```python
class PriceBookSignals:
    """Signals based on Price-to-Book ratio."""

    def __init__(
        self,
        deep_value_threshold: float = 0.8,
        value_threshold: float = 1.0,
        fair_threshold: float = 3.0
    ):
        self.deep_value_threshold = deep_value_threshold
        self.value_threshold = value_threshold
        self.fair_threshold = fair_threshold

    def pb_signals(
        self,
        price: pd.Series,
        book_value_per_share: pd.Series,
        roe: pd.Series
    ) -> pd.DataFrame:
        """
        Generate P/B signals with ROE context.

        Note: P/B should be interpreted alongside ROE.
        Low P/B + High ROE = true value
        Low P/B + Low ROE = value trap
        """
        pb_ratio = price / book_value_per_share.replace(0, np.nan)

        signals = pd.DataFrame(index=price.index)

        # Basic P/B signals
        signals['pb_deep_value'] = pb_ratio < self.deep_value_threshold
        signals['pb_value'] = pb_ratio < self.value_threshold
        signals['pb_fair'] = (
            (pb_ratio >= self.value_threshold) &
            (pb_ratio <= self.fair_threshold)
        )

        # ROE-adjusted signals
        # High quality value: cheap + good returns
        signals['quality_value'] = (pb_ratio < 1.5) & (roe > 0.12)

        # Value trap warning: cheap + poor returns
        signals['value_trap_warning'] = (pb_ratio < 1.0) & (roe < 0.05)

        # Premium justified by returns
        signals['justified_premium'] = (pb_ratio > 3.0) & (roe > 0.20)

        # Graham-style net-net
        # (Need additional data: current assets, total liabilities)

        return signals

    def tangible_book_signals(
        self,
        price: pd.Series,
        total_equity: pd.Series,
        intangible_assets: pd.Series,
        goodwill: pd.Series,
        shares_outstanding: pd.Series
    ) -> pd.DataFrame:
        """
        Generate signals from tangible book value.
        """
        tangible_equity = total_equity - intangible_assets - goodwill
        tangible_book_per_share = tangible_equity / shares_outstanding

        ptb_ratio = price / tangible_book_per_share.replace(0, np.nan)

        signals = pd.DataFrame(index=price.index)

        signals['below_tangible_book'] = ptb_ratio < 1.0
        signals['tangible_value'] = ptb_ratio < 1.5

        # Intangible intensity
        intangible_ratio = (intangible_assets + goodwill) / total_equity
        signals['high_intangible'] = intangible_ratio > 0.5
        signals['tangible_focused'] = intangible_ratio < 0.2

        return signals
```

---

## 4. Composite Value Scoring

### 4.1 Multi-Factor Value Score

**Signal Logic**:
```python
class CompositeValueScorer:
    """
    Combine multiple value metrics into composite score.
    """

    def __init__(self):
        self.factor_weights = {
            'pe_percentile': 0.20,
            'pb_percentile': 0.15,
            'ev_ebitda_percentile': 0.20,
            'fcf_yield_percentile': 0.25,
            'dividend_yield_percentile': 0.10,
            'peg_percentile': 0.10
        }

    def calculate_value_score(
        self,
        fundamentals: pd.DataFrame,
        universe: pd.Index
    ) -> pd.Series:
        """
        Calculate composite value score (0-100 scale).

        Higher score = more undervalued

        Args:
            fundamentals: DataFrame with columns for each metric
            universe: Index of securities in comparison universe

        Returns:
            Composite value score
        """
        scores = pd.DataFrame(index=universe)

        # Percentile rank each factor (inverted - lower ratio = higher score)
        if 'pe_ratio' in fundamentals.columns:
            pe = fundamentals.loc[universe, 'pe_ratio']
            pe_positive = pe[pe > 0]
            scores['pe_percentile'] = 100 - pe_positive.rank(pct=True) * 100

        if 'pb_ratio' in fundamentals.columns:
            pb = fundamentals.loc[universe, 'pb_ratio']
            pb_positive = pb[pb > 0]
            scores['pb_percentile'] = 100 - pb_positive.rank(pct=True) * 100

        if 'ev_ebitda' in fundamentals.columns:
            ev_eb = fundamentals.loc[universe, 'ev_ebitda']
            ev_positive = ev_eb[ev_eb > 0]
            scores['ev_ebitda_percentile'] = 100 - ev_positive.rank(pct=True) * 100

        if 'fcf_yield' in fundamentals.columns:
            fcf = fundamentals.loc[universe, 'fcf_yield']
            # Higher FCF yield = higher score (not inverted)
            scores['fcf_yield_percentile'] = fcf.rank(pct=True) * 100

        if 'dividend_yield' in fundamentals.columns:
            div = fundamentals.loc[universe, 'dividend_yield']
            scores['dividend_yield_percentile'] = div.rank(pct=True) * 100

        if 'peg_ratio' in fundamentals.columns:
            peg = fundamentals.loc[universe, 'peg_ratio']
            peg_valid = peg[(peg > 0) & (peg < 5)]  # Filter outliers
            scores['peg_percentile'] = 100 - peg_valid.rank(pct=True) * 100

        # Weighted composite
        composite = pd.Series(0.0, index=universe)
        total_weight = 0

        for factor, weight in self.factor_weights.items():
            if factor in scores.columns:
                composite += scores[factor].fillna(50) * weight
                total_weight += weight

        if total_weight > 0:
            composite = composite / total_weight

        return composite

    def value_quintile_signals(
        self,
        value_scores: pd.Series
    ) -> pd.DataFrame:
        """
        Generate quintile-based signals from value scores.
        """
        quintiles = pd.qcut(value_scores, q=5, labels=False, duplicates='drop')

        signals = pd.DataFrame(index=value_scores.index)

        signals['value_q1_cheapest'] = quintiles == 4
        signals['value_q2'] = quintiles == 3
        signals['value_q3_middle'] = quintiles == 2
        signals['value_q4'] = quintiles == 1
        signals['value_q5_expensive'] = quintiles == 0

        signals['value_score'] = value_scores
        signals['value_quintile'] = quintiles + 1

        return signals
```

---

## 5. Sector-Relative Value

### 5.1 Sector-Adjusted Signals

**Signal Logic**:
```python
class SectorRelativeValue:
    """
    Generate value signals relative to sector peers.
    """

    def __init__(self, discount_threshold: float = 0.20):
        self.discount_threshold = discount_threshold

    def calculate_sector_relative_value(
        self,
        metrics: pd.DataFrame,
        sector_map: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate sector-relative value metrics.

        Args:
            metrics: DataFrame with value metrics (pe, pb, ev_ebitda, etc.)
            sector_map: Series mapping ticker to sector

        Returns:
            DataFrame with relative value metrics
        """
        relative = pd.DataFrame(index=metrics.index)

        for metric in ['pe_ratio', 'pb_ratio', 'ev_ebitda', 'ev_sales']:
            if metric not in metrics.columns:
                continue

            # Calculate sector median
            sector_medians = metrics.groupby(sector_map)[metric].transform('median')

            # Relative to sector
            relative[f'{metric}_relative'] = metrics[metric] / sector_medians

            # Discount/premium
            relative[f'{metric}_discount'] = (
                relative[f'{metric}_relative'] < (1 - self.discount_threshold)
            )
            relative[f'{metric}_premium'] = (
                relative[f'{metric}_relative'] > (1 + self.discount_threshold)
            )

        # Composite sector-relative score
        discount_cols = [c for c in relative.columns if c.endswith('_discount')]
        relative['sector_discount_count'] = relative[discount_cols].sum(axis=1)

        return relative

    def identify_sector_rotations(
        self,
        sector_returns: pd.DataFrame,
        sector_valuations: pd.DataFrame,
        lookback: int = 63
    ) -> pd.DataFrame:
        """
        Identify sector rotation opportunities based on value + momentum.
        """
        signals = pd.DataFrame(index=sector_returns.columns)

        # Sector momentum
        momentum = sector_returns.rolling(lookback).sum().iloc[-1]
        signals['momentum_rank'] = momentum.rank(ascending=False)

        # Sector valuation (use average EV/EBITDA)
        avg_valuation = sector_valuations.mean()
        signals['valuation_rank'] = avg_valuation.rank()  # Lower = cheaper

        # Combined score (cheap + positive momentum)
        signals['rotation_score'] = (
            (10 - signals['valuation_rank']) +  # Cheaper is better
            (10 - signals['momentum_rank'])     # Higher momentum is better
        ) / 2

        # Signals
        signals['overweight'] = signals['rotation_score'] > signals['rotation_score'].quantile(0.75)
        signals['underweight'] = signals['rotation_score'] < signals['rotation_score'].quantile(0.25)

        return signals
```

---

## 6. Value Signal Integration

### 6.1 Production Signal Generator

```python
class ValueSignalEngine:
    """
    Production-ready value signal generation engine.
    """

    def __init__(self):
        self.pe_generator = PESignalGenerator()
        self.ev_generator = EVSignalGenerator()
        self.pb_generator = PriceBookSignals()
        self.composite_scorer = CompositeValueScorer()
        self.sector_relative = SectorRelativeValue()

    def generate_all_signals(
        self,
        market_data: pd.DataFrame,
        fundamental_data: pd.DataFrame,
        sector_map: pd.Series
    ) -> pd.DataFrame:
        """
        Generate comprehensive value signals.

        Args:
            market_data: Price and volume data
            fundamental_data: Financial statement data
            sector_map: Ticker to sector mapping

        Returns:
            DataFrame with all value signals
        """
        signals = pd.DataFrame(index=market_data.index)

        # P/E signals
        if all(c in fundamental_data.columns for c in ['eps_ttm', 'sector_median_pe']):
            pe_signals = self.pe_generator.calculate_pe_signals(
                market_data['close'],
                fundamental_data['eps_ttm'],
                fundamental_data['sector_median_pe']
            )
            signals = pd.concat([signals, pe_signals], axis=1)

        # EV signals
        if all(c in fundamental_data.columns for c in ['market_cap', 'total_debt', 'cash', 'ebitda']):
            ev = self.ev_generator.calculate_enterprise_value(
                fundamental_data['market_cap'],
                fundamental_data['total_debt'],
                fundamental_data['cash']
            )
            ev_signals = self.ev_generator.ev_ebitda_signals(
                ev,
                fundamental_data['ebitda'],
                fundamental_data.get('sector_median_ev_ebitda', ev / fundamental_data['ebitda'])
            )
            signals = pd.concat([signals, ev_signals], axis=1)

        # P/B signals
        if all(c in fundamental_data.columns for c in ['book_value_per_share', 'roe']):
            pb_signals = self.pb_generator.pb_signals(
                market_data['close'],
                fundamental_data['book_value_per_share'],
                fundamental_data['roe']
            )
            signals = pd.concat([signals, pb_signals], axis=1)

        # Composite value score
        composite_score = self.composite_scorer.calculate_value_score(
            fundamental_data,
            market_data.index
        )
        quintile_signals = self.composite_scorer.value_quintile_signals(composite_score)
        signals = pd.concat([signals, quintile_signals], axis=1)

        return signals

    def get_value_universe(
        self,
        signals: pd.DataFrame,
        min_value_score: float = 70
    ) -> pd.Index:
        """
        Get universe of value stocks based on signals.
        """
        if 'value_score' in signals.columns:
            return signals[signals['value_score'] >= min_value_score].index
        elif 'pe_value_score' in signals.columns:
            return signals[signals['pe_value_score'] >= 3].index
        else:
            return signals.index
```

---

## Signal Usage Guidelines

### Best Practices

1. **Sector Context**: Always compare within sectors; tech P/E differs from utilities
2. **Quality Filter**: Combine value with quality to avoid value traps
3. **Catalyst Awareness**: Value alone doesn't trigger mean reversion
4. **Time Horizon**: Value signals work over months/years, not days
5. **Data Quality**: Use point-in-time data to avoid lookahead bias

### Integration with Ordinis

```python
# Example integration in signal pipeline
from src.signals.fundamental import ValueSignalEngine

engine = ValueSignalEngine()
signals = engine.generate_all_signals(market_data, fundamentals, sectors)

# Filter to value universe
value_stocks = engine.get_value_universe(signals, min_value_score=70)

# Combine with technical timing
final_signals = signals.loc[value_stocks] & technical_signals.loc[value_stocks]
```

---

## Academic References

1. **Fama & French (1992)**: "The Cross-Section of Expected Stock Returns"
2. **Lakonishok, Shleifer & Vishny (1994)**: "Contrarian Investment, Extrapolation, and Risk"
3. **Piotroski (2000)**: "Value Investing: The Use of Historical Financial Statement Information"
4. **Greenblatt (2006)**: "The Little Book That Beats the Market" - Magic Formula
5. **Asness et al. (2013)**: "Value and Momentum Everywhere"
