# Alpha Research and Generation

## Overview

Alpha research systematically discovers, validates, and monitors return-predictive signals. This document covers **alpha ideation**, **signal construction**, **statistical validation**, and **decay monitoring**.

---

## 1. Alpha Discovery Process

### 1.1 Systematic Alpha Research

**Framework**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple
from scipy import stats
from enum import Enum


class AlphaCategory(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    VALUE = "value"
    QUALITY = "quality"
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    EVENT = "event"
    ALTERNATIVE = "alternative"


@dataclass
class AlphaConfig:
    """Configuration for alpha research."""

    # Testing parameters
    min_observations: int = 252
    significance_level: float = 0.05
    min_ic: float = 0.02
    min_sharpe: float = 0.5

    # Validation
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    test_ratio: float = 0.2

    # Decay monitoring
    decay_lookback: int = 60
    decay_threshold: float = 0.5  # 50% decline from peak


class AlphaResearcher:
    """Systematic alpha discovery and validation."""

    def __init__(self, config: AlphaConfig = None):
        self.config = config or AlphaConfig()
        self.alpha_library = {}

    def register_alpha(
        self,
        name: str,
        category: AlphaCategory,
        compute_func: Callable,
        description: str = ""
    ):
        """
        Register an alpha for testing.
        """
        self.alpha_library[name] = {
            'category': category,
            'compute': compute_func,
            'description': description,
            'validation_results': None
        }

    def compute_alpha(
        self,
        name: str,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        Compute alpha signal from data.
        """
        if name not in self.alpha_library:
            raise ValueError(f"Alpha '{name}' not registered")

        compute_func = self.alpha_library[name]['compute']
        return compute_func(data)

    def generate_alpha_report(
        self,
        name: str,
        alpha_signal: pd.Series,
        forward_returns: pd.Series
    ) -> Dict:
        """
        Generate comprehensive alpha analysis report.
        """
        # Align data
        aligned = pd.DataFrame({
            'signal': alpha_signal,
            'returns': forward_returns
        }).dropna()

        if len(aligned) < self.config.min_observations:
            return {'valid': False, 'reason': 'Insufficient observations'}

        report = {
            'name': name,
            'category': self.alpha_library[name]['category'].value,
            'observations': len(aligned)
        }

        # Information Coefficient
        ic = aligned['signal'].corr(aligned['returns'])
        report['ic'] = ic

        # Rolling IC
        rolling_ic = aligned['signal'].rolling(60).corr(aligned['returns'])
        report['ic_mean'] = rolling_ic.mean()
        report['ic_std'] = rolling_ic.std()
        report['ir'] = report['ic_mean'] / report['ic_std'] if report['ic_std'] > 0 else 0

        # IC t-statistic
        n = len(aligned)
        t_stat = ic * np.sqrt((n - 2) / (1 - ic**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
        report['t_stat'] = t_stat
        report['p_value'] = p_value
        report['significant'] = p_value < self.config.significance_level

        # Decile analysis
        decile_returns = self._decile_analysis(aligned)
        report['decile_spread'] = decile_returns[10] - decile_returns[1]
        report['decile_monotonicity'] = self._check_monotonicity(decile_returns)

        # Turnover estimation
        signal_turnover = aligned['signal'].diff().abs().mean() / aligned['signal'].std()
        report['signal_turnover'] = signal_turnover

        # Stability
        report['ic_stability'] = (rolling_ic > 0).mean()

        # Valid alpha check
        report['valid'] = (
            report['significant'] and
            abs(report['ic']) >= self.config.min_ic and
            report['ir'] >= self.config.min_sharpe * 0.1
        )

        return report

    def _decile_analysis(
        self,
        aligned: pd.DataFrame
    ) -> pd.Series:
        """
        Analyze returns by signal decile.
        """
        aligned['decile'] = pd.qcut(
            aligned['signal'], q=10, labels=range(1, 11), duplicates='drop'
        )
        decile_returns = aligned.groupby('decile')['returns'].mean()
        return decile_returns

    def _check_monotonicity(
        self,
        decile_returns: pd.Series
    ) -> float:
        """
        Check if returns are monotonic with signal strength.
        Returns correlation with rank.
        """
        ranks = pd.Series(range(1, len(decile_returns) + 1))
        return decile_returns.corr(ranks)
```

---

## 2. Common Alpha Templates

### 2.1 Pre-Built Alpha Signals

**Signal Logic**:
```python
class AlphaTemplates:
    """Library of common alpha constructions."""

    @staticmethod
    def momentum_alpha(
        close: pd.Series,
        lookback: int = 252,
        skip: int = 21
    ) -> pd.Series:
        """
        Classic momentum: 12-1 month return.
        """
        total_return = close / close.shift(lookback) - 1
        recent_return = close / close.shift(skip) - 1
        return total_return - recent_return

    @staticmethod
    def mean_reversion_alpha(
        close: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """
        Short-term reversal: negative recent return.
        """
        return -(close / close.shift(lookback) - 1)

    @staticmethod
    def value_alpha(
        price: pd.Series,
        book_value: pd.Series
    ) -> pd.Series:
        """
        Book-to-market ratio.
        """
        return book_value / price

    @staticmethod
    def quality_alpha(
        roe: pd.Series,
        debt_equity: pd.Series,
        accruals: pd.Series
    ) -> pd.Series:
        """
        Composite quality signal.
        """
        roe_z = (roe - roe.mean()) / roe.std()
        de_z = -(debt_equity - debt_equity.mean()) / debt_equity.std()
        acc_z = -(accruals - accruals.mean()) / accruals.std()
        return (roe_z + de_z + acc_z) / 3

    @staticmethod
    def volatility_alpha(
        returns: pd.Series,
        lookback: int = 60
    ) -> pd.Series:
        """
        Low volatility: negative realized volatility.
        """
        return -returns.rolling(lookback).std()

    @staticmethod
    def price_momentum_alpha(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """
        Price momentum relative to range.
        """
        range_position = (close - low.rolling(lookback).min()) / (
            high.rolling(lookback).max() - low.rolling(lookback).min()
        )
        return range_position

    @staticmethod
    def volume_momentum_alpha(
        volume: pd.Series,
        close: pd.Series,
        lookback: int = 20
    ) -> pd.Series:
        """
        Volume-weighted momentum.
        """
        returns = close.pct_change()
        vol_weight = volume / volume.rolling(lookback).mean()
        return (returns * vol_weight).rolling(lookback).sum()

    @staticmethod
    def earnings_momentum_alpha(
        eps: pd.Series,
        eps_estimate: pd.Series
    ) -> pd.Series:
        """
        Standardized unexpected earnings.
        """
        surprise = eps - eps_estimate
        sue = surprise / surprise.rolling(8).std()
        return sue

    @staticmethod
    def analyst_revision_alpha(
        estimate_current: pd.Series,
        estimate_prior: pd.Series
    ) -> pd.Series:
        """
        Analyst estimate revision momentum.
        """
        revision = (estimate_current - estimate_prior) / estimate_prior.abs()
        return revision

    @staticmethod
    def seasonality_alpha(
        close: pd.Series,
        month_of_year: pd.Series
    ) -> pd.Series:
        """
        Historical same-month returns.
        """
        returns = close.pct_change(21)  # Monthly
        monthly_avg = returns.groupby(month_of_year).transform('mean')
        return monthly_avg
```

---

## 3. Alpha Validation Framework

### 3.1 Rigorous Testing

**Validation Logic**:
```python
class AlphaValidator:
    """Validate alpha signals with statistical rigor."""

    def __init__(self, config: AlphaConfig = None):
        self.config = config or AlphaConfig()

    def walk_forward_validation(
        self,
        alpha_signal: pd.Series,
        forward_returns: pd.Series,
        n_splits: int = 5
    ) -> Dict:
        """
        Walk-forward validation of alpha.
        """
        results = {
            'split_ics': [],
            'split_sharpes': [],
            'consistent': True
        }

        n = len(alpha_signal)
        split_size = n // (n_splits + 1)

        for i in range(n_splits):
            train_start = i * split_size
            train_end = train_start + split_size
            test_start = train_end
            test_end = test_start + split_size

            # Test on out-of-sample
            test_signal = alpha_signal.iloc[test_start:test_end]
            test_returns = forward_returns.iloc[test_start:test_end]

            aligned = pd.DataFrame({
                'signal': test_signal,
                'returns': test_returns
            }).dropna()

            if len(aligned) > 20:
                ic = aligned['signal'].corr(aligned['returns'])
                results['split_ics'].append(ic)

                # Simple long-short return
                long_ret = aligned[aligned['signal'] > aligned['signal'].median()]['returns'].mean()
                short_ret = aligned[aligned['signal'] <= aligned['signal'].median()]['returns'].mean()
                ls_ret = long_ret - short_ret
                results['split_sharpes'].append(ls_ret * np.sqrt(252))

        # Check consistency
        ic_sign_consistency = np.mean([ic > 0 for ic in results['split_ics']])
        results['ic_consistency'] = ic_sign_consistency
        results['consistent'] = ic_sign_consistency >= 0.6

        results['avg_oos_ic'] = np.mean(results['split_ics'])
        results['avg_oos_sharpe'] = np.mean(results['split_sharpes'])

        return results

    def multiple_testing_correction(
        self,
        p_values: List[float],
        method: str = 'bonferroni'
    ) -> List[float]:
        """
        Correct for multiple testing.
        """
        n_tests = len(p_values)

        if method == 'bonferroni':
            return [min(p * n_tests, 1.0) for p in p_values]

        elif method == 'holm':
            sorted_idx = np.argsort(p_values)
            corrected = np.zeros(n_tests)
            for i, idx in enumerate(sorted_idx):
                corrected[idx] = min(p_values[idx] * (n_tests - i), 1.0)
            return list(corrected)

        elif method == 'fdr':
            sorted_idx = np.argsort(p_values)
            corrected = np.zeros(n_tests)
            for i, idx in enumerate(sorted_idx):
                corrected[idx] = p_values[idx] * n_tests / (i + 1)
            return list(np.minimum.accumulate(corrected[::-1])[::-1])

        return p_values

    def bootstrap_confidence_interval(
        self,
        alpha_signal: pd.Series,
        forward_returns: pd.Series,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Dict:
        """
        Bootstrap confidence interval for IC.
        """
        aligned = pd.DataFrame({
            'signal': alpha_signal,
            'returns': forward_returns
        }).dropna()

        n = len(aligned)
        bootstrap_ics = []

        for _ in range(n_bootstrap):
            sample_idx = np.random.choice(n, size=n, replace=True)
            sample = aligned.iloc[sample_idx]
            ic = sample['signal'].corr(sample['returns'])
            bootstrap_ics.append(ic)

        lower_pct = (1 - confidence) / 2 * 100
        upper_pct = (1 + confidence) / 2 * 100

        return {
            'ic_estimate': np.mean(bootstrap_ics),
            'ic_std': np.std(bootstrap_ics),
            'ci_lower': np.percentile(bootstrap_ics, lower_pct),
            'ci_upper': np.percentile(bootstrap_ics, upper_pct),
            'significant': np.percentile(bootstrap_ics, lower_pct) > 0
        }

    def regime_stability(
        self,
        alpha_signal: pd.Series,
        forward_returns: pd.Series,
        regime: pd.Series
    ) -> Dict:
        """
        Test alpha stability across market regimes.
        """
        results = {}

        for regime_name in regime.unique():
            mask = regime == regime_name

            regime_signal = alpha_signal[mask]
            regime_returns = forward_returns[mask]

            aligned = pd.DataFrame({
                'signal': regime_signal,
                'returns': regime_returns
            }).dropna()

            if len(aligned) > 20:
                ic = aligned['signal'].corr(aligned['returns'])
                results[regime_name] = {
                    'ic': ic,
                    'observations': len(aligned)
                }

        return results
```

---

## 4. Alpha Decay Monitoring

### 4.1 Decay Detection

**Monitoring Logic**:
```python
class AlphaDecayMonitor:
    """Monitor alpha decay over time."""

    def __init__(self, config: AlphaConfig = None):
        self.config = config or AlphaConfig()

    def calculate_rolling_metrics(
        self,
        alpha_signal: pd.Series,
        forward_returns: pd.Series,
        lookback: int = None
    ) -> pd.DataFrame:
        """
        Calculate rolling alpha metrics.
        """
        lookback = lookback or self.config.decay_lookback

        metrics = pd.DataFrame(index=alpha_signal.index)

        # Rolling IC
        metrics['rolling_ic'] = alpha_signal.rolling(lookback).corr(forward_returns)

        # Rolling IC IR
        ic_mean = metrics['rolling_ic'].rolling(lookback).mean()
        ic_std = metrics['rolling_ic'].rolling(lookback).std()
        metrics['rolling_ir'] = ic_mean / ic_std.replace(0, np.nan)

        # IC trend
        metrics['ic_trend'] = metrics['rolling_ic'].rolling(lookback).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 2 else 0
        )

        # Hit rate
        correct_sign = (alpha_signal * forward_returns) > 0
        metrics['hit_rate'] = correct_sign.rolling(lookback).mean()

        return metrics

    def detect_decay(
        self,
        rolling_metrics: pd.DataFrame
    ) -> Dict:
        """
        Detect if alpha is decaying.
        """
        ic = rolling_metrics['rolling_ic']

        # Peak IC (historical max)
        peak_ic = ic.expanding().max()

        # Current vs peak
        current_vs_peak = ic / peak_ic.replace(0, np.nan)

        # Decay signals
        decay_detected = current_vs_peak < self.config.decay_threshold

        # Time since peak
        peak_date = ic.idxmax()
        days_since_peak = (rolling_metrics.index - peak_date).days

        return {
            'decay_detected': decay_detected.iloc[-1] if len(decay_detected) > 0 else False,
            'current_ic': ic.iloc[-1] if len(ic) > 0 else np.nan,
            'peak_ic': peak_ic.iloc[-1] if len(peak_ic) > 0 else np.nan,
            'decay_ratio': current_vs_peak.iloc[-1] if len(current_vs_peak) > 0 else np.nan,
            'days_since_peak': days_since_peak[-1] if len(days_since_peak) > 0 else np.nan,
            'ic_trend': rolling_metrics['ic_trend'].iloc[-1] if 'ic_trend' in rolling_metrics.columns else np.nan
        }

    def generate_decay_alerts(
        self,
        decay_info: Dict,
        alpha_name: str
    ) -> List[str]:
        """
        Generate alerts based on decay analysis.
        """
        alerts = []

        if decay_info['decay_detected']:
            alerts.append(f"ALERT: {alpha_name} showing significant decay "
                         f"({decay_info['decay_ratio']:.1%} of peak)")

        if decay_info['ic_trend'] < -0.001:
            alerts.append(f"WARNING: {alpha_name} IC trending downward")

        if decay_info['days_since_peak'] > 180:
            alerts.append(f"INFO: {alpha_name} hasn't reached peak IC in "
                         f"{decay_info['days_since_peak']} days")

        return alerts
```

---

## 5. Production Alpha Engine

### 5.1 Integrated Research System

```python
class AlphaResearchEngine:
    """
    Production alpha research and monitoring engine.
    """

    def __init__(self, config: AlphaConfig = None):
        self.config = config or AlphaConfig()
        self.researcher = AlphaResearcher(config)
        self.validator = AlphaValidator(config)
        self.monitor = AlphaDecayMonitor(config)
        self.templates = AlphaTemplates()

        # Register standard alphas
        self._register_standard_alphas()

    def _register_standard_alphas(self):
        """Register standard alpha templates."""
        self.researcher.register_alpha(
            'momentum_12_1',
            AlphaCategory.MOMENTUM,
            lambda d: self.templates.momentum_alpha(d['close']),
            'Classic 12-1 month momentum'
        )

        self.researcher.register_alpha(
            'mean_reversion_20',
            AlphaCategory.MEAN_REVERSION,
            lambda d: self.templates.mean_reversion_alpha(d['close']),
            '20-day mean reversion'
        )

        self.researcher.register_alpha(
            'low_vol',
            AlphaCategory.QUALITY,
            lambda d: self.templates.volatility_alpha(d['returns']),
            'Low volatility'
        )

    def full_alpha_analysis(
        self,
        alpha_name: str,
        data: pd.DataFrame,
        forward_returns: pd.Series
    ) -> Dict:
        """
        Comprehensive alpha analysis.
        """
        # Compute alpha
        alpha_signal = self.researcher.compute_alpha(alpha_name, data)

        # Basic report
        report = self.researcher.generate_alpha_report(
            alpha_name, alpha_signal, forward_returns
        )

        # Walk-forward validation
        validation = self.validator.walk_forward_validation(
            alpha_signal, forward_returns
        )
        report['validation'] = validation

        # Bootstrap CI
        bootstrap = self.validator.bootstrap_confidence_interval(
            alpha_signal, forward_returns
        )
        report['bootstrap'] = bootstrap

        # Decay monitoring
        rolling_metrics = self.monitor.calculate_rolling_metrics(
            alpha_signal, forward_returns
        )
        decay_info = self.monitor.detect_decay(rolling_metrics)
        report['decay'] = decay_info

        # Alerts
        report['alerts'] = self.monitor.generate_decay_alerts(
            decay_info, alpha_name
        )

        return report
```

---

## Signal Usage Guidelines

### Alpha Quality Thresholds

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| IC | 0.02 | 0.05 | 0.10 |
| IR | 0.5 | 1.0 | 2.0 |
| t-stat | 2.0 | 3.0 | 4.0 |
| Consistency | 60% | 75% | 90% |

### Integration with Ordinis

```python
# Alpha research workflow
engine = AlphaResearchEngine()

# Analyze custom alpha
report = engine.full_alpha_analysis(
    'momentum_12_1', market_data, forward_returns
)

if report['valid'] and not report['decay']['decay_detected']:
    # Add to production signals
    production_signals['momentum'] = alpha_signal
```

---

## Academic References

1. **Qian et al. (2007)**: "Quantitative Equity Portfolio Management"
2. **Bailey & De Prado (2014)**: "The Deflated Sharpe Ratio"
3. **Harvey et al. (2016)**: "...and the Cross-Section of Expected Returns"
4. **McLean & Pontiff (2016)**: "Does Academic Research Destroy Stock Return Predictability?"
5. **Hou et al. (2020)**: "Replicating Anomalies"
