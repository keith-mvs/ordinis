# Statistical Foundations for Signal Generation

Rigorous statistical methods for hypothesis testing, multiple comparison correction, and backtest validation in trading signal development.

---

## Overview

Trading signals require statistical validation to distinguish genuine alpha from noise. Key challenges:

1. **Multiple Testing**: Testing many signals inflates false positives
2. **Data Snooping**: Overfitting to historical patterns
3. **Non-Stationarity**: Changing market dynamics
4. **Survivorship Bias**: Missing failed strategies/assets
5. **Look-Ahead Bias**: Using future information

---

## 1. Hypothesis Testing for Signals

### 1.1 Signal Significance Testing

**Null Hypothesis**: Signal has no predictive power (mean return = 0)

**Test Statistic** (t-test for mean):
$$t = \frac{\bar{r} - 0}{s / \sqrt{n}}$$

**Adjusted for Autocorrelation** (Newey-West):
$$SE_{NW} = \sqrt{\frac{1}{n}\left(\hat{\gamma}_0 + 2\sum_{j=1}^{q}\left(1-\frac{j}{q+1}\right)\hat{\gamma}_j\right)}$$

### 1.2 Python Implementation

```python
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from typing import Dict, Tuple, List


class SignalTester:
    """
    Statistical hypothesis testing for trading signals.
    """

    def __init__(self, returns: pd.Series, signal: pd.Series = None):
        """
        Initialize signal tester.

        Args:
            returns: Strategy or signal returns
            signal: Signal values (for conditional testing)
        """
        self.returns = returns
        self.signal = signal
        self.n = len(returns)

    def t_test(self, null_mean: float = 0) -> Dict:
        """
        Standard t-test for mean return.

        Args:
            null_mean: Null hypothesis mean

        Returns:
            Test results
        """
        t_stat, p_value = stats.ttest_1samp(self.returns, null_mean)

        return {
            'mean': self.returns.mean(),
            'std': self.returns.std(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant_5pct': p_value < 0.05,
            'significant_1pct': p_value < 0.01
        }

    def newey_west_test(self, lags: int = None, null_mean: float = 0) -> Dict:
        """
        t-test with Newey-West standard errors.

        Accounts for autocorrelation in returns.
        """
        if lags is None:
            lags = int(4 * (self.n / 100) ** (2/9))

        # Compute autocorrelation-adjusted variance
        r = self.returns - null_mean
        gamma_0 = np.var(r, ddof=1)

        gamma_sum = 0
        for j in range(1, lags + 1):
            weight = 1 - j / (lags + 1)  # Bartlett kernel
            gamma_j = np.cov(r[j:], r[:-j])[0, 1]
            gamma_sum += weight * gamma_j

        nw_variance = (gamma_0 + 2 * gamma_sum) / self.n
        nw_se = np.sqrt(nw_variance)

        t_stat = (self.returns.mean() - null_mean) / nw_se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), self.n - 1))

        return {
            'mean': self.returns.mean(),
            'nw_se': nw_se,
            't_statistic': t_stat,
            'p_value': p_value,
            'lags_used': lags
        }

    def sharpe_ratio_test(self, benchmark_sharpe: float = 0) -> Dict:
        """
        Test if Sharpe ratio is significantly different from benchmark.

        Uses Lo (2002) methodology for SE of Sharpe ratio.
        """
        r = self.returns
        sr = r.mean() / r.std() * np.sqrt(252)  # Annualized

        # Standard error (assuming IID normal - simplified)
        # Full formula includes skewness/kurtosis adjustments
        sr_se = np.sqrt((1 + 0.5 * sr**2) / self.n) * np.sqrt(252)

        t_stat = (sr - benchmark_sharpe) / sr_se
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return {
            'sharpe_ratio': sr,
            'se': sr_se,
            't_statistic': t_stat,
            'p_value': p_value
        }

    def information_coefficient(self) -> Dict:
        """
        Test information coefficient (IC) of signal.

        IC = correlation between signal and forward returns.
        """
        if self.signal is None:
            raise ValueError("Signal required for IC test")

        # Align signal with returns
        aligned = pd.concat([self.signal.shift(1), self.returns], axis=1).dropna()
        signal_vals = aligned.iloc[:, 0]
        return_vals = aligned.iloc[:, 1]

        # Pearson IC
        ic_pearson, p_pearson = stats.pearsonr(signal_vals, return_vals)

        # Spearman IC (rank correlation)
        ic_spearman, p_spearman = stats.spearmanr(signal_vals, return_vals)

        # IC t-stat (assuming ~252 daily observations per year)
        ic_t = ic_pearson * np.sqrt(len(signal_vals) - 2) / np.sqrt(1 - ic_pearson**2)

        return {
            'ic_pearson': ic_pearson,
            'ic_spearman': ic_spearman,
            'ic_t_stat': ic_t,
            'p_value_pearson': p_pearson,
            'p_value_spearman': p_spearman,
            'n_observations': len(signal_vals)
        }


# Example
if __name__ == "__main__":
    np.random.seed(42)
    returns = pd.Series(np.random.randn(500) * 0.01 + 0.0003)

    tester = SignalTester(returns)

    print("T-Test Results:")
    t_result = tester.t_test()
    print(f"  Mean: {t_result['mean']:.5f}")
    print(f"  t-stat: {t_result['t_statistic']:.3f}")
    print(f"  p-value: {t_result['p_value']:.4f}")

    print("\nNewey-West Results:")
    nw_result = tester.newey_west_test()
    print(f"  t-stat: {nw_result['t_statistic']:.3f}")
    print(f"  p-value: {nw_result['p_value']:.4f}")

    print("\nSharpe Ratio Test:")
    sr_result = tester.sharpe_ratio_test()
    print(f"  Sharpe: {sr_result['sharpe_ratio']:.3f}")
    print(f"  p-value: {sr_result['p_value']:.4f}")
```

---

## 2. Multiple Testing Correction

### 2.1 The Problem

Testing $m$ signals at $\alpha = 0.05$:
- Expected false positives: $m \times 0.05$
- With 100 signals: ~5 false "discoveries"

### 2.2 Correction Methods

**Bonferroni** (FWER control):
$$\alpha_{adj} = \alpha / m$$

**Holm-Bonferroni** (Step-down):
Reject if $p_{(i)} \leq \alpha / (m - i + 1)$ for ordered p-values.

**Benjamini-Hochberg** (FDR control):
Reject if $p_{(i)} \leq i \times \alpha / m$.

### 2.3 Python Implementation

```python
from statsmodels.stats.multitest import multipletests


class MultipleTestingCorrector:
    """
    Multiple testing correction for signal screening.
    """

    def __init__(self, p_values: np.ndarray, alpha: float = 0.05):
        """
        Initialize corrector.

        Args:
            p_values: Array of p-values from multiple tests
            alpha: Significance level
        """
        self.p_values = np.asarray(p_values)
        self.alpha = alpha
        self.m = len(p_values)

    def bonferroni(self) -> Dict:
        """Bonferroni correction (FWER control)."""
        adjusted_alpha = self.alpha / self.m
        reject = self.p_values < adjusted_alpha
        adjusted_p = np.minimum(self.p_values * self.m, 1.0)

        return {
            'method': 'bonferroni',
            'reject': reject,
            'adjusted_p': adjusted_p,
            'adjusted_alpha': adjusted_alpha,
            'n_rejected': reject.sum()
        }

    def holm(self) -> Dict:
        """Holm step-down procedure."""
        reject, adjusted_p, _, _ = multipletests(
            self.p_values, alpha=self.alpha, method='holm'
        )
        return {
            'method': 'holm',
            'reject': reject,
            'adjusted_p': adjusted_p,
            'n_rejected': reject.sum()
        }

    def benjamini_hochberg(self) -> Dict:
        """Benjamini-Hochberg FDR control."""
        reject, adjusted_p, _, _ = multipletests(
            self.p_values, alpha=self.alpha, method='fdr_bh'
        )
        return {
            'method': 'benjamini_hochberg',
            'reject': reject,
            'adjusted_p': adjusted_p,
            'n_rejected': reject.sum(),
            'fdr_level': self.alpha
        }

    def compare_methods(self) -> pd.DataFrame:
        """Compare all correction methods."""
        bonf = self.bonferroni()
        holm = self.holm()
        bh = self.benjamini_hochberg()

        return pd.DataFrame({
            'raw_p': self.p_values,
            'bonferroni_reject': bonf['reject'],
            'holm_reject': holm['reject'],
            'bh_reject': bh['reject'],
            'bonferroni_adj_p': bonf['adjusted_p'],
            'holm_adj_p': holm['adjusted_p'],
            'bh_adj_p': bh['adjusted_p']
        })


# Example
if __name__ == "__main__":
    np.random.seed(42)
    # Simulate p-values: some real effects, mostly null
    p_values = np.concatenate([
        np.random.uniform(0.001, 0.02, 5),   # True effects
        np.random.uniform(0, 1, 95)           # Null (uniform)
    ])

    corrector = MultipleTestingCorrector(p_values)

    print("Multiple Testing Correction:")
    print(f"  Raw rejections (p < 0.05): {(p_values < 0.05).sum()}")

    bonf = corrector.bonferroni()
    print(f"  Bonferroni rejections: {bonf['n_rejected']}")

    bh = corrector.benjamini_hochberg()
    print(f"  BH rejections: {bh['n_rejected']}")
```

---

## 3. False Discovery Rate (FDR)

### 3.1 Deflated Sharpe Ratio

Harvey & Liu (2015) - adjust for multiple testing:

$$SR_{deflated} = SR \times \sqrt{\frac{n - 1}{n - 1 + n \times SR^2 \times (trials - 1) / trials_{equiv}}}$$

### 3.2 Probabilistic Sharpe Ratio

$$PSR(SR^*) = \Phi\left(\frac{(SR - SR^*)\sqrt{n-1}}{\sqrt{1 - \gamma_3 SR + \frac{\gamma_4 - 1}{4}SR^2}}\right)$$

where $\gamma_3$ is skewness and $\gamma_4$ is kurtosis.

### 3.3 Python Implementation

```python
class BacktestStatistics:
    """
    Statistical validation for backtest results.
    """

    def __init__(self, returns: np.ndarray):
        """
        Initialize with backtest returns.

        Args:
            returns: Array of returns
        """
        self.returns = np.asarray(returns)
        self.n = len(returns)

        # Moments
        self.mean = np.mean(returns)
        self.std = np.std(returns, ddof=1)
        self.skew = stats.skew(returns)
        self.kurt = stats.kurtosis(returns) + 3  # Excess to raw

        # Sharpe ratio (annualized)
        self.sharpe = self.mean / self.std * np.sqrt(252)

    def probabilistic_sharpe_ratio(self, benchmark_sr: float = 0) -> float:
        """
        Compute PSR - probability that true SR exceeds benchmark.

        Bailey & López de Prado (2012)
        """
        sr = self.sharpe
        n = self.n

        # Adjusted standard error
        se = np.sqrt(
            (1 - self.skew * sr + (self.kurt - 1) / 4 * sr**2) / (n - 1)
        )

        psr = stats.norm.cdf((sr - benchmark_sr) / se)

        return psr

    def deflated_sharpe_ratio(
        self,
        n_trials: int,
        expected_max_sr: float = None
    ) -> Dict:
        """
        Compute deflated Sharpe ratio.

        Adjusts for multiple testing / strategy selection.

        Args:
            n_trials: Number of strategies/parameters tested
            expected_max_sr: Expected max SR under null (defaults to estimation)

        Returns:
            DSR and related statistics
        """
        if expected_max_sr is None:
            # Expected max of n_trials standard normals
            # E[max(Z_1, ..., Z_n)] ≈ sqrt(2 * log(n))
            expected_max_sr = np.sqrt(2 * np.log(n_trials)) * np.sqrt(252 / self.n)

        # DSR calculation
        psr = self.probabilistic_sharpe_ratio(expected_max_sr)

        return {
            'sharpe_ratio': self.sharpe,
            'psr': self.probabilistic_sharpe_ratio(0),
            'deflated_sr': psr,
            'expected_max_sr': expected_max_sr,
            'n_trials': n_trials,
            'significant': psr > 0.95
        }

    def minimum_track_record_length(
        self,
        target_sr: float = 1.0,
        confidence: float = 0.95
    ) -> int:
        """
        Compute minimum track record length for statistical significance.

        Bailey & López de Prado (2012)

        Args:
            target_sr: Target Sharpe ratio to detect
            confidence: Confidence level

        Returns:
            Minimum number of observations needed
        """
        z = stats.norm.ppf(confidence)

        # Simplified formula (assuming normal returns)
        min_n = (z / target_sr * np.sqrt(252))**2 * (
            1 + 0.5 * target_sr**2
        )

        return int(np.ceil(min_n))


# Example
if __name__ == "__main__":
    np.random.seed(42)
    returns = np.random.randn(500) * 0.01 + 0.0005

    stats_checker = BacktestStatistics(returns)

    print(f"Sharpe Ratio: {stats_checker.sharpe:.3f}")
    print(f"PSR (vs 0): {stats_checker.probabilistic_sharpe_ratio(0):.3f}")
    print(f"PSR (vs 0.5): {stats_checker.probabilistic_sharpe_ratio(0.5):.3f}")

    dsr = stats_checker.deflated_sharpe_ratio(n_trials=100)
    print(f"\nDeflated SR (100 trials): {dsr['deflated_sr']:.3f}")
    print(f"Significant: {dsr['significant']}")

    min_length = stats_checker.minimum_track_record_length(target_sr=1.0)
    print(f"\nMin track record for SR=1.0: {min_length} days")
```

---

## 4. Combinatorial Purged Cross-Validation

### 4.1 Problem with Standard CV

Time series data violates IID assumption:
- Temporal leakage between folds
- Overlapping labels (purging needed)
- Non-stationarity

### 4.2 CPCV Implementation

```python
class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation for backtesting.

    López de Prado (2018)
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_gap: int = 5,
        embargo_pct: float = 0.01
    ):
        """
        Initialize CPCV.

        Args:
            n_splits: Number of splits
            n_test_splits: Number of test splits per combination
            purge_gap: Gap between train and test
            embargo_pct: Embargo period as fraction
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

    def split(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices.

        Args:
            X: Feature matrix (for length)

        Returns:
            List of (train_idx, test_idx) tuples
        """
        from itertools import combinations

        n = len(X)
        fold_size = n // self.n_splits
        embargo = int(n * self.embargo_pct)

        # Create fold boundaries
        fold_bounds = [(i * fold_size, (i + 1) * fold_size)
                       for i in range(self.n_splits)]
        fold_bounds[-1] = (fold_bounds[-1][0], n)  # Last fold gets remainder

        splits = []

        # Generate all combinations of test folds
        for test_folds in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = []
            for fold in test_folds:
                start, end = fold_bounds[fold]
                test_idx.extend(range(start, end))

            # Train indices with purging and embargo
            train_idx = []
            for i in range(n):
                # Check if in test
                if i in test_idx:
                    continue

                # Check purge gap
                too_close = False
                for t_idx in test_idx:
                    if abs(i - t_idx) < self.purge_gap:
                        too_close = True
                        break

                # Check embargo
                for fold in test_folds:
                    start, end = fold_bounds[fold]
                    if end <= i < end + embargo:
                        too_close = True
                        break

                if not too_close:
                    train_idx.append(i)

            splits.append((np.array(train_idx), np.array(test_idx)))

        return splits

    def cross_val_score(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        scoring=None
    ) -> np.ndarray:
        """
        Perform CPCV and return scores.
        """
        if scoring is None:
            scoring = lambda m, X, y: m.score(X, y)

        splits = self.split(X)
        scores = []

        for train_idx, test_idx in splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            score = scoring(model, X_test, y_test)
            scores.append(score)

        return np.array(scores)


# Example
if __name__ == "__main__":
    from sklearn.linear_model import Ridge

    np.random.seed(42)
    n = 1000
    X = np.random.randn(n, 5)
    y = X[:, 0] * 0.5 + np.random.randn(n) * 0.1

    cpcv = CombinatorialPurgedCV(n_splits=5, n_test_splits=2)
    splits = cpcv.split(X)

    print(f"Number of CPCV splits: {len(splits)}")
    print(f"First split sizes: train={len(splits[0][0])}, test={len(splits[0][1])}")

    model = Ridge()
    scores = cpcv.cross_val_score(model, X, y)
    print(f"CV scores: mean={scores.mean():.4f}, std={scores.std():.4f}")
```

---

## 5. Academic References

1. **Harvey, C. R., Liu, Y., & Zhu, H. (2016)**. "...and the Cross-Section of Expected Returns." *Review of Financial Studies*.

2. **Bailey, D. H., & López de Prado, M. (2012)**. "The Sharpe Ratio Efficient Frontier." *Journal of Risk*.

3. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Wiley.

4. **Benjamini, Y., & Hochberg, Y. (1995)**. "Controlling the False Discovery Rate." *JRSS-B*.

5. **Newey, W. K., & West, K. D. (1987)**. "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." *Econometrica*.

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "published"
tags: ["hypothesis-testing", "multiple-testing", "fdr", "sharpe-ratio", "cross-validation", "backtest"]
code_lines: 450
```

---

**END OF DOCUMENT**
