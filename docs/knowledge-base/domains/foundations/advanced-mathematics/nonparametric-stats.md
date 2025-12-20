# Non-Parametric Statistics for Trading

Non-parametric methods make minimal assumptions about data distributions, providing robust statistical tools for financial analysis where normality assumptions fail.

---

## Overview

Financial returns exhibit fat tails, skewness, and regime changes that violate parametric assumptions. Non-parametric methods provide:

1. **Robust Estimation**: Distribution-free inference
2. **Density Estimation**: KDE for return distributions
3. **Regression**: LOESS/LOWESS for adaptive smoothing
4. **Bootstrap Methods**: Confidence intervals without distributional assumptions
5. **Rank-Based Tests**: Hypothesis testing robust to outliers

---

## 1. Kernel Density Estimation (KDE)

### 1.1 Theory

Estimate probability density $f(x)$ from samples $x_1, \ldots, x_n$:

$$\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

where:
- $K(\cdot)$ is the kernel function (typically Gaussian)
- $h$ is the bandwidth (smoothing parameter)

**Bandwidth Selection**:
- Silverman's rule: $h = 0.9 \cdot \min(\sigma, IQR/1.34) \cdot n^{-1/5}$
- Scott's rule: $h = 1.06 \cdot \sigma \cdot n^{-1/5}$
- Cross-validation for optimal bandwidth

### 1.2 Python Implementation

```python
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional
from dataclasses import dataclass


class AdaptiveKDE:
    """
    Kernel Density Estimation with adaptive bandwidth.

    Optimized for financial return distributions with fat tails.
    """

    def __init__(self, data: np.ndarray, bandwidth: str = 'silverman'):
        """
        Initialize KDE.

        Args:
            data: Sample data
            bandwidth: 'silverman', 'scott', 'cv', or float
        """
        self.data = np.asarray(data)
        self.n = len(data)

        if bandwidth == 'silverman':
            self.h = self._silverman_bandwidth()
        elif bandwidth == 'scott':
            self.h = self._scott_bandwidth()
        elif bandwidth == 'cv':
            self.h = self._cv_bandwidth()
        else:
            self.h = float(bandwidth)

        self.kde = gaussian_kde(data, bw_method=self.h / data.std())

    def _silverman_bandwidth(self) -> float:
        """Silverman's rule of thumb."""
        sigma = self.data.std()
        iqr = np.percentile(self.data, 75) - np.percentile(self.data, 25)
        return 0.9 * min(sigma, iqr / 1.34) * self.n ** (-0.2)

    def _scott_bandwidth(self) -> float:
        """Scott's rule."""
        return 1.06 * self.data.std() * self.n ** (-0.2)

    def _cv_bandwidth(self) -> float:
        """Leave-one-out cross-validation."""
        def cv_score(h):
            if h <= 0:
                return np.inf
            total = 0
            for i in range(self.n):
                xi = self.data[i]
                other = np.delete(self.data, i)
                kde_loo = gaussian_kde(other, bw_method=h / other.std())
                total += np.log(kde_loo(xi)[0] + 1e-10)
            return -total

        result = minimize_scalar(cv_score, bounds=(0.001, 1.0), method='bounded')
        return result.x

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate density at points x."""
        return self.kde(x)

    def cdf(self, x: float) -> float:
        """Cumulative distribution function."""
        return self.kde.integrate_box_1d(-np.inf, x)

    def quantile(self, p: float, tol: float = 1e-6) -> float:
        """Compute quantile via bisection."""
        lo, hi = self.data.min() - 3 * self.h, self.data.max() + 3 * self.h
        while hi - lo > tol:
            mid = (lo + hi) / 2
            if self.cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return mid

    def var(self, alpha: float = 0.05) -> float:
        """Value at Risk (VaR) at confidence level."""
        return -self.quantile(alpha)

    def expected_shortfall(self, alpha: float = 0.05, n_samples: int = 10000) -> float:
        """Expected Shortfall (CVaR) via Monte Carlo."""
        samples = self.kde.resample(n_samples).flatten()
        var = -self.quantile(alpha)
        tail_samples = samples[samples < -var]
        return -tail_samples.mean() if len(tail_samples) > 0 else var


# Example
if __name__ == "__main__":
    np.random.seed(42)
    returns = np.concatenate([
        np.random.normal(0.001, 0.02, 900),
        np.random.normal(-0.05, 0.05, 100)  # Fat tail
    ])

    kde = AdaptiveKDE(returns, bandwidth='silverman')
    print(f"VaR (95%): {kde.var(0.05):.4f}")
    print(f"ES (95%): {kde.expected_shortfall(0.05):.4f}")
```

---

## 2. LOESS/LOWESS Regression

### 2.1 Theory

Locally Weighted Scatterplot Smoothing fits local polynomials:

For each point $x_0$:
1. Select neighbors within bandwidth $h$
2. Weight by distance: $w_i = W\left(\frac{|x_i - x_0|}{h}\right)$
3. Fit weighted least squares: $\min \sum_i w_i (y_i - \beta_0 - \beta_1 x_i)^2$
4. Predict: $\hat{y}(x_0) = \hat{\beta}_0 + \hat{\beta}_1 x_0$

**Tricube Weight Function**:
$$W(u) = (1 - |u|^3)^3 \cdot \mathbf{1}_{|u| < 1}$$

### 2.2 Python Implementation

```python
from statsmodels.nonparametric.smoothers_lowess import lowess


class AdaptiveLOESS:
    """
    LOESS smoother for financial time series.
    """

    def __init__(self, frac: float = 0.3, it: int = 3):
        """
        Initialize LOESS.

        Args:
            frac: Fraction of data for local fit (bandwidth)
            it: Robustifying iterations
        """
        self.frac = frac
        self.it = it

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit LOESS and return smoothed values."""
        result = lowess(y, x, frac=self.frac, it=self.it, return_sorted=False)
        return result

    def smooth_returns(self, returns: pd.Series, frac: float = None) -> pd.Series:
        """Smooth return series."""
        if frac is None:
            frac = self.frac
        x = np.arange(len(returns))
        smoothed = lowess(returns.values, x, frac=frac, return_sorted=False)
        return pd.Series(smoothed, index=returns.index)

    def adaptive_trend(self, prices: pd.Series) -> pd.DataFrame:
        """
        Compute adaptive trend with multiple bandwidths.

        Returns:
            DataFrame with short/medium/long-term trends
        """
        log_prices = np.log(prices)
        x = np.arange(len(prices))

        trends = pd.DataFrame(index=prices.index)
        trends['price'] = prices

        for name, frac in [('short', 0.05), ('medium', 0.15), ('long', 0.3)]:
            smoothed = lowess(log_prices.values, x, frac=frac, return_sorted=False)
            trends[f'{name}_trend'] = np.exp(smoothed)

        return trends


# Example
if __name__ == "__main__":
    dates = pd.date_range('2020-01-01', periods=252)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(252) * 0.02)), index=dates)

    loess = AdaptiveLOESS()
    trends = loess.adaptive_trend(prices)
    print(trends.tail())
```

---

## 3. Bootstrap Methods

### 3.1 Theory

**Standard Bootstrap**: Resample with replacement to estimate sampling distribution.

**Block Bootstrap** (for time series): Resample blocks to preserve autocorrelation.
- Non-overlapping blocks
- Moving block bootstrap
- Stationary bootstrap (random block lengths)

**Confidence Intervals**:
- Percentile: $[\hat{\theta}^*_{\alpha/2}, \hat{\theta}^*_{1-\alpha/2}]$
- BCa (Bias-corrected and accelerated): Adjusts for bias and skewness

### 3.2 Python Implementation

```python
from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap


class FinancialBootstrap:
    """
    Bootstrap methods for financial time series.
    """

    def __init__(self, returns: np.ndarray, block_size: int = None):
        """
        Initialize bootstrap.

        Args:
            returns: Return series
            block_size: Average block size for stationary bootstrap
        """
        self.returns = np.asarray(returns)
        self.n = len(returns)

        if block_size is None:
            # Optimal block size (Politis & White)
            block_size = int(self.n ** (1/3))
        self.block_size = block_size

    def stationary_bootstrap(
        self,
        statistic,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> dict:
        """
        Stationary bootstrap for dependent data.

        Args:
            statistic: Function to compute on each sample
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
        """
        bs = StationaryBootstrap(self.block_size, self.returns)

        estimates = []
        for data, in bs.bootstrap(n_bootstrap):
            estimates.append(statistic(data))

        estimates = np.array(estimates)
        alpha = 1 - confidence

        return {
            'point_estimate': statistic(self.returns),
            'mean': estimates.mean(),
            'std': estimates.std(),
            'ci_lower': np.percentile(estimates, 100 * alpha / 2),
            'ci_upper': np.percentile(estimates, 100 * (1 - alpha / 2)),
            'estimates': estimates
        }

    def sharpe_ratio_ci(
        self,
        risk_free: float = 0.0,
        annualize: int = 252,
        confidence: float = 0.95
    ) -> dict:
        """
        Bootstrap confidence interval for Sharpe ratio.
        """
        def sharpe(r):
            excess = r - risk_free / annualize
            return np.sqrt(annualize) * excess.mean() / excess.std()

        return self.stationary_bootstrap(sharpe, confidence=confidence)

    def max_drawdown_ci(self, confidence: float = 0.95) -> dict:
        """
        Bootstrap CI for maximum drawdown.
        """
        def max_dd(r):
            cum_returns = np.cumprod(1 + r)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = cum_returns / running_max - 1
            return drawdowns.min()

        return self.stationary_bootstrap(max_dd, confidence=confidence)


# Example
if __name__ == "__main__":
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01 + 0.0005

    bs = FinancialBootstrap(returns)

    sharpe_ci = bs.sharpe_ratio_ci()
    print(f"Sharpe Ratio: {sharpe_ci['point_estimate']:.3f}")
    print(f"95% CI: [{sharpe_ci['ci_lower']:.3f}, {sharpe_ci['ci_upper']:.3f}]")

    dd_ci = bs.max_drawdown_ci()
    print(f"\nMax Drawdown: {dd_ci['point_estimate']:.2%}")
    print(f"95% CI: [{dd_ci['ci_lower']:.2%}, {dd_ci['ci_upper']:.2%}]")
```

---

## 4. Rank-Based Methods

### 4.1 Spearman Correlation

Rank correlation robust to outliers:

$$\rho_S = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$

where $d_i$ is the difference in ranks.

### 4.2 Non-Parametric Tests

```python
from scipy import stats


class RankBasedTests:
    """
    Rank-based statistical tests for trading.
    """

    @staticmethod
    def spearman_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Spearman rank correlation with p-value."""
        return stats.spearmanr(x, y)

    @staticmethod
    def kendall_tau(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Kendall's tau correlation."""
        return stats.kendalltau(x, y)

    @staticmethod
    def mann_whitney(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Mann-Whitney U test for comparing two distributions.

        Tests if one distribution stochastically dominates the other.
        """
        return stats.mannwhitneyu(x, y, alternative='two-sided')

    @staticmethod
    def kruskal_wallis(*groups) -> Tuple[float, float]:
        """
        Kruskal-Wallis H test for comparing multiple groups.

        Non-parametric alternative to one-way ANOVA.
        """
        return stats.kruskal(*groups)

    @staticmethod
    def sign_test(x: np.ndarray, y: np.ndarray = None) -> Tuple[float, float]:
        """
        Sign test for median.

        If y provided: paired comparison
        If y=None: test if median of x is zero
        """
        if y is not None:
            diff = x - y
        else:
            diff = x

        n_pos = np.sum(diff > 0)
        n_neg = np.sum(diff < 0)
        n = n_pos + n_neg

        # Binomial test
        p_value = 2 * stats.binom.cdf(min(n_pos, n_neg), n, 0.5)

        return n_pos - n_neg, p_value


# Example
if __name__ == "__main__":
    np.random.seed(42)
    strategy_a = np.random.randn(100) * 0.01 + 0.002
    strategy_b = np.random.randn(100) * 0.01 + 0.001

    tests = RankBasedTests()

    u_stat, p_val = tests.mann_whitney(strategy_a, strategy_b)
    print(f"Mann-Whitney: U={u_stat:.1f}, p={p_val:.4f}")

    sign_stat, p_val = tests.sign_test(strategy_a - strategy_b)
    print(f"Sign Test: stat={sign_stat}, p={p_val:.4f}")
```

---

## 5. Permutation Tests

### 5.1 Theory

Test statistic distribution under null hypothesis via permutation:

1. Compute observed statistic $T_{obs}$
2. Permute labels $B$ times, compute $T_1, \ldots, T_B$
3. p-value = $\frac{1 + \sum_{b=1}^{B} \mathbf{1}_{|T_b| \geq |T_{obs}|}}{B + 1}$

### 5.2 Python Implementation

```python
class PermutationTests:
    """
    Permutation-based hypothesis testing.
    """

    @staticmethod
    def permutation_test(
        x: np.ndarray,
        y: np.ndarray,
        statistic,
        n_permutations: int = 10000
    ) -> Tuple[float, float]:
        """
        Two-sample permutation test.

        Args:
            x, y: Two samples
            statistic: Function(x, y) -> test statistic
            n_permutations: Number of permutations

        Returns:
            (observed_statistic, p_value)
        """
        observed = statistic(x, y)

        combined = np.concatenate([x, y])
        n_x = len(x)

        count = 0
        for _ in range(n_permutations):
            perm = np.random.permutation(combined)
            perm_stat = statistic(perm[:n_x], perm[n_x:])
            if abs(perm_stat) >= abs(observed):
                count += 1

        p_value = (count + 1) / (n_permutations + 1)

        return observed, p_value

    @staticmethod
    def strategy_comparison(
        returns_a: np.ndarray,
        returns_b: np.ndarray,
        n_permutations: int = 10000
    ) -> dict:
        """
        Compare two trading strategies via permutation.
        """
        # Mean difference
        def mean_diff(x, y):
            return x.mean() - y.mean()

        # Sharpe ratio difference
        def sharpe_diff(x, y):
            sr_x = x.mean() / x.std() * np.sqrt(252)
            sr_y = y.mean() / y.std() * np.sqrt(252)
            return sr_x - sr_y

        mean_stat, mean_p = PermutationTests.permutation_test(
            returns_a, returns_b, mean_diff, n_permutations
        )

        sharpe_stat, sharpe_p = PermutationTests.permutation_test(
            returns_a, returns_b, sharpe_diff, n_permutations
        )

        return {
            'mean_difference': mean_stat,
            'mean_p_value': mean_p,
            'sharpe_difference': sharpe_stat,
            'sharpe_p_value': sharpe_p
        }


# Example
if __name__ == "__main__":
    np.random.seed(42)
    strategy_a = np.random.randn(100) * 0.01 + 0.002
    strategy_b = np.random.randn(100) * 0.01 + 0.001

    result = PermutationTests.strategy_comparison(strategy_a, strategy_b)
    print("Strategy Comparison:")
    print(f"  Mean diff: {result['mean_difference']:.5f} (p={result['mean_p_value']:.4f})")
    print(f"  Sharpe diff: {result['sharpe_difference']:.3f} (p={result['sharpe_p_value']:.4f})")
```

---

## 6. Quantile Regression

Non-parametric regression for conditional quantiles:

```python
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg


class QuantileRegressionAnalysis:
    """
    Quantile regression for conditional distribution analysis.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize with features and target.
        """
        self.X = sm.add_constant(X)
        self.y = y

    def fit_quantile(self, q: float) -> dict:
        """Fit quantile regression at quantile q."""
        model = QuantReg(self.y, self.X)
        result = model.fit(q=q)

        return {
            'quantile': q,
            'params': result.params,
            'pvalues': result.pvalues,
            'fitted': result.fittedvalues
        }

    def conditional_var(self, X_new: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Estimate conditional VaR."""
        model = QuantReg(self.y, self.X)
        result = model.fit(q=alpha)

        X_new = sm.add_constant(X_new)
        return -result.predict(X_new)
```

---

## 7. Academic References

1. **Silverman, B. W. (1986)**. *Density Estimation for Statistics and Data Analysis*. Chapman & Hall.

2. **Efron, B., & Tibshirani, R. (1993)**. *An Introduction to the Bootstrap*. Chapman & Hall.

3. **Politis, D. N., & Romano, J. P. (1994)**. "The Stationary Bootstrap." *JASA*, 89(428), 1303-1313.

4. **Cleveland, W. S. (1979)**. "Robust Locally Weighted Regression and Smoothing Scatterplots." *JASA*, 74(368), 829-836.

5. **Koenker, R., & Bassett, G. (1978)**. "Regression Quantiles." *Econometrica*, 46(1), 33-50.

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "published"
tags: ["nonparametric", "kde", "bootstrap", "loess", "rank-tests", "quantile-regression"]
code_lines: 450
```

---

**END OF DOCUMENT**
