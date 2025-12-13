# Time Series Fundamentals for Signal Generation

Core time series analysis methods for financial data: stationarity testing, autocorrelation analysis, spectral methods, cointegration, and structural break detection.

---

## Overview

Financial time series exhibit unique properties requiring specialized analysis:

1. **Non-Stationarity**: Prices are non-stationary; returns approximately stationary
2. **Volatility Clustering**: ARCH effects (autocorrelated squared returns)
3. **Fat Tails**: Leptokurtic distributions
4. **Regime Changes**: Structural breaks in dynamics
5. **Spurious Correlation**: Non-stationary series appear correlated

---

## 1. Stationarity Testing

### 1.1 Augmented Dickey-Fuller Test

Tests H0: unit root (non-stationary) vs H1: stationary.

$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{i=1}^{p} \delta_i \Delta y_{t-i} + \epsilon_t$$

Test statistic: $t_\gamma = \hat{\gamma} / SE(\hat{\gamma})$

### 1.2 Python Implementation

```python
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, Tuple, List


class StationarityTester:
    """
    Stationarity tests for financial time series.
    """

    def __init__(self, series: pd.Series):
        """
        Initialize with time series.

        Args:
            series: Pandas Series with datetime index
        """
        self.series = series
        self.n = len(series)

    def adf_test(
        self,
        regression: str = 'c',
        maxlag: int = None,
        autolag: str = 'AIC'
    ) -> Dict:
        """
        Augmented Dickey-Fuller test.

        Args:
            regression: 'c' (constant), 'ct' (constant+trend), 'n' (none)
            maxlag: Maximum lag
            autolag: Lag selection method

        Returns:
            Test results
        """
        result = adfuller(
            self.series.dropna(),
            regression=regression,
            maxlag=maxlag,
            autolag=autolag
        )

        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'stationary': result[1] < 0.05,
            'regression': regression
        }

    def kpss_test(self, regression: str = 'c', nlags: str = 'auto') -> Dict:
        """
        KPSS test (H0: stationary vs H1: unit root).

        Opposite null hypothesis to ADF.
        """
        result = kpss(self.series.dropna(), regression=regression, nlags=nlags)

        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'critical_values': result[3],
            'stationary': result[1] > 0.05  # Fail to reject H0
        }

    def combined_stationarity_test(self) -> Dict:
        """
        Combine ADF and KPSS for robust conclusion.

        Both tests agreeing provides stronger evidence.
        """
        adf = self.adf_test()
        kpss_result = self.kpss_test()

        # Interpretation
        if adf['stationary'] and kpss_result['stationary']:
            conclusion = 'stationary'
            confidence = 'high'
        elif not adf['stationary'] and not kpss_result['stationary']:
            conclusion = 'non-stationary'
            confidence = 'high'
        else:
            conclusion = 'inconclusive'
            confidence = 'low'

        return {
            'adf': adf,
            'kpss': kpss_result,
            'conclusion': conclusion,
            'confidence': confidence
        }

    def make_stationary(self, method: str = 'diff') -> Tuple[pd.Series, int]:
        """
        Transform to stationary series.

        Args:
            method: 'diff', 'log_diff', or 'detrend'

        Returns:
            (stationary_series, order of differencing)
        """
        series = self.series.copy()
        d = 0

        if method == 'log_diff':
            series = np.log(series)

        # Difference until stationary
        max_d = 3
        while d < max_d:
            tester = StationarityTester(series)
            if tester.adf_test()['stationary']:
                break
            series = series.diff().dropna()
            d += 1

        return series, d


# Example
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500)

    # Non-stationary (random walk)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(500) * 0.01)), index=dates)
    # Stationary (returns)
    returns = prices.pct_change().dropna()

    print("Price Series (Random Walk):")
    tester = StationarityTester(prices)
    result = tester.combined_stationarity_test()
    print(f"  Conclusion: {result['conclusion']} ({result['confidence']} confidence)")

    print("\nReturn Series:")
    tester = StationarityTester(returns)
    result = tester.combined_stationarity_test()
    print(f"  Conclusion: {result['conclusion']} ({result['confidence']} confidence)")
```

---

## 2. Autocorrelation Analysis

### 2.1 ACF and PACF

**Autocorrelation Function (ACF)**:
$$\rho_k = \frac{\text{Cov}(y_t, y_{t-k})}{\text{Var}(y_t)}$$

**Partial Autocorrelation Function (PACF)**:
Correlation after removing intermediate lags.

### 2.2 Python Implementation

```python
class AutocorrelationAnalysis:
    """
    Autocorrelation analysis for return predictability.
    """

    def __init__(self, series: pd.Series):
        self.series = series.dropna()
        self.n = len(self.series)

    def compute_acf(self, nlags: int = 20) -> pd.DataFrame:
        """Compute ACF with confidence bands."""
        acf_values = acf(self.series, nlags=nlags, fft=True)
        conf_int = 1.96 / np.sqrt(self.n)

        return pd.DataFrame({
            'lag': range(nlags + 1),
            'acf': acf_values,
            'ci_upper': conf_int,
            'ci_lower': -conf_int,
            'significant': np.abs(acf_values) > conf_int
        })

    def compute_pacf(self, nlags: int = 20) -> pd.DataFrame:
        """Compute PACF."""
        pacf_values = pacf(self.series, nlags=nlags)
        conf_int = 1.96 / np.sqrt(self.n)

        return pd.DataFrame({
            'lag': range(nlags + 1),
            'pacf': pacf_values,
            'ci_upper': conf_int,
            'ci_lower': -conf_int,
            'significant': np.abs(pacf_values) > conf_int
        })

    def ljung_box_test(self, lags: List[int] = None) -> pd.DataFrame:
        """
        Ljung-Box test for autocorrelation.

        H0: No autocorrelation up to lag k.
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox

        if lags is None:
            lags = [5, 10, 20]

        result = acorr_ljungbox(self.series, lags=lags, return_df=True)

        return result

    def detect_mean_reversion(self, significance: float = 0.05) -> Dict:
        """
        Detect mean reversion from negative autocorrelation.
        """
        acf_df = self.compute_acf(nlags=5)

        # Negative lag-1 autocorrelation suggests mean reversion
        lag1_acf = acf_df.loc[1, 'acf']
        lag1_sig = acf_df.loc[1, 'significant']

        return {
            'lag1_acf': lag1_acf,
            'lag1_significant': lag1_sig,
            'mean_reverting': lag1_acf < 0 and lag1_sig,
            'momentum': lag1_acf > 0 and lag1_sig
        }


# Example
if __name__ == "__main__":
    np.random.seed(42)
    returns = pd.Series(np.random.randn(500) * 0.01)

    acf_analyzer = AutocorrelationAnalysis(returns)

    print("ACF Analysis:")
    acf_df = acf_analyzer.compute_acf(nlags=10)
    print(acf_df[['lag', 'acf', 'significant']])

    print("\nLjung-Box Test:")
    lb = acf_analyzer.ljung_box_test()
    print(lb)

    print("\nMean Reversion Detection:")
    mr = acf_analyzer.detect_mean_reversion()
    print(f"  Lag-1 ACF: {mr['lag1_acf']:.4f}")
    print(f"  Mean Reverting: {mr['mean_reverting']}")
```

---

## 3. Cointegration Testing

### 3.1 Engle-Granger Two-Step

1. Regress $y_t$ on $x_t$: $y_t = \alpha + \beta x_t + \epsilon_t$
2. Test residuals for stationarity

### 3.2 Johansen Test

Tests for cointegrating vectors in multivariate systems.

### 3.3 Python Implementation

```python
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen


class CointegrationAnalysis:
    """
    Cointegration testing for pairs trading.
    """

    def __init__(self, series1: pd.Series, series2: pd.Series):
        """
        Initialize with two price series.
        """
        # Align series
        aligned = pd.concat([series1, series2], axis=1).dropna()
        self.y = aligned.iloc[:, 0].values
        self.x = aligned.iloc[:, 1].values
        self.n = len(self.y)

    def engle_granger_test(self) -> Dict:
        """
        Engle-Granger cointegration test.

        Returns:
            Test results with hedge ratio
        """
        # Cointegration test
        coint_stat, p_value, crit_values = coint(self.y, self.x)

        # Estimate hedge ratio via OLS
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools.tools import add_constant

        X = add_constant(self.x)
        model = OLS(self.y, X).fit()
        hedge_ratio = model.params[1]
        intercept = model.params[0]

        # Compute spread
        spread = self.y - hedge_ratio * self.x - intercept

        # Test spread stationarity
        spread_adf = adfuller(spread)

        return {
            'coint_statistic': coint_stat,
            'p_value': p_value,
            'critical_values': dict(zip(['1%', '5%', '10%'], crit_values)),
            'cointegrated': p_value < 0.05,
            'hedge_ratio': hedge_ratio,
            'intercept': intercept,
            'spread_adf_stat': spread_adf[0],
            'spread_adf_pvalue': spread_adf[1],
            'half_life': self._estimate_half_life(spread)
        }

    def _estimate_half_life(self, spread: np.ndarray) -> float:
        """
        Estimate mean reversion half-life.

        HL = -ln(2) / ln(phi) where phi is AR(1) coefficient.
        """
        spread_lag = spread[:-1]
        spread_diff = np.diff(spread)

        # Regress diff on lag
        X = np.column_stack([np.ones(len(spread_lag)), spread_lag])
        beta = np.linalg.lstsq(X, spread_diff, rcond=None)[0]
        phi = 1 + beta[1]

        if 0 < phi < 1:
            half_life = -np.log(2) / np.log(phi)
        else:
            half_life = np.inf

        return half_life

    def johansen_test(self, det_order: int = 0, k_ar_diff: int = 1) -> Dict:
        """
        Johansen cointegration test for multiple series.
        """
        data = np.column_stack([self.y, self.x])
        result = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

        return {
            'trace_stat': result.lr1,
            'trace_crit_90': result.cvt[:, 0],
            'trace_crit_95': result.cvt[:, 1],
            'trace_crit_99': result.cvt[:, 2],
            'eigenvalues': result.eig,
            'eigenvectors': result.evec,
            'n_cointegrating': sum(result.lr1 > result.cvt[:, 1])
        }


# Example
if __name__ == "__main__":
    np.random.seed(42)

    # Generate cointegrated pair
    n = 500
    x = np.cumsum(np.random.randn(n)) + 100
    y = 0.8 * x + np.cumsum(np.random.randn(n) * 0.1) + 20  # Cointegrated

    series1 = pd.Series(y)
    series2 = pd.Series(x)

    coint_analyzer = CointegrationAnalysis(series1, series2)
    result = coint_analyzer.engle_granger_test()

    print("Engle-Granger Cointegration Test:")
    print(f"  Cointegrated: {result['cointegrated']}")
    print(f"  p-value: {result['p_value']:.4f}")
    print(f"  Hedge ratio: {result['hedge_ratio']:.4f}")
    print(f"  Half-life: {result['half_life']:.1f} periods")
```

---

## 4. Structural Break Detection

### 4.1 CUSUM Test

Detects structural breaks in mean:

$$W_t = \sum_{i=1}^{t} (r_i - \bar{r}) / \hat{\sigma}$$

### 4.2 Python Implementation

```python
class StructuralBreakDetector:
    """
    Detect structural breaks in time series.
    """

    def __init__(self, series: pd.Series):
        self.series = series.dropna()
        self.n = len(self.series)

    def cusum_test(self) -> Dict:
        """
        CUSUM test for structural break in mean.
        """
        r = self.series.values
        r_mean = np.mean(r)
        r_std = np.std(r, ddof=1)

        # Cumulative sum of standardized residuals
        cusum = np.cumsum(r - r_mean) / r_std

        # Critical value (simplified)
        crit_value = 1.36 * np.sqrt(self.n)

        # Find potential break point
        break_idx = np.argmax(np.abs(cusum))
        break_detected = np.abs(cusum[break_idx]) > crit_value

        return {
            'cusum': cusum,
            'break_index': break_idx,
            'break_detected': break_detected,
            'max_cusum': np.abs(cusum[break_idx]),
            'critical_value': crit_value
        }

    def rolling_adf(self, window: int = 252) -> pd.Series:
        """
        Rolling ADF test for detecting regime changes.
        """
        adf_stats = []

        for i in range(window, self.n):
            window_data = self.series.iloc[i-window:i]
            try:
                result = adfuller(window_data, maxlag=5, autolag=None)
                adf_stats.append(result[0])
            except:
                adf_stats.append(np.nan)

        return pd.Series(adf_stats, index=self.series.index[window:])

    def detect_variance_breaks(self, window: int = 63) -> pd.DataFrame:
        """
        Detect volatility regime changes.
        """
        rolling_vol = self.series.rolling(window).std()

        # Z-score of volatility
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()
        vol_zscore = (rolling_vol - vol_mean) / vol_std

        return pd.DataFrame({
            'volatility': rolling_vol,
            'vol_zscore': vol_zscore,
            'high_vol_regime': vol_zscore > 2,
            'low_vol_regime': vol_zscore < -2
        })


# Example
if __name__ == "__main__":
    np.random.seed(42)

    # Series with structural break
    n1, n2 = 250, 250
    series = pd.Series(np.concatenate([
        np.random.randn(n1) * 0.01 + 0.001,  # First regime
        np.random.randn(n2) * 0.02 - 0.001   # Second regime (higher vol, negative drift)
    ]))

    detector = StructuralBreakDetector(series)

    cusum = detector.cusum_test()
    print(f"CUSUM Test:")
    print(f"  Break detected: {cusum['break_detected']}")
    print(f"  Break index: {cusum['break_index']} (actual: 250)")

    vol_breaks = detector.detect_variance_breaks()
    print(f"\nVolatility Regimes:")
    print(f"  High vol periods: {vol_breaks['high_vol_regime'].sum()}")
```

---

## 5. Spectral Analysis

### 5.1 Periodogram

Detect cyclical patterns in returns.

```python
class SpectralAnalysis:
    """
    Spectral analysis for cycle detection.
    """

    def __init__(self, series: pd.Series):
        self.series = series.dropna().values
        self.n = len(self.series)

    def compute_periodogram(self) -> pd.DataFrame:
        """
        Compute periodogram.
        """
        from scipy.fft import fft, fftfreq

        # FFT
        yf = fft(self.series - np.mean(self.series))
        xf = fftfreq(self.n, 1)

        # Periodogram (power spectrum)
        power = np.abs(yf) ** 2 / self.n

        # Keep positive frequencies
        pos_mask = xf > 0
        freqs = xf[pos_mask]
        power = power[pos_mask]

        # Convert to periods
        periods = 1 / freqs

        return pd.DataFrame({
            'frequency': freqs,
            'period': periods,
            'power': power
        })

    def dominant_cycles(self, top_n: int = 5) -> pd.DataFrame:
        """
        Find dominant cycles.
        """
        periodogram = self.compute_periodogram()
        return periodogram.nlargest(top_n, 'power')


# Example
if __name__ == "__main__":
    np.random.seed(42)

    # Create signal with cycles
    t = np.arange(500)
    signal = (np.sin(2 * np.pi * t / 21) +  # 21-day cycle
              0.5 * np.sin(2 * np.pi * t / 63) +  # 63-day cycle
              np.random.randn(500) * 0.5)

    spectral = SpectralAnalysis(pd.Series(signal))
    cycles = spectral.dominant_cycles()
    print("Dominant Cycles:")
    print(cycles[['period', 'power']])
```

---

## 6. Academic References

1. **Hamilton, J. D. (1994)**. *Time Series Analysis*. Princeton University Press.

2. **Tsay, R. S. (2010)**. *Analysis of Financial Time Series*. Wiley.

3. **Engle, R. F., & Granger, C. W. (1987)**. "Co-integration and Error Correction." *Econometrica*.

4. **Dickey, D. A., & Fuller, W. A. (1979)**. "Distribution of the Estimators..." *JASA*.

5. **Johansen, S. (1991)**. "Estimation and Hypothesis Testing of Cointegration Vectors." *Econometrica*.

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "published"
tags: ["time-series", "stationarity", "cointegration", "autocorrelation", "structural-breaks", "spectral"]
code_lines: 450
```

---

**END OF DOCUMENT**
