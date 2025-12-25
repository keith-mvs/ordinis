# Feature Engineering Mathematics

Mathematical foundations for transforming raw market data into predictive features for trading signals.

---

## Overview

Feature engineering transforms raw data into informative inputs:

1. **Technical Indicator Math**: Rigorous derivations
2. **Transformations**: Normalization, scaling, encoding
3. **Dimensionality Reduction**: PCA, factor analysis
4. **Feature Selection**: Mutual information, importance metrics
5. **Interaction Features**: Cross-asset relationships

---

## 1. Technical Indicator Mathematics

### 1.1 Moving Average Foundations

**Simple Moving Average (SMA)**:
$$SMA_n(t) = \frac{1}{n}\sum_{i=0}^{n-1} P_{t-i}$$

**Exponential Moving Average (EMA)**:
$$EMA_n(t) = \alpha P_t + (1-\alpha) EMA_n(t-1), \quad \alpha = \frac{2}{n+1}$$

**Weighted Moving Average (WMA)**:
$$WMA_n(t) = \frac{\sum_{i=0}^{n-1} (n-i) P_{t-i}}{\sum_{i=0}^{n-1}(n-i)} = \frac{\sum_{i=0}^{n-1} (n-i) P_{t-i}}{n(n+1)/2}$$

### 1.2 Python Implementation

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, List, Optional


class TechnicalIndicators:
    """
    Mathematical implementations of technical indicators.
    """

    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average."""
        return prices.rolling(window).mean()

    @staticmethod
    def ema(prices: pd.Series, span: int) -> pd.Series:
        """Exponential Moving Average."""
        return prices.ewm(span=span, adjust=False).mean()

    @staticmethod
    def wma(prices: pd.Series, window: int) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, window + 1)
        return prices.rolling(window).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        )

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.

        RSI = 100 - 100 / (1 + RS)
        RS = avg_gain / avg_loss
        """
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)

        return rsi

    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD: Moving Average Convergence Divergence.
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })

    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """
        Bollinger Bands.

        Middle = SMA
        Upper = SMA + num_std * std
        Lower = SMA - num_std * std
        """
        middle = prices.rolling(window).mean()
        std = prices.rolling(window).std()

        return pd.DataFrame({
            'middle': middle,
            'upper': middle + num_std * std,
            'lower': middle - num_std * std,
            'bandwidth': (middle + num_std * std - (middle - num_std * std)) / middle,
            'pct_b': (prices - (middle - num_std * std)) / (2 * num_std * std)
        })

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range.

        TR = max(H-L, |H-Cp|, |L-Cp|)
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr


# Example
if __name__ == "__main__":
    np.random.seed(42)
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(200) * 0.01)))

    ti = TechnicalIndicators()

    print("Technical Indicators:")
    print(f"  SMA(20): {ti.sma(prices, 20).iloc[-1]:.2f}")
    print(f"  EMA(20): {ti.ema(prices, 20).iloc[-1]:.2f}")
    print(f"  RSI(14): {ti.rsi(prices, 14).iloc[-1]:.2f}")
```

---

## 2. Feature Transformations

### 2.1 Normalization Methods

```python
class FeatureTransformations:
    """
    Feature transformation methods.
    """

    @staticmethod
    def z_score(x: pd.Series, window: int = None) -> pd.Series:
        """
        Z-score normalization.

        z = (x - mean) / std
        """
        if window:
            mean = x.rolling(window).mean()
            std = x.rolling(window).std()
        else:
            mean = x.mean()
            std = x.std()

        return (x - mean) / std

    @staticmethod
    def min_max_scale(x: pd.Series, window: int = None) -> pd.Series:
        """
        Min-max scaling to [0, 1].
        """
        if window:
            min_val = x.rolling(window).min()
            max_val = x.rolling(window).max()
        else:
            min_val = x.min()
            max_val = x.max()

        return (x - min_val) / (max_val - min_val + 1e-10)

    @staticmethod
    def percentile_rank(x: pd.Series, window: int = 252) -> pd.Series:
        """
        Rolling percentile rank.
        """
        def pct_rank(arr):
            return stats.percentileofscore(arr[:-1], arr[-1]) / 100

        return x.rolling(window).apply(pct_rank, raw=True)

    @staticmethod
    def log_transform(x: pd.Series) -> pd.Series:
        """Log transformation for skewed data."""
        return np.log1p(x.clip(lower=0))

    @staticmethod
    def box_cox(x: pd.Series, lmbda: float = None) -> Tuple[pd.Series, float]:
        """
        Box-Cox transformation.
        """
        from scipy.stats import boxcox

        # Ensure positive values
        x_pos = x - x.min() + 1

        if lmbda is None:
            transformed, lmbda = boxcox(x_pos)
        else:
            if lmbda == 0:
                transformed = np.log(x_pos)
            else:
                transformed = (x_pos ** lmbda - 1) / lmbda

        return pd.Series(transformed, index=x.index), lmbda

    @staticmethod
    def winsorize(x: pd.Series, limits: Tuple[float, float] = (0.01, 0.99)) -> pd.Series:
        """
        Winsorize outliers.
        """
        lower = x.quantile(limits[0])
        upper = x.quantile(limits[1])
        return x.clip(lower=lower, upper=upper)


# Example
if __name__ == "__main__":
    np.random.seed(42)
    returns = pd.Series(np.random.randn(500) * 0.02)

    ft = FeatureTransformations()

    print("Feature Transformations:")
    print(f"  Z-score range: [{ft.z_score(returns).min():.2f}, {ft.z_score(returns).max():.2f}]")
    print(f"  Min-max range: [{ft.min_max_scale(returns).min():.2f}, {ft.min_max_scale(returns).max():.2f}]")
```

---

## 3. Dimensionality Reduction

### 3.1 PCA for Features

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class DimensionalityReduction:
    """
    Dimensionality reduction for feature matrices.
    """

    def __init__(self, n_components: int = None, variance_threshold: float = 0.95):
        """
        Initialize reducer.

        Args:
            n_components: Fixed number of components
            variance_threshold: Cumulative variance to retain
        """
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = None

    def fit_pca(self, X: np.ndarray) -> Dict:
        """
        Fit PCA to feature matrix.
        """
        X_scaled = self.scaler.fit_transform(X)

        if self.n_components is None:
            # Find components for variance threshold
            pca_full = PCA().fit(X_scaled)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            n_comp = np.searchsorted(cumvar, self.variance_threshold) + 1
        else:
            n_comp = self.n_components

        self.pca = PCA(n_components=n_comp)
        X_pca = self.pca.fit_transform(X_scaled)

        return {
            'n_components': n_comp,
            'explained_variance': self.pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_),
            'loadings': self.pca.components_,
            'transformed': X_pca
        }

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data."""
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def factor_loadings(self, feature_names: List[str]) -> pd.DataFrame:
        """Get interpretable factor loadings."""
        loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=feature_names
        )
        return loadings


# Example
if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(500, 20)  # 20 features

    reducer = DimensionalityReduction(variance_threshold=0.90)
    result = reducer.fit_pca(X)

    print("PCA Results:")
    print(f"  Components for 90% variance: {result['n_components']}")
    print(f"  Explained variance: {result['explained_variance'][:5]}")
```

---

## 4. Feature Selection

### 4.1 Mutual Information

```python
from sklearn.feature_selection import mutual_info_regression


class FeatureSelection:
    """
    Feature selection methods for signal generation.
    """

    @staticmethod
    def mutual_information(X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> pd.DataFrame:
        """
        Compute mutual information between features and target.
        """
        mi_scores = mutual_info_regression(X, y, random_state=42)

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        return pd.DataFrame({
            'feature': feature_names,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

    @staticmethod
    def correlation_filter(X: pd.DataFrame, threshold: float = 0.8) -> List[str]:
        """
        Remove highly correlated features.
        """
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        return [col for col in X.columns if col not in to_drop]

    @staticmethod
    def variance_threshold(X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """
        Remove low-variance features.
        """
        variances = X.var()
        return list(variances[variances > threshold].index)

    @staticmethod
    def forward_selection(
        X: pd.DataFrame,
        y: pd.Series,
        max_features: int = 10,
        scoring: str = 'neg_mean_squared_error'
    ) -> List[str]:
        """
        Forward feature selection.
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        selected = []
        remaining = list(X.columns)

        for _ in range(max_features):
            best_score = -np.inf
            best_feature = None

            for feature in remaining:
                features_to_try = selected + [feature]
                X_subset = X[features_to_try]

                model = LinearRegression()
                score = cross_val_score(model, X_subset, y, cv=5, scoring=scoring).mean()

                if score > best_score:
                    best_score = score
                    best_feature = feature

            if best_feature:
                selected.append(best_feature)
                remaining.remove(best_feature)
            else:
                break

        return selected


# Example
if __name__ == "__main__":
    np.random.seed(42)

    # Create features with varying importance
    n = 500
    X = pd.DataFrame({
        'important1': np.random.randn(n),
        'important2': np.random.randn(n),
        'noise1': np.random.randn(n) * 0.1,
        'noise2': np.random.randn(n) * 0.1,
    })
    y = 0.5 * X['important1'] + 0.3 * X['important2'] + np.random.randn(n) * 0.1

    fs = FeatureSelection()

    mi_scores = fs.mutual_information(X.values, y.values, list(X.columns))
    print("Mutual Information Scores:")
    print(mi_scores)
```

---

## 5. Feature Interactions

### 5.1 Cross-Asset Features

```python
class CrossAssetFeatures:
    """
    Generate cross-asset interaction features.
    """

    @staticmethod
    def relative_strength(asset1: pd.Series, asset2: pd.Series, window: int = 20) -> pd.Series:
        """
        Relative strength: ratio of returns.
        """
        ret1 = asset1.pct_change(window)
        ret2 = asset2.pct_change(window)
        return ret1 - ret2

    @staticmethod
    def beta(asset: pd.Series, benchmark: pd.Series, window: int = 60) -> pd.Series:
        """
        Rolling beta to benchmark.
        """
        asset_ret = asset.pct_change()
        bench_ret = benchmark.pct_change()

        cov = asset_ret.rolling(window).cov(bench_ret)
        var = bench_ret.rolling(window).var()

        return cov / var

    @staticmethod
    def correlation_feature(asset1: pd.Series, asset2: pd.Series, window: int = 60) -> pd.Series:
        """
        Rolling correlation.
        """
        return asset1.rolling(window).corr(asset2)

    @staticmethod
    def spread_zscore(asset1: pd.Series, asset2: pd.Series, hedge_ratio: float = 1.0, window: int = 60) -> pd.Series:
        """
        Z-score of spread for pairs trading.
        """
        spread = asset1 - hedge_ratio * asset2
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        return (spread - mean) / std


# Example
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=500)
    asset1 = pd.Series(100 * np.exp(np.cumsum(np.random.randn(500) * 0.01)), index=dates)
    asset2 = pd.Series(100 * np.exp(np.cumsum(np.random.randn(500) * 0.01 + 0.0001)), index=dates)

    caf = CrossAssetFeatures()

    print("Cross-Asset Features:")
    print(f"  Relative strength: {caf.relative_strength(asset1, asset2).iloc[-1]:.4f}")
    print(f"  Rolling beta: {caf.beta(asset1, asset2).iloc[-1]:.4f}")
    print(f"  Rolling correlation: {caf.correlation_feature(asset1, asset2).iloc[-1]:.4f}")
```

---

## 6. Academic References

1. **López de Prado, M. (2018)**. "Feature Importance." *Advances in Financial Machine Learning*.

2. **Cover, T. M., & Thomas, J. A. (2006)**. *Elements of Information Theory*. Wiley.

3. **Murphy, J. J. (1999)**. *Technical Analysis of the Financial Markets*. NYIF.

4. **Guyon, I., & Elisseeff, A. (2003)**. "An Introduction to Variable and Feature Selection." *JMLR*.

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "published"
tags: ["feature-engineering", "technical-indicators", "normalization", "pca", "feature-selection"]
code_lines: 400
```

---

**END OF DOCUMENT**
