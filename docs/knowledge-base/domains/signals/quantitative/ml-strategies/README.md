# Machine Learning Strategies

## Overview

Machine learning applies data-driven algorithms to generate trading signals, predict returns, or classify market regimes. Unlike traditional quant methods, ML can discover non-linear patterns but requires careful handling to avoid overfitting.

---

## Strategy Types

| File | Approach | Use Case |
|------|----------|----------|
| [signal_classification.md](signal_classification.md) | Classification | Buy/sell signal prediction |
| [return_prediction.md](return_prediction.md) | Regression | Cross-sectional returns |
| [regime_classification.md](regime_classification.md) | Clustering/HMM | Market regime detection |
| [feature_engineering.md](feature_engineering.md) | Preprocessing | Financial feature creation |

---

## ML Pipeline for Trading

```
1. Feature Engineering
   - Technical indicators
   - Fundamental ratios
   - Alternative data

2. Label Creation
   - Forward returns
   - Triple-barrier labels
   - Regime labels

3. Data Splitting
   - Walk-forward validation
   - Purged cross-validation
   - Embargo periods

4. Model Training
   - Algorithm selection
   - Hyperparameter tuning
   - Regularization

5. Backtesting
   - Transaction costs
   - Slippage modeling
   - Out-of-sample testing

6. Deployment
   - Real-time prediction
   - Risk management
   - Performance monitoring
```

---

## Critical Pitfalls

### 1. Overfitting
```python
# Signs of overfitting
OVERFITTING_INDICATORS = {
    'in_sample_vs_oos_gap': 0.30,  # >30% degradation
    'parameter_sensitivity': True,  # Results change dramatically
    'too_many_features': True,      # Features > sqrt(samples)
    'perfect_fit': True             # Training accuracy too high
}

# Prevention
OVERFITTING_PREVENTION = {
    'regularization': 'L1/L2',
    'cross_validation': 'purged_kfold',
    'feature_selection': 'importance_based',
    'early_stopping': True,
    'ensemble_methods': True
}
```

### 2. Look-Ahead Bias
```python
# WRONG: Using future data
df['signal'] = df['return_tomorrow'].shift(-1)  # LOOK-AHEAD BIAS!

# CORRECT: Point-in-time features only
df['signal'] = df['return_yesterday'].shift(1)

# Always verify with:
assert all_features_available_at_signal_time(df)
```

### 3. Non-Stationarity
```python
def check_stationarity(series: pd.Series) -> bool:
    """
    Financial time series are often non-stationary.
    """
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(series.dropna())
    return result[1] < 0.05  # p-value < 0.05 = stationary

# Make features stationary
def stationarize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform features to be stationary.
    """
    transformed = pd.DataFrame()

    for col in df.columns:
        if not check_stationarity(df[col]):
            # Use returns instead of levels
            transformed[col] = df[col].pct_change()
        else:
            transformed[col] = df[col]

    return transformed
```

### 4. Survivorship Bias
```python
# WRONG: Only including currently traded stocks
universe = get_current_sp500()  # Missing delisted stocks!

# CORRECT: Point-in-time universe
universe = get_sp500_constituents(as_of=date)  # Includes delisted
```

---

## Algorithm Selection

| Algorithm | Strengths | Weaknesses | Use Case |
|-----------|-----------|------------|----------|
| Random Forest | Robust, handles non-linear | Slow prediction | Cross-sectional |
| Gradient Boosting | High accuracy | Overfitting risk | Return prediction |
| LASSO | Feature selection | Linear only | Factor models |
| Neural Networks | Non-linear patterns | Black box, overfit | Alternative data |
| SVM | Good with small data | Hard to tune | Classification |
| HMM | Regime modeling | Assumes Markov | Regime detection |

---

## Walk-Forward Validation

```python
class WalkForwardValidator:
    """
    Rolling train/test split for time series.
    """
    def __init__(
        self,
        train_size: int = 252 * 3,  # 3 years training
        test_size: int = 21,        # 1 month test
        step_size: int = 21         # Roll forward 1 month
    ):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def split(self, X: pd.DataFrame):
        """
        Generate train/test splits.
        """
        n = len(X)
        splits = []

        for start in range(0, n - self.train_size - self.test_size, self.step_size):
            train_end = start + self.train_size
            test_end = train_end + self.test_size

            train_idx = range(start, train_end)
            test_idx = range(train_end, test_end)

            splits.append((list(train_idx), list(test_idx)))

        return splits
```

---

## Purged Cross-Validation

```python
class PurgedKFold:
    """
    K-fold CV with purging for overlapping labels.
    Prevents information leakage in time series.
    """
    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        label_end_times: pd.Series
    ):
        """
        Generate purged train/test splits.

        label_end_times: For each sample, when does its label end?
        """
        n = len(X)
        embargo = int(n * self.embargo_pct)

        for fold in range(self.n_splits):
            test_start = int(n / self.n_splits * fold)
            test_end = int(n / self.n_splits * (fold + 1))

            test_idx = list(range(test_start, test_end))

            # Purge: Remove training samples that overlap with test
            train_idx = []
            for i in range(n):
                if i in test_idx:
                    continue

                # Check if this sample's label overlaps with test period
                if label_end_times.iloc[i] >= X.index[test_start]:
                    continue  # Purge this sample

                # Embargo: Skip samples just before test
                if test_start - embargo <= i < test_start:
                    continue

                train_idx.append(i)

            yield train_idx, test_idx
```

---

## Feature Importance

```python
def mean_decrease_impurity(model, feature_names: list) -> pd.Series:
    """
    Feature importance from tree-based models.
    Fast but biased toward high-cardinality features.
    """
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    )
    return importance.sort_values(ascending=False)

def mean_decrease_accuracy(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.Series:
    """
    Permutation importance - more reliable.
    """
    from sklearn.inspection import permutation_importance

    result = permutation_importance(model, X_test, y_test, n_repeats=10)

    importance = pd.Series(
        result.importances_mean,
        index=X_test.columns
    )
    return importance.sort_values(ascending=False)
```

---

## Meta-Labeling

```python
class MetaLabelingStrategy:
    """
    Two-stage approach:
    1. Primary model generates signals
    2. Meta-model predicts if signal is correct
    """
    def __init__(self, primary_model, meta_model):
        self.primary = primary_model
        self.meta = meta_model

    def train_meta_model(
        self,
        X: pd.DataFrame,
        primary_signals: pd.Series,
        actual_returns: pd.Series
    ):
        """
        Train meta-model on primary signal success.
        """
        # Label: Did primary signal make money?
        meta_labels = (primary_signals * actual_returns) > 0

        # Features: Market conditions, signal strength, etc.
        meta_features = self._extract_meta_features(X, primary_signals)

        self.meta.fit(meta_features, meta_labels)

    def generate_signal(self, X: pd.DataFrame) -> tuple:
        """
        Generate signal with confidence.
        """
        # Primary signal
        primary_signal = self.primary.predict(X)

        # Meta-model confidence
        meta_features = self._extract_meta_features(X, primary_signal)
        confidence = self.meta.predict_proba(meta_features)[:, 1]

        return primary_signal, confidence

    def position_size(self, signal: int, confidence: float) -> float:
        """
        Size position based on meta-model confidence.
        """
        if confidence < 0.5:
            return 0  # Skip trade
        elif confidence < 0.6:
            return signal * 0.5
        else:
            return signal * 1.0
```

---

## Ensemble Methods

```python
class EnsembleSignalGenerator:
    """
    Combine multiple ML models for robust signals.
    """
    def __init__(self, models: list, weights: list = None):
        self.models = models
        self.weights = weights or [1/len(models)] * len(models)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Weighted average of model predictions.
        """
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)

        return sum(predictions)

    def predict_with_confidence(self, X: pd.DataFrame) -> tuple:
        """
        Prediction with agreement-based confidence.
        """
        predictions = [model.predict(X) for model in self.models]

        # Mean prediction
        mean_pred = np.mean(predictions, axis=0)

        # Standard deviation = inverse confidence
        std_pred = np.std(predictions, axis=0)
        confidence = 1 / (1 + std_pred)

        return mean_pred, confidence
```

---

## Deflated Sharpe Ratio

```python
def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    sharpe_std: float = 1.0
) -> float:
    """
    Adjust Sharpe for multiple testing.
    After testing many strategies, some look good by chance.
    """
    from scipy.stats import norm

    # Expected maximum Sharpe from N trials under null
    expected_max_sharpe = sharpe_std * (
        (1 - 0.5772) * norm.ppf(1 - 1/n_trials) +
        0.5772 * norm.ppf(1 - 1/(n_trials * np.e))
    )

    # Deflated Sharpe
    deflated = (sharpe - expected_max_sharpe) / sharpe_std * np.sqrt(n_observations)

    return norm.cdf(deflated)  # Probability Sharpe is real
```

---

## Best Practices

1. **Simple models first**: Start with linear models, add complexity only if needed
2. **Feature engineering > algorithm selection**: Good features matter more than fancy algorithms
3. **Walk-forward always**: Never use standard cross-validation
4. **Purge overlapping labels**: Financial labels often span multiple days
5. **Account for costs**: ML turnover can be high
6. **Ensemble for robustness**: Single models are fragile
7. **Monitor for decay**: Models degrade over time

---

## Academic References

- De Prado (2018): "Advances in Financial Machine Learning"
- Bailey & de Prado (2014): "The Deflated Sharpe Ratio"
- Gu, Kelly, Xiu (2020): "Empirical Asset Pricing via Machine Learning"
- Feng, Giglio, Xiu (2020): "Taming the Factor Zoo"
