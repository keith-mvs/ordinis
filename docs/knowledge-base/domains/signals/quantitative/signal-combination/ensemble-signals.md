# Signal Combination and Ensemble Methods

## Overview

Signal combination transforms multiple individual signals into unified trading decisions. These methods provide **noise reduction**, **signal confirmation**, and **portfolio-level optimization** through systematic aggregation.

---

## 1. Linear Combination Methods

### 1.1 Weighted Signal Averaging

**Signal Logic**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from scipy import stats, optimize
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler


@dataclass
class SignalCombinationConfig:
    """Configuration for signal combination."""

    # Weighting schemes
    equal_weight: bool = False
    ic_weighted: bool = True
    vol_adjusted: bool = True

    # Normalization
    normalize_signals: bool = True
    winsorize_threshold: float = 3.0  # Standard deviations

    # Lookback for optimization
    weight_lookback: int = 252


class LinearSignalCombiner:
    """Combine signals using linear weighting methods."""

    def __init__(self, config: SignalCombinationConfig = None):
        self.config = config or SignalCombinationConfig()
        self.scaler = StandardScaler()

    def equal_weight_combination(
        self,
        signals: pd.DataFrame
    ) -> pd.Series:
        """
        Simple equal-weighted average of all signals.
        """
        if self.config.normalize_signals:
            signals = self._normalize_signals(signals)

        return signals.mean(axis=1)

    def ic_weighted_combination(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        lookback: int = None
    ) -> pd.Series:
        """
        Weight signals by their Information Coefficient (IC).

        IC = correlation between signal and forward returns
        """
        lookback = lookback or self.config.weight_lookback

        if self.config.normalize_signals:
            signals = self._normalize_signals(signals)

        # Calculate rolling IC for each signal
        ic_weights = pd.DataFrame(index=signals.index, columns=signals.columns)

        for col in signals.columns:
            rolling_ic = signals[col].rolling(lookback).corr(forward_returns)
            ic_weights[col] = rolling_ic.clip(0)  # Only use positive IC

        # Normalize weights to sum to 1
        ic_sum = ic_weights.sum(axis=1).replace(0, 1)
        ic_weights = ic_weights.div(ic_sum, axis=0)

        # Weighted combination
        combined = (signals * ic_weights).sum(axis=1)

        return combined

    def volatility_adjusted_combination(
        self,
        signals: pd.DataFrame,
        lookback: int = 60
    ) -> pd.Series:
        """
        Weight signals inversely by their volatility.

        Lower volatility signals get higher weight.
        """
        if self.config.normalize_signals:
            signals = self._normalize_signals(signals)

        # Calculate signal volatility
        signal_vol = signals.rolling(lookback).std()

        # Inverse volatility weights
        inv_vol = 1 / signal_vol.replace(0, np.nan)
        vol_weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

        # Weighted combination
        combined = (signals * vol_weights).sum(axis=1)

        return combined

    def sharpe_weighted_combination(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        lookback: int = None
    ) -> pd.Series:
        """
        Weight signals by their contribution to Sharpe ratio.
        """
        lookback = lookback or self.config.weight_lookback

        if self.config.normalize_signals:
            signals = self._normalize_signals(signals)

        # Calculate signal-return correlations and volatilities
        signal_sharpes = pd.DataFrame(index=signals.index, columns=signals.columns)

        for col in signals.columns:
            rolling_ic = signals[col].rolling(lookback).corr(forward_returns)
            rolling_vol = signals[col].rolling(lookback).std()
            signal_sharpes[col] = rolling_ic / rolling_vol.replace(0, np.nan)

        # Normalize to positive weights
        sharpe_weights = signal_sharpes.clip(lower=0)
        weight_sum = sharpe_weights.sum(axis=1).replace(0, 1)
        sharpe_weights = sharpe_weights.div(weight_sum, axis=0)

        combined = (signals * sharpe_weights).sum(axis=1)

        return combined

    def _normalize_signals(
        self,
        signals: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Normalize signals to z-scores with winsorization.
        """
        # Rolling z-score normalization
        normalized = pd.DataFrame(index=signals.index, columns=signals.columns)

        for col in signals.columns:
            rolling_mean = signals[col].rolling(60).mean()
            rolling_std = signals[col].rolling(60).std()

            z_score = (signals[col] - rolling_mean) / rolling_std.replace(0, np.nan)

            # Winsorize
            z_score = z_score.clip(
                -self.config.winsorize_threshold,
                self.config.winsorize_threshold
            )

            normalized[col] = z_score

        return normalized


class RegressionSignalCombiner:
    """Combine signals using regression-based methods."""

    def __init__(self, config: SignalCombinationConfig = None):
        self.config = config or SignalCombinationConfig()

    def ols_combination(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        train_end: int
    ) -> pd.Series:
        """
        Combine signals using OLS regression.

        Returns = alpha + beta_1*Signal_1 + beta_2*Signal_2 + ...
        """
        # Train on historical data
        X_train = signals.iloc[:train_end].dropna()
        y_train = forward_returns.iloc[:train_end].loc[X_train.index]

        # Fit model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Apply to all data
        X_all = signals.dropna()
        combined = pd.Series(
            model.predict(X_all),
            index=X_all.index
        )

        return combined

    def ridge_combination(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        train_end: int,
        alpha: float = 1.0
    ) -> pd.Series:
        """
        Combine signals using Ridge regression (L2 regularization).

        Prevents overfitting to noisy signals.
        """
        X_train = signals.iloc[:train_end].dropna()
        y_train = forward_returns.iloc[:train_end].loc[X_train.index]

        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)

        X_all = signals.dropna()
        combined = pd.Series(
            model.predict(X_all),
            index=X_all.index
        )

        return combined

    def lasso_combination(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        train_end: int,
        alpha: float = 0.1
    ) -> pd.Series:
        """
        Combine signals using Lasso regression (L1 regularization).

        Performs signal selection by driving some weights to zero.
        """
        X_train = signals.iloc[:train_end].dropna()
        y_train = forward_returns.iloc[:train_end].loc[X_train.index]

        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)

        X_all = signals.dropna()
        combined = pd.Series(
            model.predict(X_all),
            index=X_all.index
        )

        # Get selected signals
        selected_signals = [
            col for col, coef in zip(signals.columns, model.coef_)
            if abs(coef) > 1e-6
        ]

        return combined, selected_signals
```

---

## 2. Non-Linear Combination Methods

### 2.1 Machine Learning Ensemble

**Signal Logic**:
```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


class MLSignalCombiner:
    """Combine signals using machine learning methods."""

    def __init__(self, config: SignalCombinationConfig = None):
        self.config = config or SignalCombinationConfig()

    def random_forest_combination(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        train_end: int,
        n_estimators: int = 100
    ) -> pd.Series:
        """
        Combine signals using Random Forest.

        Captures non-linear interactions between signals.
        """
        X_train = signals.iloc[:train_end].dropna()
        y_train = forward_returns.iloc[:train_end].loc[X_train.index]

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=5,  # Limit depth to prevent overfitting
            min_samples_leaf=20,
            random_state=42
        )
        model.fit(X_train, y_train)

        X_all = signals.dropna()
        combined = pd.Series(
            model.predict(X_all),
            index=X_all.index
        )

        # Feature importance
        importance = dict(zip(signals.columns, model.feature_importances_))

        return combined, importance

    def gradient_boosting_combination(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        train_end: int
    ) -> pd.Series:
        """
        Combine signals using Gradient Boosting.
        """
        X_train = signals.iloc[:train_end].dropna()
        y_train = forward_returns.iloc[:train_end].loc[X_train.index]

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=20,
            random_state=42
        )
        model.fit(X_train, y_train)

        X_all = signals.dropna()
        combined = pd.Series(
            model.predict(X_all),
            index=X_all.index
        )

        return combined

    def neural_network_combination(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        train_end: int,
        hidden_layers: tuple = (32, 16)
    ) -> pd.Series:
        """
        Combine signals using Neural Network.
        """
        X_train = signals.iloc[:train_end].dropna()
        y_train = forward_returns.iloc[:train_end].loc[X_train.index]

        # Standardize inputs
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)

        X_all = signals.dropna()
        X_all_scaled = scaler.transform(X_all)

        combined = pd.Series(
            model.predict(X_all_scaled),
            index=X_all.index
        )

        return combined
```

---

## 3. Voting and Confirmation

### 3.1 Signal Voting Systems

**Signal Logic**:
```python
class SignalVotingSystem:
    """Combine signals using voting mechanisms."""

    def __init__(self, config: SignalCombinationConfig = None):
        self.config = config or SignalCombinationConfig()

    def majority_vote(
        self,
        signals: pd.DataFrame,
        threshold: float = 0.5
    ) -> pd.Series:
        """
        Take position only when majority of signals agree.

        Args:
            signals: DataFrame with signal values (-1, 0, 1 for short/neutral/long)
            threshold: Fraction of signals that must agree

        Returns:
            Combined signal
        """
        # Count positive and negative signals
        positive_count = (signals > 0).sum(axis=1)
        negative_count = (signals < 0).sum(axis=1)
        total_signals = len(signals.columns)

        combined = pd.Series(0, index=signals.index)

        # Long when enough signals are positive
        combined = np.where(
            positive_count / total_signals >= threshold, 1,
            np.where(
                negative_count / total_signals >= threshold, -1,
                0
            )
        )

        return pd.Series(combined, index=signals.index)

    def unanimous_confirmation(
        self,
        signals: pd.DataFrame
    ) -> pd.Series:
        """
        Take position only when all signals agree.
        """
        all_positive = (signals > 0).all(axis=1)
        all_negative = (signals < 0).all(axis=1)

        combined = pd.Series(0, index=signals.index)
        combined = np.where(all_positive, 1, np.where(all_negative, -1, 0))

        return pd.Series(combined, index=signals.index)

    def weighted_vote(
        self,
        signals: pd.DataFrame,
        weights: Dict[str, float]
    ) -> pd.Series:
        """
        Weighted voting where more reliable signals have higher weight.
        """
        weighted_signals = pd.DataFrame(index=signals.index)

        total_weight = sum(weights.values())

        for col in signals.columns:
            weight = weights.get(col, 1.0)
            weighted_signals[col] = np.sign(signals[col]) * weight

        weighted_sum = weighted_signals.sum(axis=1) / total_weight

        # Convert to directional signal
        combined = np.where(
            weighted_sum > 0.3, 1,
            np.where(weighted_sum < -0.3, -1, 0)
        )

        return pd.Series(combined, index=signals.index)

    def tiered_confirmation(
        self,
        primary_signal: pd.Series,
        confirming_signals: pd.DataFrame,
        min_confirmations: int = 1
    ) -> pd.Series:
        """
        Take primary signal only when confirmed by secondary signals.
        """
        # Count confirming signals
        confirmation_count = (
            (confirming_signals > 0) == (primary_signal > 0)
        ).sum(axis=1)

        # Primary signal must be confirmed
        confirmed = pd.Series(0, index=primary_signal.index)
        confirmed = np.where(
            (primary_signal != 0) & (confirmation_count >= min_confirmations),
            primary_signal,
            0
        )

        return pd.Series(confirmed, index=primary_signal.index)
```

---

## 4. Dynamic Weight Optimization

### 4.1 Adaptive Weighting

**Signal Logic**:
```python
class DynamicWeightOptimizer:
    """Dynamically optimize signal weights."""

    def __init__(self, config: SignalCombinationConfig = None):
        self.config = config or SignalCombinationConfig()

    def rolling_optimization(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series,
        lookback: int = 252,
        rebalance_freq: int = 21
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Rolling window weight optimization.
        """
        weights_history = pd.DataFrame(
            index=signals.index,
            columns=signals.columns
        )
        combined = pd.Series(index=signals.index)

        for i in range(lookback, len(signals), rebalance_freq):
            # Training window
            train_start = max(0, i - lookback)
            train_end = i

            X_train = signals.iloc[train_start:train_end].dropna()
            y_train = forward_returns.iloc[train_start:train_end].loc[X_train.index]

            # Optimize weights
            weights = self._optimize_weights(X_train, y_train)

            # Apply weights forward until next rebalance
            apply_end = min(i + rebalance_freq, len(signals))
            for j in range(i, apply_end):
                weights_history.iloc[j] = weights
                combined.iloc[j] = (signals.iloc[j] * weights).sum()

        return combined, weights_history

    def _optimize_weights(
        self,
        signals: pd.DataFrame,
        returns: pd.Series
    ) -> pd.Series:
        """
        Optimize weights to maximize IC or Sharpe.
        """
        n_signals = len(signals.columns)

        def neg_ic(weights):
            combined = (signals * weights).sum(axis=1)
            ic = combined.corr(returns)
            return -ic if not np.isnan(ic) else 0

        # Constraints: weights sum to 1, non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_signals)]

        # Initial guess: equal weights
        x0 = np.ones(n_signals) / n_signals

        result = optimize.minimize(
            neg_ic,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return pd.Series(result.x, index=signals.columns)

    def regime_adaptive_weights(
        self,
        signals: pd.DataFrame,
        regime: pd.Series,
        regime_weights: Dict[str, Dict[str, float]]
    ) -> pd.Series:
        """
        Use different weights based on market regime.
        """
        combined = pd.Series(index=signals.index)

        for i in range(len(signals)):
            current_regime = regime.iloc[i]
            weights = regime_weights.get(current_regime, {})

            if weights:
                combined.iloc[i] = sum(
                    signals[col].iloc[i] * weights.get(col, 0)
                    for col in signals.columns
                )
            else:
                combined.iloc[i] = signals.iloc[i].mean()

        return combined
```

---

## 5. Composite Signal Engine

### 5.1 Production Ensemble System

```python
class SignalEnsembleEngine:
    """
    Production signal combination engine.
    """

    def __init__(self, config: SignalCombinationConfig = None):
        self.config = config or SignalCombinationConfig()
        self.linear = LinearSignalCombiner(config)
        self.regression = RegressionSignalCombiner(config)
        self.ml = MLSignalCombiner(config)
        self.voting = SignalVotingSystem(config)
        self.dynamic = DynamicWeightOptimizer(config)

    def combine_signals(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series = None,
        method: str = 'ic_weighted'
    ) -> pd.Series:
        """
        Combine signals using specified method.

        Methods:
        - equal: Equal weighted average
        - ic_weighted: Weight by Information Coefficient
        - vol_adjusted: Inverse volatility weighting
        - sharpe_weighted: Weight by signal Sharpe contribution
        - majority_vote: Majority voting
        - ridge: Ridge regression
        - random_forest: Random Forest ensemble
        """
        if method == 'equal':
            return self.linear.equal_weight_combination(signals)

        elif method == 'ic_weighted' and forward_returns is not None:
            return self.linear.ic_weighted_combination(signals, forward_returns)

        elif method == 'vol_adjusted':
            return self.linear.volatility_adjusted_combination(signals)

        elif method == 'sharpe_weighted' and forward_returns is not None:
            return self.linear.sharpe_weighted_combination(signals, forward_returns)

        elif method == 'majority_vote':
            return self.voting.majority_vote(signals)

        elif method == 'ridge' and forward_returns is not None:
            train_end = int(len(signals) * 0.7)
            return self.regression.ridge_combination(signals, forward_returns, train_end)

        elif method == 'random_forest' and forward_returns is not None:
            train_end = int(len(signals) * 0.7)
            combined, _ = self.ml.random_forest_combination(
                signals, forward_returns, train_end
            )
            return combined

        else:
            # Default to equal weight
            return self.linear.equal_weight_combination(signals)

    def generate_meta_signal(
        self,
        signals: pd.DataFrame,
        forward_returns: pd.Series
    ) -> Dict:
        """
        Generate meta-signal combining multiple combination methods.
        """
        results = {}

        # Generate combinations using different methods
        methods = {
            'equal': self.linear.equal_weight_combination(signals),
            'ic_weighted': self.linear.ic_weighted_combination(signals, forward_returns),
            'vol_adjusted': self.linear.volatility_adjusted_combination(signals),
            'majority_vote': self.voting.majority_vote(signals)
        }

        for name, combined in methods.items():
            results[f'{name}_signal'] = combined

            # Calculate IC
            ic = combined.corr(forward_returns)
            results[f'{name}_ic'] = ic

        # Meta-combination (ensemble of ensembles)
        method_signals = pd.DataFrame(methods)
        results['meta_signal'] = method_signals.mean(axis=1)

        return results
```

---

## Signal Usage Guidelines

### Method Selection Matrix

| Method | Use When | Pros | Cons |
|--------|----------|------|------|
| Equal Weight | Signals similar quality | Simple, robust | Ignores signal quality |
| IC Weighted | Have return data | Adapts to signal value | Noisy IC estimates |
| Voting | Binary signals | Clear rules | Loses signal strength |
| Ridge | Many correlated signals | Handles multicollinearity | Requires training |
| Random Forest | Non-linear interactions | Captures complexity | Overfitting risk |

### Integration with Ordinis

```python
# Signal combination in strategy
ensemble = SignalEnsembleEngine()

# Combine momentum, value, quality signals
signals = pd.DataFrame({
    'momentum': momentum_signal,
    'value': value_signal,
    'quality': quality_signal
})

# IC-weighted combination
combined = ensemble.combine_signals(
    signals, forward_returns, method='ic_weighted'
)

# Generate positions
positions = np.sign(combined) * position_size
```

---

## Academic References

1. **Grinold & Kahn (1999)**: "Active Portfolio Management"
2. **De Prado (2018)**: "Advances in Financial Machine Learning"
3. **Bailey et al. (2014)**: "The Deflated Sharpe Ratio"
4. **Qian et al. (2007)**: "Quantitative Equity Portfolio Management"
5. **Breiman (2001)**: "Random Forests"
