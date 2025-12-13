# Signal Validation Framework

Rigorous validation methods to distinguish genuine trading signals from noise, prevent overfitting, and ensure out-of-sample performance.

---

## Overview

Signal validation prevents deploying spurious patterns:

1. **In-Sample vs Out-of-Sample**: Proper data partitioning
2. **Walk-Forward Analysis**: Rolling validation windows
3. **Cross-Validation for Time Series**: Purged K-fold methods
4. **Sharpe Ratio Statistics**: Confidence intervals and tests
5. **Drawdown Analysis**: Statistical properties of losses

---

## 1. Train-Test Splitting for Time Series

### 1.1 Proper Data Partitioning

**Standard Split**:
```
|------- Train (60%) ------|-- Val (20%) --|-- Test (20%) --|
```

**Embargo/Purge**:
```
|------- Train -------| gap |--- Test ---|
```
Gap prevents information leakage from overlapping labels.

### 1.2 Python Implementation

```python
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Generator
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Container for validation metrics."""
    train_sharpe: float
    test_sharpe: float
    train_returns: float
    test_returns: float
    overfitting_ratio: float
    p_value: float


class TimeSeriesSplitter:
    """
    Time series splitting with embargo and purging.
    """

    def __init__(
        self,
        train_pct: float = 0.6,
        val_pct: float = 0.2,
        embargo_pct: float = 0.01
    ):
        """
        Initialize splitter.

        Args:
            train_pct: Training set proportion
            val_pct: Validation set proportion
            embargo_pct: Embargo gap proportion
        """
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.test_pct = 1 - train_pct - val_pct
        self.embargo_pct = embargo_pct

    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test with embargo.

        Returns:
            (train_data, val_data, test_data)
        """
        n = len(data)
        embargo = int(n * self.embargo_pct)

        train_end = int(n * self.train_pct)
        val_start = train_end + embargo
        val_end = int(n * (self.train_pct + self.val_pct))
        test_start = val_end + embargo

        train = data.iloc[:train_end]
        val = data.iloc[val_start:val_end]
        test = data.iloc[test_start:]

        return train, val, test

    def expanding_window_split(
        self,
        data: pd.DataFrame,
        initial_train_pct: float = 0.3,
        step_pct: float = 0.1,
        test_pct: float = 0.1
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Expanding window splits.

        Yields:
            (train_data, test_data) tuples
        """
        n = len(data)
        embargo = int(n * self.embargo_pct)

        train_end = int(n * initial_train_pct)
        step = int(n * step_pct)
        test_size = int(n * test_pct)

        while train_end + embargo + test_size <= n:
            test_start = train_end + embargo
            test_end = test_start + test_size

            train = data.iloc[:train_end]
            test = data.iloc[test_start:test_end]

            yield train, test

            train_end += step

    def rolling_window_split(
        self,
        data: pd.DataFrame,
        train_size: int,
        test_size: int,
        step: int = None
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Rolling window splits (fixed train size).

        Yields:
            (train_data, test_data) tuples
        """
        if step is None:
            step = test_size

        n = len(data)
        embargo = int(n * self.embargo_pct)

        start = 0
        while start + train_size + embargo + test_size <= n:
            train_end = start + train_size
            test_start = train_end + embargo
            test_end = test_start + test_size

            train = data.iloc[start:train_end]
            test = data.iloc[test_start:test_end]

            yield train, test

            start += step


# Example
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', periods=1000)
    data = pd.DataFrame({
        'returns': np.random.randn(1000) * 0.01,
        'feature': np.random.randn(1000)
    }, index=dates)

    splitter = TimeSeriesSplitter()
    train, val, test = splitter.split(data)

    print("Data Split:")
    print(f"  Train: {len(train)} rows ({train.index[0]} to {train.index[-1]})")
    print(f"  Val: {len(val)} rows ({val.index[0]} to {val.index[-1]})")
    print(f"  Test: {len(test)} rows ({test.index[0]} to {test.index[-1]})")

    print("\nExpanding Window Splits:")
    for i, (tr, te) in enumerate(splitter.expanding_window_split(data)):
        print(f"  Split {i+1}: Train {len(tr)}, Test {len(te)}")
        if i >= 4:
            break
```

---

## 2. Walk-Forward Analysis

### 2.1 Walk-Forward Optimization

Train on period $t$, test on $t+1$, roll forward:

```
Period 1: Train [0, T1] -> Test [T1, T2]
Period 2: Train [0, T2] -> Test [T2, T3]
Period 3: Train [0, T3] -> Test [T3, T4]
```

### 2.2 Python Implementation

```python
class WalkForwardValidator:
    """
    Walk-forward analysis for strategy validation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.8,
        embargo_days: int = 5
    ):
        """
        Initialize walk-forward validator.

        Args:
            n_splits: Number of forward walks
            train_ratio: Ratio of train to total in each split
            embargo_days: Days between train and test
        """
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.embargo_days = embargo_days

    def split(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate walk-forward splits.
        """
        n = len(data)
        splits = []

        # Each split uses progressively more data
        for i in range(self.n_splits):
            # End point for this fold
            fold_end = int(n * (i + 1) / self.n_splits)

            # Train/test split within this fold
            train_end = int(fold_end * self.train_ratio)
            test_start = train_end + self.embargo_days

            if test_start < fold_end:
                train = data.iloc[:train_end]
                test = data.iloc[test_start:fold_end]
                splits.append((train, test))

        return splits

    def validate_strategy(
        self,
        data: pd.DataFrame,
        strategy_func,
        metric_func=None
    ) -> pd.DataFrame:
        """
        Run walk-forward validation on strategy.

        Args:
            data: Full dataset
            strategy_func: Function(train_data) -> model/params
            metric_func: Function(test_returns) -> metric

        Returns:
            DataFrame with validation results
        """
        if metric_func is None:
            metric_func = lambda r: r.mean() / r.std() * np.sqrt(252)

        splits = self.split(data)
        results = []

        for i, (train, test) in enumerate(splits):
            # Fit strategy on train
            model = strategy_func(train)

            # Get test returns (assuming model has predict method)
            if hasattr(model, 'predict'):
                predictions = model.predict(test)
                test_returns = predictions * test['returns']
            else:
                # Model is already returns
                test_returns = test['returns']

            # Compute metrics
            train_metric = metric_func(train['returns'])
            test_metric = metric_func(test_returns)

            results.append({
                'fold': i + 1,
                'train_start': train.index[0],
                'train_end': train.index[-1],
                'test_start': test.index[0],
                'test_end': test.index[-1],
                'train_samples': len(train),
                'test_samples': len(test),
                'train_metric': train_metric,
                'test_metric': test_metric,
                'degradation': (train_metric - test_metric) / abs(train_metric) if train_metric != 0 else 0
            })

        return pd.DataFrame(results)

    def aggregate_results(self, results: pd.DataFrame) -> Dict:
        """
        Aggregate walk-forward results.
        """
        return {
            'mean_train_metric': results['train_metric'].mean(),
            'mean_test_metric': results['test_metric'].mean(),
            'std_test_metric': results['test_metric'].std(),
            'mean_degradation': results['degradation'].mean(),
            'worst_degradation': results['degradation'].max(),
            'pct_positive_test': (results['test_metric'] > 0).mean(),
            'consistency': results['test_metric'].min() / results['test_metric'].max() if results['test_metric'].max() > 0 else 0
        }


# Example
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', periods=1000)
    data = pd.DataFrame({
        'returns': np.random.randn(1000) * 0.01 + 0.0002,
        'feature': np.random.randn(1000)
    }, index=dates)

    wf = WalkForwardValidator(n_splits=5)
    splits = wf.split(data)

    print("Walk-Forward Splits:")
    for i, (train, test) in enumerate(splits):
        print(f"  Fold {i+1}: Train {len(train)}, Test {len(test)}")
```

---

## 3. Sharpe Ratio Analysis

### 3.1 Sharpe Ratio Distribution

Under normality:
$$\hat{SR} \sim N\left(SR, \sqrt{\frac{1 + \frac{SR^2}{2}}{n}}\right)$$

### 3.2 Python Implementation

```python
from scipy import stats


class SharpeRatioAnalysis:
    """
    Statistical analysis of Sharpe ratio.
    """

    def __init__(self, returns: pd.Series, annualization: int = 252):
        """
        Initialize with returns.

        Args:
            returns: Return series
            annualization: Trading days per year
        """
        self.returns = returns.dropna()
        self.n = len(self.returns)
        self.annualization = annualization

        # Compute Sharpe
        self.sharpe = (
            self.returns.mean() / self.returns.std() *
            np.sqrt(annualization)
        )

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute confidence interval for Sharpe ratio.

        Uses Lo (2002) standard error.
        """
        # Standard error (simplified)
        se = np.sqrt((1 + 0.5 * self.sharpe**2) / self.n) * np.sqrt(self.annualization)

        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        return (self.sharpe - z * se, self.sharpe + z * se)

    def probability_positive(self) -> float:
        """
        Probability that true Sharpe > 0.
        """
        se = np.sqrt((1 + 0.5 * self.sharpe**2) / self.n) * np.sqrt(self.annualization)
        return stats.norm.cdf(self.sharpe / se)

    def minimum_track_record(self, target_sharpe: float = 1.0, confidence: float = 0.95) -> int:
        """
        Minimum track record length to detect target Sharpe.
        """
        z = stats.norm.ppf(confidence)
        n_min = (z / target_sharpe)**2 * (1 + 0.5 * target_sharpe**2) * self.annualization
        return int(np.ceil(n_min))

    def deflated_sharpe(self, n_trials: int) -> float:
        """
        Deflated Sharpe ratio accounting for multiple testing.

        Harvey & Liu (2015)
        """
        # Expected max Sharpe under null
        expected_max = np.sqrt(2 * np.log(n_trials)) * np.sqrt(self.annualization / self.n)

        # Probability true SR > expected max
        se = np.sqrt((1 + 0.5 * self.sharpe**2) / self.n) * np.sqrt(self.annualization)
        dsr = stats.norm.cdf((self.sharpe - expected_max) / se)

        return dsr

    def compare_strategies(self, other_returns: pd.Series) -> Dict:
        """
        Test if this strategy's Sharpe differs from another.
        """
        other = SharpeRatioAnalysis(other_returns, self.annualization)

        diff = self.sharpe - other.sharpe

        # Approximate SE of difference
        se1 = np.sqrt((1 + 0.5 * self.sharpe**2) / self.n)
        se2 = np.sqrt((1 + 0.5 * other.sharpe**2) / other.n)
        se_diff = np.sqrt(se1**2 + se2**2) * np.sqrt(self.annualization)

        z_stat = diff / se_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        return {
            'sharpe_1': self.sharpe,
            'sharpe_2': other.sharpe,
            'difference': diff,
            'z_statistic': z_stat,
            'p_value': p_value,
            'significant_5pct': p_value < 0.05
        }


# Example
if __name__ == "__main__":
    np.random.seed(42)
    returns = pd.Series(np.random.randn(500) * 0.01 + 0.0004)

    sr_analysis = SharpeRatioAnalysis(returns)

    print(f"Sharpe Ratio: {sr_analysis.sharpe:.3f}")
    ci = sr_analysis.confidence_interval()
    print(f"95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    print(f"P(SR > 0): {sr_analysis.probability_positive():.3f}")
    print(f"Deflated SR (100 trials): {sr_analysis.deflated_sharpe(100):.3f}")
```

---

## 4. Drawdown Analysis

### 4.1 Drawdown Statistics

```python
class DrawdownAnalysis:
    """
    Statistical analysis of drawdowns.
    """

    def __init__(self, returns: pd.Series):
        """
        Initialize with returns.
        """
        self.returns = returns.dropna()
        self.equity_curve = (1 + self.returns).cumprod()

        # Compute drawdown series
        running_max = self.equity_curve.cummax()
        self.drawdown = (self.equity_curve - running_max) / running_max

    def max_drawdown(self) -> float:
        """Maximum drawdown."""
        return self.drawdown.min()

    def drawdown_duration(self) -> pd.DataFrame:
        """
        Analyze drawdown durations.
        """
        # Identify drawdown periods
        in_drawdown = self.drawdown < 0

        # Find drawdown start/end
        dd_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
        dd_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)

        starts = self.drawdown.index[dd_starts]
        ends = self.drawdown.index[dd_ends]

        durations = []
        for i, start in enumerate(starts):
            # Find next end after this start
            future_ends = ends[ends > start]
            if len(future_ends) > 0:
                end = future_ends[0]
                duration = (end - start).days
                depth = self.drawdown[start:end].min()
                durations.append({
                    'start': start,
                    'end': end,
                    'duration_days': duration,
                    'depth': depth
                })

        return pd.DataFrame(durations)

    def calmar_ratio(self, annualization: int = 252) -> float:
        """
        Calmar ratio: annualized return / max drawdown.
        """
        annual_return = self.returns.mean() * annualization
        return -annual_return / self.max_drawdown()

    def ulcer_index(self) -> float:
        """
        Ulcer Index: RMS of drawdowns.

        Lower is better.
        """
        return np.sqrt((self.drawdown**2).mean())

    def pain_index(self) -> float:
        """
        Pain Index: mean of absolute drawdowns.
        """
        return (-self.drawdown).mean()

    def recovery_analysis(self) -> Dict:
        """
        Analyze recovery from max drawdown.
        """
        max_dd_idx = self.drawdown.idxmin()
        max_dd = self.drawdown[max_dd_idx]

        # Find recovery point
        post_dd = self.drawdown[max_dd_idx:]
        recovery_idx = post_dd[post_dd >= 0].index

        if len(recovery_idx) > 0:
            recovery_date = recovery_idx[0]
            recovery_days = (recovery_date - max_dd_idx).days
            recovered = True
        else:
            recovery_date = None
            recovery_days = None
            recovered = False

        return {
            'max_drawdown': max_dd,
            'max_dd_date': max_dd_idx,
            'recovered': recovered,
            'recovery_date': recovery_date,
            'recovery_days': recovery_days
        }


# Example
if __name__ == "__main__":
    np.random.seed(42)
    returns = pd.Series(
        np.random.randn(500) * 0.02 + 0.0003,
        index=pd.date_range('2020-01-01', periods=500)
    )

    dd_analysis = DrawdownAnalysis(returns)

    print(f"Max Drawdown: {dd_analysis.max_drawdown():.2%}")
    print(f"Calmar Ratio: {dd_analysis.calmar_ratio():.2f}")
    print(f"Ulcer Index: {dd_analysis.ulcer_index():.4f}")

    recovery = dd_analysis.recovery_analysis()
    print(f"Recovered: {recovery['recovered']}")
```

---

## 5. Overfitting Detection

### 5.1 Probability of Backtest Overfitting (PBO)

```python
class OverfittingDetector:
    """
    Detect overfitting in backtests.
    """

    def __init__(self, returns_matrix: pd.DataFrame):
        """
        Initialize with matrix of strategy returns.

        Args:
            returns_matrix: DataFrame where each column is a strategy variant
        """
        self.returns = returns_matrix
        self.n_strategies = returns_matrix.shape[1]

    def probability_of_overfitting(self, n_partitions: int = 16) -> float:
        """
        Compute Probability of Backtest Overfitting (PBO).

        Bailey et al. (2014)
        """
        from itertools import combinations

        n = len(self.returns)
        partition_size = n // n_partitions

        # Compute Sharpe for each strategy in each partition
        partition_sharpes = []
        for i in range(n_partitions):
            start = i * partition_size
            end = start + partition_size
            partition = self.returns.iloc[start:end]

            sharpes = partition.mean() / partition.std() * np.sqrt(252)
            partition_sharpes.append(sharpes)

        partition_sharpes = pd.DataFrame(partition_sharpes)

        # Count how often IS-optimal is OOS-suboptimal
        n_overfit = 0
        n_tests = 0

        for is_partitions in combinations(range(n_partitions), n_partitions // 2):
            oos_partitions = [p for p in range(n_partitions) if p not in is_partitions]

            # IS performance
            is_sharpes = partition_sharpes.iloc[list(is_partitions)].mean()
            is_best = is_sharpes.idxmax()

            # OOS performance of IS-best
            oos_sharpes = partition_sharpes.iloc[oos_partitions].mean()
            oos_rank = oos_sharpes.rank(ascending=False)[is_best]

            # Is IS-best below median OOS?
            if oos_rank > self.n_strategies / 2:
                n_overfit += 1

            n_tests += 1

            if n_tests >= 100:  # Limit iterations
                break

        return n_overfit / n_tests

    def deflated_sharpe_haircut(self, observed_sharpe: float, n_trials: int) -> float:
        """
        Haircut to apply to backtest Sharpe.
        """
        # Expected max under null
        expected_max = np.sqrt(2 * np.log(n_trials))

        haircut = expected_max / observed_sharpe if observed_sharpe > 0 else 1.0

        return min(haircut, 1.0)

    def strategy_variance_ratio(self) -> float:
        """
        Ratio of strategy variance to return variance.

        High ratio suggests overfitting.
        """
        strategy_variance = self.returns.mean(axis=0).var()
        return_variance = self.returns.var().mean()

        return strategy_variance / return_variance


# Example
if __name__ == "__main__":
    np.random.seed(42)

    # Simulate multiple strategy variants
    n_strategies = 20
    n_periods = 500

    returns_matrix = pd.DataFrame({
        f'strategy_{i}': np.random.randn(n_periods) * 0.01 + 0.0001 * (i - 10)
        for i in range(n_strategies)
    })

    detector = OverfittingDetector(returns_matrix)

    pbo = detector.probability_of_overfitting()
    print(f"Probability of Overfitting: {pbo:.2%}")

    haircut = detector.deflated_sharpe_haircut(1.5, n_strategies)
    print(f"Sharpe Haircut: {haircut:.2%}")
```

---

## 6. Validation Report Generator

```python
class ValidationReport:
    """
    Generate comprehensive validation report.
    """

    def __init__(self, train_returns: pd.Series, test_returns: pd.Series):
        self.train = train_returns
        self.test = test_returns

    def generate(self) -> Dict:
        """Generate full validation report."""
        train_sr = SharpeRatioAnalysis(self.train)
        test_sr = SharpeRatioAnalysis(self.test)

        train_dd = DrawdownAnalysis(self.train)
        test_dd = DrawdownAnalysis(self.test)

        return {
            'train': {
                'sharpe': train_sr.sharpe,
                'sharpe_ci': train_sr.confidence_interval(),
                'max_drawdown': train_dd.max_drawdown(),
                'calmar': train_dd.calmar_ratio(),
                'n_samples': len(self.train)
            },
            'test': {
                'sharpe': test_sr.sharpe,
                'sharpe_ci': test_sr.confidence_interval(),
                'max_drawdown': test_dd.max_drawdown(),
                'calmar': test_dd.calmar_ratio(),
                'n_samples': len(self.test)
            },
            'degradation': {
                'sharpe_pct': (train_sr.sharpe - test_sr.sharpe) / abs(train_sr.sharpe) if train_sr.sharpe != 0 else 0,
                'drawdown_pct': (test_dd.max_drawdown() - train_dd.max_drawdown()) / abs(train_dd.max_drawdown()) if train_dd.max_drawdown() != 0 else 0
            },
            'significance': {
                'test_sharpe_significant': test_sr.probability_positive() > 0.95,
                'train_test_different': train_sr.compare_strategies(self.test)['significant_5pct']
            }
        }
```

---

## 7. Academic References

1. **Bailey, D. H., et al. (2014)**. "The Probability of Backtest Overfitting." *Journal of Computational Finance*.

2. **López de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Wiley.

3. **Harvey, C. R., & Liu, Y. (2015)**. "Backtesting." *Journal of Portfolio Management*.

4. **Lo, A. W. (2002)**. "The Statistics of Sharpe Ratios." *Financial Analysts Journal*.

5. **Bailey, D. H., & López de Prado, M. (2012)**. "The Sharpe Ratio Efficient Frontier." *Journal of Risk*.

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "published"
tags: ["validation", "backtesting", "sharpe-ratio", "drawdown", "overfitting", "walk-forward"]
code_lines: 500
```

---

**END OF DOCUMENT**
