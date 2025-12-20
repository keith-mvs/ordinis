# Ordinis Strategy Testing Framework

> **Based on López de Prado (2018), Harvey et al. (2016), Bailey et al. (2015)**

---

**Version:** 1.0.0
**Date:** 2025-12-17
**Status:** Specification
**Target Module:** `src/ordinis/engines/proofbench/`

---

## Executive Summary

This document specifies a comprehensive testing framework for Ordinis trading strategies, grounded in academic best practices for avoiding backtest overfitting. The framework implements:

1. **Walk-Forward Testing** with proper temporal separation
2. **Purged K-Fold Cross-Validation** (López de Prado Ch. 7)
3. **Deflated Sharpe Ratio** (Bailey & López de Prado 2014)
4. **Probability of Backtest Overfitting (PBO)** (Bailey et al. 2015)
5. **Regression Testing** for strategy stability
6. **Edge Case Testing** across market regimes
7. **Optimization Framework** with multiple testing corrections

---

## Part 1: Walk-Forward Testing

### 1.1 Motivation

> "Backtesting is not a research tool... A single backtest is one realization of a stochastic process." — López de Prado

Traditional backtesting on a single historical path suffers from:
- **Path dependency**: Different sequence → different results
- **Look-ahead bias**: Subconscious selection of favorable periods
- **Overfitting**: Tuning to historical noise, not signal

Walk-forward testing addresses these by:
- Simulating realistic deployment conditions
- Never allowing future data into training
- Producing multiple out-of-sample performance windows

### 1.2 Implementation

```python
class WalkForwardTest:
    """Walk-forward backtesting with anchored or rolling windows."""

    def __init__(
        self,
        train_period: int = 252,      # 1 year
        test_period: int = 63,        # 1 quarter
        embargo_period: int = 5,      # Gap between train/test
        anchored: bool = True,        # Expanding vs rolling
        min_train_size: int = 504,    # Minimum 2 years
    ):
        self.train_period = train_period
        self.test_period = test_period
        self.embargo_period = embargo_period
        self.anchored = anchored
        self.min_train_size = min_train_size

    def generate_splits(
        self,
        data: pd.DataFrame
    ) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        """Generate train/test index pairs."""
        splits = []
        n = len(data)

        # Start position for first test window
        start = self.min_train_size

        while start + self.embargo_period + self.test_period <= n:
            # Training window
            if self.anchored:
                train_start = 0
            else:
                train_start = start - self.train_period
            train_end = start

            # Test window (after embargo)
            test_start = start + self.embargo_period
            test_end = min(test_start + self.test_period, n)

            splits.append((
                data.index[train_start:train_end],
                data.index[test_start:test_end]
            ))

            # Move forward
            start = test_end

        return splits
```

### 1.3 Walk-Forward Metrics

For each walk-forward window, compute:

| Metric | Formula | Target |
|--------|---------|--------|
| OOS Sharpe | $\frac{\mu_{oos}}{\sigma_{oos}} \times \sqrt{252}$ | > 0.5 |
| OOS Win Rate | $\frac{N_{win}}{N_{total}}$ | > 40% |
| OOS Max Drawdown | $\max_t(Peak_t - Trough_t)$ | < 20% |
| IS/OOS Ratio | $\frac{Sharpe_{oos}}{Sharpe_{is}}$ | > 0.5 |

**Critical Warning**: If OOS Sharpe is consistently < 50% of IS Sharpe, strategy is overfit.

---

## Part 2: Purged K-Fold Cross-Validation (CPCV)

### 2.1 Why Standard CV Fails in Finance

> "Standard CV doesn't work in finance... Temporal structure causes leakage." — López de Prado

**Problem**: Trading labels overlap in time. A 5-day position prediction at time $t$ knows the outcome which is determined by data at $t+5$. If test fold contains $t+5$, and training contains $t$, information leaks.

**Solution**: Purge training samples that overlap with test period, then add embargo buffer.

### 2.2 Purged K-Fold Implementation

```python
class PurgedKFold:
    """K-Fold CV with purging and embargo for financial time series."""

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,  # 1% of data as embargo
    ):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: pd.Series | None = None,
        label_times: pd.DataFrame | None = None,  # (start_time, end_time) per sample
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with purging.

        Args:
            X: Feature matrix
            y: Labels
            sample_weight: Optional weights
            label_times: DataFrame with 'start' and 'end' columns for each sample
        """
        n = len(X)
        embargo_size = int(n * self.embargo_pct)
        fold_size = n // self.n_splits

        for i in range(self.n_splits):
            # Test fold boundaries
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n
            test_indices = np.arange(test_start, test_end)

            # Get test period time boundaries
            test_start_time = X.index[test_start]
            test_end_time = X.index[test_end - 1]

            # Build training indices (exclude test + purged + embargo)
            train_indices = []

            for j in range(n):
                if j in test_indices:
                    continue

                # Check if sample label overlaps with test period
                if label_times is not None:
                    sample_end = label_times.iloc[j]['end']
                    if sample_end > test_start_time and sample_end <= test_end_time:
                        continue  # Purge this sample

                # Check embargo after test period
                if j > test_end and j <= test_end + embargo_size:
                    continue  # Embargo period

                train_indices.append(j)

            yield np.array(train_indices), test_indices
```

### 2.3 Combinatorial Purged CV (CPCV)

For path-dependent strategies, use combinatorial approach:

$$
\text{Paths} = \binom{N}{k} \text{ where } k = N - \text{test\_size}
$$

This generates multiple backtest paths, reducing single-path dependency.

**Key Insight**: CPCV produces distribution of performance metrics, not point estimates.

---

## Part 3: Deflated Sharpe Ratio

### 3.1 The Multiple Testing Problem

> "If you test 100 strategies, some will look good by chance." — Harvey et al. (2016)

With $N$ independent strategy trials, the expected maximum Sharpe ratio under the null:

$$
E[\max(SR_1, ..., SR_N)] \approx \sqrt{2 \ln N}
$$

For $N = 100$ trials: Expected max $SR \approx 3.0$ **by pure chance**.

### 3.2 Deflated Sharpe Ratio Formula

$$
DSR = SR \times \sqrt{1 - \frac{\hat{V} \times N}{T}}
$$

Where:
- $SR$ = Reported Sharpe ratio
- $\hat{V}$ = Variance of Sharpe across trials (estimated)
- $N$ = Number of strategies tested
- $T$ = Number of observations

### 3.3 Implementation

```python
from scipy import stats

def deflated_sharpe_ratio(
    sharpe: float,
    n_trials: int,
    n_observations: int,
    sharpe_variance: float = 1.0,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> tuple[float, float]:
    """
    Compute Deflated Sharpe Ratio and p-value.

    Args:
        sharpe: Reported Sharpe ratio
        n_trials: Number of strategies tested
        n_observations: Number of return observations
        sharpe_variance: Variance of Sharpe across trials
        skewness: Skewness of returns
        kurtosis: Kurtosis of returns (excess = kurtosis - 3)

    Returns:
        (deflated_sharpe, p_value)
    """
    # Adjust for non-normality (Lo, 2002)
    adj = 1 + (skewness * sharpe / 6) + ((kurtosis - 3) * sharpe**2 / 24)
    se = np.sqrt((1 + 0.5 * sharpe**2) / (n_observations - 1)) * adj

    # Expected max Sharpe under null
    e_max_sr = sharpe_variance * ((1 - np.euler_gamma) * stats.norm.ppf(1 - 1/n_trials)
               + np.euler_gamma * stats.norm.ppf(1 - 1/(n_trials * np.e)))

    # Deflate
    psr = stats.norm.cdf((sharpe - e_max_sr) / se)

    deflated = sharpe - e_max_sr

    return deflated, psr


# Usage
dsr, pval = deflated_sharpe_ratio(
    sharpe=2.0,
    n_trials=50,
    n_observations=252 * 5,  # 5 years daily
)
print(f"Deflated SR: {dsr:.2f}, p-value: {pval:.4f}")
```

### 3.4 Interpretation Thresholds

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| DSR > 0.5 | ✅ Robust | Likely genuine edge |
| DSR 0 - 0.5 | ⚠️ Marginal | May be partially overfit |
| DSR < 0 | ❌ Overfit | Expected by chance |
| p-value < 0.05 | ✅ Significant | Reject null of no skill |

---

## Part 4: Probability of Backtest Overfitting (PBO)

### 4.1 Definition

PBO measures the probability that the in-sample optimal strategy underperforms a random strategy out-of-sample.

$$
PBO = P(\text{rank}_{oos}^* > \text{median rank})
$$

Where $\text{rank}_{oos}^*$ is the out-of-sample rank of the in-sample best strategy.

### 4.2 CSCV Method

**Combinatory Symmetric Cross-Validation (CSCV)**:

1. Partition data into $N$ blocks
2. Form all $\binom{N}{N/2}$ train/test combinations
3. For each combination:
   - Select best strategy in-sample
   - Record its rank out-of-sample
4. Compute rank correlation (Spearman) between IS and OOS ranks
5. $PBO = \frac{\text{# combinations with } \rho < 0}{\text{total combinations}}$

### 4.3 Implementation

```python
from itertools import combinations
from scipy.stats import spearmanr

def compute_pbo(
    strategy_returns: pd.DataFrame,  # Columns = strategies, rows = time
    n_blocks: int = 10,
) -> tuple[float, float]:
    """
    Compute Probability of Backtest Overfitting.

    Args:
        strategy_returns: DataFrame of strategy returns
        n_blocks: Number of data blocks

    Returns:
        (pbo, avg_rank_correlation)
    """
    n = len(strategy_returns)
    block_size = n // n_blocks
    blocks = [strategy_returns.iloc[i*block_size:(i+1)*block_size]
              for i in range(n_blocks)]

    # All ways to choose half the blocks for training
    train_combos = list(combinations(range(n_blocks), n_blocks // 2))

    correlations = []
    oos_best_ranks = []

    for train_idx in train_combos:
        test_idx = tuple(i for i in range(n_blocks) if i not in train_idx)

        # Combine blocks
        train_data = pd.concat([blocks[i] for i in train_idx])
        test_data = pd.concat([blocks[i] for i in test_idx])

        # Rank strategies by in-sample Sharpe
        is_sharpe = train_data.mean() / train_data.std() * np.sqrt(252)
        is_ranks = is_sharpe.rank(ascending=False)

        # Rank strategies by out-of-sample Sharpe
        oos_sharpe = test_data.mean() / test_data.std() * np.sqrt(252)
        oos_ranks = oos_sharpe.rank(ascending=False)

        # Correlation between IS and OOS ranks
        rho, _ = spearmanr(is_ranks, oos_ranks)
        correlations.append(rho)

        # Rank of IS-best strategy in OOS
        is_best = is_sharpe.idxmax()
        oos_best_ranks.append(oos_ranks[is_best])

    # PBO = fraction of combinations where IS-best underperforms median OOS
    n_strategies = len(strategy_returns.columns)
    median_rank = (n_strategies + 1) / 2
    pbo = sum(1 for r in oos_best_ranks if r > median_rank) / len(oos_best_ranks)

    avg_correlation = np.mean(correlations)

    return pbo, avg_correlation


# Interpretation
# PBO < 0.25: Low overfitting risk
# PBO 0.25-0.50: Moderate overfitting risk
# PBO > 0.50: High overfitting risk - strategy likely overfit
```

### 4.4 PBO Thresholds

| PBO Value | Risk Level | Recommendation |
|-----------|------------|----------------|
| < 0.15 | ✅ Low | Deploy with confidence |
| 0.15 - 0.30 | ⚠️ Moderate | Additional validation needed |
| 0.30 - 0.50 | 🔶 Elevated | Simplify strategy |
| > 0.50 | ❌ High | Discard or major revision |

---

## Part 5: Regression Testing Suite

### 5.1 Purpose

Ensure strategy performance remains stable across:
- Code changes
- Dependency updates
- Data source changes
- Configuration modifications

### 5.2 Test Categories

#### 5.2.1 Determinism Tests

```python
@pytest.mark.regression
class TestStrategyDeterminism:
    """Verify strategies produce identical results given identical inputs."""

    def test_signal_reproducibility(self, strategy, sample_data):
        """Same data → same signal."""
        signal1 = strategy.generate_signal(sample_data, timestamp)
        signal2 = strategy.generate_signal(sample_data, timestamp)
        assert signal1 == signal2

    def test_seeded_randomness(self, stochastic_strategy):
        """Random strategies with same seed → same output."""
        stochastic_strategy.set_seed(42)
        result1 = stochastic_strategy.run(data)
        stochastic_strategy.set_seed(42)
        result2 = stochastic_strategy.run(data)
        assert_frame_equal(result1, result2)
```

#### 5.2.2 Baseline Performance Tests

```python
@pytest.mark.regression
class TestBaselinePerformance:
    """Verify performance doesn't degrade from baseline."""

    # Baselines established from validated backtests
    BASELINES = {
        "moving_average_crossover": {
            "sharpe": 0.8,
            "max_drawdown": 0.25,
            "win_rate": 0.42,
        },
        "rsi_mean_reversion": {
            "sharpe": 0.6,
            "max_drawdown": 0.18,
            "win_rate": 0.55,
        },
    }

    def test_sharpe_not_degraded(self, strategy_name, backtest_results):
        """Sharpe ratio within 20% of baseline."""
        baseline = self.BASELINES[strategy_name]["sharpe"]
        actual = backtest_results.sharpe_ratio
        assert actual >= baseline * 0.8, f"Sharpe degraded: {actual:.2f} vs baseline {baseline:.2f}"

    def test_drawdown_not_increased(self, strategy_name, backtest_results):
        """Max drawdown not significantly worse than baseline."""
        baseline = self.BASELINES[strategy_name]["max_drawdown"]
        actual = backtest_results.max_drawdown
        assert actual <= baseline * 1.2, f"Drawdown worse: {actual:.2%} vs baseline {baseline:.2%}"
```

#### 5.2.3 Contract Tests

```python
@pytest.mark.regression
class TestStrategyContracts:
    """Verify strategy API contracts are maintained."""

    def test_signal_schema(self, strategy, sample_data):
        """Signal output conforms to expected schema."""
        signal = strategy.generate_signal(sample_data, datetime.now())
        if signal is not None:
            assert hasattr(signal, 'direction')
            assert hasattr(signal, 'probability')
            assert 0 <= signal.probability <= 1

    def test_config_validation(self, strategy_class):
        """Invalid configs raise appropriate errors."""
        with pytest.raises(ValidationError):
            strategy_class(lookback=-1)  # Negative lookback
```

### 5.3 Regression Test Fixtures

```python
# tests/fixtures/regression_data.py

@pytest.fixture(scope="session")
def regression_dataset():
    """Fixed dataset for regression testing - never changes."""
    return pd.read_parquet("tests/fixtures/data/regression_ohlcv.parquet")

@pytest.fixture
def frozen_random_state():
    """Frozen random state for reproducible tests."""
    return np.random.RandomState(seed=12345)
```

---

## Part 6: Edge Case Testing

### 6.1 Market Regime Edge Cases

```python
@pytest.mark.edge_case
class TestMarketRegimes:
    """Test strategy behavior across extreme market conditions."""

    @pytest.mark.parametrize("regime", [
        "flash_crash",       # May 6, 2010
        "covid_crash",       # March 2020
        "volmageddon",       # Feb 2018
        "gfc_peak",          # Sep-Oct 2008
        "dotcom_burst",      # 2000-2002
    ])
    def test_extreme_regime(self, strategy, regime, regime_data):
        """Strategy doesn't blow up in extreme regimes."""
        data = regime_data[regime]
        results = run_backtest(strategy, data)

        # Must survive (not lose everything)
        assert results.final_equity > 0
        # Drawdown under catastrophic threshold
        assert results.max_drawdown < 0.80

    def test_zero_volume_handling(self, strategy):
        """Strategy handles zero volume gracefully."""
        data = create_data_with_zero_volume()
        signal = strategy.generate_signal(data, datetime.now())
        # Should return None or filtered signal, not crash
        assert signal is None or signal.probability < 0.5

    def test_gap_handling(self, strategy):
        """Strategy handles overnight gaps."""
        data = create_data_with_gaps(gap_pct=0.10)  # 10% gaps
        results = run_backtest(strategy, data)
        # Verify stop-losses respected despite gaps
        assert results.max_trade_loss < strategy.max_loss * 2
```

### 6.2 Data Quality Edge Cases

```python
@pytest.mark.edge_case
class TestDataQuality:
    """Test strategy robustness to data issues."""

    def test_missing_values(self, strategy):
        """Strategy handles NaN values."""
        data = create_data_with_nans(nan_pct=0.01)
        signal = strategy.generate_signal(data, datetime.now())
        # Should handle gracefully
        assert signal is None or isinstance(signal, Signal)

    def test_stale_prices(self, strategy):
        """Strategy detects stale/repeated prices."""
        data = create_data_with_stale_prices(stale_bars=10)
        signal = strategy.generate_signal(data, datetime.now())
        # Should not generate signal on stale data
        assert signal is None

    def test_negative_prices(self, strategy):
        """Strategy rejects invalid negative prices."""
        data = create_data_with_negatives()
        with pytest.raises(DataValidationError):
            strategy.generate_signal(data, datetime.now())
```

### 6.3 Parameter Boundary Edge Cases

```python
@pytest.mark.edge_case
class TestParameterBoundaries:
    """Test strategy at parameter extremes."""

    @pytest.mark.parametrize("lookback", [1, 2, 5, 1000, 5000])
    def test_extreme_lookbacks(self, strategy_class, lookback, sample_data):
        """Strategy handles extreme lookback periods."""
        if lookback > len(sample_data):
            # Should handle gracefully
            strategy = strategy_class(lookback=lookback)
            signal = strategy.generate_signal(sample_data, datetime.now())
            assert signal is None  # Insufficient data
        else:
            strategy = strategy_class(lookback=lookback)
            # Should not crash
            signal = strategy.generate_signal(sample_data, datetime.now())

    @pytest.mark.parametrize("threshold", [0.0, 0.001, 0.999, 1.0])
    def test_extreme_thresholds(self, strategy_class, threshold):
        """Strategy handles extreme threshold values."""
        strategy = strategy_class(threshold=threshold)
        # Should not produce infinite signals or crash
        signals = [strategy.generate_signal(d, t) for d, t in test_periods]
        valid_signals = [s for s in signals if s is not None]
        assert all(0 <= s.probability <= 1 for s in valid_signals)
```

---

## Part 7: Optimization Framework

### 7.1 Principles

> "A new factor needs t-statistic > 3.0 (not traditional 2.0) to be considered significant." — Harvey et al. (2016)

**Key Principles**:
1. **Minimize trials**: Each parameter tested increases multiple-testing penalty
2. **Use DSR/PBO**: Never optimize on raw Sharpe alone
3. **Out-of-sample validation**: Hold out data never seen during optimization
4. **Regularization**: Prefer simpler models with fewer parameters

### 7.2 Bayesian Optimization

```python
from optuna import create_study, Trial

class StrategyOptimizer:
    """Bayesian optimization with overfitting controls."""

    def __init__(
        self,
        strategy_class: type,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        n_trials: int = 50,
    ):
        self.strategy_class = strategy_class
        self.train_data = train_data
        self.val_data = val_data
        self.n_trials = n_trials
        self.trial_count = 0

    def objective(self, trial: Trial) -> float:
        """Optuna objective function."""
        # Suggest parameters
        params = {
            "lookback": trial.suggest_int("lookback", 10, 200),
            "threshold": trial.suggest_float("threshold", 0.01, 0.10),
            "atr_mult": trial.suggest_float("atr_mult", 1.0, 4.0),
        }

        # Create strategy
        strategy = self.strategy_class(**params)

        # Train-set performance
        train_results = run_backtest(strategy, self.train_data)
        train_sharpe = train_results.sharpe_ratio

        # Validation-set performance (for DSR calculation)
        val_results = run_backtest(strategy, self.val_data)
        val_sharpe = val_results.sharpe_ratio

        self.trial_count += 1

        # Compute Deflated Sharpe Ratio
        dsr, _ = deflated_sharpe_ratio(
            sharpe=val_sharpe,
            n_trials=self.trial_count,
            n_observations=len(self.val_data),
        )

        # Penalize large IS/OOS gap (overfitting signal)
        overfit_penalty = max(0, (train_sharpe - val_sharpe) / train_sharpe)

        # Combined objective: maximize DSR, penalize overfitting
        objective_value = dsr - 0.5 * overfit_penalty

        return objective_value

    def optimize(self) -> dict:
        """Run optimization."""
        study = create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)

        # Final validation on held-out test set
        best_params = study.best_params

        return {
            "params": best_params,
            "best_dsr": study.best_value,
            "n_trials": self.trial_count,
        }
```

### 7.3 Minimum t-Statistic Requirements

Based on Harvey et al. (2016), apply these t-statistic hurdles:

| Number of Trials | Required t-stat | Sharpe Equivalent |
|------------------|-----------------|-------------------|
| 10 | 2.57 | ~0.8 |
| 50 | 3.18 | ~1.0 |
| 100 | 3.39 | ~1.1 |
| 300 | 3.72 | ~1.2 |
| 1000 | 4.00 | ~1.3 |

### 7.4 Regularization Techniques

```python
def compute_complexity_penalty(strategy: BaseStrategy) -> float:
    """Penalize strategy complexity to prevent overfitting."""
    n_params = len(strategy.get_parameters())
    n_indicators = strategy.count_indicators()
    n_conditions = strategy.count_signal_conditions()

    # Log penalty (diminishing returns for complexity)
    penalty = 0.01 * (np.log(1 + n_params) +
                      np.log(1 + n_indicators) +
                      np.log(1 + n_conditions))

    return penalty


def regularized_sharpe(sharpe: float, strategy: BaseStrategy) -> float:
    """Sharpe ratio with complexity penalty."""
    penalty = compute_complexity_penalty(strategy)
    return sharpe - penalty
```

---

## Part 8: Testing Pipeline Integration

### 8.1 CI/CD Integration

```yaml
# .github/workflows/strategy-tests.yml
name: Strategy Testing Pipeline

on:
  push:
    paths:
      - 'src/ordinis/application/strategies/**'
      - 'src/ordinis/engines/proofbench/**'

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run unit tests
        run: pytest tests/test_strategies -m unit --tb=short

  regression-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - name: Run regression tests
        run: pytest tests/test_strategies -m regression --tb=short

  edge-case-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - name: Run edge case tests
        run: pytest tests/test_strategies -m edge_case --tb=short

  walk-forward-validation:
    runs-on: gpu-runner  # GPU for faster backtesting
    needs: [regression-tests, edge-case-tests]
    steps:
      - name: Run walk-forward tests
        run: python -m ordinis.engines.proofbench.validate --strategies all
      - name: Check DSR thresholds
        run: python scripts/check_dsr_thresholds.py --min-dsr 0.3
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: validation-results
          path: artifacts/validation/
```

### 8.2 Pre-Deployment Checklist

```python
class PreDeploymentValidator:
    """Final validation before strategy deployment."""

    THRESHOLDS = {
        "min_dsr": 0.3,
        "max_pbo": 0.35,
        "min_oos_is_ratio": 0.5,
        "max_drawdown": 0.25,
        "min_observations": 252 * 3,  # 3 years
    }

    def validate(self, strategy: BaseStrategy, results: BacktestResults) -> ValidationResult:
        """Run all pre-deployment checks."""
        checks = []

        # DSR check
        dsr, _ = deflated_sharpe_ratio(
            results.sharpe_ratio,
            results.n_trials_tested,
            len(results.returns)
        )
        checks.append(("DSR", dsr >= self.THRESHOLDS["min_dsr"], dsr))

        # PBO check
        pbo, _ = compute_pbo(results.strategy_returns)
        checks.append(("PBO", pbo <= self.THRESHOLDS["max_pbo"], pbo))

        # OOS/IS ratio
        ratio = results.oos_sharpe / results.is_sharpe
        checks.append(("OOS/IS", ratio >= self.THRESHOLDS["min_oos_is_ratio"], ratio))

        # Drawdown
        checks.append(("MaxDD", results.max_drawdown <= self.THRESHOLDS["max_drawdown"],
                      results.max_drawdown))

        # Sample size
        checks.append(("SampleSize", len(results.returns) >= self.THRESHOLDS["min_observations"],
                      len(results.returns)))

        passed = all(c[1] for c in checks)

        return ValidationResult(
            passed=passed,
            checks=checks,
            recommendation="DEPLOY" if passed else "REVIEW",
        )
```

---

## Part 9: References

### Core Academic Papers

1. **López de Prado, M. (2018).** *Advances in Financial Machine Learning*. Wiley.
   - Chapters 7, 11-14 for backtesting methodology

2. **Harvey, C.R., Liu, Y., & Zhu, H. (2016).** "...and the Cross-Section of Expected Returns." *RFS*.
   - t-statistic hurdles for multiple testing

3. **Bailey, D.H. & López de Prado, M. (2014).** "The Deflated Sharpe Ratio." *JPM*.
   - DSR formula and interpretation

4. **Bailey, D.H., et al. (2015).** "The Probability of Backtest Overfitting." *JCF*.
   - PBO computation and CSCV methodology

5. **Harvey, C.R. & Liu, Y. (2020).** "False (and Missed) Discoveries in Financial Economics." *JoF*.
   - Double-bootstrap for Type I/II error calibration

### Implementation Resources

- **Open Source Cross-Sectional Asset Pricing**: https://www.openassetpricing.com/
- **mlfinlab (López de Prado implementations)**: https://github.com/hudson-and-thames/mlfinlab
- **Optuna (Bayesian optimization)**: https://optuna.org/

---

## Appendix A: ProofBench Module Structure

```
src/ordinis/engines/proofbench/
├── __init__.py
├── core/
│   ├── engine.py           # Main ProofBench engine
│   ├── config.py           # Configuration
│   └── models.py           # Data models
├── validation/
│   ├── walk_forward.py     # Walk-forward testing
│   ├── purged_cv.py        # Purged K-Fold CV
│   └── cscv.py             # Combinatorial Symmetric CV
├── metrics/
│   ├── deflated_sharpe.py  # DSR calculation
│   ├── pbo.py              # PBO calculation
│   └── baseline.py         # Baseline comparisons
├── testing/
│   ├── regression.py       # Regression test framework
│   ├── edge_cases.py       # Edge case generators
│   └── fixtures.py         # Test data fixtures
└── optimization/
    ├── bayesian.py         # Bayesian optimization
    ├── regularization.py   # Complexity penalties
    └── validation.py       # Pre-deployment checks
```

---

**Document Status:** Specification Complete
**Next Steps:**
1. Implement PurgedKFold in `proofbench/validation/`
2. Add DSR/PBO to metrics module
3. Create regression test fixtures
4. Integrate with CI/CD pipeline
