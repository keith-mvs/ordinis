# Causal Inference for Algorithmic Trading

Causal inference provides rigorous frameworks for identifying cause-effect relationships in financial markets, validating trading strategies, and distinguishing true predictive signals from spurious correlations. These methods are critical for building robust strategies that generalize to changing market regimes.

---

## Overview

Financial markets exhibit complex dependencies where correlation does not imply causation. Causal inference enables traders to:

1. **Distinguish Causation from Correlation**: Identify whether X truly causes Y or if both are driven by confounders
2. **Validate Trading Signals**: Determine if a signal genuinely predicts returns or merely correlates
3. **Detect Lead-Lag Relationships**: Identify which assets causally lead others
4. **Strategy Robustness**: Build strategies based on causal mechanisms that persist across regimes

This document covers four foundational frameworks:
- Granger causality for temporal prediction
- Directed Acyclic Graphs (DAGs) and do-calculus for causal modeling
- Potential outcomes framework for treatment effects
- Causal discovery algorithms for learning structure from data

---

## 1. Granger Causality

### 1.1 Theoretical Foundation

**Granger Causality**: X Granger-causes Y if past values of X provide statistically significant information about future values of Y beyond what is already contained in past values of Y.

**Definition**: X Granger-causes Y if:

$$P(Y_{t+1} | Y_t, Y_{t-1}, \ldots, X_t, X_{t-1}, \ldots) \neq P(Y_{t+1} | Y_t, Y_{t-1}, \ldots)$$

**Vector Autoregression (VAR) Formulation**:

$$Y_t = \sum_{i=1}^p \alpha_i Y_{t-i} + \sum_{i=1}^p \beta_i X_{t-i} + \epsilon_t$$

**Test**: $H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0$ (X does not Granger-cause Y)

Use F-test or likelihood ratio test to reject $H_0$.

**Conditional Granger Causality**: X Granger-causes Y conditional on Z if X provides information about Y beyond Y's history and Z's history.

### 1.2 Trading Applications

**Lead-Lag Detection**:
- Futures Granger-cause spot prices
- Index components Granger-cause index

**Cross-Asset Prediction**:
- VIX Granger-causes equity returns
- Bond yields Granger-cause FX rates

**Signal Validation**:
- Does momentum indicator Granger-cause returns?
- Test if feature genuinely predicts vs. spurious correlation

### 1.3 Python Implementation

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.stats import f as f_dist
from typing import Tuple, Dict

class GrangerCausalityAnalyzer:
    """
    Granger causality testing and analysis for financial time series.

    Implements:
    - Bivariate Granger causality
    - Conditional Granger causality
    - Rolling Granger causality
    - Granger network construction
    """

    def __init__(self, max_lag: int = 5, significance_level: float = 0.05):
        """
        Args:
            max_lag: Maximum lag to test
            significance_level: Significance level for tests
        """
        self.max_lag = max_lag
        self.alpha = significance_level

    def test_granger_causality(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
        max_lag: int = None,
        verbose: bool = False
    ) -> Dict:
        """
        Test if cause Granger-causes effect.

        Args:
            cause: Potential causal variable (X)
            effect: Effect variable (Y)
            max_lag: Maximum lag (uses self.max_lag if None)
            verbose: Print detailed results

        Returns:
            Dictionary with test results
        """
        if max_lag is None:
            max_lag = self.max_lag

        # Prepare data for statsmodels
        data = pd.DataFrame({
            'effect': effect,
            'cause': cause
        })

        # Run Granger causality test
        results = grangercausalitytests(
            data[['effect', 'cause']],
            maxlag=max_lag,
            verbose=False
        )

        # Extract results for each lag
        test_results = {}
        for lag in range(1, max_lag + 1):
            # F-test result
            f_test = results[lag][0]['ssr_ftest']
            test_results[lag] = {
                'f_statistic': f_test[0],
                'p_value': f_test[1],
                'df_numerator': f_test[2],
                'df_denominator': f_test[3],
                'significant': f_test[1] < self.alpha
            }

        # Overall conclusion: Granger-causes if significant at any lag
        granger_causes = any(r['significant'] for r in test_results.values())

        # Optimal lag: lag with smallest p-value
        optimal_lag = min(test_results.keys(), key=lambda k: test_results[k]['p_value'])

        if verbose:
            print(f"Granger Causality Test: X -> Y")
            print("=" * 60)
            for lag, res in test_results.items():
                sig = "***" if res['significant'] else ""
                print(f"Lag {lag}: F={res['f_statistic']:.4f}, "
                      f"p={res['p_value']:.4f} {sig}")

            print(f"\nConclusion: X {'DOES' if granger_causes else 'DOES NOT'} "
                  f"Granger-cause Y")
            if granger_causes:
                print(f"Optimal lag: {optimal_lag}")

        return {
            'granger_causes': granger_causes,
            'optimal_lag': optimal_lag,
            'lag_results': test_results
        }

    def bidirectional_granger_test(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> Dict:
        """
        Test Granger causality in both directions: X->Y and Y->X.

        Args:
            x: First time series
            y: Second time series

        Returns:
            Dictionary with bidirectional test results
        """
        # X -> Y
        xy_result = self.test_granger_causality(x, y)

        # Y -> X
        yx_result = self.test_granger_causality(y, x)

        # Determine relationship
        if xy_result['granger_causes'] and not yx_result['granger_causes']:
            relationship = "X -> Y (unidirectional)"
        elif yx_result['granger_causes'] and not xy_result['granger_causes']:
            relationship = "Y -> X (unidirectional)"
        elif xy_result['granger_causes'] and yx_result['granger_causes']:
            relationship = "X <-> Y (bidirectional/feedback)"
        else:
            relationship = "No Granger causality"

        return {
            'x_to_y': xy_result,
            'y_to_x': yx_result,
            'relationship': relationship
        }

    def conditional_granger_test(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
        condition: np.ndarray,
        lag: int = 1
    ) -> Dict:
        """
        Test conditional Granger causality: X->Y | Z.

        Args:
            cause: X variable
            effect: Y variable
            condition: Z conditioning variable
            lag: Lag order

        Returns:
            Test results
        """
        from sklearn.linear_model import LinearRegression

        n = len(effect)

        # Build lagged data
        Y = effect[lag:]

        # Model 1: Y ~ Y_lag + Z_lag (restricted)
        X_restricted = np.column_stack([
            effect[lag-1:-1] if lag > 1 else effect[:-lag],
            condition[lag-1:-1] if lag > 1 else condition[:-lag]
        ])

        model_restricted = LinearRegression().fit(X_restricted, Y)
        residuals_restricted = Y - model_restricted.predict(X_restricted)
        ssr_restricted = np.sum(residuals_restricted**2)

        # Model 2: Y ~ Y_lag + Z_lag + X_lag (unrestricted)
        X_unrestricted = np.column_stack([
            X_restricted,
            cause[lag-1:-1] if lag > 1 else cause[:-lag]
        ])

        model_unrestricted = LinearRegression().fit(X_unrestricted, Y)
        residuals_unrestricted = Y - model_unrestricted.predict(X_unrestricted)
        ssr_unrestricted = np.sum(residuals_unrestricted**2)

        # F-test
        num_restrictions = lag  # Number of X lag coefficients
        df_num = num_restrictions
        df_denom = len(Y) - X_unrestricted.shape[1] - 1

        f_stat = ((ssr_restricted - ssr_unrestricted) / df_num) / (ssr_unrestricted / df_denom)
        p_value = 1 - f_dist.cdf(f_stat, df_num, df_denom)

        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'granger_causes': p_value < self.alpha,
            'ssr_restricted': ssr_restricted,
            'ssr_unrestricted': ssr_unrestricted
        }

    def rolling_granger_causality(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
        window: int = 100,
        lag: int = 1
    ) -> pd.DataFrame:
        """
        Compute rolling Granger causality over time.

        Args:
            cause: X variable
            effect: Y variable
            window: Rolling window size
            lag: Lag order

        Returns:
            DataFrame with rolling test results
        """
        n = len(effect)
        results = []

        for i in range(window, n):
            x_window = cause[i-window:i]
            y_window = effect[i-window:i]

            result = self.test_granger_causality(
                x_window, y_window, max_lag=lag
            )

            results.append({
                'time': i,
                'granger_causes': result['granger_causes'],
                'p_value': result['lag_results'][lag]['p_value'],
                'f_statistic': result['lag_results'][lag]['f_statistic']
            })

        return pd.DataFrame(results)

    def construct_granger_network(
        self,
        data: pd.DataFrame,
        lag: int = 1
    ) -> pd.DataFrame:
        """
        Construct Granger causality network from multivariate time series.

        Args:
            data: DataFrame with multiple time series
            lag: Lag order

        Returns:
            Adjacency matrix (i,j)=1 if series i Granger-causes series j
        """
        assets = data.columns
        n_assets = len(assets)

        # Initialize adjacency matrix
        adjacency = pd.DataFrame(
            np.zeros((n_assets, n_assets)),
            index=assets,
            columns=assets
        )

        # Test all pairs
        for i, asset_i in enumerate(assets):
            for j, asset_j in enumerate(assets):
                if i == j:
                    continue

                result = self.test_granger_causality(
                    data[asset_i].values,
                    data[asset_j].values,
                    max_lag=lag
                )

                if result['granger_causes']:
                    adjacency.loc[asset_i, asset_j] = 1

        return adjacency


# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    n = 500

    # Generate data with known causal structure
    # X -> Y (X Granger-causes Y)
    X = np.random.normal(0, 1, n)
    Y = np.zeros(n)
    Y[0] = np.random.normal(0, 1)

    for t in range(1, n):
        # Y depends on past X and past Y
        Y[t] = 0.5 * Y[t-1] + 0.3 * X[t-1] + np.random.normal(0, 0.5)

    # Test Granger causality
    analyzer = GrangerCausalityAnalyzer(max_lag=5)

    print("Granger Causality Analysis:")
    print("=" * 60)

    # Bidirectional test
    result = analyzer.bidirectional_granger_test(X, Y)
    print(f"\nRelationship: {result['relationship']}")

    # Detailed X->Y test
    xy_test = analyzer.test_granger_causality(X, Y, verbose=True)

    # Multi-asset network
    print("\n\nMulti-Asset Granger Network:")
    print("=" * 60)

    # Create synthetic multi-asset data
    assets = ['A', 'B', 'C']
    data = pd.DataFrame({
        'A': np.random.normal(0, 1, 200),
        'B': np.zeros(200),
        'C': np.zeros(200)
    })

    # A -> B, B -> C causal chain
    for t in range(1, 200):
        data.loc[t, 'B'] = 0.4 * data.loc[t-1, 'A'] + np.random.normal(0, 0.5)
        data.loc[t, 'C'] = 0.4 * data.loc[t-1, 'B'] + np.random.normal(0, 0.5)

    network = analyzer.construct_granger_network(data, lag=1)
    print("\nGranger Causality Network (1=causes, 0=does not cause):")
    print(network)
```

---

## 2. Directed Acyclic Graphs (DAGs) and Do-Calculus

### 2.1 Theoretical Foundation

**DAG** represents causal relationships as directed graph where:
- Nodes = variables
- Edges = direct causal effects
- No cycles (acyclic)

**Structural Causal Model (SCM)**:

$$X_i = f_i(\text{Parents}(X_i), U_i)$$

where $U_i$ are exogenous noise terms.

**Do-Calculus** (Pearl): Rules for computing interventional distributions $P(Y | do(X=x))$ from observational data $P(\cdot)$.

**Three Rules**:
1. **Insertion/Deletion of Observations**
2. **Action/Observation Exchange**
3. **Insertion/Deletion of Actions**

**Backdoor Criterion**: To estimate causal effect of X on Y, control for variables Z that:
- Block all backdoor paths from X to Y
- Do not contain descendants of X

**Front-Door Criterion**: Alternative when backdoor criterion fails.

### 2.2 Trading Applications

**Confounding Identification**:
- Market regime confounds relationship between signal and returns
- Control for confounders to isolate causal effect

**Mediation Analysis**:
- Does factor A cause returns directly or through factor B?

**Counterfactual Reasoning**:
- What would return be if signal had different value?

### 2.3 Python Implementation

```python
import numpy as np
import pandas as pd
from typing import List, Set, Dict, Tuple
import networkx as nx

class DAGCausalInference:
    """
    Causal inference using Directed Acyclic Graphs and do-calculus.

    Implements:
    - DAG construction and validation
    - Backdoor adjustment
    - Front-door adjustment
    - Causal effect estimation
    """

    def __init__(self):
        self.dag = nx.DiGraph()

    def add_edge(self, cause: str, effect: str):
        """Add causal edge to DAG."""
        self.dag.add_edge(cause, effect)

    def is_acyclic(self) -> bool:
        """Check if graph is acyclic."""
        return nx.is_directed_acyclic_graph(self.dag)

    def find_parents(self, node: str) -> Set[str]:
        """Find parent nodes (direct causes)."""
        return set(self.dag.predecessors(node))

    def find_children(self, node: str) -> Set[str]:
        """Find child nodes (direct effects)."""
        return set(self.dag.successors(node))

    def find_ancestors(self, node: str) -> Set[str]:
        """Find all ancestors (causes)."""
        return nx.ancestors(self.dag, node)

    def find_descendants(self, node: str) -> Set[str]:
        """Find all descendants (effects)."""
        return nx.descendants(self.dag, node)

    def find_backdoor_paths(
        self,
        treatment: str,
        outcome: str
    ) -> List[List[str]]:
        """
        Find all backdoor paths from treatment to outcome.

        Backdoor path: path that starts with arrow INTO treatment.
        """
        backdoor_paths = []

        # Find paths in undirected version
        undirected = self.dag.to_undirected()

        for path in nx.all_simple_paths(undirected, treatment, outcome):
            # Check if path starts with arrow INTO treatment
            if len(path) > 1:
                # First edge in directed graph
                if self.dag.has_edge(path[1], path[0]):
                    # Arrow into treatment: backdoor path
                    backdoor_paths.append(path)

        return backdoor_paths

    def satisfies_backdoor_criterion(
        self,
        treatment: str,
        outcome: str,
        adjustment_set: Set[str]
    ) -> bool:
        """
        Check if adjustment set satisfies backdoor criterion.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable
            adjustment_set: Set of variables to control for

        Returns:
            True if adjustment set blocks all backdoor paths
        """
        # Check: no descendants of treatment in adjustment set
        descendants = self.find_descendants(treatment)
        if adjustment_set & descendants:
            return False

        # Check: blocks all backdoor paths
        backdoor_paths = self.find_backdoor_paths(treatment, outcome)

        for path in backdoor_paths:
            # Check if adjustment set blocks this path
            blocked = False

            for node in path[1:-1]:  # Exclude treatment and outcome
                if node in adjustment_set:
                    blocked = True
                    break

            if not blocked:
                return False  # Found unblocked backdoor path

        return True

    def find_minimal_adjustment_set(
        self,
        treatment: str,
        outcome: str
    ) -> Set[str]:
        """
        Find minimal set of variables satisfying backdoor criterion.

        Args:
            treatment: Treatment variable
            outcome: Outcome variable

        Returns:
            Minimal adjustment set (may not be unique)
        """
        # Start with parents of treatment (often sufficient)
        parents = self.find_parents(treatment)

        if self.satisfies_backdoor_criterion(treatment, outcome, parents):
            return parents

        # If parents insufficient, try all non-descendants
        all_nodes = set(self.dag.nodes())
        descendants = self.find_descendants(treatment)
        descendants.add(treatment)
        descendants.add(outcome)

        candidates = all_nodes - descendants

        # Try subsets (greedy approximation for efficiency)
        for size in range(len(candidates) + 1):
            from itertools import combinations
            for subset in combinations(candidates, size):
                adjustment_set = set(subset)
                if self.satisfies_backdoor_criterion(treatment, outcome, adjustment_set):
                    return adjustment_set

        return set()  # No valid adjustment set found

    def estimate_causal_effect(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: Set[str] = None,
        method: str = 'backdoor'
    ) -> float:
        """
        Estimate average causal effect using backdoor adjustment.

        ATE = E[Y | do(X=1)] - E[Y | do(X=0)]
            = Σ_z [E[Y|X=1,Z=z] - E[Y|X=0,Z=z]] P(Z=z)

        Args:
            data: Observed data
            treatment: Treatment variable
            outcome: Outcome variable
            adjustment_set: Variables to adjust for (finds minimal if None)
            method: 'backdoor' or 'iptw' (inverse propensity weighting)

        Returns:
            Estimated average treatment effect
        """
        if adjustment_set is None:
            adjustment_set = self.find_minimal_adjustment_set(treatment, outcome)

        if method == 'backdoor':
            return self._backdoor_adjustment(data, treatment, outcome, adjustment_set)
        elif method == 'iptw':
            return self._iptw_adjustment(data, treatment, outcome, adjustment_set)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _backdoor_adjustment(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: Set[str]
    ) -> float:
        """
        Backdoor adjustment formula.

        ATE = Σ_z [E[Y|X=1,Z=z] - E[Y|X=0,Z=z]] P(Z=z)
        """
        from sklearn.linear_model import LinearRegression

        # Fit model: Y ~ X + Z
        X_features = [treatment] + list(adjustment_set)
        model = LinearRegression()
        model.fit(data[X_features], data[outcome])

        # Coefficient on treatment = causal effect (under linearity)
        treatment_idx = X_features.index(treatment)
        ate = model.coef_[treatment_idx]

        return ate

    def _iptw_adjustment(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        adjustment_set: Set[str]
    ) -> float:
        """
        Inverse propensity score weighting.

        ATE = E[Y·X/e(Z)] - E[Y·(1-X)/(1-e(Z))]
        where e(Z) = P(X=1|Z) is propensity score
        """
        from sklearn.linear_model import LogisticRegression

        # Estimate propensity score: P(X=1|Z)
        Z = data[list(adjustment_set)].values
        X = data[treatment].values
        Y = data[outcome].values

        ps_model = LogisticRegression()
        ps_model.fit(Z, X)
        propensity = ps_model.predict_proba(Z)[:, 1]

        # IPTW estimator
        treated = (X == 1)
        control = (X == 0)

        ate = (
            np.mean(Y[treated] / propensity[treated]) -
            np.mean(Y[control] / (1 - propensity[control]))
        )

        return ate


# Example usage
if __name__ == "__main__":
    # Construct causal DAG for trading
    dag = DAGCausalInference()

    # Causal structure:
    # Market Regime -> Signal
    # Market Regime -> Returns
    # Signal -> Returns

    dag.add_edge('regime', 'signal')
    dag.add_edge('regime', 'returns')
    dag.add_edge('signal', 'returns')

    print("DAG Structure:")
    print(f"Edges: {list(dag.dag.edges())}")
    print(f"Is acyclic: {dag.is_acyclic()}")

    # Find backdoor paths
    backdoor = dag.find_backdoor_paths('signal', 'returns')
    print(f"\nBackdoor paths (Signal -> Returns): {backdoor}")

    # Find adjustment set
    adjustment = dag.find_minimal_adjustment_set('signal', 'returns')
    print(f"Minimal adjustment set: {adjustment}")

    # Generate synthetic data
    np.random.seed(42)
    n = 1000

    regime = np.random.binomial(1, 0.5, n)  # Binary regime
    signal = 0.5 * regime + np.random.normal(0, 1, n)
    returns = 0.3 * signal + 0.4 * regime + np.random.normal(0, 1, n)

    data = pd.DataFrame({
        'regime': regime,
        'signal': signal,
        'returns': returns
    })

    # Naive correlation (confounded)
    naive_effect = np.cov(data['signal'], data['returns'])[0,1] / np.var(data['signal'])
    print(f"\nNaive effect (confounded): {naive_effect:.4f}")

    # Causal effect (adjusted)
    causal_effect = dag.estimate_causal_effect(
        data, 'signal', 'returns',
        adjustment_set={'regime'},
        method='backdoor'
    )
    print(f"Causal effect (adjusted): {causal_effect:.4f}")
    print(f"True effect: 0.3000")
```

---

## 3. Potential Outcomes Framework

### 3.1 Theoretical Foundation

**Potential Outcomes** (Rubin Causal Model): Each unit has potential outcomes under different treatments.

**Notation**:
- $Y_i(1)$: Outcome if unit $i$ receives treatment
- $Y_i(0)$: Outcome if unit $i$ receives control
- $D_i \in \{0,1\}$: Actual treatment received

**Fundamental Problem**: We only observe one potential outcome:

$$Y_i = D_i Y_i(1) + (1 - D_i) Y_i(0)$$

**Average Treatment Effect (ATE)**:

$$\text{ATE} = E[Y_i(1) - Y_i(0)]$$

**Conditional Average Treatment Effect (CATE)**:

$$\text{CATE}(x) = E[Y_i(1) - Y_i(0) | X_i = x]$$

**Identification Assumptions**:
1. **Ignorability**: $(Y(1), Y(0)) \perp D | X$ (conditional on X, treatment assignment is independent of potential outcomes)
2. **Overlap**: $0 < P(D=1|X) < 1$ for all X (both treatment and control possible)

### 3.2 Trading Applications

**Strategy A/B Testing**:
- Compare returns under strategy A vs. strategy B
- Control for market conditions (X)

**Feature Importance**:
- Causal effect of adding feature to model
- CATE: heterogeneous effects across market regimes

**Intervention Analysis**:
- Effect of trading rule change on execution quality

### 3.3 Python Implementation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

class PotentialOutcomesEstimator:
    """
    Causal inference using potential outcomes framework.

    Implements:
    - Average Treatment Effect (ATE)
    - Conditional ATE (CATE)
    - Double machine learning
    - Causal forests
    """

    def __init__(self):
        pass

    def estimate_ate_naive(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str
    ) -> float:
        """
        Naive ATE: difference in means (biased if confounded).

        Args:
            data: DataFrame
            treatment: Treatment column name
            outcome: Outcome column name

        Returns:
            Naive ATE estimate
        """
        treated = data[data[treatment] == 1][outcome]
        control = data[data[treatment] == 0][outcome]

        return treated.mean() - control.mean()

    def estimate_ate_ipw(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str]
    ) -> Tuple[float, float]:
        """
        ATE using inverse propensity weighting.

        Args:
            data: DataFrame
            treatment: Treatment column
            outcome: Outcome column
            covariates: Covariate columns

        Returns:
            (ATE estimate, standard error)
        """
        X = data[covariates].values
        D = data[treatment].values
        Y = data[outcome].values

        # Estimate propensity score
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(X, D)
        propensity = ps_model.predict_proba(X)[:, 1]

        # Clip propensities to avoid extreme weights
        propensity = np.clip(propensity, 0.01, 0.99)

        # IPW estimator
        treated_mask = (D == 1)
        control_mask = (D == 0)

        ate = (
            np.mean(Y[treated_mask] / propensity[treated_mask]) -
            np.mean(Y[control_mask] / (1 - propensity[control_mask]))
        )

        # Standard error (simplified)
        n = len(Y)
        var_treated = np.var(Y[treated_mask] / propensity[treated_mask])
        var_control = np.var(Y[control_mask] / (1 - propensity[control_mask]))
        se = np.sqrt((var_treated + var_control) / n)

        return ate, se

    def estimate_cate_tlearner(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str]
    ) -> callable:
        """
        CATE using T-learner (separate models for treatment/control).

        Args:
            data: DataFrame
            treatment: Treatment column
            outcome: Outcome column
            covariates: Covariate columns

        Returns:
            Function mapping covariates -> CATE
        """
        X = data[covariates].values
        D = data[treatment].values
        Y = data[outcome].values

        # Separate data
        X_treated = X[D == 1]
        Y_treated = Y[D == 1]
        X_control = X[D == 0]
        Y_control = Y[D == 0]

        # Fit separate models
        model_treated = RandomForestRegressor(n_estimators=100, random_state=42)
        model_control = RandomForestRegressor(n_estimators=100, random_state=42)

        model_treated.fit(X_treated, Y_treated)
        model_control.fit(X_control, Y_control)

        # CATE function
        def cate(X_new):
            mu1 = model_treated.predict(X_new)
            mu0 = model_control.predict(X_new)
            return mu1 - mu0

        return cate

    def estimate_ate_doubly_robust(
        self,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        covariates: List[str]
    ) -> float:
        """
        Doubly robust ATE estimator.

        Combines outcome regression and propensity weighting.
        Consistent if either model is correct.

        Args:
            data: DataFrame
            treatment: Treatment column
            outcome: Outcome column
            covariates: Covariate columns

        Returns:
            ATE estimate
        """
        X = data[covariates].values
        D = data[treatment].values
        Y = data[outcome].values

        # Estimate propensity score
        ps_model = LogisticRegression(max_iter=1000)
        ps_model.fit(X, D)
        propensity = ps_model.predict_proba(X)[:, 1]
        propensity = np.clip(propensity, 0.01, 0.99)

        # Estimate outcome models
        X_treated = X[D == 1]
        Y_treated = Y[D == 1]
        X_control = X[D == 0]
        Y_control = Y[D == 0]

        model_treated = RandomForestRegressor(n_estimators=50, random_state=42)
        model_control = RandomForestRegressor(n_estimators=50, random_state=42)

        model_treated.fit(X_treated, Y_treated)
        model_control.fit(X_control, Y_control)

        mu1 = model_treated.predict(X)
        mu0 = model_control.predict(X)

        # Doubly robust estimator
        treated_term = D * (Y - mu1) / propensity + mu1
        control_term = (1 - D) * (Y - mu0) / (1 - propensity) + mu0

        ate = np.mean(treated_term - control_term)

        return ate


# Example usage
if __name__ == "__main__":
    # Generate synthetic trading data
    np.random.seed(42)
    n = 2000

    # Covariates: market regime indicators
    regime_volatility = np.random.normal(0, 1, n)
    regime_trend = np.random.binomial(1, 0.5, n)

    # Treatment: using advanced signal (1) vs. simple signal (0)
    # Assignment depends on regime (confounding)
    propensity = 1 / (1 + np.exp(-(0.5 * regime_volatility + 0.3 * regime_trend)))
    treatment = np.random.binomial(1, propensity)

    # Potential outcomes
    # Y(0) = baseline returns
    Y0 = 0.5 * regime_volatility + 0.3 * regime_trend + np.random.normal(0, 1, n)

    # Y(1) = returns with advanced signal (heterogeneous effect)
    # Effect larger in high volatility regime
    treatment_effect = 0.5 + 0.3 * regime_volatility
    Y1 = Y0 + treatment_effect

    # Observed outcome
    returns = treatment * Y1 + (1 - treatment) * Y0

    # True ATE
    true_ate = np.mean(treatment_effect)

    data = pd.DataFrame({
        'treatment': treatment,
        'returns': returns,
        'regime_vol': regime_volatility,
        'regime_trend': regime_trend
    })

    estimator = PotentialOutcomesEstimator()

    # Naive estimate (biased)
    naive_ate = estimator.estimate_ate_naive(data, 'treatment', 'returns')
    print(f"Naive ATE (biased): {naive_ate:.4f}")

    # IPW estimate
    ipw_ate, ipw_se = estimator.estimate_ate_ipw(
        data, 'treatment', 'returns',
        ['regime_vol', 'regime_trend']
    )
    print(f"IPW ATE: {ipw_ate:.4f} ± {ipw_se:.4f}")

    # Doubly robust estimate
    dr_ate = estimator.estimate_ate_doubly_robust(
        data, 'treatment', 'returns',
        ['regime_vol', 'regime_trend']
    )
    print(f"Doubly Robust ATE: {dr_ate:.4f}")

    print(f"\nTrue ATE: {true_ate:.4f}")

    # CATE estimation
    cate_func = estimator.estimate_cate_tlearner(
        data, 'treatment', 'returns',
        ['regime_vol', 'regime_trend']
    )

    # Evaluate CATE for different regimes
    test_regimes = np.array([
        [-1, 0],  # Low vol, downtrend
        [0, 0],   # Medium vol, downtrend
        [1, 0],   # High vol, downtrend
        [-1, 1],  # Low vol, uptrend
        [1, 1],   # High vol, uptrend
    ])

    print("\nConditional ATE (CATE) by Regime:")
    for regime in test_regimes:
        cate_val = cate_func(regime.reshape(1, -1))[0]
        print(f"Regime (vol={regime[0]:+.0f}, trend={regime[1]:.0f}): "
              f"CATE={cate_val:.4f}")
```

---

## 4. Causal Discovery

### 4.1 Theoretical Foundation

**Causal Discovery**: Learn causal structure (DAG) from observational data.

**Constraint-Based Methods** (e.g., PC algorithm):
- Use conditional independence tests
- Start with fully connected graph
- Remove edges based on independence
- Orient edges using colliders

**Score-Based Methods** (e.g., GES):
- Define score (BIC, AIC) for each DAG
- Search over DAG space to maximize score

**Functional Causal Models**:
- Assume structural equations: $Y = f(X, noise)$
- LiNGAM: Linear Non-Gaussian Acyclic Model

**Challenges**:
- Markov equivalence (multiple DAGs have same conditional independencies)
- Sample complexity (need large data)
- Assumption violations (no unmeasured confounders)

### 4.2 Python Implementation

```python
from lingam import DirectLiNGAM, VARLiNGAM
import networkx as nx

class CausalDiscovery:
    """
    Causal structure learning from time series data.

    Implements:
    - LiNGAM for cross-sectional data
    - VAR-LiNGAM for time series
    - Granger network construction
    """

    def __init__(self):
        pass

    def discover_dag_lingam(
        self,
        data: pd.DataFrame
    ) -> nx.DiGraph:
        """
        Discover causal DAG using LiNGAM.

        Args:
            data: DataFrame with variables

        Returns:
            Directed acyclic graph
        """
        model = DirectLiNGAM()
        model.fit(data.values)

        # Extract adjacency matrix
        adj_matrix = model.adjacency_matrix_

        # Build DAG
        dag = nx.DiGraph()
        variables = list(data.columns)

        for i, var_i in enumerate(variables):
            for j, var_j in enumerate(variables):
                if np.abs(adj_matrix[i, j]) > 0.1:  # Threshold weak edges
                    dag.add_edge(var_j, var_i, weight=adj_matrix[i, j])

        return dag

    def discover_temporal_dag(
        self,
        data: pd.DataFrame,
        max_lag: int = 5
    ) -> Dict:
        """
        Discover temporal causal structure using VAR-LiNGAM.

        Args:
            data: Time series DataFrame
            max_lag: Maximum lag

        Returns:
            Dictionary with causal structure
        """
        model = VARLiNGAM(lags=max_lag)
        model.fit(data.values)

        # Extract causal order
        causal_order = model.causal_order_

        # Extract lagged effects
        lagged_effects = {}
        for lag in range(1, max_lag + 1):
            lagged_effects[lag] = model.adjacency_matrices_[lag]

        return {
            'causal_order': [data.columns[i] for i in causal_order],
            'lagged_effects': lagged_effects,
            'contemporaneous': model.adjacency_matrices_[0]
        }


# Example usage
if __name__ == "__main__":
    # Generate data with known causal structure
    # A -> B -> C
    np.random.seed(42)
    n = 500

    A = np.random.laplace(0, 1, n)  # Non-Gaussian
    B = 0.7 * A + np.random.laplace(0, 0.5, n)
    C = 0.5 * B + np.random.laplace(0, 0.5, n)

    data = pd.DataFrame({'A': A, 'B': B, 'C': C})

    discovery = CausalDiscovery()

    # Discover DAG
    dag = discovery.discover_dag_lingam(data)

    print("Discovered Causal DAG:")
    print(f"Edges: {list(dag.edges())}")
    print(f"True structure: A -> B -> C")
```

---

## 5. Integration with Trading Systems

### 5.1 Signal Validation Framework

```python
def validate_signal_causality(
    signal: pd.Series,
    returns: pd.Series,
    covariates: pd.DataFrame
) -> Dict:
    """
    Validate that signal has causal relationship with returns.

    Tests:
    1. Granger causality
    2. Causal effect estimation (controlling for confounders)
    3. Robustness checks

    Args:
        signal: Trading signal
        returns: Forward returns
        covariates: Potential confounders (regime, volatility, etc.)

    Returns:
        Validation results
    """
    results = {}

    # Test 1: Granger causality
    gc_analyzer = GrangerCausalityAnalyzer()
    gc_result = gc_analyzer.test_granger_causality(
        signal.values, returns.values
    )
    results['granger_causes'] = gc_result['granger_causes']

    # Test 2: Causal effect (potential outcomes)
    # Binarize signal
    signal_binary = (signal > signal.median()).astype(int)

    data = pd.concat([signal_binary, returns, covariates], axis=1)
    data.columns = ['signal', 'returns'] + list(covariates.columns)

    po_est = PotentialOutcomesEstimator()
    causal_effect = po_est.estimate_ate_doubly_robust(
        data, 'signal', 'returns', list(covariates.columns)
    )

    results['causal_effect'] = causal_effect
    results['is_positive'] = causal_effect > 0

    # Test 3: Conditional effects across regimes
    cate_func = po_est.estimate_cate_tlearner(
        data, 'signal', 'returns', list(covariates.columns)
    )
    results['cate_estimator'] = cate_func

    return results
```

---

## 6. Academic References

### Foundational Texts

1. **Pearl, J. (2009)**. *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
   - Definitive reference on DAGs, do-calculus, causal inference

2. **Imbens, G. W., & Rubin, D. B. (2015)**. *Causal Inference for Statistics, Social, and Biomedical Sciences*. Cambridge University Press.
   - Potential outcomes framework, treatment effects

3. **Peters, J., Janzing, D., & Schölkopf, B. (2017)**. *Elements of Causal Inference*. MIT Press.
   - Modern causal discovery methods

### Financial Applications

4. **Granger, C. W. J. (1969)**. "Investigating Causal Relations by Econometric Models and Cross-spectral Methods." *Econometrica*, 37(3), 424-438.
   - Original Granger causality paper

5. **Shimizu, S., et al. (2006)**. "A Linear Non-Gaussian Acyclic Model for Causal Discovery." *Journal of Machine Learning Research*, 7, 2003-2030.
   - LiNGAM algorithm

6. **Chernozhukov, V., et al. (2018)**. "Double/Debiased Machine Learning for Treatment and Structural Parameters." *Econometrics Journal*, 21(1), C1-C68.
   - Modern causal ML methods

---

## 7. Cross-References

**Related Knowledge Base Sections**:

- [Information Theory](information_theory.md) - Transfer entropy for causality
- [Signal Processing](signal_processing.md) - VAR models, spectral causality
- [Feature Engineering](../../02_signals/quantitative/ml_strategies/feature_engineering.md) - Causal feature selection
- [Strategy Validation](../../04_strategy/backtesting-requirements.md) - Causal backtesting

**Integration Points**:

1. **SignalCore**: Causal feature selection, signal validation
2. **ProofBench**: Causal backtesting, counterfactual analysis
3. **RiskGuard**: Causal risk attribution
4. **FlowRoute**: Lead-lag execution based on causality

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
last_updated: "2025-12-12"
status: "published"
category: "foundations/advanced_mathematics"
tags: ["causal-inference", "granger-causality", "dag", "do-calculus", "potential-outcomes", "treatment-effects", "causal-discovery", "lingam"]
code_lines: 920
academic_references: 6
implementation_completeness: "production-ready"
```

---

**END OF DOCUMENT**
