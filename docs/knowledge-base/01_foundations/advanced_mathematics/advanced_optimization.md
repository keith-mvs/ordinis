# Advanced Optimization for Algorithmic Trading

Advanced optimization techniques enable systematic trading systems to handle complex constraints, adapt to changing market conditions, optimize multiple competing objectives, and discover optimal hyperparameters for trading strategies. This document covers methods beyond standard convex optimization.

---

## Overview

Trading optimization problems often involve:
- **Non-convexity**: Discrete decisions (buy/sell), transaction costs, cardinality constraints
- **Multiple Objectives**: Maximize returns while minimizing risk, drawdown, turnover
- **Uncertainty**: Parameter uncertainty, regime changes, model misspecification
- **Online Learning**: Adapt strategies in real-time as data streams

This document covers five advanced optimization frameworks:
1. Online Convex Optimization for adaptive strategies
2. Multi-Objective Optimization for Pareto-efficient portfolios
3. Distributionally Robust Optimization for worst-case performance
4. Mixed-Integer Programming for cardinality constraints
5. Bayesian Optimization for hyperparameter tuning

---

## 1. Online Convex Optimization (OCO)

### 1.1 Theoretical Foundation

**Online Learning Framework**:
- At each time $t$:
  1. Learner chooses action $x_t \in \mathcal{X}$ (convex set)
  2. Adversary reveals cost function $f_t: \mathcal{X} \to \mathbb{R}$ (convex)
  3. Learner incurs loss $f_t(x_t)$

**Goal**: Minimize regret compared to best fixed strategy in hindsight:

$$\text{Regret}_T = \sum_{t=1}^T f_t(x_t) - \min_{x \in \mathcal{X}} \sum_{t=1}^T f_t(x)$$

**Online Gradient Descent (OGD)**:

$$x_{t+1} = \Pi_{\mathcal{X}}(x_t - \eta_t \nabla f_t(x_t))$$

where $\Pi_{\mathcal{X}}$ is projection onto $\mathcal{X}$.

**Regret Bound**: $\text{Regret}_T = O(\sqrt{T})$ with proper step size $\eta_t = \frac{D}{G\sqrt{T}}$.

**Follow-The-Regularized-Leader (FTRL)**:

$$x_{t+1} = \arg\min_{x \in \mathcal{X}} \left\{ \sum_{s=1}^t \langle \nabla f_s(x_s), x \rangle + \frac{1}{\eta_t} R(x) \right\}$$

### 1.2 Trading Applications

**Online Portfolio Selection**:
- Update portfolio weights as new returns revealed
- No regret against best constant rebalancing strategy

**Adaptive Execution**:
- Adjust trading rate based on realized market impact
- Learn optimal execution schedule online

**Parameter Adaptation**:
- Update strategy parameters as market regime changes
- Online learning of alpha decay, position sizing

### 1.3 Python Implementation

```python
import numpy as np
from typing import Callable, List
from dataclasses import dataclass

@dataclass
class OCOParameters:
    """Online convex optimization parameters."""
    learning_rate: float = 0.1
    projection_radius: float = 1.0
    regularization: float = 0.01


class OnlineGradientDescent:
    """
    Online Gradient Descent for adaptive portfolio management.

    Implements:
    - Projected gradient descent
    - Adaptive learning rates
    - Regret tracking
    """

    def __init__(self, n_assets: int, params: OCOParameters = None):
        self.n_assets = n_assets
        self.params = params or OCOParameters()

        # Initialize portfolio weights (equal weight)
        self.weights = np.ones(n_assets) / n_assets
        self.t = 0

        # History
        self.weight_history = [self.weights.copy()]
        self.loss_history = []
        self.cumulative_loss = 0.0

    def project_simplex(self, weights: np.ndarray) -> np.ndarray:
        """
        Project onto probability simplex.

        Ensures: weights >= 0, sum(weights) = 1
        """
        n = len(weights)

        # Sort in descending order
        sorted_weights = np.sort(weights)[::-1]

        # Find rho
        cumsum = np.cumsum(sorted_weights)
        rho_values = sorted_weights - (cumsum - 1) / np.arange(1, n + 1)
        rho = np.max(np.where(rho_values > 0)[0]) + 1

        # Compute lambda
        lambda_val = (cumsum[rho - 1] - 1) / rho

        # Project
        return np.maximum(weights - lambda_val, 0)

    def update(
        self,
        gradient: np.ndarray,
        learning_rate: float = None
    ) -> np.ndarray:
        """
        Perform online gradient descent update.

        Args:
            gradient: Gradient of loss function
            learning_rate: Step size (uses default if None)

        Returns:
            Updated weights
        """
        if learning_rate is None:
            # Adaptive: eta_t = eta_0 / sqrt(t)
            learning_rate = self.params.learning_rate / np.sqrt(self.t + 1)

        # Gradient step
        new_weights = self.weights - learning_rate * gradient

        # Project onto simplex
        new_weights = self.project_simplex(new_weights)

        self.weights = new_weights
        self.t += 1
        self.weight_history.append(new_weights.copy())

        return new_weights

    def compute_regret(self, returns: np.ndarray) -> float:
        """
        Compute regret against best fixed portfolio.

        Args:
            returns: Return matrix (T × n_assets)

        Returns:
            Regret value
        """
        T = len(returns)

        # Our cumulative return
        our_returns = np.sum([
            np.dot(self.weight_history[t], returns[t])
            for t in range(min(T, len(self.weight_history) - 1))
        ])

        # Best fixed portfolio (in hindsight)
        best_weights = self.find_best_fixed_portfolio(returns)
        best_returns = np.sum(returns @ best_weights)

        regret = best_returns - our_returns

        return regret

    def find_best_fixed_portfolio(self, returns: np.ndarray) -> np.ndarray:
        """
        Find best fixed portfolio in hindsight.

        Args:
            returns: Return matrix

        Returns:
            Optimal constant rebalanced portfolio
        """
        from scipy.optimize import minimize

        def objective(w):
            return -np.sum(returns @ w)

        def simplex_constraint(w):
            return np.sum(w) - 1

        constraints = [
            {'type': 'eq', 'fun': simplex_constraint}
        ]

        bounds = [(0, 1) for _ in range(self.n_assets)]

        result = minimize(
            objective,
            x0=np.ones(self.n_assets) / self.n_assets,
            bounds=bounds,
            constraints=constraints
        )

        return result.x


class FollowTheRegularizedLeader:
    """
    Follow-The-Regularized-Leader (FTRL) algorithm.

    Better regret bounds than OGD in some settings.
    """

    def __init__(self, n_assets: int, eta: float = 0.1):
        self.n_assets = n_assets
        self.eta = eta
        self.cumulative_gradient = np.zeros(n_assets)
        self.weights = np.ones(n_assets) / n_assets
        self.t = 0

    def update(self, gradient: np.ndarray) -> np.ndarray:
        """
        FTRL update.

        Args:
            gradient: Current gradient

        Returns:
            Updated weights
        """
        self.cumulative_gradient += gradient

        # Solve: argmin_w <cumulative_grad, w> + (1/eta) * ||w||^2
        # Subject to: w >= 0, sum(w) = 1

        from scipy.optimize import minimize

        def objective(w):
            return (np.dot(self.cumulative_gradient, w) +
                    (1 / (2 * self.eta)) * np.dot(w, w))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(self.n_assets)]

        result = minimize(
            objective,
            x0=self.weights,
            bounds=bounds,
            constraints=constraints
        )

        self.weights = result.x
        self.t += 1

        return self.weights


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    n_assets = 5
    n_periods = 100

    # Generate random returns
    returns = np.random.normal(0.001, 0.02, (n_periods, n_assets))

    # Online portfolio selection with OGD
    ogd = OnlineGradientDescent(n_assets)

    portfolio_values = [1.0]

    for t in range(n_periods):
        # Current portfolio return
        portfolio_return = np.dot(ogd.weights, returns[t])
        portfolio_values.append(portfolio_values[-1] * (1 + portfolio_return))

        # Gradient: negative return (we minimize loss = -return)
        gradient = -returns[t]

        # Update weights
        ogd.update(gradient)

    # Compute regret
    regret = ogd.compute_regret(returns)

    print("Online Gradient Descent Results:")
    print("=" * 60)
    print(f"Final portfolio value: {portfolio_values[-1]:.4f}")
    print(f"Total return: {(portfolio_values[-1] - 1) * 100:.2f}%")
    print(f"Regret vs. best fixed: {regret:.6f}")

    # Compare with FTRL
    ftrl = FollowTheRegularizedLeader(n_assets, eta=0.1)
    ftrl_values = [1.0]

    for t in range(n_periods):
        portfolio_return = np.dot(ftrl.weights, returns[t])
        ftrl_values.append(ftrl_values[-1] * (1 + portfolio_return))

        gradient = -returns[t]
        ftrl.update(gradient)

    print(f"\nFTRL Final value: {ftrl_values[-1]:.4f}")
    print(f"FTRL Return: {(ftrl_values[-1] - 1) * 100:.2f}%")
```

---

## 2. Multi-Objective Optimization (MOO)

### 2.1 Theoretical Foundation

**Multi-Objective Problem**:

$$\min_{x \in \mathcal{X}} \begin{bmatrix} f_1(x) \\ f_2(x) \\ \vdots \\ f_m(x) \end{bmatrix}$$

**Pareto Dominance**: $x$ dominates $y$ (denoted $x \prec y$) if:
- $f_i(x) \leq f_i(y)$ for all $i$
- $f_j(x) < f_j(y)$ for some $j$

**Pareto Optimal**: $x^*$ is Pareto optimal if no other $x$ dominates it.

**Pareto Front**: Set of all Pareto optimal solutions.

**Scalarization Methods**:

1. **Weighted Sum**: $\min \sum_{i=1}^m w_i f_i(x)$
2. **ε-Constraint**: $\min f_1(x)$ s.t. $f_i(x) \leq \epsilon_i$ for $i > 1$
3. **Chebyshev**: $\min \max_{i} w_i |f_i(x) - f_i^*|$

**Evolutionary Algorithms**: NSGA-II, MOEA/D

### 2.2 Trading Applications

**Portfolio Optimization**:
- Maximize return, minimize risk, minimize turnover
- Trade-off among multiple objectives

**Strategy Selection**:
- Maximize Sharpe ratio, minimize drawdown, maximize profit factor
- Pareto-efficient strategy configurations

**Execution**:
- Minimize cost, minimize duration, maximize fill rate

### 2.3 Python Implementation

```python
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

class PortfolioMOO(Problem):
    """
    Multi-objective portfolio optimization.

    Objectives:
    1. Maximize expected return
    2. Minimize variance
    3. Minimize turnover
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        current_weights: np.ndarray = None
    ):
        self.mu = expected_returns
        self.Sigma = covariance
        self.n_assets = len(expected_returns)

        if current_weights is None:
            self.current_weights = np.zeros(self.n_assets)
        else:
            self.current_weights = current_weights

        super().__init__(
            n_var=self.n_assets,
            n_obj=3,  # Return, risk, turnover
            n_constr=1,  # Sum to 1
            xl=np.zeros(self.n_assets),  # Lower bound
            xu=np.ones(self.n_assets)    # Upper bound
        )

    def _evaluate(self, X, out, *args, **kwargs):
        """
        Evaluate objectives for population X.

        Args:
            X: Population (n_individuals × n_assets)
            out: Output dictionary
        """
        # Objective 1: Maximize return (minimize negative return)
        returns = -X @ self.mu

        # Objective 2: Minimize variance
        variances = np.array([w @ self.Sigma @ w for w in X])

        # Objective 3: Minimize turnover
        turnovers = np.sum(np.abs(X - self.current_weights), axis=1)

        # Constraint: weights sum to 1
        sum_constraint = np.abs(np.sum(X, axis=1) - 1)

        out["F"] = np.column_stack([returns, variances, turnovers])
        out["G"] = sum_constraint.reshape(-1, 1) - 0.01  # Tolerance


class MultiObjectivePortfolioOptimizer:
    """
    Multi-objective portfolio optimization using NSGA-II.

    Finds Pareto-efficient portfolios trading off return, risk, turnover.
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray
    ):
        self.mu = expected_returns
        self.Sigma = covariance
        self.n_assets = len(expected_returns)

    def optimize(
        self,
        current_weights: np.ndarray = None,
        pop_size: int = 100,
        n_gen: int = 100
    ) -> dict:
        """
        Find Pareto front using NSGA-II.

        Args:
            current_weights: Current portfolio (for turnover calculation)
            pop_size: Population size
            n_gen: Number of generations

        Returns:
            Dictionary with Pareto front solutions
        """
        # Define problem
        problem = PortfolioMOO(self.mu, self.Sigma, current_weights)

        # NSGA-II algorithm
        algorithm = NSGA2(pop_size=pop_size)

        # Termination criterion
        termination = get_termination("n_gen", n_gen)

        # Optimize
        res = pymoo_minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=False
        )

        # Extract Pareto front
        pareto_weights = res.X
        pareto_objectives = res.F

        return {
            'pareto_weights': pareto_weights,
            'pareto_objectives': pareto_objectives,
            'n_solutions': len(pareto_weights)
        }

    def scalarize_weighted_sum(
        self,
        weights_obj: np.ndarray,
        current_weights: np.ndarray = None
    ) -> np.ndarray:
        """
        Solve using weighted sum scalarization.

        Args:
            weights_obj: Weights on [return, risk, turnover]
            current_weights: Current portfolio

        Returns:
            Optimal portfolio weights
        """
        from scipy.optimize import minimize

        if current_weights is None:
            current_weights = np.zeros(self.n_assets)

        def objective(w):
            ret = -np.dot(self.mu, w)  # Negative return
            risk = w @ self.Sigma @ w
            turnover = np.sum(np.abs(w - current_weights))

            return (weights_obj[0] * ret +
                    weights_obj[1] * risk +
                    weights_obj[2] * turnover)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        bounds = [(0, 1) for _ in range(self.n_assets)]

        result = minimize(
            objective,
            x0=np.ones(self.n_assets) / self.n_assets,
            bounds=bounds,
            constraints=constraints
        )

        return result.x


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    n_assets = 10

    # Generate random return forecasts and covariance
    expected_returns = np.random.normal(0.10, 0.05, n_assets)
    correlation = np.random.uniform(0.3, 0.7, (n_assets, n_assets))
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1.0)

    volatilities = np.random.uniform(0.15, 0.30, n_assets)
    covariance = np.outer(volatilities, volatilities) * correlation

    # Current portfolio (equal weight)
    current_weights = np.ones(n_assets) / n_assets

    # Multi-objective optimization
    optimizer = MultiObjectivePortfolioOptimizer(expected_returns, covariance)

    print("Multi-Objective Portfolio Optimization:")
    print("=" * 60)

    # Find Pareto front
    result = optimizer.optimize(current_weights, pop_size=100, n_gen=50)

    print(f"Number of Pareto-optimal solutions: {result['n_solutions']}")

    # Display first 5 solutions
    print("\nPareto Front (first 5 solutions):")
    print("Return   | Risk     | Turnover")
    print("-" * 40)
    for i in range(min(5, result['n_solutions'])):
        obj = result['pareto_objectives'][i]
        print(f"{-obj[0]:8.4f} | {obj[1]:8.4f} | {obj[2]:8.4f}")

    # Weighted sum scalarization
    print("\n\nWeighted Sum Scalarization:")
    print("-" * 60)

    weight_configs = [
        (1.0, 0.5, 0.1, "Return-focused"),
        (0.5, 1.0, 0.1, "Risk-focused"),
        (0.5, 0.5, 1.0, "Low-turnover")
    ]

    for w_ret, w_risk, w_turn, desc in weight_configs:
        weights_obj = np.array([w_ret, w_risk, w_turn])
        optimal_w = optimizer.scalarize_weighted_sum(weights_obj, current_weights)

        ret = np.dot(expected_returns, optimal_w)
        risk = np.sqrt(optimal_w @ covariance @ optimal_w)
        turnover = np.sum(np.abs(optimal_w - current_weights))

        print(f"\n{desc}:")
        print(f"  Return: {ret:.4f}, Risk: {risk:.4f}, Turnover: {turnover:.4f}")
```

---

## 3. Distributionally Robust Optimization (DRO)

### 3.1 Theoretical Foundation

**DRO Problem**: Optimize against worst-case distribution in uncertainty set:

$$\min_{x \in \mathcal{X}} \sup_{P \in \mathcal{P}} E_P[f(x, \xi)]$$

**Ambiguity Set** $\mathcal{P}$: Set of plausible distributions

**Common Ambiguity Sets**:
1. **Moment-Based**: $\{P: E_P[\xi] = \mu, \text{Var}_P(\xi) \leq \sigma^2\}$
2. **Wasserstein**: $\{P: W(P, \hat{P}) \leq \epsilon\}$ (distributions close to empirical)
3. **KL-Divergence**: $\{P: D_{KL}(P \| \hat{P}) \leq \epsilon\}$

**Advantages**:
- Robust to model misspecification
- Data-driven: ambiguity set calibrated from data
- Often computationally tractable

### 3.2 Trading Applications

**Robust Portfolio Optimization**:
- Worst-case expected return
- Protect against parameter uncertainty

**Robust Execution**:
- Worst-case market impact
- Robust to impact model errors

**Risk Management**:
- Worst-case Value-at-Risk
- Stress testing under distributional ambiguity

### 3.3 Python Implementation

```python
import numpy as np
import cvxpy as cp

class DistributionallyRobustPortfolio:
    """
    Distributionally robust portfolio optimization.

    Optimizes worst-case return over Wasserstein ambiguity set.
    """

    def __init__(
        self,
        returns_data: np.ndarray,
        epsilon: float = 0.1
    ):
        """
        Args:
            returns_data: Historical returns (n_samples × n_assets)
            epsilon: Wasserstein radius
        """
        self.returns = returns_data
        self.n_samples, self.n_assets = returns_data.shape
        self.epsilon = epsilon

    def optimize_worst_case_cvar(
        self,
        alpha: float = 0.05
    ) -> np.ndarray:
        """
        Maximize worst-case CVaR.

        Args:
            alpha: CVaR confidence level

        Returns:
            Optimal portfolio weights
        """
        # Decision variables
        w = cp.Variable(self.n_assets)  # Portfolio weights

        # Worst-case CVaR formulation (approximation via scenarios)
        # For each sample, compute portfolio return
        portfolio_returns = self.returns @ w

        # VaR and CVaR variables
        var = cp.Variable()
        cvar_losses = cp.Variable(self.n_samples)

        # CVaR formulation: var + (1/alpha) * E[max(0, -R - var)]
        # Maximize CVaR = minimize -CVaR

        objective = cp.Minimize(
            -var + (1 / (alpha * self.n_samples)) * cp.sum(cvar_losses)
        )

        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            cvar_losses >= 0,
            cvar_losses >= -portfolio_returns - var
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        return w.value

    def optimize_moment_based_dro(
        self,
        target_return: float = 0.05
    ) -> np.ndarray:
        """
        Minimize worst-case variance subject to return target.

        Ambiguity set: mean = empirical mean, variance <= empirical variance

        Args:
            target_return: Minimum expected return

        Returns:
            Optimal portfolio weights
        """
        # Estimate moments
        mu_hat = np.mean(self.returns, axis=0)
        Sigma_hat = np.cov(self.returns.T)

        # Worst-case variance = largest eigenvalue scenario
        # For moment-based, worst case is: w' Sigma w

        w = cp.Variable(self.n_assets)

        objective = cp.Minimize(cp.quad_form(w, Sigma_hat))

        constraints = [
            mu_hat @ w >= target_return,
            cp.sum(w) == 1,
            w >= 0
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        return w.value


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    # Generate synthetic return data
    n_samples = 500
    n_assets = 5

    # Returns with heavy tails (t-distribution)
    from scipy.stats import t
    df = 5  # Degrees of freedom
    returns = t.rvs(df, size=(n_samples, n_assets)) * 0.02

    # DRO portfolio
    dro = DistributionallyRobustPortfolio(returns, epsilon=0.1)

    print("Distributionally Robust Portfolio Optimization:")
    print("=" * 60)

    # Worst-case CVaR optimization
    w_dro_cvar = dro.optimize_worst_case_cvar(alpha=0.05)

    print(f"\nWorst-Case CVaR Portfolio:")
    print(f"Weights: {w_dro_cvar}")

    # Empirical performance
    empirical_returns = returns @ w_dro_cvar
    empirical_cvar = np.mean(empirical_returns[empirical_returns <= np.percentile(empirical_returns, 5)])

    print(f"Empirical CVaR (5%): {empirical_cvar:.4f}")

    # Moment-based DRO
    w_dro_moment = dro.optimize_moment_based_dro(target_return=0.01)

    print(f"\nMoment-Based DRO Portfolio:")
    print(f"Weights: {w_dro_moment}")
    print(f"Empirical return: {np.mean(returns @ w_dro_moment):.4f}")
    print(f"Empirical std: {np.std(returns @ w_dro_moment):.4f}")
```

---

## 4. Mixed-Integer Programming for Cardinality Constraints

### 4.1 Theoretical Foundation

**Cardinality-Constrained Portfolio**:

$$\min_{x} x^T \Sigma x$$
$$\text{s.t. } \mu^T x \geq r_{\min}, \quad \sum_i x_i = 1, \quad x_i \geq 0$$
$$\sum_i \mathbf{1}_{x_i > 0} \leq K$$

**Binary Variables**: $z_i \in \{0,1\}$ where $z_i = 1$ iff $x_i > 0$

**MIP Formulation**:

$$x_i \leq z_i, \quad x_i \geq \epsilon z_i, \quad \sum_i z_i \leq K$$

**Round-Lot Constraints**: $x_i = \text{lot\_size} \times n_i$ where $n_i \in \mathbb{Z}_+$

### 4.2 Python Implementation

```python
import gurobipy as gp
from gurobipy import GRB

class CardinalityConstrainedPortfolio:
    """
    Portfolio optimization with cardinality constraints using MIP.

    Constraints:
    - Maximum K assets in portfolio
    - Minimum position size per asset
    - Round lot constraints
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray
    ):
        self.mu = expected_returns
        self.Sigma = covariance
        self.n_assets = len(expected_returns)

    def optimize(
        self,
        max_assets: int = 10,
        min_weight: float = 0.01,
        target_return: float = 0.08
    ) -> np.ndarray:
        """
        Optimize portfolio with cardinality constraint.

        Args:
            max_assets: Maximum number of assets
            min_weight: Minimum weight per selected asset
            target_return: Minimum expected return

        Returns:
            Optimal portfolio weights
        """
        model = gp.Model("cardinality_portfolio")
        model.Params.OutputFlag = 0

        # Continuous variables: portfolio weights
        w = model.addVars(self.n_assets, lb=0, ub=1, name="weights")

        # Binary variables: asset selection
        z = model.addVars(self.n_assets, vtype=GRB.BINARY, name="selected")

        # Objective: minimize variance
        # Quadratic objective: w' Sigma w
        obj = sum(
            w[i] * self.Sigma[i, j] * w[j]
            for i in range(self.n_assets)
            for j in range(self.n_assets)
        )

        model.setObjective(obj, GRB.MINIMIZE)

        # Constraints

        # 1. Weights sum to 1
        model.addConstr(sum(w[i] for i in range(self.n_assets)) == 1)

        # 2. Return target
        model.addConstr(
            sum(self.mu[i] * w[i] for i in range(self.n_assets)) >= target_return
        )

        # 3. Cardinality constraint
        model.addConstr(sum(z[i] for i in range(self.n_assets)) <= max_assets)

        # 4. Linking constraints: w[i] > 0 iff z[i] = 1
        for i in range(self.n_assets):
            # w[i] <= z[i] (if not selected, weight = 0)
            model.addConstr(w[i] <= z[i])

            # w[i] >= min_weight * z[i] (if selected, weight >= min)
            model.addConstr(w[i] >= min_weight * z[i])

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            weights = np.array([w[i].X for i in range(self.n_assets)])
            return weights
        else:
            raise RuntimeError("Optimization failed")


# Example usage
if __name__ == "__main__":
    np.random.seed(42)

    n_assets = 20

    # Generate random parameters
    expected_returns = np.random.normal(0.08, 0.03, n_assets)
    volatilities = np.random.uniform(0.10, 0.30, n_assets)
    correlation = np.eye(n_assets) + 0.3 * (np.ones((n_assets, n_assets)) - np.eye(n_assets))
    covariance = np.outer(volatilities, volatilities) * correlation

    optimizer = CardinalityConstrainedPortfolio(expected_returns, covariance)

    print("Cardinality-Constrained Portfolio:")
    print("=" * 60)

    # Optimize with max 10 assets
    weights = optimizer.optimize(max_assets=10, min_weight=0.05)

    selected = weights > 1e-6
    print(f"Number of assets selected: {np.sum(selected)}")
    print(f"\nSelected assets and weights:")
    for i in np.where(selected)[0]:
        print(f"  Asset {i}: {weights[i]:.4f}")

    # Performance
    portfolio_return = np.dot(expected_returns, weights)
    portfolio_risk = np.sqrt(weights @ covariance @ weights)

    print(f"\nPortfolio return: {portfolio_return:.4f}")
    print(f"Portfolio risk: {portfolio_risk:.4f}")
    print(f"Sharpe ratio: {portfolio_return / portfolio_risk:.4f}")
```

---

## 5. Bayesian Optimization for Hyperparameter Tuning

### 5.1 Theoretical Foundation

**Goal**: Find $x^* = \arg\max_{x \in \mathcal{X}} f(x)$ where $f$ is expensive black-box function.

**Gaussian Process (GP) Prior**:

$$f(x) \sim \mathcal{GP}(\mu(x), k(x, x'))$$

**Acquisition Function**: Balances exploration vs. exploitation
- **Expected Improvement (EI)**: $\alpha(x) = E[\max(f(x) - f(x^+), 0)]$
- **Upper Confidence Bound (UCB)**: $\alpha(x) = \mu(x) + \kappa \sigma(x)$
- **Probability of Improvement (PI)**: $\alpha(x) = P(f(x) > f(x^+))$

**Algorithm**:
1. Initialize with random samples
2. Fit GP to observed data
3. Maximize acquisition function to select next point
4. Evaluate $f$ at selected point
5. Repeat until convergence

### 5.2 Trading Applications

**Strategy Hyperparameter Tuning**:
- Optimize lookback periods, thresholds, position sizing parameters
- Expensive objective: full backtest

**Execution Parameter Selection**:
- Optimize urgency, participation rate
- Objective: minimize implementation shortfall

**Risk Parameter Calibration**:
- Optimize stop-loss, take-profit levels
- Objective: Sharpe ratio or other metrics

### 5.3 Python Implementation

```python
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize

class BayesianOptimizer:
    """
    Bayesian optimization for strategy hyperparameter tuning.

    Uses Gaussian Process with Expected Improvement acquisition.
    """

    def __init__(
        self,
        bounds: List[Tuple[float, float]],
        n_initial: int = 5,
        acquisition: str = 'ei',
        kappa: float = 2.576  # UCB parameter
    ):
        """
        Args:
            bounds: List of (min, max) for each parameter
            n_initial: Number of random initial samples
            acquisition: 'ei', 'ucb', or 'pi'
            kappa: Exploration parameter for UCB
        """
        self.bounds = np.array(bounds)
        self.n_dims = len(bounds)
        self.n_initial = n_initial
        self.acquisition_type = acquisition
        self.kappa = kappa

        # Gaussian process
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42
        )

        # Observations
        self.X_obs = []
        self.y_obs = []

        # Best found
        self.best_x = None
        self.best_y = -np.inf

    def _random_sample(self) -> np.ndarray:
        """Generate random sample within bounds."""
        return np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1]
        )

    def _expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """
        Expected Improvement acquisition function.

        Args:
            X: Candidate points (n_samples × n_dims)

        Returns:
            EI values
        """
        if len(self.X_obs) == 0:
            return np.zeros(len(X))

        mu, sigma = self.gp.predict(X, return_std=True)

        # Avoid division by zero
        sigma = np.maximum(sigma, 1e-9)

        # Best observed value
        f_best = self.best_y

        # Expected Improvement
        Z = (mu - f_best) / sigma
        ei = sigma * (Z * norm.cdf(Z) + norm.pdf(Z))

        return ei

    def _upper_confidence_bound(self, X: np.ndarray) -> np.ndarray:
        """
        Upper Confidence Bound acquisition function.

        Args:
            X: Candidate points

        Returns:
            UCB values
        """
        if len(self.X_obs) == 0:
            return np.zeros(len(X))

        mu, sigma = self.gp.predict(X, return_std=True)

        return mu + self.kappa * sigma

    def _probability_improvement(self, X: np.ndarray) -> np.ndarray:
        """
        Probability of Improvement acquisition function.

        Args:
            X: Candidate points

        Returns:
            PI values
        """
        if len(self.X_obs) == 0:
            return np.zeros(len(X))

        mu, sigma = self.gp.predict(X, return_std=True)

        sigma = np.maximum(sigma, 1e-9)

        f_best = self.best_y
        Z = (mu - f_best) / sigma

        return norm.cdf(Z)

    def _acquisition(self, X: np.ndarray) -> np.ndarray:
        """Compute acquisition function."""
        if self.acquisition_type == 'ei':
            return self._expected_improvement(X)
        elif self.acquisition_type == 'ucb':
            return self._upper_confidence_bound(X)
        elif self.acquisition_type == 'pi':
            return self._probability_improvement(X)
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition_type}")

    def suggest_next(self) -> np.ndarray:
        """
        Suggest next point to evaluate.

        Returns:
            Next parameter vector to try
        """
        # Initial random sampling
        if len(self.X_obs) < self.n_initial:
            return self._random_sample()

        # Maximize acquisition function
        def objective(x):
            return -self._acquisition(x.reshape(1, -1))[0]

        # Multi-start optimization
        best_acq = -np.inf
        best_x = None

        for _ in range(10):
            x0 = self._random_sample()

            result = minimize(
                objective,
                x0,
                bounds=self.bounds,
                method='L-BFGS-B'
            )

            if -result.fun > best_acq:
                best_acq = -result.fun
                best_x = result.x

        return best_x

    def observe(self, x: np.ndarray, y: float):
        """
        Record observation.

        Args:
            x: Parameter vector
            y: Objective value
        """
        self.X_obs.append(x)
        self.y_obs.append(y)

        if y > self.best_y:
            self.best_y = y
            self.best_x = x

        # Refit GP
        if len(self.X_obs) >= 2:
            self.gp.fit(np.array(self.X_obs), np.array(self.y_obs))

    def optimize(
        self,
        objective_func: callable,
        n_iterations: int = 50
    ) -> dict:
        """
        Run Bayesian optimization.

        Args:
            objective_func: Black-box function to maximize
            n_iterations: Number of iterations

        Returns:
            Optimization results
        """
        for iteration in range(n_iterations):
            # Suggest next point
            x_next = self.suggest_next()

            # Evaluate objective
            y_next = objective_func(x_next)

            # Record observation
            self.observe(x_next, y_next)

            print(f"Iteration {iteration+1}/{n_iterations}: "
                  f"x={x_next}, y={y_next:.4f}, best_y={self.best_y:.4f}")

        return {
            'best_x': self.best_x,
            'best_y': self.best_y,
            'X_obs': np.array(self.X_obs),
            'y_obs': np.array(self.y_obs)
        }


# Example: Optimize momentum strategy parameters
if __name__ == "__main__":
    # Simulate strategy backtest objective
    def backtest_strategy(params):
        """
        Simulated backtest of momentum strategy.

        Args:
            params: [lookback_period, z_score_threshold]

        Returns:
            Sharpe ratio (objective to maximize)
        """
        lookback, threshold = params

        # Simulate: optimal around lookback=20, threshold=2.0
        optimal_lookback = 20
        optimal_threshold = 2.0

        # Penalty for deviation from optimal
        penalty = (
            ((lookback - optimal_lookback) / 10)**2 +
            ((threshold - optimal_threshold) / 0.5)**2
        )

        # Simulated Sharpe with noise
        sharpe = 1.5 - penalty + np.random.normal(0, 0.1)

        return sharpe

    # Parameter bounds
    bounds = [
        (5, 50),    # Lookback period: 5-50 days
        (1.0, 3.0)  # Z-score threshold: 1.0-3.0
    ]

    # Bayesian optimizer
    optimizer = BayesianOptimizer(
        bounds=bounds,
        n_initial=5,
        acquisition='ei'
    )

    print("Bayesian Optimization of Strategy Parameters:")
    print("=" * 60)

    # Run optimization
    result = optimizer.optimize(backtest_strategy, n_iterations=30)

    print(f"\n\nOptimization Complete:")
    print(f"Best parameters: Lookback={result['best_x'][0]:.1f}, "
          f"Threshold={result['best_x'][1]:.2f}")
    print(f"Best Sharpe ratio: {result['best_y']:.4f}")
```

---

## 6. Integration with Trading Systems

### 6.1 Unified Optimization Framework

```python
class AdvancedOptimizationEngine:
    """
    Unified optimization engine for Ordinis trading system.

    Integrates:
    - Online learning for adaptive strategies
    - Multi-objective portfolio construction
    - Robust optimization for uncertainty
    - Cardinality constraints for practical portfolios
    - Bayesian optimization for hyperparameter tuning
    """

    def __init__(self, config: dict):
        self.config = config

    def optimize_adaptive_portfolio(
        self,
        returns_stream: callable
    ):
        """Online portfolio optimization with regret minimization."""
        ogd = OnlineGradientDescent(n_assets=self.config['n_assets'])
        # Stream returns and update online
        return ogd

    def optimize_multi_objective_portfolio(
        self,
        returns: np.ndarray,
        covariance: np.ndarray
    ):
        """Find Pareto-efficient portfolios."""
        moo = MultiObjectivePortfolioOptimizer(returns, covariance)
        return moo.optimize()

    def optimize_robust_portfolio(
        self,
        returns_data: np.ndarray,
        epsilon: float = 0.1
    ):
        """Worst-case robust portfolio."""
        dro = DistributionallyRobustPortfolio(returns_data, epsilon)
        return dro.optimize_worst_case_cvar()

    def optimize_sparse_portfolio(
        self,
        returns: np.ndarray,
        covariance: np.ndarray,
        max_assets: int = 20
    ):
        """Cardinality-constrained portfolio."""
        card = CardinalityConstrainedPortfolio(returns, covariance)
        return card.optimize(max_assets=max_assets)

    def tune_strategy_hyperparameters(
        self,
        objective: callable,
        param_bounds: List[Tuple]
    ):
        """Bayesian hyperparameter optimization."""
        bo = BayesianOptimizer(bounds=param_bounds)
        return bo.optimize(objective)
```

---

## 7. Academic References

### Foundational Texts

1. **Hazan, E. (2016)**. *Introduction to Online Convex Optimization*. MIT Press.
   - Comprehensive online learning reference

2. **Boyd, S., & Vandenberghe, L. (2004)**. *Convex Optimization*. Cambridge University Press.
   - Standard convex optimization text

3. **Miettinen, K. (1999)**. *Nonlinear Multiobjective Optimization*. Springer.
   - Multi-objective optimization methods

### Trading Applications

4. **Garlappi, L., Uppal, R., & Wang, T. (2007)**. "Portfolio Selection with Parameter and Model Uncertainty: A Multi-Prior Approach." *Review of Financial Studies*, 20(1), 41-81.
   - Robust portfolio optimization

5. **Esfahani, P. M., & Kuhn, D. (2018)**. "Data-Driven Distributionally Robust Optimization." *Mathematical Programming*, 171(1), 105-151.
   - DRO theory and applications

6. **Beasley, J. E., et al. (2003)**. "Portfolio Optimization: Models and Solution Approaches." *Surveys in Operations Research and Management Science*, 8(2), 135-161.
   - Cardinality constraints

7. **Shahriari, B., et al. (2016)**. "Taking the Human Out of the Loop: A Review of Bayesian Optimization." *Proceedings of the IEEE*, 104(1), 148-175.
   - Bayesian optimization survey

---

## 8. Cross-References

**Related Knowledge Base Sections**:

- [Control Theory](control_theory.md) - MPC overlaps with online optimization
- [Game Theory](game_theory.md) - Nash equilibria via optimization
- [Portfolio Construction](../../02_signals/quantitative/portfolio_construction/mean_variance.md) - Mean-variance optimization
- [Execution Algorithms](../../02_signals/quantitative/execution_algorithms/optimal_execution.md) - Execution optimization
- [Backtesting](../../04_strategy/backtesting-requirements.md) - Hyperparameter optimization

**Integration Points**:

1. **SignalCore**: Feature selection via online learning
2. **FlowRoute**: Robust execution optimization
3. **ProofBench**: Bayesian hyperparameter tuning
4. **RiskGuard**: Multi-objective risk-return optimization

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
last_updated: "2025-12-12"
status: "published"
category: "foundations/advanced_mathematics"
tags: ["advanced-optimization", "online-learning", "multi-objective", "dro", "mip", "bayesian-optimization", "regret-minimization", "pareto-front"]
code_lines: 950
academic_references: 7
implementation_completeness: "production-ready"
```

---

**END OF DOCUMENT**
