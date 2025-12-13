# Extreme Value Theory for Tail Risk Management

Extreme Value Theory (EVT) provides rigorous methods for modeling rare events, estimating tail risk measures, and understanding extreme market movements.

---

## Overview

Financial markets exhibit fat-tailed distributions where extreme events occur more frequently than Gaussian models predict. EVT enables:

1. **Tail Estimation**: Model extreme returns beyond historical data
2. **Risk Measures**: VaR and ES at extreme confidence levels
3. **Tail Dependence**: Joint extreme behavior via copulas
4. **Stress Testing**: Scenario generation for extreme events
5. **Insurance Pricing**: Pricing tail risk protection

---

## 1. Generalized Extreme Value (GEV) Distribution

### 1.1 Fisher-Tippett Theorem

Block maxima converge to GEV distribution:

$$G(x; \mu, \sigma, \xi) = \exp\left\{-\left[1 + \xi\left(\frac{x-\mu}{\sigma}\right)\right]^{-1/\xi}\right\}$$

where:
- $\mu$ = location parameter
- $\sigma$ = scale parameter
- $\xi$ = shape parameter (tail index)

**Tail Types**:
- $\xi > 0$: Fréchet (heavy tail, power law)
- $\xi = 0$: Gumbel (light tail, exponential)
- $\xi < 0$: Weibull (bounded tail)

### 1.2 Python Implementation

```python
import numpy as np
import pandas as pd
from scipy.stats import genextreme, genpareto
from scipy.optimize import minimize
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class EVTResult:
    """Container for EVT estimation results."""
    shape: float
    scale: float
    location: float
    var_estimates: Dict[float, float]
    es_estimates: Dict[float, float]
    tail_index: float


class GEVAnalysis:
    """
    Generalized Extreme Value analysis via block maxima.
    """

    def __init__(self, block_size: int = 21):
        """
        Initialize GEV analyzer.

        Args:
            block_size: Size of blocks for maxima extraction (21 = monthly)
        """
        self.block_size = block_size
        self.params = None

    def extract_block_maxima(
        self,
        returns: np.ndarray,
        use_minima: bool = False
    ) -> np.ndarray:
        """
        Extract block maxima (or minima for losses).

        Args:
            returns: Return series
            use_minima: If True, extract minima (for loss distribution)

        Returns:
            Array of block extremes
        """
        n_blocks = len(returns) // self.block_size
        extremes = []

        for i in range(n_blocks):
            block = returns[i * self.block_size:(i + 1) * self.block_size]
            if use_minima:
                extremes.append(block.min())
            else:
                extremes.append(block.max())

        return np.array(extremes)

    def fit(self, returns: np.ndarray, use_losses: bool = True) -> EVTResult:
        """
        Fit GEV distribution to block extremes.

        Args:
            returns: Return series
            use_losses: If True, fit to losses (negative returns)

        Returns:
            EVTResult with fitted parameters
        """
        if use_losses:
            data = -returns
            extremes = self.extract_block_maxima(data, use_minima=False)
        else:
            extremes = self.extract_block_maxima(returns, use_minima=False)

        # Fit GEV
        # scipy's genextreme uses -xi convention
        shape, loc, scale = genextreme.fit(extremes)

        self.params = {
            'shape': -shape,  # Convert to standard convention
            'location': loc,
            'scale': scale
        }

        # Compute risk measures
        var_levels = [0.95, 0.99, 0.995, 0.999]
        var_estimates = {}
        es_estimates = {}

        for alpha in var_levels:
            var_estimates[alpha] = self.var(alpha)
            es_estimates[alpha] = self.expected_shortfall(alpha)

        return EVTResult(
            shape=-shape,
            scale=scale,
            location=loc,
            var_estimates=var_estimates,
            es_estimates=es_estimates,
            tail_index=-shape if shape < 0 else np.inf
        )

    def var(self, alpha: float) -> float:
        """
        Compute VaR at confidence level alpha.
        """
        if self.params is None:
            raise ValueError("Must fit model first")

        # Block VaR
        xi = self.params['shape']
        sigma = self.params['scale']
        mu = self.params['location']

        y_p = -np.log(alpha)

        if abs(xi) < 1e-10:
            block_var = mu - sigma * np.log(y_p)
        else:
            block_var = mu + sigma * (y_p ** (-xi) - 1) / xi

        # Convert to daily VaR (approximate)
        daily_var = block_var / np.sqrt(self.block_size)

        return daily_var

    def expected_shortfall(self, alpha: float) -> float:
        """
        Compute Expected Shortfall at confidence level alpha.
        """
        if self.params is None:
            raise ValueError("Must fit model first")

        xi = self.params['shape']
        var = self.var(alpha)

        if xi >= 1:
            return np.inf

        # ES = VaR / (1 - xi) + (sigma - xi * mu) / (1 - xi)
        sigma = self.params['scale']
        mu = self.params['location']

        es = var / (1 - xi) + (sigma - xi * mu) / (1 - xi)

        return es / np.sqrt(self.block_size)


# Example
if __name__ == "__main__":
    np.random.seed(42)
    # Simulate fat-tailed returns
    returns = np.concatenate([
        np.random.normal(0.0005, 0.02, 2000),
        np.random.standard_t(3, 100) * 0.05  # Fat tail events
    ])
    np.random.shuffle(returns)

    gev = GEVAnalysis(block_size=21)
    result = gev.fit(returns, use_losses=True)

    print("GEV Analysis Results:")
    print(f"  Shape (xi): {result.shape:.4f}")
    print(f"  Scale: {result.scale:.4f}")
    print(f"  Location: {result.location:.4f}")
    print("\nRisk Measures:")
    for alpha, var in result.var_estimates.items():
        es = result.es_estimates[alpha]
        print(f"  {alpha*100:.1f}% VaR: {var:.4f}, ES: {es:.4f}")
```

---

## 2. Peaks Over Threshold (POT)

### 2.1 Theory

Model exceedances over threshold $u$ with Generalized Pareto Distribution (GPD):

$$F_u(y) = P(X - u \leq y | X > u) = 1 - \left(1 + \xi\frac{y}{\sigma}\right)^{-1/\xi}$$

**Advantages over Block Maxima**:
- Uses more data (all exceedances)
- More efficient parameter estimation
- Better for risk measure estimation

### 2.2 Python Implementation

```python
class POTAnalysis:
    """
    Peaks Over Threshold analysis using GPD.
    """

    def __init__(self, threshold_quantile: float = 0.95):
        """
        Initialize POT analyzer.

        Args:
            threshold_quantile: Quantile for threshold selection
        """
        self.threshold_quantile = threshold_quantile
        self.threshold = None
        self.params = None
        self.exceedance_rate = None

    def select_threshold(
        self,
        data: np.ndarray,
        method: str = 'quantile'
    ) -> float:
        """
        Select threshold for POT.

        Args:
            data: Data (typically losses)
            method: 'quantile' or 'mean_excess'

        Returns:
            Threshold value
        """
        if method == 'quantile':
            return np.percentile(data, self.threshold_quantile * 100)
        elif method == 'mean_excess':
            # Find threshold where mean excess plot becomes linear
            return self._mean_excess_threshold(data)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _mean_excess_threshold(self, data: np.ndarray) -> float:
        """Select threshold using mean excess plot."""
        sorted_data = np.sort(data)
        n = len(sorted_data)

        # Compute mean excess for various thresholds
        thresholds = sorted_data[int(0.5 * n):int(0.98 * n)]
        mean_excesses = []

        for u in thresholds:
            exceedances = data[data > u] - u
            if len(exceedances) > 10:
                mean_excesses.append(exceedances.mean())
            else:
                mean_excesses.append(np.nan)

        # Find where slope stabilizes (simplified)
        # In practice, visual inspection is recommended
        return thresholds[int(len(thresholds) * 0.3)]

    def fit(self, returns: np.ndarray) -> Dict:
        """
        Fit GPD to exceedances.

        Args:
            returns: Return series (fits to losses = -returns)

        Returns:
            Fitted parameters and diagnostics
        """
        losses = -returns

        # Select threshold
        self.threshold = self.select_threshold(losses)

        # Extract exceedances
        exceedances = losses[losses > self.threshold] - self.threshold
        n_exceed = len(exceedances)
        n_total = len(losses)

        self.exceedance_rate = n_exceed / n_total

        # Fit GPD
        # scipy's genpareto uses c = xi
        shape, loc, scale = genpareto.fit(exceedances, floc=0)

        self.params = {
            'shape': shape,
            'scale': scale,
            'threshold': self.threshold
        }

        return {
            'params': self.params,
            'n_exceedances': n_exceed,
            'exceedance_rate': self.exceedance_rate,
            'exceedances': exceedances
        }

    def var(self, alpha: float, n_obs: int = None) -> float:
        """
        Compute VaR at confidence level alpha.

        Args:
            alpha: Confidence level (e.g., 0.99)
            n_obs: Number of observations (for rate estimation)

        Returns:
            VaR estimate
        """
        if self.params is None:
            raise ValueError("Must fit model first")

        xi = self.params['shape']
        sigma = self.params['scale']
        u = self.params['threshold']
        zeta_u = self.exceedance_rate

        # VaR formula for GPD
        if abs(xi) < 1e-10:
            var = u + sigma * np.log(zeta_u / (1 - alpha))
        else:
            var = u + (sigma / xi) * ((zeta_u / (1 - alpha)) ** xi - 1)

        return var

    def expected_shortfall(self, alpha: float) -> float:
        """
        Compute Expected Shortfall at confidence level alpha.
        """
        if self.params is None:
            raise ValueError("Must fit model first")

        xi = self.params['shape']
        sigma = self.params['scale']
        var = self.var(alpha)

        if xi >= 1:
            return np.inf

        es = var / (1 - xi) + (sigma - xi * self.params['threshold']) / (1 - xi)

        return es

    def tail_probability(self, x: float) -> float:
        """
        Estimate tail probability P(Loss > x).
        """
        if x <= self.threshold:
            # Below threshold, use empirical
            return self.exceedance_rate

        xi = self.params['shape']
        sigma = self.params['scale']
        u = self.threshold

        if abs(xi) < 1e-10:
            return self.exceedance_rate * np.exp(-(x - u) / sigma)
        else:
            return self.exceedance_rate * (1 + xi * (x - u) / sigma) ** (-1 / xi)


# Example
if __name__ == "__main__":
    np.random.seed(42)
    returns = np.random.standard_t(4, 2500) * 0.02

    pot = POTAnalysis(threshold_quantile=0.95)
    result = pot.fit(returns)

    print("POT Analysis Results:")
    print(f"  Threshold: {pot.threshold:.4f}")
    print(f"  Shape (xi): {pot.params['shape']:.4f}")
    print(f"  Scale: {pot.params['scale']:.4f}")
    print(f"  Exceedance rate: {pot.exceedance_rate:.4f}")

    print("\nRisk Measures:")
    for alpha in [0.95, 0.99, 0.995, 0.999]:
        var = pot.var(alpha)
        es = pot.expected_shortfall(alpha)
        print(f"  {alpha*100:.1f}% VaR: {var:.4f}, ES: {es:.4f}")
```

---

## 3. Copulas and Tail Dependence

### 3.1 Theory

Copulas model joint dependence structure separate from marginals:

$$F(x, y) = C(F_X(x), F_Y(y))$$

**Tail Dependence Coefficients**:

Upper tail:
$$\lambda_U = \lim_{u \to 1} P(Y > F_Y^{-1}(u) | X > F_X^{-1}(u))$$

Lower tail:
$$\lambda_L = \lim_{u \to 0} P(Y < F_Y^{-1}(u) | X < F_X^{-1}(u))$$

### 3.2 Python Implementation

```python
from scipy.stats import kendalltau, norm


class CopulaAnalysis:
    """
    Copula-based tail dependence analysis.
    """

    def __init__(self):
        self.copula_type = None
        self.params = None

    def fit_gaussian_copula(self, u: np.ndarray, v: np.ndarray) -> Dict:
        """
        Fit Gaussian copula.

        Note: Gaussian copula has zero tail dependence.

        Args:
            u, v: Uniform marginals (pseudo-observations)

        Returns:
            Fitted parameters
        """
        # Transform to normal
        x = norm.ppf(np.clip(u, 0.001, 0.999))
        y = norm.ppf(np.clip(v, 0.001, 0.999))

        # Correlation
        rho = np.corrcoef(x, y)[0, 1]

        self.copula_type = 'gaussian'
        self.params = {'rho': rho}

        return {
            'copula': 'gaussian',
            'rho': rho,
            'lambda_upper': 0,  # No tail dependence
            'lambda_lower': 0
        }

    def fit_clayton_copula(self, u: np.ndarray, v: np.ndarray) -> Dict:
        """
        Fit Clayton copula (lower tail dependence).

        Args:
            u, v: Uniform marginals

        Returns:
            Fitted parameters including tail dependence
        """
        # Estimate theta via Kendall's tau
        tau, _ = kendalltau(u, v)

        # Clayton: tau = theta / (theta + 2)
        # theta = 2 * tau / (1 - tau)
        if tau <= 0:
            theta = 0.01  # Minimum positive
        else:
            theta = 2 * tau / (1 - tau)

        # Lower tail dependence
        lambda_lower = 2 ** (-1 / theta) if theta > 0 else 0

        self.copula_type = 'clayton'
        self.params = {'theta': theta}

        return {
            'copula': 'clayton',
            'theta': theta,
            'kendall_tau': tau,
            'lambda_upper': 0,
            'lambda_lower': lambda_lower
        }

    def fit_gumbel_copula(self, u: np.ndarray, v: np.ndarray) -> Dict:
        """
        Fit Gumbel copula (upper tail dependence).

        Args:
            u, v: Uniform marginals

        Returns:
            Fitted parameters including tail dependence
        """
        tau, _ = kendalltau(u, v)

        # Gumbel: tau = 1 - 1/theta
        # theta = 1 / (1 - tau)
        if tau >= 1:
            theta = 10  # Cap at high value
        elif tau <= 0:
            theta = 1  # Independence
        else:
            theta = 1 / (1 - tau)

        theta = max(1, theta)  # Gumbel requires theta >= 1

        # Upper tail dependence
        lambda_upper = 2 - 2 ** (1 / theta)

        self.copula_type = 'gumbel'
        self.params = {'theta': theta}

        return {
            'copula': 'gumbel',
            'theta': theta,
            'kendall_tau': tau,
            'lambda_upper': lambda_upper,
            'lambda_lower': 0
        }

    def empirical_tail_dependence(
        self,
        u: np.ndarray,
        v: np.ndarray,
        threshold: float = 0.95
    ) -> Dict:
        """
        Compute empirical tail dependence.

        Args:
            u, v: Uniform marginals
            threshold: Threshold for tail definition

        Returns:
            Empirical tail dependence estimates
        """
        n = len(u)

        # Upper tail
        upper_joint = np.sum((u > threshold) & (v > threshold))
        upper_marginal = np.sum(u > threshold)
        lambda_upper = upper_joint / upper_marginal if upper_marginal > 0 else 0

        # Lower tail
        lower_threshold = 1 - threshold
        lower_joint = np.sum((u < lower_threshold) & (v < lower_threshold))
        lower_marginal = np.sum(u < lower_threshold)
        lambda_lower = lower_joint / lower_marginal if lower_marginal > 0 else 0

        return {
            'lambda_upper_empirical': lambda_upper,
            'lambda_lower_empirical': lambda_lower,
            'threshold': threshold
        }


def pseudo_observations(x: np.ndarray) -> np.ndarray:
    """Convert to pseudo-observations (empirical CDF)."""
    n = len(x)
    ranks = np.argsort(np.argsort(x)) + 1
    return ranks / (n + 1)


# Example
if __name__ == "__main__":
    np.random.seed(42)

    # Generate correlated returns with tail dependence
    n = 2000
    # Using t-copula implicitly via multivariate t
    from scipy.stats import multivariate_t

    df = 3  # Low df = high tail dependence
    corr = np.array([[1, 0.6], [0.6, 1]])
    mv_t = multivariate_t(loc=[0, 0], shape=corr, df=df)
    samples = mv_t.rvs(n)

    u = pseudo_observations(samples[:, 0])
    v = pseudo_observations(samples[:, 1])

    copula = CopulaAnalysis()

    # Fit different copulas
    gauss = copula.fit_gaussian_copula(u, v)
    print(f"Gaussian Copula: rho={gauss['rho']:.3f}")

    clayton = copula.fit_clayton_copula(u, v)
    print(f"Clayton Copula: theta={clayton['theta']:.3f}, lambda_L={clayton['lambda_lower']:.3f}")

    gumbel = copula.fit_gumbel_copula(u, v)
    print(f"Gumbel Copula: theta={gumbel['theta']:.3f}, lambda_U={gumbel['lambda_upper']:.3f}")

    empirical = copula.empirical_tail_dependence(u, v)
    print(f"Empirical: lambda_U={empirical['lambda_upper_empirical']:.3f}, "
          f"lambda_L={empirical['lambda_lower_empirical']:.3f}")
```

---

## 4. Risk Applications

### 4.1 Integrated EVT Risk Manager

```python
class EVTRiskManager:
    """
    EVT-based risk management for trading.
    """

    def __init__(self):
        self.gev = GEVAnalysis()
        self.pot = POTAnalysis()
        self.is_fitted = False

    def fit(self, returns: pd.Series) -> Dict:
        """
        Fit EVT models to return series.
        """
        returns_array = returns.values

        # GEV (block maxima)
        gev_result = self.gev.fit(returns_array)

        # POT
        pot_result = self.pot.fit(returns_array)

        self.is_fitted = True

        return {
            'gev': gev_result,
            'pot': pot_result
        }

    def risk_report(self, confidence_levels: List[float] = None) -> pd.DataFrame:
        """
        Generate risk report with EVT-based measures.
        """
        if not self.is_fitted:
            raise ValueError("Must fit models first")

        if confidence_levels is None:
            confidence_levels = [0.95, 0.99, 0.995, 0.999]

        results = []
        for alpha in confidence_levels:
            results.append({
                'confidence': f"{alpha*100:.1f}%",
                'var_gev': self.gev.var(alpha),
                'var_pot': self.pot.var(alpha),
                'es_gev': self.gev.expected_shortfall(alpha),
                'es_pot': self.pot.expected_shortfall(alpha)
            })

        return pd.DataFrame(results)

    def stress_scenarios(self, n_scenarios: int = 1000) -> np.ndarray:
        """
        Generate stress scenarios from fitted tail.
        """
        if self.pot.params is None:
            raise ValueError("Must fit models first")

        xi = self.pot.params['shape']
        sigma = self.pot.params['scale']

        # Generate from GPD
        gpd_samples = genpareto.rvs(xi, scale=sigma, size=n_scenarios)

        # Add threshold
        return gpd_samples + self.pot.threshold


# Example
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000)
    returns = pd.Series(
        np.random.standard_t(4, 1000) * 0.015 + 0.0003,
        index=dates
    )

    risk_mgr = EVTRiskManager()
    risk_mgr.fit(returns)

    report = risk_mgr.risk_report()
    print("EVT Risk Report:")
    print(report.to_string(index=False))

    scenarios = risk_mgr.stress_scenarios(1000)
    print(f"\nStress Scenarios (losses):")
    print(f"  Mean: {scenarios.mean():.4f}")
    print(f"  95th percentile: {np.percentile(scenarios, 95):.4f}")
    print(f"  Max: {scenarios.max():.4f}")
```

---

## 5. Academic References

1. **Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997)**. *Modelling Extremal Events for Insurance and Finance*. Springer.

2. **McNeil, A. J., Frey, R., & Embrechts, P. (2015)**. *Quantitative Risk Management: Concepts, Techniques and Tools*. Princeton.

3. **Coles, S. (2001)**. *An Introduction to Statistical Modeling of Extreme Values*. Springer.

4. **Joe, H. (2014)**. *Dependence Modeling with Copulas*. Chapman & Hall.

5. **Balkema, A. A., & de Haan, L. (1974)**. "Residual Life Time at Great Age." *Annals of Probability*.

6. **Pickands, J. (1975)**. "Statistical Inference Using Extreme Order Statistics." *Annals of Statistics*.

---

## 6. Cross-References

**Related Knowledge Base Sections**:

- [Network Theory](network_theory.md) - Tail dependence in correlation networks
- [Nonparametric Statistics](nonparametric_stats.md) - KDE for tail estimation
- [Advanced Risk Methods](../../03_risk/advanced_risk_methods.md) - VaR/ES implementation

**Integration Points**:

1. **RiskGuard**: EVT-based VaR/ES limits
2. **ProofBench**: Stress testing with EVT scenarios
3. **SignalCore**: Extreme event detection signals

---

## Document Metadata

```yaml
version: "1.0.0"
created: "2025-12-12"
status: "published"
tags: ["evt", "tail-risk", "var", "expected-shortfall", "copulas", "gpd", "gev"]
code_lines: 500
```

---

**END OF DOCUMENT**
