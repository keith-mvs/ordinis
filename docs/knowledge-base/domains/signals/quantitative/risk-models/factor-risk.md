# Factor Risk Models

## Overview

Factor risk models decompose portfolio risk into systematic factors and idiosyncratic components. These models enable **risk attribution**, **factor exposure management**, and **hedging optimization** for quantitative strategies.

---

## 1. Factor Model Framework

### 1.1 Linear Factor Model

**Mathematical Foundation**:
```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy import stats, optimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


@dataclass
class FactorModelConfig:
    """Configuration for factor risk models."""

    # Factor estimation
    lookback_estimation: int = 252 * 2  # 2 years
    min_observations: int = 60

    # Risk calculation
    annualization_factor: float = 252.0
    confidence_level: float = 0.95

    # Factor types
    use_statistical_factors: bool = True
    use_fundamental_factors: bool = True
    n_statistical_factors: int = 5


class LinearFactorModel:
    """
    Linear factor model: r_i = alpha_i + sum(beta_ij * f_j) + epsilon_i
    """

    def __init__(self, config: FactorModelConfig = None):
        self.config = config or FactorModelConfig()
        self.factor_returns = None
        self.factor_loadings = None
        self.factor_covariance = None
        self.specific_variance = None

    def estimate_model(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame
    ):
        """
        Estimate factor model parameters.

        Args:
            asset_returns: DataFrame of asset returns (assets x time)
            factor_returns: DataFrame of factor returns (factors x time)
        """
        self.factor_returns = factor_returns

        # Align data
        common_idx = asset_returns.index.intersection(factor_returns.index)
        R = asset_returns.loc[common_idx]
        F = factor_returns.loc[common_idx]

        # Estimate factor loadings via regression
        self.factor_loadings = pd.DataFrame(
            index=R.columns,
            columns=['alpha'] + list(F.columns)
        )
        self.specific_variance = pd.Series(index=R.columns)

        for asset in R.columns:
            y = R[asset].values
            X = np.column_stack([np.ones(len(F)), F.values])

            # OLS regression
            model = LinearRegression(fit_intercept=False)
            model.fit(X, y)

            self.factor_loadings.loc[asset] = model.coef_
            residuals = y - model.predict(X)
            self.specific_variance[asset] = np.var(residuals)

        # Factor covariance matrix
        self.factor_covariance = F.cov() * self.config.annualization_factor

    def calculate_risk_decomposition(
        self,
        weights: pd.Series
    ) -> Dict:
        """
        Decompose portfolio risk into factor and specific components.

        Args:
            weights: Portfolio weights (sums to 1)

        Returns:
            Dictionary with risk decomposition
        """
        # Ensure alignment
        assets = weights.index
        betas = self.factor_loadings.loc[assets].drop('alpha', axis=1)
        specific_var = self.specific_variance.loc[assets]

        # Portfolio factor exposure
        portfolio_betas = (weights.values @ betas.values)
        portfolio_betas = pd.Series(portfolio_betas, index=betas.columns)

        # Factor risk contribution
        factor_variance = (
            portfolio_betas.values @
            self.factor_covariance.loc[betas.columns, betas.columns].values @
            portfolio_betas.values
        )

        # Specific risk
        specific_variance = (weights ** 2 * specific_var).sum()

        # Total variance
        total_variance = factor_variance + specific_variance

        return {
            'total_risk': np.sqrt(total_variance),
            'factor_risk': np.sqrt(factor_variance),
            'specific_risk': np.sqrt(specific_variance),
            'factor_contribution_pct': factor_variance / total_variance * 100,
            'specific_contribution_pct': specific_variance / total_variance * 100,
            'portfolio_betas': portfolio_betas
        }

    def calculate_factor_contribution(
        self,
        weights: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate marginal risk contribution of each factor.
        """
        assets = weights.index
        betas = self.factor_loadings.loc[assets].drop('alpha', axis=1)

        # Portfolio factor exposure
        portfolio_betas = (weights.values @ betas.values)
        portfolio_betas = pd.Series(portfolio_betas, index=betas.columns)

        # Factor covariance contribution
        F_cov = self.factor_covariance.loc[betas.columns, betas.columns]
        factor_variance = portfolio_betas.values @ F_cov.values @ portfolio_betas.values

        contributions = pd.DataFrame(index=betas.columns)

        for factor in betas.columns:
            # Marginal contribution
            marginal = (
                2 * portfolio_betas[factor] *
                (F_cov.loc[factor] * portfolio_betas).sum()
            )
            contributions.loc[factor, 'marginal_contribution'] = marginal

            # Percentage contribution
            contributions.loc[factor, 'pct_contribution'] = (
                marginal / (2 * factor_variance) * 100 if factor_variance > 0 else 0
            )

        return contributions
```

---

## 2. Statistical Factor Models

### 2.1 PCA-Based Factor Model

**Implementation**:
```python
class StatisticalFactorModel:
    """Statistical factor model using PCA."""

    def __init__(self, config: FactorModelConfig = None):
        self.config = config or FactorModelConfig()
        self.pca = None
        self.factor_loadings = None
        self.factor_returns = None
        self.specific_variance = None

    def estimate_factors(
        self,
        asset_returns: pd.DataFrame,
        n_factors: int = None
    ) -> pd.DataFrame:
        """
        Extract statistical factors using PCA.
        """
        n_factors = n_factors or self.config.n_statistical_factors

        # Standardize returns
        returns_centered = asset_returns - asset_returns.mean()
        returns_std = returns_centered / returns_centered.std()

        # PCA
        self.pca = PCA(n_components=n_factors)
        factor_scores = self.pca.fit_transform(returns_std)

        # Factor returns (unstandardized)
        self.factor_returns = pd.DataFrame(
            factor_scores * returns_centered.std().mean(),
            index=asset_returns.index,
            columns=[f'PC{i+1}' for i in range(n_factors)]
        )

        # Factor loadings
        self.factor_loadings = pd.DataFrame(
            self.pca.components_.T,
            index=asset_returns.columns,
            columns=self.factor_returns.columns
        )

        # Explained variance
        explained_var = self.pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)

        return {
            'factor_returns': self.factor_returns,
            'loadings': self.factor_loadings,
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var
        }

    def calculate_specific_risk(
        self,
        asset_returns: pd.DataFrame
    ):
        """
        Calculate specific (idiosyncratic) variance.
        """
        # Reconstruct returns from factors
        reconstructed = self.factor_returns @ self.factor_loadings.T

        # Residuals
        residuals = asset_returns - reconstructed.loc[asset_returns.index]

        # Specific variance
        self.specific_variance = residuals.var()

        return self.specific_variance


class AsymmetricFactorModel:
    """Factor model with asymmetric risk (different up/down betas)."""

    def __init__(self, config: FactorModelConfig = None):
        self.config = config or FactorModelConfig()
        self.up_betas = None
        self.down_betas = None

    def estimate_asymmetric_betas(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.Series
    ) -> Dict:
        """
        Estimate separate betas for up and down markets.
        """
        results = {}

        for asset in asset_returns.columns:
            y = asset_returns[asset]
            X = factor_returns

            # Align data
            aligned = pd.DataFrame({'y': y, 'X': X}).dropna()

            # Up market
            up_mask = aligned['X'] > 0
            if up_mask.sum() > 20:
                up_beta = aligned.loc[up_mask, 'y'].cov(aligned.loc[up_mask, 'X']) / aligned.loc[up_mask, 'X'].var()
            else:
                up_beta = np.nan

            # Down market
            down_mask = aligned['X'] < 0
            if down_mask.sum() > 20:
                down_beta = aligned.loc[down_mask, 'y'].cov(aligned.loc[down_mask, 'X']) / aligned.loc[down_mask, 'X'].var()
            else:
                down_beta = np.nan

            results[asset] = {
                'up_beta': up_beta,
                'down_beta': down_beta,
                'beta_asymmetry': up_beta - down_beta if pd.notna(up_beta) and pd.notna(down_beta) else np.nan
            }

        return pd.DataFrame(results).T
```

---

## 3. Fundamental Factor Models

### 3.1 Cross-Sectional Factor Model

**Implementation**:
```python
class FundamentalFactorModel:
    """
    Fundamental factor model (Barra-style).
    """

    def __init__(self, config: FactorModelConfig = None):
        self.config = config or FactorModelConfig()
        self.factor_returns_history = None

    def prepare_factor_exposures(
        self,
        characteristics: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Standardize factor exposures cross-sectionally.

        Args:
            characteristics: DataFrame with columns for each characteristic
                            (market_cap, book_to_price, momentum, etc.)
        """
        exposures = pd.DataFrame(index=characteristics.index)

        for col in characteristics.columns:
            values = characteristics[col]

            # Cross-sectional z-score
            mean = values.mean()
            std = values.std()
            z_score = (values - mean) / std if std > 0 else 0

            # Winsorize
            z_score = z_score.clip(-3, 3)

            exposures[col] = z_score

        return exposures

    def estimate_factor_returns(
        self,
        asset_returns: pd.Series,
        factor_exposures: pd.DataFrame,
        weights: pd.Series = None
    ) -> Dict:
        """
        Estimate factor returns via cross-sectional regression.

        r_i = sum(X_ij * f_j) + epsilon_i

        Solved for f_j using WLS.
        """
        # Align data
        common = asset_returns.index.intersection(factor_exposures.index)
        y = asset_returns.loc[common].values
        X = factor_exposures.loc[common].values

        if weights is not None:
            W = np.diag(weights.loc[common].values)
        else:
            W = np.eye(len(y))

        # WLS: (X'WX)^-1 X'Wy
        XtW = X.T @ W
        XtWX = XtW @ X
        XtWy = XtW @ y

        try:
            factor_returns = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            factor_returns = np.linalg.lstsq(XtWX, XtWy, rcond=None)[0]

        # Residuals
        predicted = X @ factor_returns
        residuals = y - predicted

        return {
            'factor_returns': pd.Series(factor_returns, index=factor_exposures.columns),
            'residuals': pd.Series(residuals, index=common),
            'r_squared': 1 - np.var(residuals) / np.var(y)
        }

    def build_factor_covariance(
        self,
        factor_returns_history: pd.DataFrame,
        half_life: int = 60
    ) -> pd.DataFrame:
        """
        Build factor covariance matrix with exponential weighting.
        """
        # Exponential weights
        n = len(factor_returns_history)
        decay = 0.5 ** (1 / half_life)
        weights = np.array([decay ** (n - 1 - i) for i in range(n)])
        weights = weights / weights.sum()

        # Weighted covariance
        returns_centered = factor_returns_history - factor_returns_history.mean()

        cov_matrix = pd.DataFrame(
            index=factor_returns_history.columns,
            columns=factor_returns_history.columns
        )

        for i, col_i in enumerate(factor_returns_history.columns):
            for j, col_j in enumerate(factor_returns_history.columns):
                if j <= i:
                    weighted_cov = (
                        weights *
                        returns_centered[col_i].values *
                        returns_centered[col_j].values
                    ).sum()
                    cov_matrix.loc[col_i, col_j] = weighted_cov
                    cov_matrix.loc[col_j, col_i] = weighted_cov

        return cov_matrix.astype(float) * self.config.annualization_factor
```

---

## 4. Risk Attribution

### 4.1 Portfolio Risk Attribution

**Implementation**:
```python
class RiskAttributor:
    """Attribute portfolio risk to various sources."""

    def __init__(self, config: FactorModelConfig = None):
        self.config = config or FactorModelConfig()

    def attribute_risk(
        self,
        weights: pd.Series,
        factor_loadings: pd.DataFrame,
        factor_covariance: pd.DataFrame,
        specific_variance: pd.Series
    ) -> Dict:
        """
        Full risk attribution analysis.
        """
        assets = weights.index
        betas = factor_loadings.loc[assets]

        # Portfolio factor exposure
        portfolio_betas = weights.values @ betas.values
        portfolio_betas = pd.Series(portfolio_betas, index=betas.columns)

        # Factor contribution to variance
        F_cov = factor_covariance.loc[betas.columns, betas.columns]
        factor_var = portfolio_betas.values @ F_cov.values @ portfolio_betas.values

        # Specific variance
        spec_var = (weights ** 2 * specific_variance.loc[assets]).sum()

        # Total variance
        total_var = factor_var + spec_var
        total_risk = np.sqrt(total_var)

        # Factor risk contributions
        factor_contributions = {}
        for factor in betas.columns:
            # Marginal contribution
            mrc = 2 * portfolio_betas[factor] * (F_cov.loc[factor] * portfolio_betas).sum()
            # Risk contribution
            rc = portfolio_betas[factor] * (F_cov.loc[factor] * portfolio_betas).sum()
            factor_contributions[factor] = {
                'exposure': portfolio_betas[factor],
                'marginal_risk_contribution': mrc / (2 * total_risk),
                'risk_contribution': rc / total_var,
                'risk_contribution_pct': rc / total_var * 100
            }

        # Asset risk contributions
        asset_contributions = {}
        for asset in assets:
            # Marginal risk from this asset
            asset_beta = betas.loc[asset].values
            asset_factor_risk = asset_beta @ F_cov.values @ portfolio_betas.values
            asset_spec_risk = weights[asset] * specific_variance.loc[asset]
            asset_total = asset_factor_risk + asset_spec_risk

            asset_contributions[asset] = {
                'weight': weights[asset],
                'factor_risk_contribution': asset_factor_risk / total_var,
                'specific_risk_contribution': asset_spec_risk / total_var,
                'total_risk_contribution': asset_total / total_var
            }

        return {
            'total_risk': total_risk,
            'factor_risk': np.sqrt(factor_var),
            'specific_risk': np.sqrt(spec_var),
            'factor_risk_pct': factor_var / total_var * 100,
            'specific_risk_pct': spec_var / total_var * 100,
            'factor_contributions': pd.DataFrame(factor_contributions).T,
            'asset_contributions': pd.DataFrame(asset_contributions).T
        }

    def calculate_tracking_error(
        self,
        active_weights: pd.Series,
        factor_loadings: pd.DataFrame,
        factor_covariance: pd.DataFrame,
        specific_variance: pd.Series
    ) -> Dict:
        """
        Calculate tracking error vs benchmark.
        """
        assets = active_weights.index
        betas = factor_loadings.loc[assets]

        # Active factor exposure
        active_betas = active_weights.values @ betas.values
        active_betas = pd.Series(active_betas, index=betas.columns)

        # Factor tracking error
        F_cov = factor_covariance.loc[betas.columns, betas.columns]
        factor_te_var = active_betas.values @ F_cov.values @ active_betas.values

        # Specific tracking error
        spec_te_var = (active_weights ** 2 * specific_variance.loc[assets]).sum()

        # Total tracking error
        total_te_var = factor_te_var + spec_te_var
        tracking_error = np.sqrt(total_te_var)

        return {
            'tracking_error': tracking_error,
            'factor_te': np.sqrt(factor_te_var),
            'specific_te': np.sqrt(spec_te_var),
            'active_factor_exposures': active_betas
        }
```

---

## 5. Production Risk Engine

### 5.1 Integrated Factor Risk System

```python
class FactorRiskEngine:
    """
    Production factor risk engine.
    """

    def __init__(self, config: FactorModelConfig = None):
        self.config = config or FactorModelConfig()
        self.linear_model = LinearFactorModel(config)
        self.statistical_model = StatisticalFactorModel(config)
        self.fundamental_model = FundamentalFactorModel(config)
        self.attributor = RiskAttributor(config)

    def estimate_full_model(
        self,
        asset_returns: pd.DataFrame,
        factor_returns: pd.DataFrame = None,
        characteristics: pd.DataFrame = None
    ):
        """
        Estimate complete factor model.
        """
        results = {}

        # Statistical factors (PCA)
        if self.config.use_statistical_factors:
            stat_result = self.statistical_model.estimate_factors(asset_returns)
            results['statistical'] = stat_result

        # Fundamental factors
        if self.config.use_fundamental_factors and characteristics is not None:
            exposures = self.fundamental_model.prepare_factor_exposures(characteristics)
            results['fundamental_exposures'] = exposures

        # Linear model with provided factors
        if factor_returns is not None:
            self.linear_model.estimate_model(asset_returns, factor_returns)
            results['linear_model'] = {
                'loadings': self.linear_model.factor_loadings,
                'factor_covariance': self.linear_model.factor_covariance,
                'specific_variance': self.linear_model.specific_variance
            }

        return results

    def analyze_portfolio_risk(
        self,
        weights: pd.Series
    ) -> Dict:
        """
        Comprehensive portfolio risk analysis.
        """
        if self.linear_model.factor_loadings is None:
            raise ValueError("Model not estimated. Call estimate_full_model first.")

        # Risk decomposition
        decomposition = self.linear_model.calculate_risk_decomposition(weights)

        # Factor contributions
        factor_contrib = self.linear_model.calculate_factor_contribution(weights)

        # Full attribution
        attribution = self.attributor.attribute_risk(
            weights,
            self.linear_model.factor_loadings.drop('alpha', axis=1),
            self.linear_model.factor_covariance,
            self.linear_model.specific_variance
        )

        return {
            'decomposition': decomposition,
            'factor_contributions': factor_contrib,
            'attribution': attribution
        }

    def optimize_factor_exposure(
        self,
        current_weights: pd.Series,
        target_exposures: Dict[str, float],
        max_turnover: float = 0.2
    ) -> pd.Series:
        """
        Optimize weights to achieve target factor exposures.
        """
        assets = current_weights.index
        betas = self.linear_model.factor_loadings.loc[assets].drop('alpha', axis=1)
        n_assets = len(assets)

        def objective(w):
            # Minimize deviation from target exposures
            portfolio_betas = w @ betas.values
            deviation = 0
            for factor, target in target_exposures.items():
                if factor in betas.columns:
                    idx = list(betas.columns).index(factor)
                    deviation += (portfolio_betas[idx] - target) ** 2
            return deviation

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Full investment
            {'type': 'ineq', 'fun': lambda w: max_turnover - np.sum(np.abs(w - current_weights.values))}  # Turnover
        ]

        bounds = [(0, 0.1) for _ in range(n_assets)]  # Max 10% per asset

        result = optimize.minimize(
            objective,
            current_weights.values,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return pd.Series(result.x, index=assets)
```

---

## Signal Usage Guidelines

### Common Risk Factors

| Factor | Description | Typical Beta |
|--------|-------------|--------------|
| Market | Overall market exposure | 0.8 - 1.2 |
| Size | Small vs large cap | -0.5 to 0.5 |
| Value | Book-to-market | -0.3 to 0.3 |
| Momentum | 12-1 month returns | -0.3 to 0.3 |
| Volatility | Low vs high vol | -0.5 to 0.5 |

### Integration with Ordinis

```python
# Risk analysis in portfolio construction
risk_engine = FactorRiskEngine()

# Estimate model
risk_engine.estimate_full_model(
    asset_returns, factor_returns
)

# Analyze proposed portfolio
analysis = risk_engine.analyze_portfolio_risk(proposed_weights)

# Check factor exposures
if abs(analysis['decomposition']['portfolio_betas']['momentum']) > 0.5:
    print("Warning: High momentum exposure")

# Optimize to target exposures
optimal = risk_engine.optimize_factor_exposure(
    current_weights,
    {'market': 1.0, 'momentum': 0.2}
)
```

---

## Academic References

1. **Sharpe (1964)**: "Capital Asset Prices" (CAPM)
2. **Ross (1976)**: "Arbitrage Theory of Capital Asset Pricing" (APT)
3. **Fama & French (1993)**: "Common Risk Factors in Returns"
4. **Menchero (2010)**: "The Barra Risk Model Handbook"
5. **Connor & Korajczyk (1993)**: "A Test for the Number of Factors"
