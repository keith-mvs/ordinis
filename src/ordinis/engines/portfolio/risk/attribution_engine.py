"""
Risk Attribution Engine - Factor-based risk decomposition.

Provides detailed attribution of portfolio risk to:
- Factor exposures (market, size, value, momentum, quality, volatility)
- Individual securities (idiosyncratic risk)
- Sectors and industries
- Asset classes

Gap Addressed: Missing risk attribution (factor-based decomposition).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
import logging

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# Risk Factor Definitions
# ============================================================================


class RiskFactor(str, Enum):
    """Standard risk factors for attribution."""

    # Market factors
    MARKET = "market"  # Equity market beta
    SIZE = "size"  # SMB (small minus big)
    VALUE = "value"  # HML (high minus low book-to-market)
    MOMENTUM = "momentum"  # UMD (up minus down)
    QUALITY = "quality"  # QMJ (quality minus junk)
    VOLATILITY = "volatility"  # Low volatility factor

    # Fixed income factors
    DURATION = "duration"
    CREDIT = "credit"
    INFLATION = "inflation"

    # Macro factors
    GROWTH = "growth"
    RATES = "rates"
    FX = "fx"
    COMMODITY = "commodity"

    # Alternative factors
    LIQUIDITY = "liquidity"
    CARRY = "carry"


@dataclass
class FactorExposure:
    """Exposure to a single risk factor."""

    factor: RiskFactor
    beta: float  # Factor loading
    contribution_pct: float = 0.0  # % of total risk
    contribution_var: float = 0.0  # Variance contribution
    t_stat: float = 0.0  # Statistical significance
    confidence: float = 0.0  # R-squared or similar


@dataclass
class FactorReturns:
    """Historical factor returns for attribution."""

    factor: RiskFactor
    returns: np.ndarray  # Time series of returns
    mean_return: float = 0.0
    volatility: float = 0.0
    sharpe: float = 0.0


# ============================================================================
# Risk Attribution Results
# ============================================================================


@dataclass
class SecurityAttribution:
    """Risk attribution for a single security."""

    symbol: str
    weight: float
    total_risk_contribution: float  # % of portfolio risk
    systematic_risk: float  # Factor-explained risk
    idiosyncratic_risk: float  # Security-specific risk
    factor_exposures: dict[RiskFactor, float] = field(default_factory=dict)


@dataclass
class SectorAttribution:
    """Risk attribution for a sector."""

    sector: str
    weight: float
    total_risk_contribution: float
    active_risk: float  # Risk vs benchmark
    securities: list[str] = field(default_factory=list)


@dataclass
class RiskAttributionResult:
    """Complete risk attribution analysis."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Total portfolio risk
    total_volatility: float = 0.0  # Annualized volatility
    total_var_95: float = 0.0  # 1-day 95% VaR
    total_cvar_95: float = 0.0  # 1-day 95% CVaR (Expected Shortfall)

    # Factor attribution
    factor_exposures: list[FactorExposure] = field(default_factory=list)
    systematic_risk_pct: float = 0.0  # % explained by factors
    idiosyncratic_risk_pct: float = 0.0  # % unexplained

    # Security attribution
    security_attributions: list[SecurityAttribution] = field(default_factory=list)

    # Sector attribution
    sector_attributions: list[SectorAttribution] = field(default_factory=list)

    # Marginal risk
    marginal_var: dict[str, float] = field(default_factory=dict)
    marginal_cvar: dict[str, float] = field(default_factory=dict)

    # Concentration metrics
    herfindahl_index: float = 0.0  # Position concentration
    top_5_risk_contribution: float = 0.0  # % from top 5 positions

    def get_top_risk_contributors(self, n: int = 5) -> list[SecurityAttribution]:
        """Get top N risk contributors."""
        sorted_attrs = sorted(
            self.security_attributions,
            key=lambda x: x.total_risk_contribution,
            reverse=True,
        )
        return sorted_attrs[:n]

    def get_factor_summary(self) -> dict[str, float]:
        """Get summary of factor contributions."""
        return {
            exp.factor.value: exp.contribution_pct
            for exp in self.factor_exposures
        }


# ============================================================================
# Risk Attribution Engine
# ============================================================================


class RiskAttributionEngine:
    """Engine for decomposing portfolio risk.

    Supports:
    - Factor-based attribution (Fama-French, Barra-style)
    - Marginal VaR/CVaR calculation
    - Sector and security-level decomposition
    - Historical and parametric methods

    Example:
        >>> engine = RiskAttributionEngine()
        >>> engine.set_factor_returns(market_returns, factor_returns)
        >>> result = engine.attribute_risk(weights, returns)
        >>> print(result.get_top_risk_contributors())
    """

    def __init__(
        self,
        lookback_days: int = 252,
        confidence_level: float = 0.95,
        use_parametric: bool = True,
    ) -> None:
        """Initialize risk attribution engine.

        Args:
            lookback_days: Days of history for calculations
            confidence_level: VaR/CVaR confidence level
            use_parametric: Use parametric (True) or historical (False) VaR
        """
        self.lookback_days = lookback_days
        self.confidence_level = confidence_level
        self.use_parametric = use_parametric

        # Factor returns storage
        self._factor_returns: dict[RiskFactor, FactorReturns] = {}

        # Covariance matrix
        self._cov_matrix: np.ndarray | None = None
        self._cov_symbols: list[str] = []

    def set_factor_returns(
        self,
        factor: RiskFactor,
        returns: np.ndarray,
    ) -> None:
        """Set historical returns for a factor.

        Args:
            factor: Risk factor
            returns: Array of historical returns
        """
        mean_ret = float(np.mean(returns))
        vol = float(np.std(returns) * np.sqrt(252))
        sharpe = mean_ret / vol * np.sqrt(252) if vol > 0 else 0

        self._factor_returns[factor] = FactorReturns(
            factor=factor,
            returns=returns,
            mean_return=mean_ret,
            volatility=vol,
            sharpe=sharpe,
        )

    def set_covariance_matrix(
        self,
        cov_matrix: np.ndarray,
        symbols: list[str],
    ) -> None:
        """Set the covariance matrix.

        Args:
            cov_matrix: NxN covariance matrix
            symbols: List of N symbols in matrix order
        """
        self._cov_matrix = cov_matrix
        self._cov_symbols = symbols

    def estimate_covariance(
        self,
        returns: np.ndarray,
        symbols: list[str],
        shrinkage: float = 0.1,
    ) -> None:
        """Estimate covariance matrix from returns.

        Uses Ledoit-Wolf shrinkage toward identity matrix.

        Args:
            returns: TxN array of returns (T periods, N assets)
            symbols: List of N symbols
            shrinkage: Shrinkage intensity (0-1)
        """
        sample_cov = np.cov(returns, rowvar=False)

        # Ledoit-Wolf shrinkage
        n = sample_cov.shape[0]
        mu = np.trace(sample_cov) / n
        target = mu * np.eye(n)

        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * target

        self._cov_matrix = shrunk_cov
        self._cov_symbols = symbols

    def calculate_portfolio_variance(
        self,
        weights: dict[str, float],
    ) -> float:
        """Calculate portfolio variance.

        Args:
            weights: Symbol -> weight mapping

        Returns:
            Portfolio variance
        """
        if self._cov_matrix is None:
            logger.warning("Covariance matrix not set, using equal correlations")
            avg_var = 0.04  # 20% vol assumption
            return sum(w**2 for w in weights.values()) * avg_var

        # Build weight vector aligned with covariance matrix
        w = np.zeros(len(self._cov_symbols))
        for i, sym in enumerate(self._cov_symbols):
            w[i] = weights.get(sym, 0)

        # Portfolio variance = w' * Σ * w
        return float(w @ self._cov_matrix @ w)

    def calculate_marginal_risk(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Calculate marginal VaR for each position.

        Marginal VaR measures incremental risk from small weight increase.

        Args:
            weights: Symbol -> weight mapping

        Returns:
            Symbol -> marginal VaR mapping
        """
        if self._cov_matrix is None:
            return dict.fromkeys(weights, 0.0)

        port_vol = np.sqrt(self.calculate_portfolio_variance(weights))
        z_score = 1.645 if abs(self.confidence_level - 0.95) < 0.001 else 2.326

        marginal: dict[str, float] = {}
        for i, sym in enumerate(self._cov_symbols):
            if sym not in weights:
                continue

            w = np.zeros(len(self._cov_symbols))
            for j, s in enumerate(self._cov_symbols):
                w[j] = weights.get(s, 0)

            # Marginal contribution = (Σ * w)_i / σ_p
            cov_contribution = self._cov_matrix[i, :] @ w
            marginal_vol = cov_contribution / port_vol if port_vol > 0 else 0

            marginal[sym] = float(marginal_vol * z_score)

        return marginal

    def calculate_component_risk(
        self,
        weights: dict[str, float],
    ) -> dict[str, float]:
        """Calculate component VaR for each position.

        Component VaR sums to total VaR.

        Args:
            weights: Symbol -> weight mapping

        Returns:
            Symbol -> component VaR mapping
        """
        marginal = self.calculate_marginal_risk(weights)
        return {
            sym: marginal.get(sym, 0) * weights.get(sym, 0)
            for sym in weights
        }

    def run_factor_regression(
        self,
        asset_returns: np.ndarray,
        factor_returns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Run multivariate regression for factor loadings.

        Args:
            asset_returns: Tx1 array of asset returns
            factor_returns: TxK array of factor returns

        Returns:
            Tuple of (betas, residuals, r_squared)
        """
        # Add intercept
        X = np.column_stack([np.ones(len(factor_returns)), factor_returns])
        y = asset_returns

        # OLS: beta = (X'X)^-1 X'y
        try:
            betas = np.linalg.lstsq(X, y, rcond=None)[0]
            predictions = X @ betas
            residuals = y - predictions

            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return betas[1:], residuals, float(r_squared)
        except np.linalg.LinAlgError:
            return np.zeros(factor_returns.shape[1]), asset_returns, 0.0

    def attribute_risk(
        self,
        weights: dict[str, float],
        returns: np.ndarray | None = None,
        symbols: list[str] | None = None,
        sector_mapping: dict[str, str] | None = None,
    ) -> RiskAttributionResult:
        """Perform complete risk attribution analysis.

        Args:
            weights: Symbol -> weight mapping
            returns: TxN array of historical returns
            symbols: List of N symbols corresponding to returns columns
            sector_mapping: Symbol -> sector mapping

        Returns:
            Complete risk attribution result
        """
        result = RiskAttributionResult()

        # Calculate total portfolio volatility
        port_var = self.calculate_portfolio_variance(weights)
        result.total_volatility = float(np.sqrt(port_var) * np.sqrt(252))

        # Calculate VaR and CVaR
        z_score = 1.645 if abs(self.confidence_level - 0.95) < 0.001 else 2.326
        result.total_var_95 = result.total_volatility / np.sqrt(252) * z_score
        result.total_cvar_95 = result.total_var_95 * 1.25  # Approximate

        # Marginal risk
        result.marginal_var = self.calculate_marginal_risk(weights)

        # Component risk for security attribution
        component_risk = self.calculate_component_risk(weights)
        total_component = sum(abs(v) for v in component_risk.values())

        if total_component > 0:
            for sym, weight in weights.items():
                comp_risk = component_risk.get(sym, 0)
                risk_contrib = abs(comp_risk) / total_component * 100

                result.security_attributions.append(
                    SecurityAttribution(
                        symbol=sym,
                        weight=weight,
                        total_risk_contribution=risk_contrib,
                        systematic_risk=risk_contrib * 0.7,  # Placeholder
                        idiosyncratic_risk=risk_contrib * 0.3,
                    )
                )

        # Factor attribution if returns available
        if returns is not None and symbols is not None and self._factor_returns:
            result.factor_exposures = self._calculate_factor_exposures(
                weights, returns, symbols
            )

        # Sector attribution
        if sector_mapping:
            result.sector_attributions = self._calculate_sector_attribution(
                weights, component_risk, sector_mapping
            )

        # Concentration metrics
        result.herfindahl_index = sum(w**2 for w in weights.values())

        top_contributors = result.get_top_risk_contributors(5)
        result.top_5_risk_contribution = sum(
            c.total_risk_contribution for c in top_contributors
        )

        # Systematic vs idiosyncratic split
        if result.factor_exposures:
            result.systematic_risk_pct = sum(
                exp.contribution_pct for exp in result.factor_exposures
            )
            result.idiosyncratic_risk_pct = 100 - result.systematic_risk_pct

        return result

    def _calculate_factor_exposures(
        self,
        weights: dict[str, float],
        returns: np.ndarray,
        symbols: list[str],
    ) -> list[FactorExposure]:
        """Calculate portfolio factor exposures."""
        exposures: list[FactorExposure] = []

        # Build portfolio return series
        w = np.array([weights.get(s, 0) for s in symbols])
        port_returns = returns @ w

        # Stack available factor returns
        available_factors: list[RiskFactor] = []
        factor_list: list[np.ndarray] = []

        for factor, factor_data in self._factor_returns.items():
            if len(factor_data.returns) >= len(port_returns):
                available_factors.append(factor)
                factor_list.append(factor_data.returns[:len(port_returns)])

        if not available_factors:
            return exposures

        factor_matrix = np.column_stack(factor_list)

        # Run regression
        betas, _, r_squared = self.run_factor_regression(
            port_returns, factor_matrix
        )

        # Calculate contribution of each factor
        total_var = np.var(port_returns)
        if total_var > 0:
            for i, factor in enumerate(available_factors):
                factor_var = betas[i]**2 * np.var(factor_matrix[:, i])
                contribution = factor_var / total_var * 100

                exposures.append(
                    FactorExposure(
                        factor=factor,
                        beta=float(betas[i]),
                        contribution_pct=float(contribution),
                        contribution_var=float(factor_var),
                        t_stat=0.0,  # Would need std errors
                        confidence=float(r_squared),
                    )
                )

        return exposures

    def _calculate_sector_attribution(
        self,
        weights: dict[str, float],
        component_risk: dict[str, float],
        sector_mapping: dict[str, str],
    ) -> list[SectorAttribution]:
        """Calculate sector-level risk attribution."""
        sector_data: dict[str, dict[str, Any]] = {}

        for sym, weight in weights.items():
            sector = sector_mapping.get(sym, "Unknown")

            if sector not in sector_data:
                sector_data[sector] = {
                    "weight": 0.0,
                    "risk": 0.0,
                    "securities": [],
                }

            sector_data[sector]["weight"] += weight
            sector_data[sector]["risk"] += abs(component_risk.get(sym, 0))
            sector_data[sector]["securities"].append(sym)

        # Normalize risk contributions
        total_risk = sum(d["risk"] for d in sector_data.values())

        attributions = []
        for sector, data in sector_data.items():
            risk_pct = (data["risk"] / total_risk * 100) if total_risk > 0 else 0

            attributions.append(
                SectorAttribution(
                    sector=sector,
                    weight=data["weight"],
                    total_risk_contribution=risk_pct,
                    active_risk=0.0,  # Would need benchmark
                    securities=data["securities"],
                )
            )

        return sorted(
            attributions,
            key=lambda x: x.total_risk_contribution,
            reverse=True,
        )


# ============================================================================
# Factory Functions
# ============================================================================


def create_fama_french_engine(
    market_returns: np.ndarray,
    smb_returns: np.ndarray,
    hml_returns: np.ndarray,
    momentum_returns: np.ndarray | None = None,
) -> RiskAttributionEngine:
    """Create engine with Fama-French factor returns.

    Args:
        market_returns: Market factor returns
        smb_returns: Size factor returns (SMB)
        hml_returns: Value factor returns (HML)
        momentum_returns: Optional momentum factor returns

    Returns:
        Configured RiskAttributionEngine
    """
    engine = RiskAttributionEngine()

    engine.set_factor_returns(RiskFactor.MARKET, market_returns)
    engine.set_factor_returns(RiskFactor.SIZE, smb_returns)
    engine.set_factor_returns(RiskFactor.VALUE, hml_returns)

    if momentum_returns is not None:
        engine.set_factor_returns(RiskFactor.MOMENTUM, momentum_returns)

    return engine


def create_simple_attribution_engine() -> RiskAttributionEngine:
    """Create a simple attribution engine for testing.

    Uses synthetic factor returns.

    Returns:
        Configured RiskAttributionEngine
    """
    engine = RiskAttributionEngine()

    # Synthetic factor returns (252 days)
    rng = np.random.default_rng(42)
    n_days = 252

    # Market factor: ~10% annual return, 16% vol
    engine.set_factor_returns(
        RiskFactor.MARKET,
        rng.normal(0.0004, 0.01, n_days),
    )
    # Size factor: ~2.5% annual
    engine.set_factor_returns(
        RiskFactor.SIZE,
        rng.normal(0.0001, 0.005, n_days),
    )
    # Value factor: ~3.8% annual
    engine.set_factor_returns(
        RiskFactor.VALUE,
        rng.normal(0.00015, 0.006, n_days),
    )
    # Momentum factor: ~5% annual
    engine.set_factor_returns(
        RiskFactor.MOMENTUM,
        rng.normal(0.0002, 0.008, n_days),
    )

    return engine
