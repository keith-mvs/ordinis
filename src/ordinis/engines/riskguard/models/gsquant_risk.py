"""
Goldman Sachs gs-quant risk management integration.

Reference: https://github.com/goldmansachs/gs-quant
Documentation: https://developer.gs.com/docs/gsquant/
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# Optional gs-quant import with graceful degradation
try:
    from gs_quant.markets.portfolio import Portfolio
    from gs_quant.risk import RiskMeasure
    from gs_quant.session import GsSession

    GS_QUANT_AVAILABLE = True
except ImportError:
    GS_QUANT_AVAILABLE = False
    Portfolio = None
    RiskMeasure = None
    GsSession = None


@dataclass
class RiskFactorExposure:
    """Exposure to a single risk factor."""

    factor_name: str
    exposure: float  # Dollar exposure
    sensitivity: float  # Percent change per factor unit
    contribution_to_var: float  # Contribution to portfolio VaR
    factor_volatility: float


@dataclass
class PortfolioRiskMetrics:
    """Comprehensive portfolio risk metrics."""

    # Value at Risk
    var_95: float  # 95% VaR (1-day)
    var_99: float  # 99% VaR (1-day)
    cvar_95: float  # Conditional VaR (Expected Shortfall)

    # Greeks (for options portfolios)
    delta: float
    gamma: float
    vega: float
    theta: float

    # Portfolio-level
    beta: float  # Market beta
    volatility: float  # Annualized volatility
    sharpe_ratio: float
    max_drawdown: float

    # Factor exposures
    factor_exposures: list[RiskFactorExposure] = field(default_factory=list)

    # Metadata
    calculation_timestamp: datetime = field(default_factory=datetime.utcnow)
    model_name: str = "gsquant_risk"


@dataclass
class ScenarioResult:
    """Result of scenario analysis."""

    scenario_name: str
    scenario_description: str
    portfolio_pnl: float
    portfolio_pnl_percent: float
    worst_position: str
    worst_position_pnl: float
    best_position: str
    best_position_pnl: float
    positions_affected: int


class VaRCalculator:
    """
    Value at Risk calculator using multiple methodologies.

    Supports:
    - Historical simulation
    - Parametric (variance-covariance)
    - Monte Carlo simulation
    """

    def __init__(
        self,
        confidence_levels: list[float] | None = None,
        lookback_days: int = 252,
        monte_carlo_simulations: int = 10000,
    ):
        """
        Initialize VaR calculator.

        Args:
            confidence_levels: VaR confidence levels (default: [0.95, 0.99])
            lookback_days: Historical lookback period
            monte_carlo_simulations: Number of MC simulations
        """
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self.lookback_days = lookback_days
        self.monte_carlo_simulations = monte_carlo_simulations

    def calculate_historical_var(
        self,
        returns: pd.Series | pd.DataFrame,
        portfolio_value: float,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """
        Calculate VaR using historical simulation.

        Args:
            returns: Historical returns (daily)
            portfolio_value: Current portfolio value
            confidence: Confidence level (0.95 = 95%)

        Returns:
            Dict with VaR and CVaR values
        """
        if isinstance(returns, pd.DataFrame):
            # Assume equal-weighted portfolio if DataFrame
            returns = returns.mean(axis=1)

        returns = returns.dropna()

        if len(returns) < 20:
            # Insufficient data - return conservative estimate
            return {
                "var": portfolio_value * 0.05,
                "cvar": portfolio_value * 0.08,
                "method": "conservative_fallback",
                "data_points": len(returns),
            }

        # Historical VaR
        var_percentile = 1 - confidence
        var_return = np.percentile(returns, var_percentile * 100)
        var_dollar = abs(var_return) * portfolio_value

        # CVaR (Expected Shortfall)
        tail_returns = returns[returns <= var_return]
        cvar_return = tail_returns.mean() if len(tail_returns) > 0 else var_return
        cvar_dollar = abs(cvar_return) * portfolio_value

        return {
            "var": var_dollar,
            "var_percent": abs(var_return),
            "cvar": cvar_dollar,
            "cvar_percent": abs(cvar_return),
            "method": "historical_simulation",
            "data_points": len(returns),
            "confidence": confidence,
        }

    def calculate_parametric_var(
        self,
        returns: pd.Series | pd.DataFrame,
        portfolio_value: float,
        confidence: float = 0.95,
    ) -> dict[str, Any]:
        """
        Calculate VaR using parametric (variance-covariance) method.

        Assumes normal distribution of returns.
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.mean(axis=1)

        returns = returns.dropna()

        if len(returns) < 20:
            return self.calculate_historical_var(returns, portfolio_value, confidence)

        # Calculate moments
        mean_return = returns.mean()
        std_return = returns.std()

        # Z-score for confidence level
        from scipy import stats

        z_score = stats.norm.ppf(1 - confidence)

        # Parametric VaR
        var_return = mean_return + z_score * std_return
        var_dollar = abs(var_return) * portfolio_value

        # Parametric CVaR (expected shortfall under normality)
        pdf_z = stats.norm.pdf(z_score)
        cvar_multiplier = pdf_z / (1 - confidence)
        cvar_return = mean_return - cvar_multiplier * std_return
        cvar_dollar = abs(cvar_return) * portfolio_value

        return {
            "var": var_dollar,
            "var_percent": abs(var_return),
            "cvar": cvar_dollar,
            "cvar_percent": abs(cvar_return),
            "method": "parametric",
            "mean_return": mean_return,
            "volatility": std_return,
            "confidence": confidence,
        }

    def calculate_monte_carlo_var(
        self,
        returns: pd.Series | pd.DataFrame,
        portfolio_value: float,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> dict[str, Any]:
        """
        Calculate VaR using Monte Carlo simulation.

        Simulates future returns based on historical distribution.
        """
        if isinstance(returns, pd.DataFrame):
            returns = returns.mean(axis=1)

        returns = returns.dropna()

        if len(returns) < 20:
            return self.calculate_historical_var(returns, portfolio_value, confidence)

        # Fit distribution parameters
        mean_return = returns.mean()
        std_return = returns.std()

        # Simulate returns
        np.random.seed(42)  # Reproducibility
        simulated_returns = np.random.normal(
            mean_return * horizon_days,
            std_return * np.sqrt(horizon_days),
            self.monte_carlo_simulations,
        )

        # Calculate VaR from simulations
        var_percentile = 1 - confidence
        var_return = np.percentile(simulated_returns, var_percentile * 100)
        var_dollar = abs(var_return) * portfolio_value

        # CVaR
        tail_returns = simulated_returns[simulated_returns <= var_return]
        cvar_return = tail_returns.mean() if len(tail_returns) > 0 else var_return
        cvar_dollar = abs(cvar_return) * portfolio_value

        return {
            "var": var_dollar,
            "var_percent": abs(var_return),
            "cvar": cvar_dollar,
            "cvar_percent": abs(cvar_return),
            "method": "monte_carlo",
            "simulations": self.monte_carlo_simulations,
            "horizon_days": horizon_days,
            "confidence": confidence,
        }


class GSQuantRiskManager:
    """
    Risk management using Goldman Sachs gs-quant library.

    Provides:
    - Portfolio risk analytics (VaR, Greeks, factor exposures)
    - Scenario analysis
    - Risk factor decomposition
    - Position-level risk attribution
    """

    # Standard risk factors for equities
    EQUITY_FACTORS = [
        "market",
        "size",
        "value",
        "momentum",
        "quality",
        "volatility",
        "dividend_yield",
    ]

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        use_api: bool = False,
    ):
        """
        Initialize GS Quant risk manager.

        Args:
            client_id: GS API client ID (for API access)
            client_secret: GS API client secret
            use_api: Whether to use GS Marquee API
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self.use_api = use_api and GS_QUANT_AVAILABLE
        self._session = None
        self._var_calculator = VaRCalculator()

        # Initialize session if credentials provided
        if self.use_api and client_id and client_secret:
            self._initialize_session()

    def _initialize_session(self) -> None:
        """Initialize GS Quant API session."""
        if not GS_QUANT_AVAILABLE:
            return

        try:
            self._session = GsSession.get(
                client_id=self.client_id,
                client_secret=self.client_secret,
            )
        except Exception:
            self._session = None

    def calculate_portfolio_risk(
        self,
        positions: dict[str, dict[str, Any]],
        returns_data: pd.DataFrame,
        portfolio_value: float,
    ) -> PortfolioRiskMetrics:
        """
        Calculate comprehensive portfolio risk metrics.

        Args:
            positions: Dict of symbol -> {quantity, value, weight}
            returns_data: Historical returns DataFrame (symbols as columns)
            portfolio_value: Total portfolio value

        Returns:
            PortfolioRiskMetrics with all risk measures
        """
        if self.use_api and self._session:
            return self._calculate_risk_api(positions, portfolio_value)

        return self._calculate_risk_local(positions, returns_data, portfolio_value)

    def _calculate_risk_local(
        self,
        positions: dict[str, dict[str, Any]],
        returns_data: pd.DataFrame,
        portfolio_value: float,
    ) -> PortfolioRiskMetrics:
        """Calculate risk metrics using local computation."""
        # Calculate portfolio returns
        weights = {}
        for symbol, pos in positions.items():
            if "weight" in pos:
                weights[symbol] = pos["weight"]
            elif "value" in pos:
                weights[symbol] = pos["value"] / portfolio_value
            else:
                weights[symbol] = 1.0 / len(positions)

        # Get available symbols
        available_symbols = [s for s in weights if s in returns_data.columns]

        if not available_symbols:
            return self._empty_risk_metrics()

        # Calculate weighted portfolio returns
        portfolio_returns = sum(
            returns_data[symbol] * weights[symbol]
            for symbol in available_symbols
            if symbol in returns_data.columns
        )

        # VaR calculations
        var_95 = self._var_calculator.calculate_historical_var(
            portfolio_returns, portfolio_value, 0.95
        )
        var_99 = self._var_calculator.calculate_historical_var(
            portfolio_returns, portfolio_value, 0.99
        )

        # Volatility and beta
        annualized_vol = portfolio_returns.std() * np.sqrt(252)

        # Calculate beta if market returns available
        beta = 1.0
        if "SPY" in returns_data.columns or "market" in returns_data.columns:
            market_col = "SPY" if "SPY" in returns_data.columns else "market"
            market_returns = returns_data[market_col].dropna()
            aligned_returns = portfolio_returns.reindex(market_returns.index).dropna()

            if len(aligned_returns) > 20:
                covariance = np.cov(aligned_returns, market_returns.loc[aligned_returns.index])[
                    0, 1
                ]
                market_variance = market_returns.loc[aligned_returns.index].var()
                if market_variance > 0:
                    beta = covariance / market_variance

        # Sharpe ratio (assuming 0% risk-free rate)
        mean_return = portfolio_returns.mean() * 252  # Annualized
        sharpe = mean_return / annualized_vol if annualized_vol > 0 else 0.0

        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Factor exposures (simplified)
        factor_exposures = self._estimate_factor_exposures(positions, returns_data, portfolio_value)

        return PortfolioRiskMetrics(
            var_95=var_95["var"],
            var_99=var_99["var"],
            cvar_95=var_95["cvar"],
            delta=1.0,  # For equities, delta is ~1
            gamma=0.0,  # No gamma for linear instruments
            vega=0.0,
            theta=0.0,
            beta=beta,
            volatility=annualized_vol,
            sharpe_ratio=sharpe,
            max_drawdown=abs(max_drawdown),
            factor_exposures=factor_exposures,
            calculation_timestamp=datetime.utcnow(),
            model_name="local_risk_model",
        )

    def _calculate_risk_api(
        self,
        positions: dict[str, dict[str, Any]],
        portfolio_value: float,
    ) -> PortfolioRiskMetrics:
        """Calculate risk metrics using GS Quant API."""
        if not GS_QUANT_AVAILABLE or not self._session:
            return self._empty_risk_metrics()

        try:
            # Create GS Portfolio object
            # Note: This is a simplified example - actual implementation
            # would need proper instrument definitions
            return self._empty_risk_metrics()  # Placeholder for API implementation

        except Exception:
            return self._empty_risk_metrics()

    def _estimate_factor_exposures(
        self,
        positions: dict[str, dict[str, Any]],
        returns_data: pd.DataFrame,
        portfolio_value: float,
    ) -> list[RiskFactorExposure]:
        """
        Estimate factor exposures using returns regression.

        Simple factor model using market as the primary factor.
        """
        exposures = []

        # Market factor (beta)
        if "SPY" in returns_data.columns or "market" in returns_data.columns:
            market_col = "SPY" if "SPY" in returns_data.columns else "market"
            market_returns = returns_data[market_col].dropna()
            market_vol = market_returns.std() * np.sqrt(252)

            # Estimate market exposure
            total_exposure = sum(pos.get("value", 0) for pos in positions.values())

            exposures.append(
                RiskFactorExposure(
                    factor_name="market",
                    exposure=total_exposure,
                    sensitivity=1.0,  # Simplified - would regress for actual beta
                    contribution_to_var=total_exposure * market_vol * 1.65 / np.sqrt(252),
                    factor_volatility=market_vol,
                )
            )

        return exposures

    def _empty_risk_metrics(self) -> PortfolioRiskMetrics:
        """Return empty risk metrics when calculation not possible."""
        return PortfolioRiskMetrics(
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            beta=1.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            factor_exposures=[],
            model_name="empty",
        )

    def run_scenario_analysis(
        self,
        positions: dict[str, dict[str, Any]],
        scenarios: list[dict[str, Any]],
    ) -> list[ScenarioResult]:
        """
        Run scenario analysis on portfolio.

        Args:
            positions: Dict of symbol -> {quantity, value, current_price}
            scenarios: List of scenario definitions with factor shocks

        Returns:
            List of ScenarioResult for each scenario
        """
        results = []

        for scenario in scenarios:
            scenario_name = scenario.get("name", "Unnamed")
            scenario_desc = scenario.get("description", "")
            market_shock = scenario.get("market_shock", 0.0)
            volatility_shock = scenario.get("volatility_shock", 0.0)
            sector_shocks = scenario.get("sector_shocks", {})

            # Calculate P&L for each position
            position_pnls = {}
            for symbol, pos in positions.items():
                value = pos.get("value", 0)
                sector = pos.get("sector", "other")

                # Apply shocks
                shock = market_shock
                if sector in sector_shocks:
                    shock += sector_shocks[sector]

                pnl = value * shock
                position_pnls[symbol] = pnl

            # Aggregate results
            total_pnl = sum(position_pnls.values())
            portfolio_value = sum(pos.get("value", 0) for pos in positions.values())

            worst_symbol = (
                min(position_pnls, key=lambda x: position_pnls[x]) if position_pnls else ""
            )
            best_symbol = (
                max(position_pnls, key=lambda x: position_pnls[x]) if position_pnls else ""
            )

            results.append(
                ScenarioResult(
                    scenario_name=scenario_name,
                    scenario_description=scenario_desc,
                    portfolio_pnl=total_pnl,
                    portfolio_pnl_percent=total_pnl / portfolio_value if portfolio_value > 0 else 0,
                    worst_position=worst_symbol,
                    worst_position_pnl=position_pnls.get(worst_symbol, 0),
                    best_position=best_symbol,
                    best_position_pnl=position_pnls.get(best_symbol, 0),
                    positions_affected=len([p for p in position_pnls.values() if p != 0]),
                )
            )

        return results

    def get_standard_scenarios(self) -> list[dict[str, Any]]:
        """Get standard stress test scenarios."""
        return [
            {
                "name": "Market Crash (-20%)",
                "description": "Broad market decline of 20%",
                "market_shock": -0.20,
                "volatility_shock": 0.50,
            },
            {
                "name": "Market Correction (-10%)",
                "description": "Standard market correction",
                "market_shock": -0.10,
                "volatility_shock": 0.25,
            },
            {
                "name": "Flash Crash (-5%)",
                "description": "Rapid intraday decline",
                "market_shock": -0.05,
                "volatility_shock": 1.00,
            },
            {
                "name": "Tech Selloff",
                "description": "Technology sector decline of 15%",
                "market_shock": -0.05,
                "sector_shocks": {"technology": -0.10},
            },
            {
                "name": "Rate Shock",
                "description": "Interest rate increase impact",
                "market_shock": -0.03,
                "sector_shocks": {
                    "financials": 0.05,
                    "utilities": -0.08,
                    "real_estate": -0.10,
                },
            },
            {
                "name": "Bull Market (+15%)",
                "description": "Strong market rally",
                "market_shock": 0.15,
                "volatility_shock": -0.20,
            },
        ]

    def calculate_position_risk_contribution(
        self,
        positions: dict[str, dict[str, Any]],
        returns_data: pd.DataFrame,
    ) -> dict[str, dict[str, float]]:
        """
        Calculate each position's contribution to portfolio risk.

        Args:
            positions: Dict of symbol -> {quantity, value, weight}
            returns_data: Historical returns DataFrame

        Returns:
            Dict of symbol -> {marginal_var, component_var, risk_contribution_pct}
        """
        contributions = {}

        # Calculate portfolio variance
        available = [s for s in positions if s in returns_data.columns]
        if len(available) < 2:
            return {
                s: {"marginal_var": 0, "component_var": 0, "risk_contribution_pct": 0}
                for s in positions
            }

        portfolio_value = sum(pos.get("value", 0) for pos in positions.values())
        weights = np.array([positions[s].get("value", 0) / portfolio_value for s in available])

        returns_subset = returns_data[available].dropna()
        cov_matrix = returns_subset.cov().values * 252  # Annualized

        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)

        for i, symbol in enumerate(available):
            # Marginal VaR contribution
            marginal_cov = cov_matrix[i] @ weights
            marginal_var = marginal_cov / portfolio_vol if portfolio_vol > 0 else 0

            # Component VaR
            component_var = weights[i] * marginal_var

            # Percentage contribution
            total_component_var = (
                sum(
                    weights[j] * (cov_matrix[j] @ weights) / portfolio_vol
                    for j in range(len(available))
                )
                if portfolio_vol > 0
                else 1
            )

            risk_pct = component_var / total_component_var if total_component_var > 0 else 0

            contributions[symbol] = {
                "marginal_var": float(marginal_var * 1.65 * portfolio_value),  # 95% VaR
                "component_var": float(component_var * 1.65 * portfolio_value),
                "risk_contribution_pct": float(risk_pct),
            }

        return contributions

    def describe(self) -> dict[str, Any]:
        """Get risk manager description."""
        return {
            "model": "GSQuantRiskManager",
            "gs_quant_available": GS_QUANT_AVAILABLE,
            "api_enabled": self.use_api,
            "session_active": self._session is not None,
            "supported_measures": [
                "VaR (Historical, Parametric, Monte Carlo)",
                "CVaR / Expected Shortfall",
                "Beta",
                "Volatility",
                "Sharpe Ratio",
                "Max Drawdown",
                "Factor Exposures",
                "Scenario Analysis",
                "Risk Attribution",
            ],
            "standard_scenarios": len(self.get_standard_scenarios()),
        }
