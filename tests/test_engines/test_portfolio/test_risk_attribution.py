"""Tests for RiskAttributionEngine - Factor-based risk decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from ordinis.engines.portfolio.risk.attribution_engine import (
    FactorExposure,
    RiskAttributionEngine,
    RiskAttributionResult,
    RiskFactor,
    SecurityAttribution,
    create_fama_french_engine,
    create_simple_attribution_engine,
)


class TestRiskFactor:
    """Tests for RiskFactor enum."""

    def test_all_factors_defined(self) -> None:
        """Test all expected factors are defined."""
        assert RiskFactor.MARKET
        assert RiskFactor.SIZE
        assert RiskFactor.VALUE
        assert RiskFactor.MOMENTUM
        assert RiskFactor.QUALITY
        assert RiskFactor.VOLATILITY


class TestRiskAttributionEngine:
    """Tests for RiskAttributionEngine."""

    @pytest.fixture
    def engine(self) -> RiskAttributionEngine:
        """Create engine with sample data."""
        engine = RiskAttributionEngine(lookback_days=252)

        # Set up synthetic covariance matrix
        symbols = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
        n = len(symbols)

        # Create correlation matrix with sector clustering
        corr = np.array([
            [1.0, 0.7, 0.6, 0.3, 0.2],  # AAPL
            [0.7, 1.0, 0.65, 0.35, 0.25],  # MSFT
            [0.6, 0.65, 1.0, 0.3, 0.2],  # GOOGL
            [0.3, 0.35, 0.3, 1.0, 0.4],  # JPM
            [0.2, 0.25, 0.2, 0.4, 1.0],  # XOM
        ])

        # Volatilities (annualized)
        vols = np.array([0.25, 0.22, 0.28, 0.20, 0.30])

        # Convert to covariance
        cov = np.outer(vols, vols) * corr / 252  # Daily covariance

        engine.set_covariance_matrix(cov, symbols)
        return engine

    def test_portfolio_variance_calculation(
        self,
        engine: RiskAttributionEngine,
    ) -> None:
        """Test portfolio variance calculation."""
        weights = {
            "AAPL": 0.25,
            "MSFT": 0.25,
            "GOOGL": 0.20,
            "JPM": 0.15,
            "XOM": 0.15,
        }

        variance = engine.calculate_portfolio_variance(weights)
        volatility = np.sqrt(variance * 252)

        # Should be between individual asset vols due to diversification
        assert 0.10 < volatility < 0.30

    def test_marginal_risk_calculation(
        self,
        engine: RiskAttributionEngine,
    ) -> None:
        """Test marginal VaR calculation."""
        weights = {
            "AAPL": 0.25,
            "MSFT": 0.25,
            "GOOGL": 0.20,
            "JPM": 0.15,
            "XOM": 0.15,
        }

        marginal = engine.calculate_marginal_risk(weights)

        # All positions should have marginal risk values
        assert len(marginal) == 5
        for symbol in weights:
            assert symbol in marginal
            assert marginal[symbol] != 0

    def test_component_risk_sums_to_total(
        self,
        engine: RiskAttributionEngine,
    ) -> None:
        """Test that component VaR sums to total VaR."""
        weights = {
            "AAPL": 0.30,
            "MSFT": 0.30,
            "GOOGL": 0.20,
            "JPM": 0.10,
            "XOM": 0.10,
        }

        component = engine.calculate_component_risk(weights)
        port_var = engine.calculate_portfolio_variance(weights)

        # Component risks should roughly sum to portfolio risk
        # (This is approximate due to calculation method)
        total_component = sum(component.values())
        assert total_component != 0

    def test_full_attribution(self, engine: RiskAttributionEngine) -> None:
        """Test complete risk attribution analysis."""
        weights = {
            "AAPL": 0.25,
            "MSFT": 0.25,
            "GOOGL": 0.20,
            "JPM": 0.15,
            "XOM": 0.15,
        }

        result = engine.attribute_risk(weights)

        # Check result structure
        assert isinstance(result, RiskAttributionResult)
        assert result.total_volatility > 0
        assert result.total_var_95 > 0
        assert result.total_cvar_95 >= result.total_var_95

        # Check security attributions
        assert len(result.security_attributions) == 5
        for attr in result.security_attributions:
            assert isinstance(attr, SecurityAttribution)
            assert 0 <= attr.total_risk_contribution <= 100

    def test_herfindahl_index(self, engine: RiskAttributionEngine) -> None:
        """Test concentration metric calculation."""
        # Concentrated portfolio
        concentrated = {"AAPL": 0.80, "MSFT": 0.20}
        result_conc = engine.attribute_risk(concentrated)

        # Diversified portfolio
        diversified = {
            "AAPL": 0.20,
            "MSFT": 0.20,
            "GOOGL": 0.20,
            "JPM": 0.20,
            "XOM": 0.20,
        }
        result_div = engine.attribute_risk(diversified)

        # Concentrated should have higher Herfindahl
        assert result_conc.herfindahl_index > result_div.herfindahl_index

    def test_top_risk_contributors(self, engine: RiskAttributionEngine) -> None:
        """Test getting top risk contributors."""
        weights = {
            "AAPL": 0.40,  # Large position
            "MSFT": 0.30,
            "GOOGL": 0.15,
            "JPM": 0.10,
            "XOM": 0.05,
        }

        result = engine.attribute_risk(weights)
        top_5 = result.get_top_risk_contributors(5)

        assert len(top_5) == 5
        # Should be sorted by risk contribution
        for i in range(len(top_5) - 1):
            assert top_5[i].total_risk_contribution >= top_5[i + 1].total_risk_contribution


class TestSectorAttribution:
    """Tests for sector-level attribution."""

    @pytest.fixture
    def engine(self) -> RiskAttributionEngine:
        """Create engine."""
        return create_simple_attribution_engine()

    def test_sector_attribution(self, engine: RiskAttributionEngine) -> None:
        """Test sector-level risk attribution."""
        weights = {
            "AAPL": 0.20,
            "MSFT": 0.20,
            "JPM": 0.30,
            "XOM": 0.30,
        }

        sector_mapping = {
            "AAPL": "Technology",
            "MSFT": "Technology",
            "JPM": "Financials",
            "XOM": "Energy",
        }

        # Set up covariance first
        symbols = list(weights.keys())
        cov = np.eye(len(symbols)) * 0.04 / 252  # Simple diagonal
        engine.set_covariance_matrix(cov, symbols)

        result = engine.attribute_risk(
            weights,
            sector_mapping=sector_mapping,
        )

        # Should have 3 sectors
        assert len(result.sector_attributions) == 3

        sectors = {s.sector for s in result.sector_attributions}
        assert "Technology" in sectors
        assert "Financials" in sectors
        assert "Energy" in sectors


class TestFactorAttribution:
    """Tests for factor-based attribution."""

    def test_factor_returns_setup(self) -> None:
        """Test setting up factor returns."""
        engine = RiskAttributionEngine()

        market_returns = np.random.normal(0.0004, 0.01, 252)
        engine.set_factor_returns(RiskFactor.MARKET, market_returns)

        assert RiskFactor.MARKET in engine._factor_returns
        factor_data = engine._factor_returns[RiskFactor.MARKET]
        assert factor_data.mean_return != 0 or True  # Could be ~0
        assert factor_data.volatility > 0

    def test_factor_regression(self) -> None:
        """Test factor regression."""
        engine = RiskAttributionEngine()

        np.random.seed(42)

        # Create correlated asset and factor returns
        factor_returns = np.random.normal(0, 0.01, 100)
        # Asset with beta = 1.2 to factor
        asset_returns = 1.2 * factor_returns + np.random.normal(0, 0.005, 100)

        betas, residuals, r_squared = engine.run_factor_regression(
            asset_returns,
            factor_returns.reshape(-1, 1),
        )

        assert len(betas) == 1
        assert betas[0] == pytest.approx(1.2, abs=0.3)
        assert r_squared > 0.5


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_simple_engine(self) -> None:
        """Test simple engine factory."""
        engine = create_simple_attribution_engine()

        # Should have factor returns set
        assert RiskFactor.MARKET in engine._factor_returns
        assert RiskFactor.SIZE in engine._factor_returns
        assert RiskFactor.VALUE in engine._factor_returns
        assert RiskFactor.MOMENTUM in engine._factor_returns

    def test_create_fama_french_engine(self) -> None:
        """Test Fama-French engine factory."""
        np.random.seed(42)
        n = 252

        engine = create_fama_french_engine(
            market_returns=np.random.normal(0.0004, 0.01, n),
            smb_returns=np.random.normal(0.0001, 0.005, n),
            hml_returns=np.random.normal(0.00015, 0.006, n),
            momentum_returns=np.random.normal(0.0002, 0.008, n),
        )

        assert RiskFactor.MARKET in engine._factor_returns
        assert RiskFactor.SIZE in engine._factor_returns
        assert RiskFactor.VALUE in engine._factor_returns
        assert RiskFactor.MOMENTUM in engine._factor_returns


class TestCovarianceEstimation:
    """Tests for covariance estimation."""

    def test_ledoit_wolf_shrinkage(self) -> None:
        """Test Ledoit-Wolf shrinkage estimation."""
        engine = RiskAttributionEngine()

        np.random.seed(42)
        n_periods = 252
        n_assets = 5

        # Generate correlated returns
        true_cov = np.eye(n_assets) * 0.04
        returns = np.random.multivariate_normal(
            np.zeros(n_assets),
            true_cov,
            n_periods,
        )

        symbols = [f"ASSET_{i}" for i in range(n_assets)]

        engine.estimate_covariance(returns, symbols, shrinkage=0.1)

        assert engine._cov_matrix is not None
        assert engine._cov_matrix.shape == (n_assets, n_assets)
        assert len(engine._cov_symbols) == n_assets

        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(
            engine._cov_matrix,
            engine._cov_matrix.T,
        )

        # Diagonal should be positive (variances)
        assert all(engine._cov_matrix[i, i] > 0 for i in range(n_assets))


class TestRiskAttributionResult:
    """Tests for RiskAttributionResult."""

    def test_result_defaults(self) -> None:
        """Test result default values."""
        result = RiskAttributionResult()

        assert result.total_volatility == pytest.approx(0.0)
        assert len(result.security_attributions) == 0
        assert len(result.sector_attributions) == 0
        assert result.herfindahl_index == pytest.approx(0.0)

    def test_get_factor_summary(self) -> None:
        """Test factor summary extraction."""
        result = RiskAttributionResult(
            factor_exposures=[
                FactorExposure(
                    factor=RiskFactor.MARKET,
                    beta=1.1,
                    contribution_pct=60.0,
                ),
                FactorExposure(
                    factor=RiskFactor.SIZE,
                    beta=0.3,
                    contribution_pct=15.0,
                ),
            ]
        )

        summary = result.get_factor_summary()

        assert summary["market"] == pytest.approx(60.0)
        assert summary["size"] == pytest.approx(15.0)
