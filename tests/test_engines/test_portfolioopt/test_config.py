"""Tests for PortfolioOptEngineConfig.

This module tests the configuration dataclass and validation logic.
"""

from pathlib import Path

from ordinis.engines.portfolioopt import PortfolioOptEngineConfig


class TestPortfolioOptEngineConfig:
    """Test PortfolioOptEngineConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = PortfolioOptEngineConfig()

        assert config.engine_id == "portfolioopt"
        assert config.engine_name == "PortfolioOpt Engine"
        assert config.qpo_src is None
        assert config.default_api == "cvxpy"
        assert config.target_return == 0.001
        assert config.max_weight == 0.20
        assert config.risk_aversion == 0.5
        assert config.n_paths == 1000
        assert config.simulation_method == "log_gbm"
        assert config.enable_governance is True
        assert config.require_preflight is True

    def test_default_constraints(self) -> None:
        """Test default constraint values."""
        config = PortfolioOptEngineConfig()

        assert config.min_weight == 0.0
        assert config.max_concentration == 0.25
        assert config.min_diversification == 5
        assert config.max_cvar == 0.10
        assert config.max_volatility == 0.25

    def test_default_solver_settings(self) -> None:
        """Test default solver settings."""
        config = PortfolioOptEngineConfig()

        assert config.solver_timeout == 60.0
        assert config.solver_verbose is False

    def test_custom_engine_id(self) -> None:
        """Test custom engine ID."""
        config = PortfolioOptEngineConfig(engine_id="custom-opt")

        assert config.engine_id == "custom-opt"

    def test_custom_qpo_src(self) -> None:
        """Test custom QPO source path."""
        custom_path = Path("/custom/qpo/path")
        config = PortfolioOptEngineConfig(qpo_src=custom_path)

        assert config.qpo_src == custom_path

    def test_custom_solver_api_cvxpy(self) -> None:
        """Test setting solver to cvxpy."""
        config = PortfolioOptEngineConfig(default_api="cvxpy")

        assert config.default_api == "cvxpy"

    def test_custom_solver_api_cuopt(self) -> None:
        """Test setting solver to cuopt."""
        config = PortfolioOptEngineConfig(default_api="cuopt")

        assert config.default_api == "cuopt"

    def test_custom_optimization_params(self) -> None:
        """Test custom optimization parameters."""
        config = PortfolioOptEngineConfig(
            target_return=0.002,
            max_weight=0.15,
            risk_aversion=1.0,
        )

        assert config.target_return == 0.002
        assert config.max_weight == 0.15
        assert config.risk_aversion == 1.0

    def test_custom_simulation_params(self) -> None:
        """Test custom simulation parameters."""
        config = PortfolioOptEngineConfig(
            n_paths=5000,
            simulation_method="geometric_brownian",
        )

        assert config.n_paths == 5000
        assert config.simulation_method == "geometric_brownian"

    def test_custom_risk_constraints(self) -> None:
        """Test custom risk constraints."""
        config = PortfolioOptEngineConfig(
            max_concentration=0.30,
            min_diversification=10,
            max_cvar=0.05,
            max_volatility=0.20,
        )

        assert config.max_concentration == 0.30
        assert config.min_diversification == 10
        assert config.max_cvar == 0.05
        assert config.max_volatility == 0.20

    def test_disable_governance(self) -> None:
        """Test disabling governance."""
        config = PortfolioOptEngineConfig(
            enable_governance=False,
            require_preflight=False,
        )

        assert config.enable_governance is False
        assert config.require_preflight is False

    def test_solver_settings(self) -> None:
        """Test custom solver settings."""
        config = PortfolioOptEngineConfig(
            solver_timeout=120.0,
            solver_verbose=True,
        )

        assert config.solver_timeout == 120.0
        assert config.solver_verbose is True


class TestConfigValidation:
    """Test configuration validation logic."""

    def test_valid_config(self) -> None:
        """Test validation of valid configuration."""
        config = PortfolioOptEngineConfig(
            target_return=0.001,
            max_weight=0.20,
            risk_aversion=0.5,
            n_paths=1000,
            max_concentration=0.25,
            min_diversification=5,
            max_cvar=0.10,
            solver_timeout=60.0,
        )

        errors = config.validate()
        assert len(errors) == 0

    def test_negative_target_return_invalid(self) -> None:
        """Test negative target return is invalid."""
        config = PortfolioOptEngineConfig(target_return=-0.001)

        errors = config.validate()
        assert any("target_return must be non-negative" in e for e in errors)

    def test_max_weight_zero_invalid(self) -> None:
        """Test max_weight of zero is invalid."""
        config = PortfolioOptEngineConfig(max_weight=0.0)

        errors = config.validate()
        assert any("max_weight must be between 0 and 1" in e for e in errors)

    def test_max_weight_above_one_invalid(self) -> None:
        """Test max_weight above 1 is invalid."""
        config = PortfolioOptEngineConfig(max_weight=1.5)

        errors = config.validate()
        assert any("max_weight must be between 0 and 1" in e for e in errors)

    def test_negative_risk_aversion_invalid(self) -> None:
        """Test negative risk aversion is invalid."""
        config = PortfolioOptEngineConfig(risk_aversion=-0.5)

        errors = config.validate()
        assert any("risk_aversion must be non-negative" in e for e in errors)

    def test_low_n_paths_invalid(self) -> None:
        """Test n_paths below 100 is invalid."""
        config = PortfolioOptEngineConfig(n_paths=50)

        errors = config.validate()
        assert any("n_paths should be at least 100" in e for e in errors)

    def test_max_concentration_zero_invalid(self) -> None:
        """Test max_concentration of zero is invalid."""
        config = PortfolioOptEngineConfig(max_concentration=0.0)

        errors = config.validate()
        assert any("max_concentration must be between 0 and 1" in e for e in errors)

    def test_max_concentration_above_one_invalid(self) -> None:
        """Test max_concentration above 1 is invalid."""
        config = PortfolioOptEngineConfig(max_concentration=1.2)

        errors = config.validate()
        assert any("max_concentration must be between 0 and 1" in e for e in errors)

    def test_min_diversification_zero_invalid(self) -> None:
        """Test min_diversification of zero is invalid."""
        config = PortfolioOptEngineConfig(min_diversification=0)

        errors = config.validate()
        assert any("min_diversification must be at least 1" in e for e in errors)

    def test_max_cvar_zero_invalid(self) -> None:
        """Test max_cvar of zero is invalid."""
        config = PortfolioOptEngineConfig(max_cvar=0.0)

        errors = config.validate()
        assert any("max_cvar must be positive" in e for e in errors)

    def test_negative_max_cvar_invalid(self) -> None:
        """Test negative max_cvar is invalid."""
        config = PortfolioOptEngineConfig(max_cvar=-0.05)

        errors = config.validate()
        assert any("max_cvar must be positive" in e for e in errors)

    def test_solver_timeout_zero_invalid(self) -> None:
        """Test solver_timeout of zero is invalid."""
        config = PortfolioOptEngineConfig(solver_timeout=0.0)

        errors = config.validate()
        assert any("solver_timeout must be positive" in e for e in errors)

    def test_negative_solver_timeout_invalid(self) -> None:
        """Test negative solver_timeout is invalid."""
        config = PortfolioOptEngineConfig(solver_timeout=-10.0)

        errors = config.validate()
        assert any("solver_timeout must be positive" in e for e in errors)

    def test_multiple_validation_errors(self) -> None:
        """Test multiple validation errors are reported."""
        config = PortfolioOptEngineConfig(
            target_return=-0.001,
            max_weight=1.5,
            risk_aversion=-0.5,
            n_paths=50,
            max_concentration=0.0,
            min_diversification=0,
            max_cvar=0.0,
            solver_timeout=0.0,
        )

        errors = config.validate()
        assert len(errors) >= 8  # All validation rules should fail

    def test_edge_case_max_weight_one(self) -> None:
        """Test max_weight of exactly 1.0 is valid."""
        config = PortfolioOptEngineConfig(max_weight=1.0)

        errors = config.validate()
        assert not any("max_weight" in e for e in errors)

    def test_edge_case_n_paths_exactly_100(self) -> None:
        """Test n_paths of exactly 100 is valid."""
        config = PortfolioOptEngineConfig(n_paths=100)

        errors = config.validate()
        assert not any("n_paths" in e for e in errors)

    def test_edge_case_zero_target_return(self) -> None:
        """Test target_return of zero is valid."""
        config = PortfolioOptEngineConfig(target_return=0.0)

        errors = config.validate()
        assert not any("target_return" in e for e in errors)

    def test_edge_case_zero_risk_aversion(self) -> None:
        """Test risk_aversion of zero is valid."""
        config = PortfolioOptEngineConfig(risk_aversion=0.0)

        errors = config.validate()
        assert not any("risk_aversion" in e for e in errors)


class TestConfigInheritance:
    """Test configuration inheritance from BaseEngineConfig."""

    def test_inherits_base_config(self) -> None:
        """Test PortfolioOptEngineConfig inherits from BaseEngineConfig."""
        from ordinis.engines.base import BaseEngineConfig

        config = PortfolioOptEngineConfig()

        assert isinstance(config, BaseEngineConfig)

    def test_base_config_attributes(self) -> None:
        """Test inherited base config attributes."""
        config = PortfolioOptEngineConfig(
            engine_name="TestPortfolioOpt",
            enabled=True,
            log_level="DEBUG",
        )

        assert config.enabled is True
        assert config.engine_name == "TestPortfolioOpt"
        assert config.log_level == "DEBUG"

    def test_override_base_attributes(self) -> None:
        """Test overriding base config attributes."""
        config = PortfolioOptEngineConfig(
            enabled=False,
            metrics_enabled=False,
            enable_governance=False,
        )

        assert config.enabled is False
        assert config.metrics_enabled is False
        assert config.enable_governance is False
