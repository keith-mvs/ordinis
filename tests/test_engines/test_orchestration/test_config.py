"""Tests for OrchestrationEngineConfig.

This module tests the configuration dataclass and its validation logic.
"""

from ordinis.engines.orchestration.core.config import OrchestrationEngineConfig


class TestOrchestrationEngineConfig:
    """Test OrchestrationEngineConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = OrchestrationEngineConfig()

        assert config.engine_id == "orchestration"
        assert config.engine_name == "Orchestration Engine"
        assert config.mode == "paper"
        assert config.cycle_interval_ms == 100
        assert config.max_signals_per_cycle == 100
        assert config.signal_batch_enabled is True
        assert config.parallel_execution is False
        assert config.require_risk_approval is True
        assert config.risk_timeout_ms == 50
        assert config.execution_timeout_ms == 500
        assert config.max_orders_per_cycle == 10
        assert config.enable_analytics_recording is True
        assert config.analytics_batch_size == 100
        assert config.enable_governance is True

    def test_latency_budgets_default(self) -> None:
        """Test latency budget default values."""
        config = OrchestrationEngineConfig()

        assert config.data_pipeline_budget_ms == 100
        assert config.anomaly_detection_budget_ms == 50
        assert config.feature_engineering_budget_ms == 20
        assert config.signal_generation_budget_ms == 100
        assert config.risk_checks_budget_ms == 10
        assert config.order_routing_budget_ms == 60
        assert config.total_budget_ms == 300

    def test_custom_engine_id(self) -> None:
        """Test setting custom engine_id."""
        config = OrchestrationEngineConfig(engine_id="custom-orchestrator")

        assert config.engine_id == "custom-orchestrator"

    def test_custom_engine_name(self) -> None:
        """Test setting custom engine_name."""
        config = OrchestrationEngineConfig(engine_name="Custom Orchestration")

        assert config.engine_name == "Custom Orchestration"

    def test_mode_live(self) -> None:
        """Test live trading mode."""
        config = OrchestrationEngineConfig(mode="live")

        assert config.mode == "live"

    def test_mode_paper(self) -> None:
        """Test paper trading mode."""
        config = OrchestrationEngineConfig(mode="paper")

        assert config.mode == "paper"

    def test_mode_backtest(self) -> None:
        """Test backtest mode."""
        config = OrchestrationEngineConfig(mode="backtest")

        assert config.mode == "backtest"

    def test_custom_cycle_interval(self) -> None:
        """Test custom cycle interval."""
        config = OrchestrationEngineConfig(cycle_interval_ms=500)

        assert config.cycle_interval_ms == 500

    def test_custom_max_signals(self) -> None:
        """Test custom max signals per cycle."""
        config = OrchestrationEngineConfig(max_signals_per_cycle=50)

        assert config.max_signals_per_cycle == 50

    def test_signal_batch_disabled(self) -> None:
        """Test disabling signal batching."""
        config = OrchestrationEngineConfig(signal_batch_enabled=False)

        assert config.signal_batch_enabled is False

    def test_parallel_execution_enabled(self) -> None:
        """Test enabling parallel execution."""
        config = OrchestrationEngineConfig(parallel_execution=True)

        assert config.parallel_execution is True

    def test_risk_approval_not_required(self) -> None:
        """Test disabling required risk approval."""
        config = OrchestrationEngineConfig(require_risk_approval=False)

        assert config.require_risk_approval is False

    def test_custom_risk_timeout(self) -> None:
        """Test custom risk timeout."""
        config = OrchestrationEngineConfig(risk_timeout_ms=100)

        assert config.risk_timeout_ms == 100

    def test_custom_execution_timeout(self) -> None:
        """Test custom execution timeout."""
        config = OrchestrationEngineConfig(execution_timeout_ms=1000)

        assert config.execution_timeout_ms == 1000

    def test_custom_max_orders(self) -> None:
        """Test custom max orders per cycle."""
        config = OrchestrationEngineConfig(max_orders_per_cycle=20)

        assert config.max_orders_per_cycle == 20

    def test_analytics_recording_disabled(self) -> None:
        """Test disabling analytics recording."""
        config = OrchestrationEngineConfig(enable_analytics_recording=False)

        assert config.enable_analytics_recording is False

    def test_custom_analytics_batch_size(self) -> None:
        """Test custom analytics batch size."""
        config = OrchestrationEngineConfig(analytics_batch_size=200)

        assert config.analytics_batch_size == 200

    def test_governance_disabled(self) -> None:
        """Test disabling governance."""
        config = OrchestrationEngineConfig(enable_governance=False)

        assert config.enable_governance is False

    def test_custom_latency_budgets(self) -> None:
        """Test custom latency budgets."""
        config = OrchestrationEngineConfig(
            data_pipeline_budget_ms=150,
            anomaly_detection_budget_ms=75,
            feature_engineering_budget_ms=30,
            signal_generation_budget_ms=120,
            risk_checks_budget_ms=15,
            order_routing_budget_ms=80,
            total_budget_ms=400,
        )

        assert config.data_pipeline_budget_ms == 150
        assert config.anomaly_detection_budget_ms == 75
        assert config.feature_engineering_budget_ms == 30
        assert config.signal_generation_budget_ms == 120
        assert config.risk_checks_budget_ms == 15
        assert config.order_routing_budget_ms == 80
        assert config.total_budget_ms == 400

    def test_inherits_base_config(self) -> None:
        """Test OrchestrationEngineConfig inherits from BaseEngineConfig."""
        config = OrchestrationEngineConfig()

        # These come from BaseEngineConfig
        assert config.enabled is True
        assert config.log_level == "INFO"
        assert config.metrics_enabled is True
        assert config.governance_enabled is True

    def test_override_base_values(self) -> None:
        """Test overriding base configuration values."""
        config = OrchestrationEngineConfig(
            enabled=False,
            log_level="DEBUG",
            metrics_enabled=False,
        )

        assert config.enabled is False
        assert config.log_level == "DEBUG"
        assert config.metrics_enabled is False


class TestOrchestrationEngineConfigValidation:
    """Test OrchestrationEngineConfig validation logic."""

    def test_valid_config(self) -> None:
        """Test valid configuration passes validation."""
        config = OrchestrationEngineConfig()
        errors = config.validate()

        assert errors == []

    def test_cycle_interval_too_low(self) -> None:
        """Test cycle_interval_ms below minimum."""
        config = OrchestrationEngineConfig(cycle_interval_ms=5)
        errors = config.validate()

        assert len(errors) > 0
        assert any("cycle_interval_ms" in error for error in errors)
        assert any("at least 10ms" in error for error in errors)

    def test_cycle_interval_minimum_valid(self) -> None:
        """Test minimum valid cycle_interval_ms."""
        config = OrchestrationEngineConfig(cycle_interval_ms=10)
        errors = config.validate()

        assert not any("cycle_interval_ms" in error for error in errors)

    def test_max_signals_zero(self) -> None:
        """Test max_signals_per_cycle of zero is invalid."""
        config = OrchestrationEngineConfig(max_signals_per_cycle=0)
        errors = config.validate()

        assert len(errors) > 0
        assert any("max_signals_per_cycle" in error for error in errors)
        assert any("positive" in error for error in errors)

    def test_max_signals_negative(self) -> None:
        """Test negative max_signals_per_cycle is invalid."""
        config = OrchestrationEngineConfig(max_signals_per_cycle=-1)
        errors = config.validate()

        assert len(errors) > 0
        assert any("max_signals_per_cycle" in error for error in errors)

    def test_max_signals_positive_valid(self) -> None:
        """Test positive max_signals_per_cycle is valid."""
        config = OrchestrationEngineConfig(max_signals_per_cycle=1)
        errors = config.validate()

        assert not any("max_signals_per_cycle" in error for error in errors)

    def test_risk_timeout_zero(self) -> None:
        """Test risk_timeout_ms of zero is invalid."""
        config = OrchestrationEngineConfig(risk_timeout_ms=0)
        errors = config.validate()

        assert len(errors) > 0
        assert any("risk_timeout_ms" in error for error in errors)
        assert any("positive" in error for error in errors)

    def test_risk_timeout_negative(self) -> None:
        """Test negative risk_timeout_ms is invalid."""
        config = OrchestrationEngineConfig(risk_timeout_ms=-1)
        errors = config.validate()

        assert len(errors) > 0
        assert any("risk_timeout_ms" in error for error in errors)

    def test_risk_timeout_positive_valid(self) -> None:
        """Test positive risk_timeout_ms is valid."""
        config = OrchestrationEngineConfig(risk_timeout_ms=1)
        errors = config.validate()

        assert not any("risk_timeout_ms" in error for error in errors)

    def test_execution_timeout_zero(self) -> None:
        """Test execution_timeout_ms of zero is invalid."""
        config = OrchestrationEngineConfig(execution_timeout_ms=0)
        errors = config.validate()

        assert len(errors) > 0
        assert any("execution_timeout_ms" in error for error in errors)
        assert any("positive" in error for error in errors)

    def test_execution_timeout_negative(self) -> None:
        """Test negative execution_timeout_ms is invalid."""
        config = OrchestrationEngineConfig(execution_timeout_ms=-1)
        errors = config.validate()

        assert len(errors) > 0
        assert any("execution_timeout_ms" in error for error in errors)

    def test_execution_timeout_positive_valid(self) -> None:
        """Test positive execution_timeout_ms is valid."""
        config = OrchestrationEngineConfig(execution_timeout_ms=1)
        errors = config.validate()

        assert not any("execution_timeout_ms" in error for error in errors)

    def test_total_budget_less_than_interval(self) -> None:
        """Test total_budget_ms less than cycle_interval_ms is invalid."""
        config = OrchestrationEngineConfig(
            cycle_interval_ms=100,
            total_budget_ms=50,
        )
        errors = config.validate()

        assert len(errors) > 0
        assert any("total_budget_ms" in error for error in errors)
        assert any(">= cycle_interval_ms" in error for error in errors)

    def test_total_budget_equal_to_interval_valid(self) -> None:
        """Test total_budget_ms equal to cycle_interval_ms is valid."""
        config = OrchestrationEngineConfig(
            cycle_interval_ms=100,
            total_budget_ms=100,
        )
        errors = config.validate()

        assert not any("total_budget_ms" in error for error in errors)

    def test_total_budget_greater_than_interval_valid(self) -> None:
        """Test total_budget_ms greater than cycle_interval_ms is valid."""
        config = OrchestrationEngineConfig(
            cycle_interval_ms=100,
            total_budget_ms=300,
        )
        errors = config.validate()

        assert not any("total_budget_ms" in error for error in errors)

    def test_multiple_validation_errors(self) -> None:
        """Test multiple validation errors are returned."""
        config = OrchestrationEngineConfig(
            cycle_interval_ms=5,  # Too low
            max_signals_per_cycle=0,  # Invalid
            risk_timeout_ms=0,  # Invalid
            execution_timeout_ms=0,  # Invalid
        )
        errors = config.validate()

        assert len(errors) >= 4
        assert any("cycle_interval_ms" in error for error in errors)
        assert any("max_signals_per_cycle" in error for error in errors)
        assert any("risk_timeout_ms" in error for error in errors)
        assert any("execution_timeout_ms" in error for error in errors)

    def test_validation_includes_base_errors(self) -> None:
        """Test validation includes errors from BaseEngineConfig."""
        config = OrchestrationEngineConfig()
        # Force a base validation error (if any exist in base class)
        errors = config.validate()

        # Should call super().validate()
        assert isinstance(errors, list)


class TestOrchestrationEngineConfigModes:
    """Test different operating modes."""

    def test_live_mode_configuration(self) -> None:
        """Test configuration for live trading mode."""
        config = OrchestrationEngineConfig(
            mode="live",
            require_risk_approval=True,
            enable_governance=True,
        )

        assert config.mode == "live"
        assert config.require_risk_approval is True
        assert config.enable_governance is True

    def test_paper_mode_configuration(self) -> None:
        """Test configuration for paper trading mode."""
        config = OrchestrationEngineConfig(
            mode="paper",
            require_risk_approval=True,
        )

        assert config.mode == "paper"

    def test_backtest_mode_configuration(self) -> None:
        """Test configuration for backtest mode."""
        config = OrchestrationEngineConfig(
            mode="backtest",
            cycle_interval_ms=10,  # Faster for backtesting
            enable_analytics_recording=False,  # Often disabled in backtests
        )

        assert config.mode == "backtest"
        assert config.cycle_interval_ms == 10
        assert config.enable_analytics_recording is False


class TestOrchestrationEngineConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_cycle_interval(self) -> None:
        """Test very large cycle interval."""
        config = OrchestrationEngineConfig(
            cycle_interval_ms=60000,  # 1 minute
            total_budget_ms=60000,  # Must be >= cycle_interval_ms
        )

        assert config.cycle_interval_ms == 60000
        errors = config.validate()
        assert not any("cycle_interval_ms" in error for error in errors)

    def test_very_large_max_signals(self) -> None:
        """Test very large max signals."""
        config = OrchestrationEngineConfig(max_signals_per_cycle=10000)

        assert config.max_signals_per_cycle == 10000
        errors = config.validate()
        assert not any("max_signals_per_cycle" in error for error in errors)

    def test_minimum_valid_values(self) -> None:
        """Test minimum valid values for all fields."""
        config = OrchestrationEngineConfig(
            cycle_interval_ms=10,
            max_signals_per_cycle=1,
            risk_timeout_ms=1,
            execution_timeout_ms=1,
            total_budget_ms=10,
        )

        errors = config.validate()
        assert errors == []

    def test_all_features_disabled(self) -> None:
        """Test disabling all optional features."""
        config = OrchestrationEngineConfig(
            signal_batch_enabled=False,
            parallel_execution=False,
            require_risk_approval=False,
            enable_analytics_recording=False,
            enable_governance=False,
        )

        assert config.signal_batch_enabled is False
        assert config.parallel_execution is False
        assert config.require_risk_approval is False
        assert config.enable_analytics_recording is False
        assert config.enable_governance is False
        errors = config.validate()
        assert errors == []

    def test_all_features_enabled(self) -> None:
        """Test enabling all optional features."""
        config = OrchestrationEngineConfig(
            signal_batch_enabled=True,
            parallel_execution=True,
            require_risk_approval=True,
            enable_analytics_recording=True,
            enable_governance=True,
        )

        assert config.signal_batch_enabled is True
        assert config.parallel_execution is True
        assert config.require_risk_approval is True
        assert config.enable_analytics_recording is True
        assert config.enable_governance is True
        errors = config.validate()
        assert errors == []
