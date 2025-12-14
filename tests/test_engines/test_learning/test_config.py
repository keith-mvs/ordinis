"""Tests for LearningEngine configuration.

This module tests the LearningEngineConfig dataclass and its validation logic.
"""

from pathlib import Path

from ordinis.engines.learning.core.config import LearningEngineConfig
from ordinis.engines.learning.core.models import RolloutStrategy


class TestLearningEngineConfigDefaults:
    """Test default configuration values."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LearningEngineConfig()

        assert config.engine_id == "learning"
        assert config.engine_name == "Learning Engine"
        assert config.data_dir is None
        assert config.max_events_memory == 10000
        assert config.flush_interval_seconds == 60.0

    def test_event_collection_defaults(self) -> None:
        """Test default event collection settings."""
        config = LearningEngineConfig()

        assert config.collect_signals is True
        assert config.collect_executions is True
        assert config.collect_portfolio is True
        assert config.collect_risk is True
        assert config.collect_predictions is True

    def test_training_defaults(self) -> None:
        """Test default training settings."""
        config = LearningEngineConfig()

        assert config.min_samples_for_training == 1000
        assert config.training_batch_size == 256
        assert config.max_concurrent_jobs == 2

    def test_evaluation_defaults(self) -> None:
        """Test default evaluation settings."""
        config = LearningEngineConfig()

        assert config.require_eval_pass is True
        assert config.eval_benchmark_threshold == 0.95
        assert config.cross_validation_folds == 5

    def test_drift_detection_defaults(self) -> None:
        """Test default drift detection settings."""
        config = LearningEngineConfig()

        assert config.enable_drift_detection is True
        assert config.drift_check_interval_seconds == 3600.0
        assert config.drift_threshold_pct == 0.10

    def test_rollout_defaults(self) -> None:
        """Test default rollout settings."""
        config = LearningEngineConfig()

        assert config.default_rollout_strategy == RolloutStrategy.CANARY
        assert config.canary_percentage == 0.10
        assert config.shadow_mode_duration_seconds == 86400.0

    def test_retention_defaults(self) -> None:
        """Test default retention settings."""
        config = LearningEngineConfig()

        assert config.event_retention_days == 90
        assert config.model_version_retention == 10

    def test_governance_defaults(self) -> None:
        """Test default governance settings."""
        config = LearningEngineConfig()

        assert config.enable_governance is True


class TestLearningEngineConfigCustomization:
    """Test custom configuration values."""

    def test_custom_engine_id(self) -> None:
        """Test setting custom engine ID."""
        config = LearningEngineConfig(engine_id="custom_learning")

        assert config.engine_id == "custom_learning"

    def test_custom_data_dir(self, temp_data_dir: Path) -> None:
        """Test setting custom data directory."""
        config = LearningEngineConfig(data_dir=temp_data_dir)

        assert config.data_dir == temp_data_dir

    def test_custom_memory_settings(self) -> None:
        """Test custom memory and flush settings."""
        config = LearningEngineConfig(
            max_events_memory=5000,
            flush_interval_seconds=30.0,
        )

        assert config.max_events_memory == 5000
        assert config.flush_interval_seconds == 30.0

    def test_disable_event_collection(self) -> None:
        """Test disabling specific event types."""
        config = LearningEngineConfig(
            collect_signals=False,
            collect_risk=False,
        )

        assert config.collect_signals is False
        assert config.collect_executions is True
        assert config.collect_portfolio is True
        assert config.collect_risk is False
        assert config.collect_predictions is True

    def test_custom_training_settings(self) -> None:
        """Test custom training settings."""
        config = LearningEngineConfig(
            min_samples_for_training=500,
            training_batch_size=128,
            max_concurrent_jobs=4,
        )

        assert config.min_samples_for_training == 500
        assert config.training_batch_size == 128
        assert config.max_concurrent_jobs == 4

    def test_custom_evaluation_settings(self) -> None:
        """Test custom evaluation settings."""
        config = LearningEngineConfig(
            require_eval_pass=False,
            eval_benchmark_threshold=0.90,
            cross_validation_folds=3,
        )

        assert config.require_eval_pass is False
        assert config.eval_benchmark_threshold == 0.90
        assert config.cross_validation_folds == 3

    def test_custom_drift_settings(self) -> None:
        """Test custom drift detection settings."""
        config = LearningEngineConfig(
            enable_drift_detection=False,
            drift_check_interval_seconds=1800.0,
            drift_threshold_pct=0.15,
        )

        assert config.enable_drift_detection is False
        assert config.drift_check_interval_seconds == 1800.0
        assert config.drift_threshold_pct == 0.15

    def test_custom_rollout_settings(self) -> None:
        """Test custom rollout settings."""
        config = LearningEngineConfig(
            default_rollout_strategy=RolloutStrategy.BLUE_GREEN,
            canary_percentage=0.05,
            shadow_mode_duration_seconds=43200.0,
        )

        assert config.default_rollout_strategy == RolloutStrategy.BLUE_GREEN
        assert config.canary_percentage == 0.05
        assert config.shadow_mode_duration_seconds == 43200.0

    def test_custom_retention_settings(self) -> None:
        """Test custom retention settings."""
        config = LearningEngineConfig(
            event_retention_days=30,
            model_version_retention=5,
        )

        assert config.event_retention_days == 30
        assert config.model_version_retention == 5

    def test_disable_governance(self) -> None:
        """Test disabling governance."""
        config = LearningEngineConfig(enable_governance=False)

        assert config.enable_governance is False


class TestLearningEngineConfigValidation:
    """Test configuration validation logic."""

    def test_validate_success(self) -> None:
        """Test validation passes with valid config."""
        config = LearningEngineConfig()
        errors = config.validate()

        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_validate_max_events_memory_too_low(self) -> None:
        """Test validation fails with too low max_events_memory."""
        config = LearningEngineConfig(max_events_memory=50)
        errors = config.validate()

        assert len(errors) > 0
        assert any("max_events_memory" in error for error in errors)

    def test_validate_flush_interval_too_low(self) -> None:
        """Test validation fails with too low flush_interval."""
        config = LearningEngineConfig(flush_interval_seconds=0.5)
        errors = config.validate()

        assert len(errors) > 0
        assert any("flush_interval_seconds" in error for error in errors)

    def test_validate_min_samples_too_low(self) -> None:
        """Test validation fails with too few min_samples."""
        config = LearningEngineConfig(min_samples_for_training=5)
        errors = config.validate()

        assert len(errors) > 0
        assert any("min_samples_for_training" in error for error in errors)

    def test_validate_batch_size_invalid(self) -> None:
        """Test validation fails with invalid batch_size."""
        config = LearningEngineConfig(training_batch_size=0)
        errors = config.validate()

        assert len(errors) > 0
        assert any("training_batch_size" in error for error in errors)

    def test_validate_eval_threshold_too_low(self) -> None:
        """Test validation fails with eval_threshold <= 0."""
        config = LearningEngineConfig(eval_benchmark_threshold=0.0)
        errors = config.validate()

        assert len(errors) > 0
        assert any("eval_benchmark_threshold" in error for error in errors)

    def test_validate_eval_threshold_too_high(self) -> None:
        """Test validation fails with eval_threshold > 2."""
        config = LearningEngineConfig(eval_benchmark_threshold=2.5)
        errors = config.validate()

        assert len(errors) > 0
        assert any("eval_benchmark_threshold" in error for error in errors)

    def test_validate_canary_percentage_too_low(self) -> None:
        """Test validation fails with canary_percentage <= 0."""
        config = LearningEngineConfig(canary_percentage=0.0)
        errors = config.validate()

        assert len(errors) > 0
        assert any("canary_percentage" in error for error in errors)

    def test_validate_canary_percentage_too_high(self) -> None:
        """Test validation fails with canary_percentage > 1."""
        config = LearningEngineConfig(canary_percentage=1.5)
        errors = config.validate()

        assert len(errors) > 0
        assert any("canary_percentage" in error for error in errors)

    def test_validate_drift_threshold_negative(self) -> None:
        """Test validation fails with negative drift_threshold."""
        config = LearningEngineConfig(drift_threshold_pct=-0.1)
        errors = config.validate()

        assert len(errors) > 0
        assert any("drift_threshold_pct" in error for error in errors)

    def test_validate_retention_days_too_low(self) -> None:
        """Test validation fails with retention_days < 1."""
        config = LearningEngineConfig(event_retention_days=0)
        errors = config.validate()

        assert len(errors) > 0
        assert any("event_retention_days" in error for error in errors)

    def test_validate_multiple_errors(self) -> None:
        """Test validation accumulates multiple errors."""
        config = LearningEngineConfig(
            max_events_memory=50,
            training_batch_size=0,
            drift_threshold_pct=-0.1,
        )
        errors = config.validate()

        assert len(errors) >= 3

    def test_validate_edge_case_valid_values(self) -> None:
        """Test validation passes with edge case valid values."""
        config = LearningEngineConfig(
            max_events_memory=100,  # Minimum valid
            flush_interval_seconds=1.0,  # Minimum valid
            min_samples_for_training=10,  # Minimum valid
            training_batch_size=1,  # Minimum valid
            eval_benchmark_threshold=0.01,  # Just above zero
            canary_percentage=0.01,  # Just above zero
            drift_threshold_pct=0.0,  # Minimum valid
            event_retention_days=1,  # Minimum valid
        )
        errors = config.validate()

        assert isinstance(errors, list)
        assert len(errors) == 0


class TestLearningEngineConfigInheritance:
    """Test configuration inheritance from BaseEngineConfig."""

    def test_inherits_base_config(self) -> None:
        """Test LearningEngineConfig inherits from BaseEngineConfig."""
        config = LearningEngineConfig()

        # Base config properties
        assert hasattr(config, "enabled")
        assert hasattr(config, "name")
        assert hasattr(config, "log_level")
        assert hasattr(config, "metrics_enabled")
        assert hasattr(config, "governance_enabled")

    def test_base_config_defaults(self) -> None:
        """Test base config default values."""
        config = LearningEngineConfig()

        assert config.enabled is True
        assert config.log_level == "INFO"
        assert config.metrics_enabled is True

    def test_override_base_config(self) -> None:
        """Test overriding base config values."""
        config = LearningEngineConfig(
            enabled=False,
            name="CustomLearning",
            log_level="DEBUG",
            metrics_enabled=False,
        )

        assert config.enabled is False
        assert config.name == "CustomLearning"
        assert config.log_level == "DEBUG"
        assert config.metrics_enabled is False

    def test_base_config_to_dict(self) -> None:
        """Test to_dict includes base config fields."""
        config = LearningEngineConfig()
        config_dict = config.to_dict()

        # Base fields
        assert "enabled" in config_dict
        assert "name" in config_dict
        assert "log_level" in config_dict

        # Note: to_dict() in BaseEngineConfig only includes base fields
        # Subclasses don't override to_dict, so learning-specific fields won't be present
        # This is testing that the base method works correctly


class TestLearningEngineConfigEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_concurrent_jobs(self) -> None:
        """Test with zero concurrent jobs."""
        config = LearningEngineConfig(max_concurrent_jobs=0)

        # Should be valid (though impractical)
        errors = config.validate()
        # No validation rule against zero concurrent jobs
        assert config.max_concurrent_jobs == 0
        assert isinstance(errors, list)

    def test_very_large_memory_buffer(self) -> None:
        """Test with very large memory buffer."""
        config = LearningEngineConfig(max_events_memory=1000000)

        assert config.max_events_memory == 1000000
        errors = config.validate()
        assert isinstance(errors, list)
        assert len(errors) == 0

    def test_all_event_types_disabled(self) -> None:
        """Test with all event types disabled."""
        config = LearningEngineConfig(
            collect_signals=False,
            collect_executions=False,
            collect_portfolio=False,
            collect_risk=False,
            collect_predictions=False,
        )

        # Should be valid (though not useful)
        assert config.collect_signals is False
        assert config.collect_executions is False
        assert config.collect_portfolio is False
        assert config.collect_risk is False
        assert config.collect_predictions is False

    def test_all_rollout_strategies(self) -> None:
        """Test setting each rollout strategy."""
        for strategy in RolloutStrategy:
            config = LearningEngineConfig(default_rollout_strategy=strategy)
            assert config.default_rollout_strategy == strategy

    def test_path_handling(self, temp_data_dir: Path) -> None:
        """Test path configuration and handling."""
        config = LearningEngineConfig(data_dir=temp_data_dir)

        assert isinstance(config.data_dir, Path)
        assert config.data_dir == temp_data_dir

    def test_none_data_dir(self) -> None:
        """Test with None data_dir."""
        config = LearningEngineConfig(data_dir=None)

        assert config.data_dir is None
