"""
SignalCore engine configuration.

Provides SignalCoreEngineConfig extending BaseEngineConfig for standardized
engine configuration with signal generation-specific settings.
"""

from dataclasses import dataclass, field

from ordinis.engines.base import BaseEngineConfig


@dataclass
class SignalCoreEngineConfig(BaseEngineConfig):
    """Configuration for SignalCore signal generation engine.

    Extends BaseEngineConfig with signal-specific settings including
    model management, generation thresholds, and governance controls.

    Attributes:
        engine_id: Unique identifier (default: "signalcore")
        engine_name: Display name (default: "SignalCore Engine")
        min_probability: Minimum signal probability threshold
        min_score: Minimum absolute score threshold
        enable_batch_generation: Allow batch signal generation
        max_batch_size: Maximum symbols per batch
        default_lookback: Default lookback period in trading days
        enable_governance: Enable governance hooks for all operations
    """

    engine_id: str = "signalcore"
    engine_name: str = "SignalCore Engine"

    # Signal thresholds
    min_probability: float = 0.6
    min_score: float = 0.3

    # Batch settings
    enable_batch_generation: bool = True
    max_batch_size: int = 100

    # Data settings
    default_lookback: int = 252  # Trading days
    min_data_points: int = 100

    # Governance
    enable_governance: bool = True

    # Ensemble settings
    enable_ensemble: bool = False
    ensemble_strategy: str = "voting"  # voting, weighted_average, highest_confidence

    # Confidence filter settings (per optimization findings)
    # Key Finding: 80%+ confidence has 51.3% win rate vs 44.7% baseline
    enable_confidence_filter: bool = True
    min_confidence: float = 0.80
    min_agreeing_models: int = 4
    apply_volatility_adjustment: bool = True

    # Model tracking
    registered_models: list[str] = field(default_factory=list)

    def validate(self) -> list[str]:
        """Validate signalcore engine configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = super().validate()

        if not 0 <= self.min_probability <= 1:
            errors.append("min_probability must be between 0 and 1")

        if not 0 <= self.min_score <= 1:
            errors.append("min_score must be between 0 and 1")

        if self.max_batch_size < 1:
            errors.append("max_batch_size must be at least 1")

        if self.min_data_points < 1:
            errors.append("min_data_points must be at least 1")

        return errors
