"""
Learning Engine - Continual Model Improvement.

Captures trading events, trains models, evaluates against benchmarks,
and manages model lifecycle with drift detection and controlled rollout.

Key Components:
- LearningEngine: Main engine for event capture and model management
- LearningEngineConfig: Configuration with training and evaluation settings
- LearningGovernanceHook: Validation for training and promotion operations

Example:
    >>> from ordinis.engines.learning import (
    ...     LearningEngine,
    ...     LearningEngineConfig,
    ...     LearningEvent,
    ...     EventType,
    ... )
    >>> config = LearningEngineConfig()
    >>> engine = LearningEngine(config)
    >>> await engine.initialize()
    >>> engine.record_event(LearningEvent(
    ...     event_type=EventType.SIGNAL_GENERATED,
    ...     source_engine="signalcore",
    ...     payload={"probability": 0.75}
    ... ))
"""

from .collectors import (
    CircuitBreakerMonitor,
    CircuitBreakerState,
    ErrorWindow,
    FeedbackCollector,
    FeedbackPriority,
    FeedbackRecord,
    FeedbackType,
)
from .core import (
    DriftAlert,
    EvaluationResult,
    EventType,
    LearningEngine,
    LearningEngineConfig,
    LearningEvent,
    ModelStage,
    ModelVersion,
    RolloutStrategy,
    TrainingJob,
    TrainingStatus,
)
from .hooks import (
    LearningGovernanceHook,
    PromotionValidationRule,
    TrainingValidationRule,
)

__all__ = [
    # Core
    "DriftAlert",
    "EvaluationResult",
    "EventType",
    "LearningEngine",
    "LearningEngineConfig",
    "LearningEvent",
    # Collectors
    "CircuitBreakerMonitor",
    "CircuitBreakerState",
    "ErrorWindow",
    "FeedbackCollector",
    "FeedbackPriority",
    "FeedbackRecord",
    "FeedbackType",
    # Hooks
    "LearningGovernanceHook",
    "ModelStage",
    "ModelVersion",
    "PromotionValidationRule",
    "RolloutStrategy",
    "TrainingJob",
    "TrainingStatus",
    "TrainingValidationRule",
]
