"""ProofBench validation module.

Provides unified strategy validation harness for GTM strategies.
"""

from .strategy_validation import (
    AcceptanceCriteria,
    BootstrapResult,
    CostAnalysis,
    StressTestResult,
    StrategyValidationResult,
    StrategyValidator,
    ValidationStatus,
    WalkForwardPeriod,
    WalkForwardSummary,
    create_default_validator,
)

__all__ = [
    "AcceptanceCriteria",
    "BootstrapResult",
    "CostAnalysis",
    "StressTestResult",
    "StrategyValidationResult",
    "StrategyValidator",
    "ValidationStatus",
    "WalkForwardPeriod",
    "WalkForwardSummary",
    "create_default_validator",
]
