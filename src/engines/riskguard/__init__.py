"""
RiskGuard rule-based risk management engine.

Provides deterministic, auditable risk constraints for trading decisions.
"""

from .core.engine import RiskGuardEngine
from .core.rules import RiskCheckResult, RiskRule, RuleCategory
from .rules.standard import STANDARD_RISK_RULES

__all__ = [
    "RiskGuardEngine",
    "RiskRule",
    "RuleCategory",
    "RiskCheckResult",
    "STANDARD_RISK_RULES",
]
