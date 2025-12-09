"""
RiskGuard rule-based risk management engine.

Provides deterministic, auditable risk constraints for trading decisions.
"""

from .core.engine import PortfolioState, Position, ProposedTrade, RiskGuardEngine
from .core.llm_enhanced import LLMEnhancedRiskGuard, LLMRiskAnalyzer
from .core.rules import RiskCheckResult, RiskRule, RuleCategory
from .rules.standard import (
    STANDARD_RISK_RULES,
    create_aggressive_ruleset,
    create_conservative_ruleset,
    create_day_trading_ruleset,
    create_swing_trading_ruleset,
)

__all__ = [
    # Core engine
    "RiskGuardEngine",
    "PortfolioState",
    "Position",
    "ProposedTrade",
    # Rules framework
    "RiskRule",
    "RuleCategory",
    "RiskCheckResult",
    # Standard rule sets
    "STANDARD_RISK_RULES",
    "create_conservative_ruleset",
    "create_aggressive_ruleset",
    "create_day_trading_ruleset",
    "create_swing_trading_ruleset",
    # LLM enhanced
    "LLMEnhancedRiskGuard",
    "LLMRiskAnalyzer",
]
