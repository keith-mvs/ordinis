"""
Futures trading risk rules.

Specific rules for futures contracts including margin and expiry checks.
"""

from ..core.rules import RiskRule, RuleCategory

FUTURES_RISK_RULES = {
    # ==========================================================================
    # FUTURES SPECIFIC RULES
    # ==========================================================================
    "max_leverage": RiskRule(
        rule_id="FUT001",
        category=RuleCategory.PRE_TRADE,
        name="Max Account Leverage",
        description="Maximum total account leverage allowed",
        condition="total_exposure / portfolio_equity <= threshold",
        threshold=5.0,  # 5x max leverage
        comparison="<=",
        action_on_breach="reject",
        severity="critical",
    ),
    "margin_check": RiskRule(
        rule_id="FUT002",
        category=RuleCategory.PRE_TRADE,
        name="Initial Margin Check",
        description="Ensure sufficient equity for initial margin",
        condition="available_margin >= required_margin",
        threshold=0.0,  # Must be positive
        comparison=">=",
        action_on_breach="reject",
        severity="critical",
    ),
    "expiry_proximity": RiskRule(
        rule_id="FUT003",
        category=RuleCategory.PRE_TRADE,
        name="Expiry Proximity Check",
        description="Prevent opening positions too close to expiry",
        condition="days_to_expiry > threshold",
        threshold=2,  # Don't trade if < 2 days to expiry
        comparison=">",
        action_on_breach="reject",
        severity="high",
    ),
}
