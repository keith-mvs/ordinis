"""
Standard risk rule set for RiskGuard.

Based on industry best practices and regulatory requirements.
"""

from ..core.rules import RiskRule, RuleCategory

# Standard risk rules as defined in system architecture
STANDARD_RISK_RULES = {
    # Per-trade limits
    "max_position_pct": RiskRule(
        rule_id="RT001",
        category=RuleCategory.PRE_TRADE,
        name="Max Position Size",
        description="Maximum position as percentage of equity",
        condition="position_value / portfolio_equity <= threshold",
        threshold=0.10,  # 10% max position size
        comparison="<=",
        action_on_breach="resize",
        severity="high",
    ),
    "max_risk_per_trade": RiskRule(
        rule_id="RT002",
        category=RuleCategory.PRE_TRADE,
        name="Max Risk Per Trade",
        description="Maximum capital at risk per trade",
        condition="(entry_price - stop_price) * quantity / portfolio_equity <= threshold",
        threshold=0.01,  # 1% max risk per trade
        comparison="<=",
        action_on_breach="resize",
        severity="high",
    ),
    # Portfolio limits
    "max_positions": RiskRule(
        rule_id="RP001",
        category=RuleCategory.PORTFOLIO_LIMIT,
        name="Max Open Positions",
        description="Maximum number of concurrent positions",
        condition="count(open_positions) <= threshold",
        threshold=10.0,  # Max 10 positions
        comparison="<=",
        action_on_breach="reject",
        severity="medium",
    ),
    "max_sector_concentration": RiskRule(
        rule_id="RP002",
        category=RuleCategory.PORTFOLIO_LIMIT,
        name="Max Sector Concentration",
        description="Maximum exposure to single sector",
        condition="sector_exposure / portfolio_equity <= threshold",
        threshold=0.30,  # 30% max sector exposure
        comparison="<=",
        action_on_breach="reject",
        severity="high",
    ),
    "max_correlation_exposure": RiskRule(
        rule_id="RP003",
        category=RuleCategory.PORTFOLIO_LIMIT,
        name="Max Correlated Exposure",
        description="Maximum exposure to highly correlated assets (r > 0.7)",
        condition="correlated_exposure / portfolio_equity <= threshold",
        threshold=0.40,  # 40% max correlated exposure
        comparison="<=",
        action_on_breach="warn",
        severity="medium",
    ),
    # Kill switches
    "daily_loss_limit": RiskRule(
        rule_id="RK001",
        category=RuleCategory.KILL_SWITCH,
        name="Daily Loss Limit",
        description="Maximum daily loss before halt",
        condition="daily_pnl / portfolio_equity >= threshold",
        threshold=-0.03,  # -3% daily loss limit
        comparison=">=",
        action_on_breach="halt",
        severity="critical",
    ),
    "max_drawdown": RiskRule(
        rule_id="RK002",
        category=RuleCategory.KILL_SWITCH,
        name="Max Drawdown",
        description="Maximum drawdown from peak before halt",
        condition="(equity - peak_equity) / peak_equity >= threshold",
        threshold=-0.15,  # -15% max drawdown
        comparison=">=",
        action_on_breach="halt",
        severity="critical",
    ),
    # Sanity checks
    "price_deviation": RiskRule(
        rule_id="RS001",
        category=RuleCategory.SANITY_CHECK,
        name="Price Deviation Check",
        description="Order price deviation from last trade",
        condition="abs(order_price - last_price) / last_price <= threshold",
        threshold=0.05,  # 5% max price deviation
        comparison="<=",
        action_on_breach="reject",
        severity="high",
    ),
    "liquidity_check": RiskRule(
        rule_id="RS002",
        category=RuleCategory.SANITY_CHECK,
        name="Liquidity Check",
        description="Order size relative to average daily volume",
        condition="order_quantity / avg_daily_volume <= threshold",
        threshold=0.01,  # 1% of ADV max
        comparison="<=",
        action_on_breach="resize",
        severity="medium",
    ),
}


def create_conservative_ruleset() -> dict[str, RiskRule]:
    """
    Create conservative risk rule set.

    Tighter limits for risk-averse trading.
    """
    conservative = STANDARD_RISK_RULES.copy()

    # Reduce limits
    conservative["max_position_pct"].threshold = 0.05  # 5% max position
    conservative["max_risk_per_trade"].threshold = 0.005  # 0.5% max risk
    conservative["max_positions"].threshold = 5.0  # 5 max positions
    conservative["daily_loss_limit"].threshold = -0.02  # -2% daily loss
    conservative["max_drawdown"].threshold = -0.10  # -10% max drawdown

    return conservative


def create_aggressive_ruleset() -> dict[str, RiskRule]:
    """
    Create aggressive risk rule set.

    Looser limits for higher risk tolerance.
    """
    aggressive = STANDARD_RISK_RULES.copy()

    # Increase limits
    aggressive["max_position_pct"].threshold = 0.20  # 20% max position
    aggressive["max_risk_per_trade"].threshold = 0.02  # 2% max risk
    aggressive["max_positions"].threshold = 20.0  # 20 max positions
    aggressive["daily_loss_limit"].threshold = -0.05  # -5% daily loss
    aggressive["max_drawdown"].threshold = -0.25  # -25% max drawdown

    return aggressive
