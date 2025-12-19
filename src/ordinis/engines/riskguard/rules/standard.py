"""
Standard risk rule set for RiskGuard.

Based on industry best practices and regulatory requirements.
"""

from ..core.rules import RiskRule, RuleCategory

# Standard risk rules as defined in system architecture
STANDARD_RISK_RULES = {
    # ==========================================================================
    # PRE-TRADE RULES (RT) - Evaluated before order generation
    # ==========================================================================
    "max_position_pct": RiskRule(
        rule_id="RT001",
        category=RuleCategory.PRE_TRADE,
        name="Max Position Size",
        description="Maximum position as percentage of equity",
        condition="position_value / portfolio_equity <= threshold",
        threshold=0.20,  # 20% max position size
        comparison="<=",
        action_on_breach="resize",
        severity="high",
    ),
    "max_risk_per_trade": RiskRule(
        rule_id="RT002",
        category=RuleCategory.PRE_TRADE,
        name="Max Risk Per Trade",
        description="Maximum capital at risk per trade (entry to stop)",
        condition="risk_per_trade / portfolio_equity <= threshold",
        threshold=0.02,  # 2% max risk per trade
        comparison="<=",
        action_on_breach="resize",
        severity="high",
    ),
    "min_position_size": RiskRule(
        rule_id="RT003",
        category=RuleCategory.PRE_TRADE,
        name="Min Position Size",
        description="Minimum position value to avoid excessive fragmentation",
        condition="position_value >= threshold",
        threshold=1000.0,  # $1,000 minimum position
        comparison=">=",
        action_on_breach="reject",
        severity="low",
    ),
    "max_single_stock_pct": RiskRule(
        rule_id="RT004",
        category=RuleCategory.PRE_TRADE,
        name="Max Single Stock Exposure",
        description="Maximum total exposure to any single stock",
        condition="total_symbol_exposure / portfolio_equity <= threshold",
        threshold=0.15,  # 15% max single stock
        comparison="<=",
        action_on_breach="resize",
        severity="high",
    ),
    # ==========================================================================
    # POSITION LIMIT RULES (PL) - Per-position constraints
    # ==========================================================================
    "max_leverage_per_position": RiskRule(
        rule_id="PL001",
        category=RuleCategory.POSITION_LIMIT,
        name="Max Leverage Per Position",
        description="Maximum leverage for any single position",
        condition="position_leverage <= threshold",
        threshold=2.0,  # 2x max leverage
        comparison="<=",
        action_on_breach="resize",
        severity="high",
    ),
    "max_notional_value": RiskRule(
        rule_id="PL002",
        category=RuleCategory.POSITION_LIMIT,
        name="Max Notional Value",
        description="Maximum notional value for any single position",
        condition="position_notional <= threshold",
        threshold=50000.0,  # $50,000 max notional
        comparison="<=",
        action_on_breach="resize",
        severity="medium",
    ),
    # ==========================================================================
    # PORTFOLIO LIMIT RULES (RP) - Portfolio-wide constraints
    # ==========================================================================
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
    "max_total_exposure": RiskRule(
        rule_id="RP004",
        category=RuleCategory.PORTFOLIO_LIMIT,
        name="Max Total Exposure",
        description="Maximum total portfolio exposure (gross)",
        condition="total_exposure / portfolio_equity <= threshold",
        threshold=1.0,  # 100% max (no leverage)
        comparison="<=",
        action_on_breach="reject",
        severity="high",
    ),
    "max_long_exposure": RiskRule(
        rule_id="RP005",
        category=RuleCategory.PORTFOLIO_LIMIT,
        name="Max Long Exposure",
        description="Maximum total long exposure",
        condition="long_exposure / portfolio_equity <= threshold",
        threshold=1.0,  # 100% max long
        comparison="<=",
        action_on_breach="reject",
        severity="high",
    ),
    "max_short_exposure": RiskRule(
        rule_id="RP006",
        category=RuleCategory.PORTFOLIO_LIMIT,
        name="Max Short Exposure",
        description="Maximum total short exposure",
        condition="short_exposure / portfolio_equity <= threshold",
        threshold=0.0,  # No shorting by default
        comparison="<=",
        action_on_breach="reject",
        severity="high",
    ),
    "min_cash_buffer": RiskRule(
        rule_id="RP007",
        category=RuleCategory.PORTFOLIO_LIMIT,
        name="Min Cash Buffer",
        description="Minimum cash reserve as percentage of equity",
        condition="cash / portfolio_equity >= threshold",
        threshold=0.05,  # 5% minimum cash
        comparison=">=",
        action_on_breach="reject",
        severity="medium",
    ),
    "max_daily_trades": RiskRule(
        rule_id="RP008",
        category=RuleCategory.PORTFOLIO_LIMIT,
        name="Max Daily Trades",
        description="Maximum number of trades per day",
        condition="daily_trades <= threshold",
        threshold=20.0,  # Max 20 trades/day
        comparison="<=",
        action_on_breach="reject",
        severity="medium",
    ),
    # ==========================================================================
    # KILL SWITCH RULES (RK) - Emergency halt conditions
    # ==========================================================================
    "daily_loss_limit": RiskRule(
        rule_id="RK001",
        category=RuleCategory.KILL_SWITCH,
        name="Daily Loss Limit",
        description="Maximum daily loss before halt - CRITICAL",
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
        description="Maximum drawdown from peak before halt - CRITICAL",
        condition="(equity - peak_equity) / peak_equity >= threshold",
        threshold=-0.15,  # -15% max drawdown
        comparison=">=",
        action_on_breach="halt",
        severity="critical",
    ),
    "weekly_loss_limit": RiskRule(
        rule_id="RK003",
        category=RuleCategory.KILL_SWITCH,
        name="Weekly Loss Limit",
        description="Maximum weekly loss before halt",
        condition="weekly_pnl / portfolio_equity >= threshold",
        threshold=-0.05,  # -5% weekly loss limit
        comparison=">=",
        action_on_breach="halt",
        severity="critical",
    ),
    "monthly_loss_limit": RiskRule(
        rule_id="RK004",
        category=RuleCategory.KILL_SWITCH,
        name="Monthly Loss Limit",
        description="Maximum monthly loss before halt",
        condition="monthly_pnl / portfolio_equity >= threshold",
        threshold=-0.10,  # -10% monthly loss limit
        comparison=">=",
        action_on_breach="halt",
        severity="critical",
    ),
    "consecutive_losses": RiskRule(
        rule_id="RK005",
        category=RuleCategory.KILL_SWITCH,
        name="Consecutive Losses",
        description="Maximum consecutive losing trades before halt",
        condition="consecutive_losses <= threshold",
        threshold=5.0,  # 5 consecutive losses max
        comparison="<=",
        action_on_breach="halt",
        severity="critical",
    ),
    "market_hours_only": RiskRule(
        rule_id="RK006",
        category=RuleCategory.KILL_SWITCH,
        name="Market Hours Only",
        description="Trading only during market hours",
        condition="market_open == threshold",
        threshold=1.0,  # 1 = True (market open)
        comparison="==",
        action_on_breach="halt",
        severity="critical",
    ),
    "connectivity_check": RiskRule(
        rule_id="RK007",
        category=RuleCategory.KILL_SWITCH,
        name="Connectivity Check",
        description="Halt trading if connectivity lost",
        condition="connectivity_ok == threshold",
        threshold=1.0,  # 1 = True (connected)
        comparison="==",
        action_on_breach="halt",
        severity="critical",
    ),
    # ==========================================================================
    # SANITY CHECK RULES (RS) - Data validation
    # ==========================================================================
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
    "stale_data_check": RiskRule(
        rule_id="RS003",
        category=RuleCategory.SANITY_CHECK,
        name="Stale Data Check",
        description="Reject trades based on stale market data",
        condition="data_age_seconds <= threshold",
        threshold=60.0,  # 60 seconds max data age
        comparison="<=",
        action_on_breach="reject",
        severity="high",
    ),
    "bid_ask_spread": RiskRule(
        rule_id="RS004",
        category=RuleCategory.SANITY_CHECK,
        name="Bid-Ask Spread Check",
        description="Reject trades with excessive bid-ask spread",
        condition="bid_ask_spread_pct <= threshold",
        threshold=0.02,  # 2% max spread
        comparison="<=",
        action_on_breach="reject",
        severity="medium",
    ),
}


def create_conservative_ruleset() -> dict[str, RiskRule]:
    """
    Create conservative risk rule set.

    Tighter limits for risk-averse trading.
    """
    import copy

    conservative = copy.deepcopy(STANDARD_RISK_RULES)

    # Pre-trade: Tighter position sizing
    conservative["max_position_pct"].threshold = 0.05  # 5% max position
    conservative["max_risk_per_trade"].threshold = 0.005  # 0.5% max risk
    conservative["max_single_stock_pct"].threshold = 0.08  # 8% max single stock
    conservative["min_position_size"].threshold = 2000.0  # $2,000 min position

    # Portfolio: Fewer positions, more diversification
    conservative["max_positions"].threshold = 5.0  # 5 max positions
    conservative["max_sector_concentration"].threshold = 0.20  # 20% max sector
    conservative["max_correlation_exposure"].threshold = 0.25  # 25% max correlated
    conservative["min_cash_buffer"].threshold = 0.10  # 10% minimum cash
    conservative["max_daily_trades"].threshold = 10.0  # 10 max trades/day

    # Kill switches: Tighter thresholds
    conservative["daily_loss_limit"].threshold = -0.02  # -2% daily loss
    conservative["max_drawdown"].threshold = -0.10  # -10% max drawdown
    conservative["weekly_loss_limit"].threshold = -0.03  # -3% weekly loss
    conservative["monthly_loss_limit"].threshold = -0.07  # -7% monthly loss
    conservative["consecutive_losses"].threshold = 3.0  # 3 consecutive losses

    return conservative


def create_aggressive_ruleset() -> dict[str, RiskRule]:
    """
    Create aggressive risk rule set.

    Looser limits for higher risk tolerance.
    """
    import copy

    aggressive = copy.deepcopy(STANDARD_RISK_RULES)

    # Pre-trade: Larger positions allowed
    aggressive["max_position_pct"].threshold = 0.20  # 20% max position
    aggressive["max_risk_per_trade"].threshold = 0.02  # 2% max risk
    aggressive["max_single_stock_pct"].threshold = 0.25  # 25% max single stock
    aggressive["min_position_size"].threshold = 500.0  # $500 min position

    # Portfolio: More positions, higher concentration
    aggressive["max_positions"].threshold = 20.0  # 20 max positions
    aggressive["max_sector_concentration"].threshold = 0.50  # 50% max sector
    aggressive["max_correlation_exposure"].threshold = 0.60  # 60% max correlated
    aggressive["min_cash_buffer"].threshold = 0.02  # 2% minimum cash
    aggressive["max_daily_trades"].threshold = 50.0  # 50 max trades/day

    # Kill switches: Looser thresholds
    aggressive["daily_loss_limit"].threshold = -0.05  # -5% daily loss
    aggressive["max_drawdown"].threshold = -0.25  # -25% max drawdown
    aggressive["weekly_loss_limit"].threshold = -0.08  # -8% weekly loss
    aggressive["monthly_loss_limit"].threshold = -0.15  # -15% monthly loss
    aggressive["consecutive_losses"].threshold = 7.0  # 7 consecutive losses

    return aggressive


def create_day_trading_ruleset() -> dict[str, RiskRule]:
    """
    Create day trading risk rule set.

    Optimized for intraday strategies with no overnight positions.
    """
    import copy

    day_trading = copy.deepcopy(STANDARD_RISK_RULES)

    # Pre-trade: Quick entries/exits
    day_trading["max_position_pct"].threshold = 0.15  # 15% max position
    day_trading["max_risk_per_trade"].threshold = 0.005  # 0.5% max risk (tight stops)
    day_trading["min_position_size"].threshold = 500.0  # Lower minimum

    # Portfolio: Higher turnover allowed
    day_trading["max_positions"].threshold = 3.0  # Max 3 positions at once
    day_trading["max_daily_trades"].threshold = 100.0  # High turnover
    day_trading["min_cash_buffer"].threshold = 0.25  # 25% cash for PDT requirement

    # Kill switches: Tight daily, looser monthly
    day_trading["daily_loss_limit"].threshold = -0.02  # -2% daily (strict)
    day_trading["max_drawdown"].threshold = -0.10  # -10% max drawdown
    day_trading["consecutive_losses"].threshold = 4.0  # 4 consecutive losses

    return day_trading


def create_swing_trading_ruleset() -> dict[str, RiskRule]:
    """
    Create swing trading risk rule set.

    Optimized for multi-day holds with wider stops.
    """
    import copy

    swing = copy.deepcopy(STANDARD_RISK_RULES)

    # Pre-trade: Larger positions, wider stops
    swing["max_position_pct"].threshold = 0.12  # 12% max position
    swing["max_risk_per_trade"].threshold = 0.015  # 1.5% max risk (wider stops)
    swing["min_position_size"].threshold = 1500.0  # Larger minimum

    # Portfolio: Longer holds
    swing["max_positions"].threshold = 8.0  # 8 max positions
    swing["max_daily_trades"].threshold = 5.0  # Low turnover
    swing["min_cash_buffer"].threshold = 0.10  # 10% cash reserve

    # Kill switches: Balanced
    swing["daily_loss_limit"].threshold = -0.04  # -4% daily (more room)
    swing["max_drawdown"].threshold = -0.15  # -15% max drawdown
    swing["weekly_loss_limit"].threshold = -0.06  # -6% weekly

    return swing
