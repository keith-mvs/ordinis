"""
Broker Terms of Service Compliance Engine.

Ensures trading operations comply with broker/API agreements and terms of service.
Supports multiple brokers with extensible policy definitions.

Supported Brokers:
- Alpaca Markets (paper and live trading)
- Interactive Brokers (planned)
- TD Ameritrade (planned)
- Others via extensible framework
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
from typing import Any

_logger = logging.getLogger(__name__)


class Broker(Enum):
    """Supported broker integrations."""

    ALPACA = "alpaca"
    ALPACA_PAPER = "alpaca_paper"
    INTERACTIVE_BROKERS = "interactive_brokers"
    TD_AMERITRADE = "td_ameritrade"
    CUSTOM = "custom"


class ComplianceCategory(Enum):
    """Categories of broker compliance requirements."""

    RATE_LIMITING = "rate_limiting"
    DATA_USAGE = "data_usage"
    ORDER_RESTRICTIONS = "order_restrictions"
    ACCOUNT_REQUIREMENTS = "account_requirements"
    TRADING_PATTERNS = "trading_patterns"
    API_USAGE = "api_usage"
    REPORTING = "reporting"
    MARKET_DATA = "market_data"


class ViolationSeverity(Enum):
    """Severity levels for compliance violations."""

    INFO = "info"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


@dataclass
class BrokerPolicy:
    """Defines a broker-specific compliance policy."""

    broker: Broker
    category: ComplianceCategory
    name: str
    description: str
    threshold: float | None = None
    limit: int | None = None
    time_window: timedelta | None = None
    blocking: bool = False
    severity: ViolationSeverity = ViolationSeverity.WARNING


@dataclass
class ComplianceCheckResult:
    """Result of a broker compliance check."""

    policy: BrokerPolicy
    passed: bool
    current_value: float | None = None
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RateLimitState:
    """Tracks rate limiting state for API calls."""

    endpoint: str
    requests_made: int = 0
    window_start: datetime = field(default_factory=datetime.utcnow)
    window_seconds: int = 60
    max_requests: int = 200

    def can_make_request(self) -> tuple[bool, int]:
        """Check if request is allowed within rate limits."""
        now = datetime.utcnow()
        window_elapsed = (now - self.window_start).total_seconds()

        if window_elapsed >= self.window_seconds:
            # Reset window
            self.requests_made = 0
            self.window_start = now
            return True, self.max_requests - 1

        if self.requests_made >= self.max_requests:
            wait_time = int(self.window_seconds - window_elapsed)
            return False, wait_time

        return True, self.max_requests - self.requests_made - 1

    def record_request(self) -> None:
        """Record that a request was made."""
        now = datetime.utcnow()
        window_elapsed = (now - self.window_start).total_seconds()

        if window_elapsed >= self.window_seconds:
            self.requests_made = 1
            self.window_start = now
        else:
            self.requests_made += 1


class BrokerComplianceEngine:
    """
    Engine for enforcing broker terms of service and API agreements.

    Implements compliance checking for:
    - Rate limiting (API call quotas)
    - Order restrictions (PDT, position limits)
    - Data usage policies
    - Trading pattern rules
    - Account requirements
    """

    def __init__(
        self,
        broker: Broker = Broker.ALPACA_PAPER,
        account_type: str = "paper",
    ):
        """Initialize broker compliance engine.

        Args:
            broker: Target broker for compliance checking.
            account_type: Account type (paper, live, margin).
        """
        self.broker = broker
        self.account_type = account_type

        # Policy registry
        self._policies: dict[str, BrokerPolicy] = {}

        # Rate limiting state
        self._rate_limits: dict[str, RateLimitState] = {}

        # Trading activity tracking
        self._day_trades: list[dict[str, Any]] = []
        self._orders_today: list[dict[str, Any]] = []
        self._api_calls_today: int = 0

        # Violation tracking
        self._violations: list[ComplianceCheckResult] = []
        self._violation_callbacks: list[Callable] = []

        # Load broker-specific policies
        self._load_broker_policies()

    def _load_broker_policies(self) -> None:
        """Load policies specific to the configured broker."""
        if self.broker in (Broker.ALPACA, Broker.ALPACA_PAPER):
            self._load_alpaca_policies()
        elif self.broker == Broker.INTERACTIVE_BROKERS:
            self._load_ib_policies()
        # Add more brokers as needed

    def _load_alpaca_policies(self) -> None:
        """Load Alpaca Markets specific compliance policies.

        Based on Alpaca's Terms of Service and API documentation:
        https://alpaca.markets/docs/api-references/
        """
        policies = [
            # Rate Limiting Policies
            BrokerPolicy(
                broker=Broker.ALPACA,
                category=ComplianceCategory.RATE_LIMITING,
                name="api_rate_limit",
                description="API calls limited to 200 requests per minute",
                limit=200,
                time_window=timedelta(minutes=1),
                blocking=True,
                severity=ViolationSeverity.CRITICAL,
            ),
            BrokerPolicy(
                broker=Broker.ALPACA,
                category=ComplianceCategory.RATE_LIMITING,
                name="order_rate_limit",
                description="Order submissions limited per minute",
                limit=100,
                time_window=timedelta(minutes=1),
                blocking=True,
                severity=ViolationSeverity.CRITICAL,
            ),
            # Order Restriction Policies
            BrokerPolicy(
                broker=Broker.ALPACA,
                category=ComplianceCategory.ORDER_RESTRICTIONS,
                name="pdt_rule",
                description="Pattern Day Trader rule - max 3 day trades in 5 days for accounts under $25k",
                limit=3,
                time_window=timedelta(days=5),
                blocking=True,
                severity=ViolationSeverity.CRITICAL,
            ),
            BrokerPolicy(
                broker=Broker.ALPACA,
                category=ComplianceCategory.ORDER_RESTRICTIONS,
                name="fractional_shares",
                description="Fractional shares allowed for supported symbols",
                blocking=False,
                severity=ViolationSeverity.INFO,
            ),
            BrokerPolicy(
                broker=Broker.ALPACA,
                category=ComplianceCategory.ORDER_RESTRICTIONS,
                name="short_selling",
                description="Short selling requires margin account approval",
                blocking=True,
                severity=ViolationSeverity.VIOLATION,
            ),
            # Account Requirement Policies
            BrokerPolicy(
                broker=Broker.ALPACA,
                category=ComplianceCategory.ACCOUNT_REQUIREMENTS,
                name="margin_account",
                description="Margin trading requires approved margin account",
                blocking=True,
                severity=ViolationSeverity.CRITICAL,
            ),
            BrokerPolicy(
                broker=Broker.ALPACA,
                category=ComplianceCategory.ACCOUNT_REQUIREMENTS,
                name="buying_power",
                description="Orders must not exceed buying power",
                blocking=True,
                severity=ViolationSeverity.CRITICAL,
            ),
            # Trading Pattern Policies
            BrokerPolicy(
                broker=Broker.ALPACA,
                category=ComplianceCategory.TRADING_PATTERNS,
                name="wash_sale",
                description="Wash sale rule awareness for tax implications",
                blocking=False,
                severity=ViolationSeverity.WARNING,
            ),
            BrokerPolicy(
                broker=Broker.ALPACA,
                category=ComplianceCategory.TRADING_PATTERNS,
                name="market_manipulation",
                description="Prohibited: wash trading, spoofing, layering",
                blocking=True,
                severity=ViolationSeverity.CRITICAL,
            ),
            # Data Usage Policies
            BrokerPolicy(
                broker=Broker.ALPACA,
                category=ComplianceCategory.DATA_USAGE,
                name="market_data_redistribution",
                description="Market data redistribution prohibited without license",
                blocking=True,
                severity=ViolationSeverity.CRITICAL,
            ),
            BrokerPolicy(
                broker=Broker.ALPACA,
                category=ComplianceCategory.DATA_USAGE,
                name="data_storage",
                description="Historical data storage subject to exchange agreements",
                blocking=False,
                severity=ViolationSeverity.WARNING,
            ),
            # API Usage Policies
            BrokerPolicy(
                broker=Broker.ALPACA,
                category=ComplianceCategory.API_USAGE,
                name="api_key_security",
                description="API keys must be kept secure, not shared or exposed",
                blocking=True,
                severity=ViolationSeverity.CRITICAL,
            ),
            BrokerPolicy(
                broker=Broker.ALPACA,
                category=ComplianceCategory.API_USAGE,
                name="paper_vs_live",
                description="Paper trading uses separate endpoints from live trading",
                blocking=True,
                severity=ViolationSeverity.CRITICAL,
            ),
        ]

        for policy in policies:
            self._policies[policy.name] = policy

        # Initialize rate limiters
        self._rate_limits["api"] = RateLimitState(
            endpoint="api",
            max_requests=200,
            window_seconds=60,
        )
        self._rate_limits["orders"] = RateLimitState(
            endpoint="orders",
            max_requests=100,
            window_seconds=60,
        )

    def _load_ib_policies(self) -> None:
        """Load Interactive Brokers specific policies."""
        policies = [
            BrokerPolicy(
                broker=Broker.INTERACTIVE_BROKERS,
                category=ComplianceCategory.RATE_LIMITING,
                name="ib_rate_limit",
                description="IB API message rate limiting",
                limit=50,
                time_window=timedelta(seconds=1),
                blocking=True,
                severity=ViolationSeverity.CRITICAL,
            ),
            BrokerPolicy(
                broker=Broker.INTERACTIVE_BROKERS,
                category=ComplianceCategory.ORDER_RESTRICTIONS,
                name="ib_pdt_rule",
                description="Pattern Day Trader restrictions",
                limit=3,
                time_window=timedelta(days=5),
                blocking=True,
                severity=ViolationSeverity.CRITICAL,
            ),
        ]

        for policy in policies:
            self._policies[policy.name] = policy

    def check_rate_limit(
        self,
        endpoint: str = "api",
    ) -> ComplianceCheckResult:
        """Check if API call is within rate limits.

        Args:
            endpoint: The API endpoint category to check.

        Returns:
            ComplianceCheckResult with pass/fail status.
        """
        if endpoint not in self._rate_limits:
            self._rate_limits[endpoint] = RateLimitState(endpoint=endpoint)

        rate_state = self._rate_limits[endpoint]
        allowed, remaining = rate_state.can_make_request()

        policy = self._policies.get(
            "api_rate_limit",
            BrokerPolicy(
                broker=self.broker,
                category=ComplianceCategory.RATE_LIMITING,
                name="api_rate_limit",
                description="Default rate limit",
                limit=200,
            ),
        )

        result = ComplianceCheckResult(
            policy=policy,
            passed=allowed,
            current_value=float(rate_state.requests_made),
            message=f"Rate limit: {rate_state.requests_made}/{rate_state.max_requests} (remaining: {remaining})"
            if allowed
            else f"Rate limit exceeded. Wait {remaining} seconds.",
            details={
                "endpoint": endpoint,
                "requests_made": rate_state.requests_made,
                "max_requests": rate_state.max_requests,
                "remaining": remaining,
            },
        )

        if allowed:
            rate_state.record_request()
        else:
            self._record_violation(result)

        return result

    def check_pdt_compliance(
        self,
        account_value: float,
        is_day_trade: bool = False,
    ) -> ComplianceCheckResult:
        """Check Pattern Day Trader rule compliance.

        PDT Rule: Accounts under $25,000 are limited to 3 day trades
        within any 5-business-day period.

        Args:
            account_value: Current account value in USD.
            is_day_trade: Whether this order would be a day trade.

        Returns:
            ComplianceCheckResult with compliance status.
        """
        policy = self._policies.get("pdt_rule")
        if not policy:
            return ComplianceCheckResult(
                policy=BrokerPolicy(
                    broker=self.broker,
                    category=ComplianceCategory.ORDER_RESTRICTIONS,
                    name="pdt_rule",
                    description="PDT rule not configured",
                ),
                passed=True,
                message="PDT rule checking not configured",
            )

        # PDT only applies to accounts under $25,000
        if account_value >= 25000:
            return ComplianceCheckResult(
                policy=policy,
                passed=True,
                current_value=account_value,
                message="PDT rule not applicable - account value >= $25,000",
                details={"account_value": account_value, "pdt_exempt": True},
            )

        # Count day trades in last 5 business days
        cutoff = datetime.now(UTC) - timedelta(days=5)
        recent_day_trades = [
            dt
            for dt in self._day_trades
            if dt.get("timestamp", datetime.min.replace(tzinfo=UTC)) > cutoff
        ]

        current_count = len(recent_day_trades)
        would_exceed = current_count >= 3 and is_day_trade

        result = ComplianceCheckResult(
            policy=policy,
            passed=not would_exceed,
            current_value=float(current_count),
            message=f"Day trades: {current_count}/3 in last 5 days"
            if not would_exceed
            else "PDT limit reached - cannot execute day trade",
            details={
                "day_trade_count": current_count,
                "limit": 3,
                "account_value": account_value,
                "is_day_trade": is_day_trade,
            },
        )

        if would_exceed:
            self._record_violation(result)

        return result

    def check_buying_power(
        self,
        buying_power: float,
        order_value: float,
    ) -> ComplianceCheckResult:
        """Check if order is within buying power limits.

        Args:
            buying_power: Available buying power in USD.
            order_value: Total value of the proposed order.

        Returns:
            ComplianceCheckResult with compliance status.
        """
        policy = self._policies.get(
            "buying_power",
            BrokerPolicy(
                broker=self.broker,
                category=ComplianceCategory.ACCOUNT_REQUIREMENTS,
                name="buying_power",
                description="Buying power check",
            ),
        )

        passed = order_value <= buying_power

        result = ComplianceCheckResult(
            policy=policy,
            passed=passed,
            current_value=order_value,
            message=f"Order ${order_value:,.2f} within buying power ${buying_power:,.2f}"
            if passed
            else f"Insufficient buying power: need ${order_value:,.2f}, have ${buying_power:,.2f}",
            details={
                "buying_power": buying_power,
                "order_value": order_value,
                "shortfall": max(0, order_value - buying_power),
            },
        )

        if not passed:
            self._record_violation(result)

        return result

    def check_short_selling(
        self,
        is_short: bool,
        has_margin_approval: bool,
        symbol: str,
    ) -> ComplianceCheckResult:
        """Check short selling compliance.

        Args:
            is_short: Whether this is a short sale.
            has_margin_approval: Whether account has margin approval.
            symbol: Symbol being traded.

        Returns:
            ComplianceCheckResult with compliance status.
        """
        policy = self._policies.get(
            "short_selling",
            BrokerPolicy(
                broker=self.broker,
                category=ComplianceCategory.ORDER_RESTRICTIONS,
                name="short_selling",
                description="Short selling check",
            ),
        )

        if not is_short:
            return ComplianceCheckResult(
                policy=policy,
                passed=True,
                message="Not a short sale",
                details={"is_short": False},
            )

        passed = has_margin_approval

        result = ComplianceCheckResult(
            policy=policy,
            passed=passed,
            message=f"Short sale of {symbol} approved with margin account"
            if passed
            else f"Short sale of {symbol} requires margin approval",
            details={
                "symbol": symbol,
                "is_short": True,
                "has_margin_approval": has_margin_approval,
            },
        )

        if not passed:
            self._record_violation(result)

        return result

    def check_data_usage(
        self,
        action: str,
        data_type: str = "market_data",
    ) -> ComplianceCheckResult:
        """Check data usage compliance.

        Args:
            action: What is being done with the data (store, redistribute, display).
            data_type: Type of data being used.

        Returns:
            ComplianceCheckResult with compliance status.
        """
        prohibited_actions = ["redistribute", "resell", "broadcast"]

        policy = self._policies.get(
            "market_data_redistribution",
            BrokerPolicy(
                broker=self.broker,
                category=ComplianceCategory.DATA_USAGE,
                name="market_data_redistribution",
                description="Data usage check",
            ),
        )

        is_prohibited = action.lower() in prohibited_actions

        result = ComplianceCheckResult(
            policy=policy,
            passed=not is_prohibited,
            message=f"Data usage '{action}' is permitted"
            if not is_prohibited
            else f"Data action '{action}' is prohibited without proper licensing",
            details={
                "action": action,
                "data_type": data_type,
                "prohibited": is_prohibited,
            },
        )

        if is_prohibited:
            self._record_violation(result)

        return result

    def check_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        account_info: dict[str, Any],
    ) -> tuple[bool, list[ComplianceCheckResult]]:
        """Run all order-related compliance checks.

        Args:
            symbol: Trading symbol.
            side: Order side (buy/sell).
            quantity: Order quantity.
            order_type: Order type (market, limit, etc.).
            account_info: Account details including:
                - buying_power: Available buying power
                - account_value: Total account value
                - has_margin: Whether margin is enabled
                - positions: Current positions

        Returns:
            Tuple of (all_passed, list of check results).
        """
        results = []

        # Rate limit check
        results.append(self.check_rate_limit("orders"))

        # Buying power check (simplified - would need price in real impl)
        estimated_value = quantity * 100  # Placeholder
        results.append(
            self.check_buying_power(
                buying_power=account_info.get("buying_power", 0),
                order_value=estimated_value,
            )
        )

        # PDT check
        is_day_trade = self._would_be_day_trade(symbol, side, account_info.get("positions", {}))
        results.append(
            self.check_pdt_compliance(
                account_value=account_info.get("account_value", 0),
                is_day_trade=is_day_trade,
            )
        )

        # Short selling check
        is_short = side.lower() == "sell" and symbol not in account_info.get("positions", {})
        results.append(
            self.check_short_selling(
                is_short=is_short,
                has_margin_approval=account_info.get("has_margin", False),
                symbol=symbol,
            )
        )

        all_passed = all(r.passed for r in results)
        blocking_failures = [r for r in results if not r.passed and r.policy.blocking]

        return len(blocking_failures) == 0, results

    def _would_be_day_trade(
        self,
        symbol: str,
        side: str,
        positions: dict[str, Any],
    ) -> bool:
        """Determine if an order would constitute a day trade.

        A day trade is opening and closing a position in the same day.
        """
        # Check if we have an existing position opened today
        if symbol in positions:
            position = positions[symbol]
            opened_today = position.get("opened_today", False)

            # If we're closing a position opened today, it's a day trade
            if opened_today and side.lower() == "sell":
                return True

        return False

    def record_day_trade(
        self,
        symbol: str,
        buy_time: datetime,
        sell_time: datetime,
        profit_loss: float,
    ) -> None:
        """Record a day trade for PDT tracking.

        Args:
            symbol: Symbol that was day traded.
            buy_time: Time of buy order.
            sell_time: Time of sell order.
            profit_loss: Realized P&L from the trade.
        """
        self._day_trades.append(
            {
                "symbol": symbol,
                "buy_time": buy_time,
                "sell_time": sell_time,
                "profit_loss": profit_loss,
                "timestamp": datetime.now(UTC),
            }
        )

    def _record_violation(self, result: ComplianceCheckResult) -> None:
        """Record a compliance violation."""
        self._violations.append(result)

        # Trigger callbacks
        for callback in self._violation_callbacks:
            try:
                callback(result)
            except Exception:
                _logger.debug("Violation callback error isolated", exc_info=True)

    def register_violation_callback(
        self,
        callback: Callable[[ComplianceCheckResult], None],
    ) -> None:
        """Register a callback for compliance violations.

        Args:
            callback: Function to call when violation occurs.
        """
        self._violation_callbacks.append(callback)

    def get_compliance_summary(self) -> dict[str, Any]:
        """Get summary of compliance status.

        Returns:
            Dictionary with compliance metrics and violation history.
        """
        # Group violations by category
        by_category: dict[str, int] = {}
        for violation in self._violations:
            cat = violation.policy.category.value
            by_category[cat] = by_category.get(cat, 0) + 1

        # Rate limit status
        rate_status = {}
        for name, state in self._rate_limits.items():
            rate_status[name] = {
                "requests_made": state.requests_made,
                "max_requests": state.max_requests,
                "utilization": state.requests_made / state.max_requests
                if state.max_requests > 0
                else 0,
            }

        return {
            "broker": self.broker.value,
            "account_type": self.account_type,
            "total_violations": len(self._violations),
            "violations_by_category": by_category,
            "day_trades_count": len(self._day_trades),
            "rate_limit_status": rate_status,
            "policies_loaded": len(self._policies),
            "critical_violations": len(
                [v for v in self._violations if v.policy.severity == ViolationSeverity.CRITICAL]
            ),
        }

    def get_broker_policies(self) -> list[BrokerPolicy]:
        """Get all loaded policies for the current broker.

        Returns:
            List of BrokerPolicy objects.
        """
        return list(self._policies.values())

    def add_custom_policy(self, policy: BrokerPolicy) -> None:
        """Add a custom compliance policy.

        Args:
            policy: Custom policy to add.
        """
        self._policies[policy.name] = policy

    def reset_daily_counters(self) -> None:
        """Reset daily counters (call at market open)."""
        self._orders_today = []
        self._api_calls_today = 0

        # Keep day trades for PDT tracking (5-day window)
        cutoff = datetime.now(UTC) - timedelta(days=5)
        self._day_trades = [
            dt
            for dt in self._day_trades
            if dt.get("timestamp", datetime.min.replace(tzinfo=UTC)) > cutoff
        ]
