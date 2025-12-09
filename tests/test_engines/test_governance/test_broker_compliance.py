"""Tests for Broker Compliance Engine."""

from datetime import datetime, timedelta

from src.engines.governance.core.broker_compliance import (
    Broker,
    BrokerComplianceEngine,
    BrokerPolicy,
    ComplianceCategory,
    ComplianceCheckResult,
    RateLimitState,
    ViolationSeverity,
)


class TestRateLimitState:
    """Tests for RateLimitState."""

    def test_rate_limit_allows_requests(self) -> None:
        """Test rate limiter allows requests within limit."""
        state = RateLimitState(
            endpoint="test",
            max_requests=10,
            window_seconds=60,
        )

        for _ in range(5):
            allowed, remaining = state.can_make_request()
            assert allowed is True
            state.record_request()

        assert state.requests_made == 5

    def test_rate_limit_blocks_excess(self) -> None:
        """Test rate limiter blocks after limit reached."""
        state = RateLimitState(
            endpoint="test",
            max_requests=3,
            window_seconds=60,
        )

        # Use up all requests
        for _ in range(3):
            state.record_request()

        allowed, wait_time = state.can_make_request()
        assert allowed is False
        assert wait_time > 0

    def test_rate_limit_resets_after_window(self) -> None:
        """Test rate limiter resets after time window."""
        state = RateLimitState(
            endpoint="test",
            max_requests=3,
            window_seconds=1,  # 1 second window
        )

        # Use up all requests
        for _ in range(3):
            state.record_request()

        # Manually reset window by backdating
        state.window_start = datetime.utcnow() - timedelta(seconds=2)

        allowed, remaining = state.can_make_request()
        assert allowed is True


class TestBrokerComplianceEngine:
    """Tests for BrokerComplianceEngine."""

    def test_engine_initialization_alpaca(self) -> None:
        """Test engine initializes with Alpaca policies."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        assert engine.broker == Broker.ALPACA
        assert len(engine._policies) > 0
        assert "api_rate_limit" in engine._policies
        assert "pdt_rule" in engine._policies

    def test_engine_initialization_paper(self) -> None:
        """Test engine initializes for paper trading."""
        engine = BrokerComplianceEngine(
            broker=Broker.ALPACA_PAPER,
            account_type="paper",
        )

        assert engine.account_type == "paper"

    def test_check_rate_limit_pass(self) -> None:
        """Test rate limit check passes when under limit."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        result = engine.check_rate_limit("api")

        assert result.passed is True
        assert result.policy.category == ComplianceCategory.RATE_LIMITING

    def test_check_rate_limit_fail(self) -> None:
        """Test rate limit check fails when exceeded."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        # Exhaust rate limit
        engine._rate_limits["api"].requests_made = 200

        result = engine.check_rate_limit("api")

        assert result.passed is False
        assert "exceeded" in result.message.lower()

    def test_check_pdt_exempt_large_account(self) -> None:
        """Test PDT check exempt for accounts over $25k."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        result = engine.check_pdt_compliance(
            account_value=50000,
            is_day_trade=True,
        )

        assert result.passed is True
        assert result.details.get("pdt_exempt") is True

    def test_check_pdt_under_limit(self) -> None:
        """Test PDT check passes when under 3 day trades."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        result = engine.check_pdt_compliance(
            account_value=10000,
            is_day_trade=True,
        )

        assert result.passed is True
        assert result.details.get("day_trade_count") == 0

    def test_check_pdt_at_limit(self) -> None:
        """Test PDT check fails at 3 day trades."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        # Record 3 day trades
        for i in range(3):
            engine.record_day_trade(
                symbol=f"TEST{i}",
                buy_time=datetime.utcnow(),
                sell_time=datetime.utcnow(),
                profit_loss=100.0,
            )

        result = engine.check_pdt_compliance(
            account_value=10000,
            is_day_trade=True,
        )

        assert result.passed is False
        assert "PDT limit reached" in result.message

    def test_check_buying_power_sufficient(self) -> None:
        """Test buying power check with sufficient funds."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        result = engine.check_buying_power(
            buying_power=50000,
            order_value=10000,
        )

        assert result.passed is True

    def test_check_buying_power_insufficient(self) -> None:
        """Test buying power check with insufficient funds."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        result = engine.check_buying_power(
            buying_power=5000,
            order_value=10000,
        )

        assert result.passed is False
        assert result.details.get("shortfall") == 5000

    def test_check_short_selling_with_margin(self) -> None:
        """Test short selling check with margin approval."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        result = engine.check_short_selling(
            is_short=True,
            has_margin_approval=True,
            symbol="AAPL",
        )

        assert result.passed is True

    def test_check_short_selling_without_margin(self) -> None:
        """Test short selling check without margin approval."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        result = engine.check_short_selling(
            is_short=True,
            has_margin_approval=False,
            symbol="AAPL",
        )

        assert result.passed is False
        assert "requires margin" in result.message.lower()

    def test_check_short_selling_not_short(self) -> None:
        """Test short selling check for non-short orders."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        result = engine.check_short_selling(
            is_short=False,
            has_margin_approval=False,
            symbol="AAPL",
        )

        assert result.passed is True

    def test_check_data_usage_allowed(self) -> None:
        """Test data usage check for allowed actions."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        result = engine.check_data_usage(action="store", data_type="market_data")

        assert result.passed is True

    def test_check_data_usage_prohibited(self) -> None:
        """Test data usage check for prohibited actions."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        result = engine.check_data_usage(action="redistribute", data_type="market_data")

        assert result.passed is False
        assert "prohibited" in result.message.lower()

    def test_check_order_comprehensive(self) -> None:
        """Test comprehensive order compliance check."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        passed, results = engine.check_order(
            symbol="AAPL",
            side="buy",
            quantity=100,
            order_type="market",
            account_info={
                "buying_power": 50000,
                "account_value": 100000,
                "has_margin": False,
                "positions": {},
            },
        )

        assert passed is True
        assert len(results) >= 3  # Rate limit, buying power, PDT, short check

    def test_check_order_fails_buying_power(self) -> None:
        """Test order check fails on insufficient buying power."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        passed, results = engine.check_order(
            symbol="AAPL",
            side="buy",
            quantity=1000,
            order_type="market",
            account_info={
                "buying_power": 1000,  # Not enough
                "account_value": 5000,
                "has_margin": False,
                "positions": {},
            },
        )

        # Should fail due to buying power
        buying_power_results = [r for r in results if r.policy.name == "buying_power"]
        assert len(buying_power_results) > 0
        assert buying_power_results[0].passed is False

    def test_record_day_trade(self) -> None:
        """Test recording day trades."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        engine.record_day_trade(
            symbol="AAPL",
            buy_time=datetime.utcnow(),
            sell_time=datetime.utcnow(),
            profit_loss=150.0,
        )

        assert len(engine._day_trades) == 1
        assert engine._day_trades[0]["symbol"] == "AAPL"

    def test_violation_callback(self) -> None:
        """Test violation callback is triggered."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)
        violations = []

        engine.register_violation_callback(lambda v: violations.append(v))

        # Trigger a violation
        engine.check_buying_power(buying_power=100, order_value=10000)

        assert len(violations) == 1

    def test_compliance_summary(self) -> None:
        """Test compliance summary generation."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        # Perform some checks
        engine.check_rate_limit("api")
        engine.check_buying_power(buying_power=1000, order_value=5000)  # Violation

        summary = engine.get_compliance_summary()

        assert summary["broker"] == "alpaca"
        assert "total_violations" in summary
        assert "violations_by_category" in summary
        assert "rate_limit_status" in summary

    def test_get_broker_policies(self) -> None:
        """Test getting broker policies."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        policies = engine.get_broker_policies()

        assert len(policies) > 0
        assert all(isinstance(p, BrokerPolicy) for p in policies)

    def test_add_custom_policy(self) -> None:
        """Test adding custom policy."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        custom = BrokerPolicy(
            broker=Broker.ALPACA,
            category=ComplianceCategory.TRADING_PATTERNS,
            name="custom_rule",
            description="Custom trading rule",
            blocking=True,
        )

        engine.add_custom_policy(custom)

        assert "custom_rule" in engine._policies

    def test_reset_daily_counters(self) -> None:
        """Test daily counter reset."""
        engine = BrokerComplianceEngine(broker=Broker.ALPACA)

        # Add some activity
        engine._orders_today.append({"test": "order"})
        engine._api_calls_today = 100

        # Old day trade (should be kept)
        engine._day_trades.append(
            {
                "symbol": "TEST",
                "timestamp": datetime.utcnow() - timedelta(days=3),
            }
        )

        # Very old day trade (should be removed)
        engine._day_trades.append(
            {
                "symbol": "OLD",
                "timestamp": datetime.utcnow() - timedelta(days=10),
            }
        )

        engine.reset_daily_counters()

        assert len(engine._orders_today) == 0
        assert engine._api_calls_today == 0
        assert len(engine._day_trades) == 1  # Only recent one kept


class TestBrokerPolicy:
    """Tests for BrokerPolicy dataclass."""

    def test_policy_creation(self) -> None:
        """Test basic policy creation."""
        policy = BrokerPolicy(
            broker=Broker.ALPACA,
            category=ComplianceCategory.RATE_LIMITING,
            name="test_policy",
            description="Test policy description",
            limit=100,
            blocking=True,
            severity=ViolationSeverity.CRITICAL,
        )

        assert policy.broker == Broker.ALPACA
        assert policy.limit == 100
        assert policy.blocking is True

    def test_policy_with_time_window(self) -> None:
        """Test policy with time window."""
        policy = BrokerPolicy(
            broker=Broker.ALPACA,
            category=ComplianceCategory.ORDER_RESTRICTIONS,
            name="pdt_test",
            description="PDT test",
            limit=3,
            time_window=timedelta(days=5),
        )

        assert policy.time_window == timedelta(days=5)


class TestComplianceCheckResult:
    """Tests for ComplianceCheckResult dataclass."""

    def test_result_creation(self) -> None:
        """Test result creation."""
        policy = BrokerPolicy(
            broker=Broker.ALPACA,
            category=ComplianceCategory.RATE_LIMITING,
            name="test",
            description="test",
        )

        result = ComplianceCheckResult(
            policy=policy,
            passed=True,
            current_value=50.0,
            message="Check passed",
        )

        assert result.passed is True
        assert result.current_value == 50.0

    def test_result_with_details(self) -> None:
        """Test result with detailed info."""
        policy = BrokerPolicy(
            broker=Broker.ALPACA,
            category=ComplianceCategory.ORDER_RESTRICTIONS,
            name="test",
            description="test",
        )

        result = ComplianceCheckResult(
            policy=policy,
            passed=False,
            message="Insufficient funds",
            details={
                "required": 10000,
                "available": 5000,
                "shortfall": 5000,
            },
        )

        assert result.passed is False
        assert result.details["shortfall"] == 5000
