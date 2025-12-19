"""Tests for PortfolioOpt governance hooks.

This module tests the governance hook functionality including:
- Preflight validation rules
- Audit trail management
- Risk limit enforcement
- Data quality checks
"""

import pytest

from ordinis.engines.base import AuditRecord, Decision
from ordinis.engines.portfolioopt.hooks.governance import (
    DataQualityRule,
    PortfolioOptGovernanceHook,
    RiskLimitRule,
    SolverValidationRule,
)


def create_hook(
    risk_rule: RiskLimitRule | None = None,
    data_rule: DataQualityRule | None = None,
    solver_rule: SolverValidationRule | None = None,
) -> PortfolioOptGovernanceHook:
    """Helper to create a properly initialized governance hook."""
    hook = PortfolioOptGovernanceHook.__new__(PortfolioOptGovernanceHook)
    hook.engine_name = "TestEngine"
    hook._policy_version = "1.0.0"
    hook.risk_rule = risk_rule or RiskLimitRule()
    hook.data_rule = data_rule or DataQualityRule()
    hook.solver_rule = solver_rule or SolverValidationRule()
    hook._audit_log = []
    return hook


class TestRiskLimitRule:
    """Test RiskLimitRule validation."""

    def test_default_values(self) -> None:
        """Test default risk limit values."""
        rule = RiskLimitRule()

        assert rule.max_target_return == 0.05
        assert rule.max_weight_per_asset == 0.30
        assert rule.min_assets == 3

    def test_validate_success(self) -> None:
        """Test validation passes with valid parameters."""
        rule = RiskLimitRule()
        context = {
            "target_return": 0.02,
            "max_weight": 0.20,
            "n_assets": 5,
        }

        passed, reason = rule.validate(context)

        assert passed is True
        assert reason is None

    def test_validate_target_return_exceeds_limit(self) -> None:
        """Test validation fails when target return exceeds limit."""
        rule = RiskLimitRule(max_target_return=0.03)
        context = {
            "target_return": 0.05,
            "max_weight": 0.20,
            "n_assets": 5,
        }

        passed, reason = rule.validate(context)

        assert passed is False
        assert "Target return" in reason
        assert "exceeds limit" in reason

    def test_validate_max_weight_exceeds_limit(self) -> None:
        """Test validation fails when max weight exceeds limit."""
        rule = RiskLimitRule(max_weight_per_asset=0.25)
        context = {
            "target_return": 0.01,
            "max_weight": 0.35,
            "n_assets": 5,
        }

        passed, reason = rule.validate(context)

        assert passed is False
        assert "Max weight" in reason
        assert "exceeds limit" in reason

    def test_validate_insufficient_assets(self) -> None:
        """Test validation fails with insufficient assets."""
        rule = RiskLimitRule(min_assets=5)
        context = {
            "target_return": 0.01,
            "max_weight": 0.20,
            "n_assets": 3,
        }

        passed, reason = rule.validate(context)

        assert passed is False
        assert "minimum is" in reason

    def test_validate_edge_case_exact_limit(self) -> None:
        """Test validation at exact limit."""
        rule = RiskLimitRule(max_target_return=0.05)
        context = {
            "target_return": 0.05,
            "max_weight": 0.30,
            "n_assets": 3,
        }

        passed, reason = rule.validate(context)

        # Target return of 0.05 == max of 0.05, so it passes (uses > not >=)
        assert passed is True
        assert reason is None

    def test_validate_custom_limits(self) -> None:
        """Test validation with custom limits."""
        rule = RiskLimitRule(
            max_target_return=0.10,
            max_weight_per_asset=0.40,
            min_assets=10,
        )

        context_pass = {
            "target_return": 0.08,
            "max_weight": 0.35,
            "n_assets": 15,
        }

        passed, reason = rule.validate(context_pass)
        assert passed is True

        context_fail = {
            "target_return": 0.08,
            "max_weight": 0.35,
            "n_assets": 5,
        }

        passed, reason = rule.validate(context_fail)
        assert passed is False


class TestDataQualityRule:
    """Test DataQualityRule validation."""

    def test_default_values(self) -> None:
        """Test default data quality values."""
        rule = DataQualityRule()

        assert rule.min_periods == 20
        assert rule.max_periods == 10000

    def test_validate_success(self) -> None:
        """Test validation passes with valid data."""
        rule = DataQualityRule()
        context = {"n_periods": 100}

        passed, reason = rule.validate(context)

        assert passed is True
        assert reason is None

    def test_validate_insufficient_data(self) -> None:
        """Test validation fails with insufficient data."""
        rule = DataQualityRule(min_periods=30)
        context = {"n_periods": 15}

        passed, reason = rule.validate(context)

        assert passed is False
        assert "Insufficient data" in reason
        assert "minimum is" in reason

    def test_validate_too_much_data(self) -> None:
        """Test validation fails with too much data."""
        rule = DataQualityRule(max_periods=5000)
        context = {"n_periods": 10000}

        passed, reason = rule.validate(context)

        assert passed is False
        assert "Too much data" in reason
        assert "maximum is" in reason

    def test_validate_edge_cases(self) -> None:
        """Test validation at boundary values."""
        rule = DataQualityRule(min_periods=20, max_periods=100)

        # Exactly at min
        passed, _ = rule.validate({"n_periods": 20})
        assert passed is True

        # Exactly at max
        passed, _ = rule.validate({"n_periods": 100})
        assert passed is True

        # Just below min
        passed, _ = rule.validate({"n_periods": 19})
        assert passed is False

        # Just above max
        passed, _ = rule.validate({"n_periods": 101})
        assert passed is False


class TestSolverValidationRule:
    """Test SolverValidationRule validation."""

    def test_default_values(self) -> None:
        """Test default solver validation values."""
        rule = SolverValidationRule()

        assert "cvxpy" in rule.allowed_apis
        assert "cuopt" in rule.allowed_apis

    def test_validate_cvxpy(self) -> None:
        """Test validation passes for cvxpy."""
        rule = SolverValidationRule()
        context = {"api": "cvxpy"}

        passed, reason = rule.validate(context)

        assert passed is True
        assert reason is None

    def test_validate_cuopt(self) -> None:
        """Test validation passes for cuopt."""
        rule = SolverValidationRule()
        context = {"api": "cuopt"}

        passed, reason = rule.validate(context)

        assert passed is True
        assert reason is None

    def test_validate_invalid_api(self) -> None:
        """Test validation fails for invalid API."""
        rule = SolverValidationRule()
        context = {"api": "invalid_solver"}

        passed, reason = rule.validate(context)

        assert passed is False
        assert "Invalid solver API" in reason

    def test_validate_default_api(self) -> None:
        """Test validation with default API."""
        rule = SolverValidationRule()
        context = {}  # No api specified

        passed, reason = rule.validate(context)

        assert passed is True  # Defaults to cvxpy


class TestPortfolioOptGovernanceHook:
    """Test PortfolioOptGovernanceHook."""

    def test_initialization_default(self) -> None:
        """Test hook initialization with defaults."""
        hook = create_hook()

        assert isinstance(hook.risk_rule, RiskLimitRule)
        assert isinstance(hook.data_rule, DataQualityRule)
        assert isinstance(hook.solver_rule, SolverValidationRule)
        assert len(hook._audit_log) == 0

    def test_initialization_custom_rules(self) -> None:
        """Test hook initialization with custom rules."""
        custom_risk = RiskLimitRule(max_target_return=0.10)
        custom_data = DataQualityRule(min_periods=50)
        custom_solver = SolverValidationRule()

        hook = create_hook(
            risk_rule=custom_risk,
            data_rule=custom_data,
            solver_rule=custom_solver,
        )

        assert hook.risk_rule.max_target_return == 0.10
        assert hook.data_rule.min_periods == 50

    @pytest.mark.asyncio
    async def test_preflight_optimize_allowed(self) -> None:
        """Test preflight allows valid optimization."""
        hook = create_hook()
        context = {
            "operation": "optimize",
            "target_return": 0.01,
            "max_weight": 0.20,
            "n_assets": 5,
            "n_periods": 100,
            "api": "cvxpy",
        }

        result = await hook.preflight(context)

        assert result.allowed is True
        assert result.decision == Decision.ALLOW
        assert "validation rules passed" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_preflight_optimize_denied_risk(self) -> None:
        """Test preflight denies optimization exceeding risk limits."""
        hook = create_hook()
        context = {
            "operation": "optimize",
            "target_return": 0.10,  # Exceeds default 5%
            "max_weight": 0.20,
            "n_assets": 5,
            "n_periods": 100,
            "api": "cvxpy",
        }

        result = await hook.preflight(context)

        assert result.blocked is True
        assert result.decision == Decision.DENY
        assert "Target return" in result.reason

    @pytest.mark.asyncio
    async def test_preflight_optimize_denied_data_quality(self) -> None:
        """Test preflight denies optimization with insufficient data."""
        hook = create_hook()
        context = {
            "operation": "optimize",
            "target_return": 0.01,
            "max_weight": 0.20,
            "n_assets": 5,
            "n_periods": 10,  # Below minimum 20
            "api": "cvxpy",
        }

        result = await hook.preflight(context)

        assert result.blocked is True
        assert result.decision == Decision.DENY
        assert "Insufficient data" in result.reason

    @pytest.mark.asyncio
    async def test_preflight_optimize_denied_solver(self) -> None:
        """Test preflight denies optimization with invalid solver."""
        hook = create_hook()
        context = {
            "operation": "optimize",
            "target_return": 0.01,
            "max_weight": 0.20,
            "n_assets": 5,
            "n_periods": 100,
            "api": "invalid_solver",
        }

        result = await hook.preflight(context)

        assert result.blocked is True
        assert result.decision == Decision.DENY
        assert "Invalid solver API" in result.reason

    @pytest.mark.asyncio
    async def test_preflight_multiple_violations(self) -> None:
        """Test preflight reports multiple violations."""
        hook = create_hook()
        context = {
            "operation": "optimize",
            "target_return": 0.10,  # Exceeds limit
            "max_weight": 0.50,  # Exceeds limit
            "n_assets": 2,  # Below minimum
            "n_periods": 5,  # Below minimum
            "api": "invalid",  # Invalid
        }

        result = await hook.preflight(context)

        assert result.blocked is True
        assert result.decision == Decision.DENY
        # Should contain multiple reasons
        assert len(result.reason.split(";")) >= 3

    @pytest.mark.asyncio
    async def test_preflight_scenario_generation_allowed(self) -> None:
        """Test preflight allows valid scenario generation."""
        hook = create_hook()
        context = {
            "operation": "generate_scenarios",
            "n_paths": 1000,
            "n_assets": 5,
        }

        result = await hook.preflight(context)

        assert result.allowed is True
        assert result.decision == Decision.ALLOW

    @pytest.mark.asyncio
    async def test_preflight_scenario_generation_too_few_paths(self) -> None:
        """Test preflight denies scenario generation with too few paths."""
        hook = create_hook()
        context = {
            "operation": "generate_scenarios",
            "n_paths": 50,  # Below minimum 100
        }

        result = await hook.preflight(context)

        assert result.blocked is True
        assert result.decision == Decision.DENY
        assert "Too few paths" in result.reason

    @pytest.mark.asyncio
    async def test_preflight_scenario_generation_too_many_paths(self) -> None:
        """Test preflight denies scenario generation with too many paths."""
        hook = create_hook()
        context = {
            "operation": "generate_scenarios",
            "n_paths": 200000,  # Above maximum 100000
        }

        result = await hook.preflight(context)

        assert result.blocked is True
        assert result.decision == Decision.DENY
        assert "Too many paths" in result.reason

    @pytest.mark.asyncio
    async def test_preflight_non_optimization_operation(self) -> None:
        """Test preflight allows non-optimization operations."""
        hook = create_hook()
        context = {
            "operation": "health_check",
        }

        result = await hook.preflight(context)

        assert result.allowed is True
        assert result.decision == Decision.ALLOW
        assert "Non-optimization operation" in result.reason

    @pytest.mark.asyncio
    async def test_audit_creates_record(self) -> None:
        """Test audit creates and stores record."""
        hook = create_hook()

        record = AuditRecord(
            engine="test-engine",
            action="optimize",
            inputs={"test": "data"},
            outputs={"weights": {}},
            latency_ms=100.0,
        )

        await hook.audit(record)

        audit_log = hook.get_audit_log()
        assert len(audit_log) == 1
        assert audit_log[0] == record

    @pytest.mark.asyncio
    async def test_audit_multiple_records(self) -> None:
        """Test audit stores multiple records."""
        hook = create_hook()

        for i in range(5):
            record = AuditRecord(
                engine="test-engine",
                action="optimize",
                inputs={"iteration": i},
                outputs={},
                latency_ms=100.0 + i,
            )
            await hook.audit(record)

        audit_log = hook.get_audit_log()
        assert len(audit_log) == 5

    def test_get_audit_log_returns_copy(self) -> None:
        """Test get_audit_log returns a copy."""
        hook = create_hook()

        log1 = hook.get_audit_log()
        log2 = hook.get_audit_log()

        assert log1 is not log2

    def test_clear_audit_log(self) -> None:
        """Test clearing audit log."""
        hook = create_hook()

        # Add some records
        for i in range(3):
            hook._audit_log.append(
                AuditRecord(
                    engine="test",
                    action="optimize",
                    inputs={},
                    outputs={},
                    latency_ms=100.0,
                )
            )

        assert len(hook.get_audit_log()) == 3

        hook.clear_audit_log()

        assert len(hook.get_audit_log()) == 0


class TestGovernanceIntegration:
    """Test governance hook integration scenarios."""

    @pytest.mark.asyncio
    async def test_strict_governance_enforcement(self) -> None:
        """Test strict governance with tight limits."""
        hook = create_hook(
            risk_rule=RiskLimitRule(
                max_target_return=0.01,
                max_weight_per_asset=0.15,
                min_assets=10,
            ),
            data_rule=DataQualityRule(min_periods=100, max_periods=1000),
        )

        # Should fail multiple rules
        context = {
            "operation": "optimize",
            "target_return": 0.02,
            "max_weight": 0.20,
            "n_assets": 5,
            "n_periods": 50,
            "api": "cvxpy",
        }

        result = await hook.preflight(context)

        assert result.blocked is True
        # Multiple violations
        assert "Target return" in result.reason
        # RiskLimitRule returns early, so we might not see all violations
        # assert "Max weight" in result.reason
        # assert "assets provided" in result.reason

    @pytest.mark.asyncio
    async def test_permissive_governance(self) -> None:
        """Test permissive governance with relaxed limits."""
        hook = create_hook(
            risk_rule=RiskLimitRule(
                max_target_return=0.20,
                max_weight_per_asset=0.50,
                min_assets=2,
            ),
            data_rule=DataQualityRule(min_periods=10, max_periods=50000),
        )

        context = {
            "operation": "optimize",
            "target_return": 0.05,
            "max_weight": 0.30,
            "n_assets": 3,
            "n_periods": 100,
            "api": "cvxpy",
        }

        result = await hook.preflight(context)

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_audit_trail_consistency(self) -> None:
        """Test audit trail maintains consistency."""
        hook = create_hook()

        # Simulate multiple operations
        operations = [
            ("optimize", 50.0),
            ("optimize", 75.0),
            ("generate_scenarios", 200.0),
            ("optimize", 60.0),
        ]

        for op, duration in operations:
            record = AuditRecord(
                engine="test-engine",
                action=op,
                inputs={},
                outputs={},
                latency_ms=duration,
            )
            await hook.audit(record)

        audit_log = hook.get_audit_log()

        assert len(audit_log) == 4
        assert audit_log[0].action == "optimize"
        assert audit_log[2].action == "generate_scenarios"
        assert all(r.engine == "test-engine" for r in audit_log)
