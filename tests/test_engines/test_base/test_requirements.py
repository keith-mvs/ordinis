"""Tests for requirements framework.

This module tests the requirement tracking, verification, and
compliance reporting functionality.
"""

from datetime import datetime

import pytest

from ordinis.engines.base import (
    Requirement,
    RequirementCategory,
    RequirementPriority,
    RequirementRegistry,
    RequirementStatus,
    RequirementVerification,
    verifies,
)


class TestRequirementEnums:
    """Test requirement enum types."""

    def test_requirement_category_values(self) -> None:
        """Test RequirementCategory enum values."""
        assert RequirementCategory.FUNC.value == "functional"
        assert RequirementCategory.PERF.value == "performance"
        assert RequirementCategory.SEC.value == "security"
        assert RequirementCategory.REL.value == "reliability"
        assert RequirementCategory.GOV.value == "governance"
        assert RequirementCategory.INT.value == "integration"

    def test_requirement_priority_values(self) -> None:
        """Test RequirementPriority enum values."""
        assert RequirementPriority.CRITICAL.value == "critical"
        assert RequirementPriority.HIGH.value == "high"
        assert RequirementPriority.MEDIUM.value == "medium"
        assert RequirementPriority.LOW.value == "low"

    def test_requirement_status_values(self) -> None:
        """Test RequirementStatus enum values."""
        assert RequirementStatus.DRAFT.value == "draft"
        assert RequirementStatus.APPROVED.value == "approved"
        assert RequirementStatus.IN_PROGRESS.value == "in_progress"
        assert RequirementStatus.IMPLEMENTED.value == "implemented"
        assert RequirementStatus.VERIFIED.value == "verified"
        assert RequirementStatus.DEFERRED.value == "deferred"
        assert RequirementStatus.REJECTED.value == "rejected"


class TestRequirement:
    """Test Requirement dataclass."""

    def test_minimal_requirement(self) -> None:
        """Test creating requirement with minimal fields."""
        req = Requirement(
            id="TEST-FUNC-001",
            title="Test requirement",
            description="A test requirement",
            category=RequirementCategory.FUNC,
        )

        assert req.id == "TEST-FUNC-001"
        assert req.title == "Test requirement"
        assert req.description == "A test requirement"
        assert req.category == RequirementCategory.FUNC
        assert req.priority == RequirementPriority.MEDIUM
        assert req.status == RequirementStatus.DRAFT
        assert req.acceptance_criteria == []
        assert req.test_ids == []
        assert req.dependencies == []
        assert req.version == "1.0.0"

    def test_full_requirement(self) -> None:
        """Test creating requirement with all fields."""
        created = datetime.utcnow()
        req = Requirement(
            id="TEST-FUNC-001",
            title="Test requirement",
            description="A test requirement",
            category=RequirementCategory.FUNC,
            priority=RequirementPriority.HIGH,
            status=RequirementStatus.IMPLEMENTED,
            acceptance_criteria=["Criterion 1", "Criterion 2"],
            test_ids=["test_1", "test_2"],
            dependencies=["TEST-FUNC-000"],
            version="2.0.0",
            created_at=created,
            created_by="test_user",
            rationale="Test rationale",
            references=["DOC-001"],
        )

        assert req.priority == RequirementPriority.HIGH
        assert req.status == RequirementStatus.IMPLEMENTED
        assert len(req.acceptance_criteria) == 2
        assert len(req.test_ids) == 2
        assert len(req.dependencies) == 1
        assert req.version == "2.0.0"
        assert req.created_by == "test_user"
        assert req.rationale == "Test rationale"

    def test_requirement_id_validation(self) -> None:
        """Test requirement ID format validation."""
        with pytest.raises(ValueError, match="Invalid requirement ID format"):
            Requirement(
                id="INVALID-ID",
                title="Test",
                description="Test",
                category=RequirementCategory.FUNC,
            )

        with pytest.raises(ValueError, match="Invalid requirement ID format"):
            Requirement(
                id="TEST-FUNC-001-EXTRA",
                title="Test",
                description="Test",
                category=RequirementCategory.FUNC,
            )

    def test_requirement_engine_property(self) -> None:
        """Test extracting engine name from requirement ID."""
        req = Requirement(
            id="SIGNAL-FUNC-001",
            title="Test",
            description="Test",
            category=RequirementCategory.FUNC,
        )

        assert req.engine == "SIGNAL"

    def test_requirement_is_testable(self) -> None:
        """Test is_testable property."""
        req_no_tests = Requirement(
            id="TEST-FUNC-001",
            title="Test",
            description="Test",
            category=RequirementCategory.FUNC,
        )
        assert not req_no_tests.is_testable

        req_with_tests = Requirement(
            id="TEST-FUNC-002",
            title="Test",
            description="Test",
            category=RequirementCategory.FUNC,
            test_ids=["test_1"],
        )
        assert req_with_tests.is_testable

    def test_requirement_is_verified(self) -> None:
        """Test is_verified property."""
        req_draft = Requirement(
            id="TEST-FUNC-001",
            title="Test",
            description="Test",
            category=RequirementCategory.FUNC,
            status=RequirementStatus.DRAFT,
        )
        assert not req_draft.is_verified

        req_verified = Requirement(
            id="TEST-FUNC-002",
            title="Test",
            description="Test",
            category=RequirementCategory.FUNC,
            status=RequirementStatus.VERIFIED,
        )
        assert req_verified.is_verified


class TestRequirementVerification:
    """Test RequirementVerification dataclass."""

    def test_minimal_verification(self) -> None:
        """Test creating verification with minimal fields."""
        verification = RequirementVerification(
            requirement_id="TEST-FUNC-001",
            passed=True,
        )

        assert verification.requirement_id == "TEST-FUNC-001"
        assert verification.passed is True
        assert verification.test_results == {}
        assert verification.verified_by == ""
        assert verification.evidence == []
        assert verification.notes == ""

    def test_full_verification(self) -> None:
        """Test creating verification with all fields."""
        verified = datetime.utcnow()
        verification = RequirementVerification(
            requirement_id="TEST-FUNC-001",
            passed=True,
            test_results={"test_1": True, "test_2": True},
            verified_at=verified,
            verified_by="CI",
            evidence=["log1.txt", "log2.txt"],
            notes="All tests passed",
        )

        assert verification.passed is True
        assert len(verification.test_results) == 2
        assert verification.verified_by == "CI"
        assert len(verification.evidence) == 2
        assert verification.notes == "All tests passed"

    def test_verification_coverage(self) -> None:
        """Test coverage calculation."""
        verification_no_tests = RequirementVerification(
            requirement_id="TEST-FUNC-001",
            passed=True,
        )
        assert verification_no_tests.coverage == 0.0

        verification_all_pass = RequirementVerification(
            requirement_id="TEST-FUNC-001",
            passed=True,
            test_results={"test_1": True, "test_2": True, "test_3": True},
        )
        assert verification_all_pass.coverage == 100.0

        verification_partial = RequirementVerification(
            requirement_id="TEST-FUNC-001",
            passed=False,
            test_results={"test_1": True, "test_2": False},
        )
        assert verification_partial.coverage == 50.0


class TestRequirementRegistry:
    """Test RequirementRegistry."""

    def test_registry_initialization(self) -> None:
        """Test registry initialization."""
        registry = RequirementRegistry("SIGNAL")

        assert registry.engine_name == "SIGNAL"

    def test_register_requirement(self) -> None:
        """Test registering a requirement."""
        registry = RequirementRegistry("SIGNAL")
        req = Requirement(
            id="SIGNAL-FUNC-001",
            title="Test",
            description="Test",
            category=RequirementCategory.FUNC,
        )

        registry.register(req)

        assert registry.get("SIGNAL-FUNC-001") == req

    def test_register_duplicate_requirement(self) -> None:
        """Test registering duplicate requirement raises error."""
        registry = RequirementRegistry("SIGNAL")
        req = Requirement(
            id="SIGNAL-FUNC-001",
            title="Test",
            description="Test",
            category=RequirementCategory.FUNC,
        )

        registry.register(req)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(req)

    def test_register_wrong_engine(self) -> None:
        """Test registering requirement for wrong engine."""
        registry = RequirementRegistry("SIGNAL")
        req = Requirement(
            id="RISK-FUNC-001",
            title="Test",
            description="Test",
            category=RequirementCategory.FUNC,
        )

        with pytest.raises(ValueError, match="belongs to RISK, not SIGNAL"):
            registry.register(req)

    def test_create_requirement(self) -> None:
        """Test creating requirement with auto-generated ID."""
        registry = RequirementRegistry("SIGNAL")

        req = registry.create(
            category=RequirementCategory.FUNC,
            title="Test requirement",
            description="Test description",
        )

        assert req.id == "SIGNAL-FUNC-001"
        assert req.title == "Test requirement"
        assert registry.get("SIGNAL-FUNC-001") == req

    def test_create_multiple_requirements(self) -> None:
        """Test creating multiple requirements increments counter."""
        registry = RequirementRegistry("SIGNAL")

        req1 = registry.create(
            category=RequirementCategory.FUNC,
            title="Req 1",
            description="Desc 1",
        )
        req2 = registry.create(
            category=RequirementCategory.FUNC,
            title="Req 2",
            description="Desc 2",
        )

        assert req1.id == "SIGNAL-FUNC-001"
        assert req2.id == "SIGNAL-FUNC-002"

    def test_create_different_categories(self) -> None:
        """Test creating requirements in different categories."""
        registry = RequirementRegistry("SIGNAL")

        func_req = registry.create(
            category=RequirementCategory.FUNC,
            title="Func",
            description="Func",
        )
        perf_req = registry.create(
            category=RequirementCategory.PERF,
            title="Perf",
            description="Perf",
        )

        assert func_req.id == "SIGNAL-FUNC-001"
        assert perf_req.id == "SIGNAL-PERF-001"

    def test_get_nonexistent_requirement(self) -> None:
        """Test getting nonexistent requirement returns None."""
        registry = RequirementRegistry("SIGNAL")

        assert registry.get("SIGNAL-FUNC-999") is None

    def test_list_by_category(self) -> None:
        """Test listing requirements by category."""
        registry = RequirementRegistry("SIGNAL")

        registry.create(
            category=RequirementCategory.FUNC,
            title="Func 1",
            description="Desc",
        )
        registry.create(
            category=RequirementCategory.FUNC,
            title="Func 2",
            description="Desc",
        )
        registry.create(
            category=RequirementCategory.PERF,
            title="Perf 1",
            description="Desc",
        )

        func_reqs = registry.list_by_category(RequirementCategory.FUNC)
        perf_reqs = registry.list_by_category(RequirementCategory.PERF)

        assert len(func_reqs) == 2
        assert len(perf_reqs) == 1

    def test_list_by_status(self) -> None:
        """Test listing requirements by status."""
        registry = RequirementRegistry("SIGNAL")

        registry.create(
            category=RequirementCategory.FUNC,
            title="Draft",
            description="Desc",
            status=RequirementStatus.DRAFT,
        )
        registry.create(
            category=RequirementCategory.FUNC,
            title="Verified",
            description="Desc",
            status=RequirementStatus.VERIFIED,
        )

        draft_reqs = registry.list_by_status(RequirementStatus.DRAFT)
        verified_reqs = registry.list_by_status(RequirementStatus.VERIFIED)

        assert len(draft_reqs) == 1
        assert len(verified_reqs) == 1

    def test_list_by_priority(self) -> None:
        """Test listing requirements by priority."""
        registry = RequirementRegistry("SIGNAL")

        registry.create(
            category=RequirementCategory.FUNC,
            title="High",
            description="Desc",
            priority=RequirementPriority.HIGH,
        )
        registry.create(
            category=RequirementCategory.FUNC,
            title="Low",
            description="Desc",
            priority=RequirementPriority.LOW,
        )

        high_reqs = registry.list_by_priority(RequirementPriority.HIGH)
        low_reqs = registry.list_by_priority(RequirementPriority.LOW)

        assert len(high_reqs) == 1
        assert len(low_reqs) == 1

    def test_record_verification(self) -> None:
        """Test recording verification result."""
        registry = RequirementRegistry("SIGNAL")
        req = registry.create(
            category=RequirementCategory.FUNC,
            title="Test",
            description="Desc",
        )

        verification = RequirementVerification(
            requirement_id=req.id,
            passed=True,
        )
        registry.record_verification(verification)

        verifications = registry.get_verifications(req.id)
        assert len(verifications) == 1
        assert verifications[0].passed is True

    def test_record_verification_updates_status(self) -> None:
        """Test recording verification updates requirement status."""
        registry = RequirementRegistry("SIGNAL")
        req = registry.create(
            category=RequirementCategory.FUNC,
            title="Test",
            description="Desc",
            status=RequirementStatus.IMPLEMENTED,
        )

        verification = RequirementVerification(
            requirement_id=req.id,
            passed=True,
        )
        registry.record_verification(verification)

        updated_req = registry.get(req.id)
        assert updated_req is not None
        assert updated_req.status == RequirementStatus.VERIFIED

    def test_get_latest_verification(self) -> None:
        """Test getting latest verification."""
        registry = RequirementRegistry("SIGNAL")
        req = registry.create(
            category=RequirementCategory.FUNC,
            title="Test",
            description="Desc",
        )

        v1 = RequirementVerification(
            requirement_id=req.id,
            passed=False,
            verified_at=datetime(2024, 1, 1),
        )
        v2 = RequirementVerification(
            requirement_id=req.id,
            passed=True,
            verified_at=datetime(2024, 1, 2),
        )

        registry.record_verification(v1)
        registry.record_verification(v2)

        latest = registry.get_latest_verification(req.id)
        assert latest is not None
        assert latest.passed is True

    def test_get_latest_verification_no_verifications(self) -> None:
        """Test getting latest verification when none exist."""
        registry = RequirementRegistry("SIGNAL")

        latest = registry.get_latest_verification("SIGNAL-FUNC-999")
        assert latest is None

    def test_compliance_report_empty(self) -> None:
        """Test compliance report for empty registry."""
        registry = RequirementRegistry("SIGNAL")

        report = registry.compliance_report()

        assert report["total"] == 0
        assert report["coverage"] == 0.0
        assert report["verified"] == 0

    def test_compliance_report(self) -> None:
        """Test compliance report generation."""
        registry = RequirementRegistry("SIGNAL")

        req1 = registry.create(
            category=RequirementCategory.FUNC,
            title="Req 1",
            description="Desc",
            status=RequirementStatus.VERIFIED,
            test_ids=["test_1"],
        )
        req2 = registry.create(
            category=RequirementCategory.FUNC,
            title="Req 2",
            description="Desc",
            status=RequirementStatus.IMPLEMENTED,
            test_ids=["test_2"],
        )
        registry.create(
            category=RequirementCategory.PERF,
            title="Req 3",
            description="Desc",
            status=RequirementStatus.DRAFT,
        )

        report = registry.compliance_report()

        assert report["engine"] == "SIGNAL"
        assert report["total"] == 3
        assert report["verified"] == 1
        assert report["implemented"] == 1
        assert report["testable"] == 2
        assert report["coverage_pct"] == pytest.approx(33.33, rel=0.1)
        assert report["testability_pct"] == pytest.approx(66.67, rel=0.1)
        assert report["by_category"]["FUNC"] == 2
        assert report["by_category"]["PERF"] == 1

    def test_to_dict(self) -> None:
        """Test exporting registry to dictionary."""
        registry = RequirementRegistry("SIGNAL")

        registry.create(
            category=RequirementCategory.FUNC,
            title="Req 1",
            description="Desc",
        )

        data = registry.to_dict()

        assert data["engine"] == "SIGNAL"
        assert "requirements" in data
        assert "SIGNAL-FUNC-001" in data["requirements"]
        assert data["requirements"]["SIGNAL-FUNC-001"]["title"] == "Req 1"


class TestVerifiesDecorator:
    """Test @verifies decorator."""

    def test_decorator_adds_metadata(self) -> None:
        """Test decorator adds requirement metadata to function."""

        @verifies("TEST-FUNC-001", "TEST-FUNC-002")
        def test_function():
            pass

        assert hasattr(test_function, "_verifies_requirements")
        assert test_function._verifies_requirements == ("TEST-FUNC-001", "TEST-FUNC-002")

    def test_decorator_single_requirement(self) -> None:
        """Test decorator with single requirement."""

        @verifies("TEST-FUNC-001")
        def test_function():
            pass

        assert test_function._verifies_requirements == ("TEST-FUNC-001",)

    def test_decorator_preserves_function(self) -> None:
        """Test decorator preserves function behavior."""

        @verifies("TEST-FUNC-001")
        def test_function():
            return "result"

        assert test_function() == "result"

    def test_decorator_with_pytest_test(self) -> None:
        """Test decorator works with pytest test functions."""

        @verifies("TEST-FUNC-001")
        def test_example():
            assert True

        test_example()
        assert hasattr(test_example, "_verifies_requirements")
