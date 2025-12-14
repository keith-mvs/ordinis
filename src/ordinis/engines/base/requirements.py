"""Engine requirements framework for traceable, verifiable, testable specifications.

This module provides a structured way to define, track, and verify
engine requirements with unique IDs for audit and compliance.

Requirements follow the pattern: {ENGINE}-{CATEGORY}-{NUMBER}
Example: SIGNAL-FUNC-001, RISK-PERF-003, EXEC-SEC-002

Categories:
- FUNC: Functional requirements (what it does)
- PERF: Performance requirements (how fast/efficient)
- SEC: Security requirements (access, data protection)
- REL: Reliability requirements (uptime, recovery)
- GOV: Governance requirements (audit, compliance)
- INT: Integration requirements (interfaces, dependencies)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RequirementCategory(Enum):
    """Requirement categories for classification."""

    FUNC = "functional"
    PERF = "performance"
    SEC = "security"
    REL = "reliability"
    GOV = "governance"
    INT = "integration"


class RequirementPriority(Enum):
    """Requirement priority levels."""

    CRITICAL = "critical"  # Must have, blocks release
    HIGH = "high"  # Should have, significant impact
    MEDIUM = "medium"  # Nice to have, moderate impact
    LOW = "low"  # Optional, minor impact


class RequirementStatus(Enum):
    """Requirement implementation status."""

    DRAFT = "draft"  # Not yet approved
    APPROVED = "approved"  # Approved, not implemented
    IN_PROGRESS = "in_progress"  # Implementation started
    IMPLEMENTED = "implemented"  # Code complete
    VERIFIED = "verified"  # Tests passing
    DEFERRED = "deferred"  # Postponed
    REJECTED = "rejected"  # Will not implement


@dataclass
class Requirement:
    """A traceable, verifiable requirement specification.

    Attributes:
        id: Unique requirement ID (e.g., SIGNAL-FUNC-001).
        title: Short descriptive title.
        description: Detailed requirement description.
        category: Requirement category.
        priority: Priority level.
        status: Implementation status.
        acceptance_criteria: List of testable acceptance criteria.
        test_ids: Associated test case IDs.
        dependencies: IDs of dependent requirements.
        version: Requirement version.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        created_by: Author identifier.
        rationale: Why this requirement exists.
        references: External references (docs, tickets, etc.).
    """

    id: str
    title: str
    description: str
    category: RequirementCategory
    priority: RequirementPriority = RequirementPriority.MEDIUM
    status: RequirementStatus = RequirementStatus.DRAFT
    acceptance_criteria: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = ""
    rationale: str = ""
    references: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate requirement ID format."""
        parts = self.id.split("-")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid requirement ID format: {self.id}. "
                "Expected: ENGINE-CATEGORY-NUMBER (e.g., SIGNAL-FUNC-001)"
            )

    @property
    def engine(self) -> str:
        """Extract engine name from ID."""
        return self.id.split("-")[0]

    @property
    def is_testable(self) -> bool:
        """Check if requirement has associated tests."""
        return len(self.test_ids) > 0

    @property
    def is_verified(self) -> bool:
        """Check if requirement is verified."""
        return self.status == RequirementStatus.VERIFIED


@dataclass
class RequirementVerification:
    """Verification result for a requirement.

    Attributes:
        requirement_id: ID of the verified requirement.
        passed: Whether verification passed.
        test_results: Results from each test.
        verified_at: Verification timestamp.
        verified_by: Verifier identifier (human or CI).
        evidence: Supporting evidence (logs, screenshots, etc.).
        notes: Additional verification notes.
    """

    requirement_id: str
    passed: bool
    test_results: dict[str, bool] = field(default_factory=dict)
    verified_at: datetime = field(default_factory=datetime.utcnow)
    verified_by: str = ""
    evidence: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def coverage(self) -> float:
        """Calculate test coverage percentage."""
        if not self.test_results:
            return 0.0
        passed = sum(1 for v in self.test_results.values() if v)
        return (passed / len(self.test_results)) * 100


class RequirementRegistry:
    """Registry for managing engine requirements.

    Provides centralized storage and querying of requirements
    with verification tracking.
    """

    def __init__(self, engine_name: str) -> None:
        """Initialize registry for an engine.

        Args:
            engine_name: Name of the engine (e.g., SIGNAL, RISK).
        """
        self.engine_name = engine_name.upper()
        self._requirements: dict[str, Requirement] = {}
        self._verifications: dict[str, list[RequirementVerification]] = {}
        self._counters: dict[str, int] = {}

    def register(self, requirement: Requirement) -> None:
        """Register a requirement.

        Args:
            requirement: The requirement to register.

        Raises:
            ValueError: If requirement ID already exists or wrong engine.
        """
        if requirement.engine != self.engine_name:
            raise ValueError(
                f"Requirement {requirement.id} belongs to {requirement.engine}, "
                f"not {self.engine_name}"
            )
        if requirement.id in self._requirements:
            raise ValueError(f"Requirement {requirement.id} already registered")
        self._requirements[requirement.id] = requirement

    def create(
        self,
        category: RequirementCategory,
        title: str,
        description: str,
        **kwargs: Any,
    ) -> Requirement:
        """Create and register a new requirement with auto-generated ID.

        Args:
            category: Requirement category.
            title: Short title.
            description: Detailed description.
            **kwargs: Additional requirement attributes.

        Returns:
            The created and registered requirement.
        """
        cat_key = category.name
        self._counters.setdefault(cat_key, 0)
        self._counters[cat_key] += 1
        req_id = f"{self.engine_name}-{cat_key}-{self._counters[cat_key]:03d}"

        requirement = Requirement(
            id=req_id,
            title=title,
            description=description,
            category=category,
            **kwargs,
        )
        self.register(requirement)
        return requirement

    def get(self, req_id: str) -> Requirement | None:
        """Get a requirement by ID."""
        return self._requirements.get(req_id)

    def list_by_category(self, category: RequirementCategory) -> list[Requirement]:
        """List requirements by category."""
        return [r for r in self._requirements.values() if r.category == category]

    def list_by_status(self, status: RequirementStatus) -> list[Requirement]:
        """List requirements by status."""
        return [r for r in self._requirements.values() if r.status == status]

    def list_by_priority(self, priority: RequirementPriority) -> list[Requirement]:
        """List requirements by priority."""
        return [r for r in self._requirements.values() if r.priority == priority]

    def record_verification(self, verification: RequirementVerification) -> None:
        """Record a verification result.

        Args:
            verification: The verification result to record.
        """
        req_id = verification.requirement_id
        if req_id not in self._verifications:
            self._verifications[req_id] = []
        self._verifications[req_id].append(verification)

        # Update requirement status if verified
        if verification.passed and req_id in self._requirements:
            self._requirements[req_id].status = RequirementStatus.VERIFIED
            self._requirements[req_id].updated_at = datetime.utcnow()

    def get_verifications(self, req_id: str) -> list[RequirementVerification]:
        """Get all verifications for a requirement."""
        return self._verifications.get(req_id, [])

    def get_latest_verification(self, req_id: str) -> RequirementVerification | None:
        """Get the most recent verification for a requirement."""
        verifications = self._verifications.get(req_id, [])
        if not verifications:
            return None
        return max(verifications, key=lambda v: v.verified_at)

    def compliance_report(self) -> dict[str, Any]:
        """Generate a compliance report for all requirements.

        Returns:
            Dictionary with compliance statistics and details.
        """
        total = len(self._requirements)
        if total == 0:
            return {"total": 0, "coverage": 0.0, "verified": 0}

        verified = len(self.list_by_status(RequirementStatus.VERIFIED))
        implemented = len(self.list_by_status(RequirementStatus.IMPLEMENTED))
        testable = sum(1 for r in self._requirements.values() if r.is_testable)

        by_category = {cat.name: len(self.list_by_category(cat)) for cat in RequirementCategory}
        by_priority = {pri.name: len(self.list_by_priority(pri)) for pri in RequirementPriority}
        by_status = {status.name: len(self.list_by_status(status)) for status in RequirementStatus}

        return {
            "engine": self.engine_name,
            "total": total,
            "verified": verified,
            "implemented": implemented,
            "testable": testable,
            "coverage_pct": (verified / total) * 100 if total > 0 else 0.0,
            "testability_pct": (testable / total) * 100 if total > 0 else 0.0,
            "by_category": by_category,
            "by_priority": by_priority,
            "by_status": by_status,
            "generated_at": datetime.utcnow().isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Export all requirements to dictionary."""
        return {
            "engine": self.engine_name,
            "requirements": {
                req_id: {
                    "id": req.id,
                    "title": req.title,
                    "description": req.description,
                    "category": req.category.name,
                    "priority": req.priority.name,
                    "status": req.status.name,
                    "acceptance_criteria": req.acceptance_criteria,
                    "test_ids": req.test_ids,
                    "dependencies": req.dependencies,
                    "version": req.version,
                }
                for req_id, req in self._requirements.items()
            },
        }


# Decorator for linking tests to requirements
def verifies(*requirement_ids: str) -> Callable:
    """Decorator to link a test function to requirements.

    Usage:
        @verifies("SIGNAL-FUNC-001", "SIGNAL-FUNC-002")
        def test_signal_generation():
            ...

    Args:
        *requirement_ids: Requirement IDs that this test verifies.

    Returns:
        Decorated function with requirement metadata.
    """

    def decorator(func: Callable) -> Callable:
        func._verifies_requirements = requirement_ids  # type: ignore
        return func

    return decorator
