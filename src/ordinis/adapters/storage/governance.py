"""
Storage governance hooks.

Implements governance rules for storage operations including:
- Path validation (ensuring storage is within allowed directories)
- Database initialization preflight checks
- ChromaDB path validation
- Audit logging for storage operations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from ordinis.engines.base import (
    AuditRecord,
    Decision,
    GovernanceHook,
    PreflightContext,
    PreflightResult,
)

_logger = logging.getLogger(__name__)


class StorageRule(ABC):
    """Abstract base class for storage governance rules."""

    @abstractmethod
    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check if rule passes.

        Args:
            context: Preflight context with operation details

        Returns:
            Tuple of (passed: bool, reason: str)
        """
        ...


@dataclass
class PathValidationRule(StorageRule):
    """Rule enforcing storage path restrictions.

    Attributes:
        allowed_base_dirs: Base directories where storage is allowed
        project_root: Project root directory for relative path resolution
    """

    allowed_base_dirs: list[str] | None = None
    project_root: Path | None = None

    def __post_init__(self):
        """Initialize default allowed directories."""
        if self.allowed_base_dirs is None:
            self.allowed_base_dirs = ["data/", "artifacts/", "tests/fixtures/"]
        if self.project_root is None:
            # Try to detect project root
            self.project_root = Path(__file__).parent.parent.parent.parent

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Validate storage paths are within allowed directories.

        Args:
            context: Preflight context with path parameter

        Returns:
            Tuple of (passed, reason)
        """
        params = context.parameters
        path_str = params.get("path", params.get("db_path", ""))

        if not path_str:
            return True, "No path specified, skipping validation"

        # Convert to Path and resolve
        try:
            path = Path(path_str)
            if not path.is_absolute():
                path = self.project_root / path
            path = path.resolve()
        except Exception as e:
            return False, f"Invalid path: {e}"

        # Check if path is within allowed directories
        for allowed_dir in self.allowed_base_dirs:
            allowed_path = (self.project_root / allowed_dir).resolve()
            try:
                path.relative_to(allowed_path)
                return True, f"Path validated: {path} is within {allowed_dir}"
            except ValueError:
                continue

        return False, (
            f"Path {path} is not within allowed directories: {self.allowed_base_dirs}"
        )


@dataclass
class DatabaseIntegrityRule(StorageRule):
    """Rule enforcing database integrity requirements.

    Attributes:
        require_wal_mode: Require WAL journal mode
        require_foreign_keys: Require foreign key enforcement
    """

    require_wal_mode: bool = True
    require_foreign_keys: bool = True

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Check database configuration requirements.

        Args:
            context: Preflight context

        Returns:
            Tuple of (passed, reason)
        """
        params = context.parameters
        journal_mode = params.get("journal_mode", "WAL")
        foreign_keys = params.get("foreign_keys", True)

        if self.require_wal_mode and journal_mode != "WAL":
            return False, f"WAL mode required, got: {journal_mode}"

        if self.require_foreign_keys and not foreign_keys:
            return False, "Foreign key enforcement required"

        return True, "Database configuration validated"


@dataclass
class ChromaDBPathRule(StorageRule):
    """Rule enforcing ChromaDB storage path restrictions.

    Attributes:
        required_base_path: Required base directory for ChromaDB
        disallow_root_storage: Disallow ChromaDB in project root
    """

    required_base_path: str = "data/chromadb"
    disallow_root_storage: bool = True

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Validate ChromaDB path configuration.

        Args:
            context: Preflight context with chroma_path parameter

        Returns:
            Tuple of (passed, reason)
        """
        params = context.parameters
        chroma_path = params.get("chroma_path", "")

        if not chroma_path:
            # No ChromaDB path specified
            return True, "No ChromaDB path to validate"

        path = Path(chroma_path)

        # Check for root storage
        if self.disallow_root_storage:
            if path.name == "chroma.sqlite3" and len(path.parts) <= 2:
                return False, "ChromaDB storage in project root is not allowed"

        # Check for correct base path
        if self.required_base_path not in str(path):
            return False, (
                f"ChromaDB must be stored in {self.required_base_path}, got: {path}"
            )

        return True, f"ChromaDB path validated: {path}"


@dataclass
class BackupValidationRule(StorageRule):
    """Rule enforcing backup requirements.

    Attributes:
        max_backup_age_days: Maximum age for backups before cleanup
        min_backup_count: Minimum number of backups to retain
    """

    max_backup_age_days: int = 30
    min_backup_count: int = 3

    def check(self, context: PreflightContext) -> tuple[bool, str]:
        """Validate backup configuration.

        Args:
            context: Preflight context

        Returns:
            Tuple of (passed, reason)
        """
        params = context.parameters
        keep_count = params.get("keep_backups", 10)

        if keep_count < self.min_backup_count:
            return False, (
                f"Backup count {keep_count} below minimum {self.min_backup_count}"
            )

        return True, f"Backup configuration validated (keeping {keep_count} backups)"


class StorageGovernanceHook(GovernanceHook):
    """Governance hook for storage operations.

    Implements preflight checks and audit logging for all storage operations
    including database initialization, ChromaDB setup, and backup operations.

    Example:
        >>> from ordinis.adapters.storage.governance import (
        ...     StorageGovernanceHook,
        ...     PathValidationRule,
        ...     DatabaseIntegrityRule,
        ... )
        >>> hook = StorageGovernanceHook(
        ...     rules=[
        ...         PathValidationRule(),
        ...         DatabaseIntegrityRule(),
        ...     ]
        ... )
        >>> # Use with DatabaseManager or other storage components
    """

    def __init__(
        self,
        rules: list[StorageRule] | None = None,
        audit_all_operations: bool = True,
    ) -> None:
        """Initialize storage governance hook.

        Args:
            rules: List of governance rules to enforce
            audit_all_operations: Whether to audit all operations (not just failures)
        """
        self._rules = rules or [
            PathValidationRule(),
            DatabaseIntegrityRule(),
            ChromaDBPathRule(),
            BackupValidationRule(),
        ]
        self._audit_all = audit_all_operations
        self._audit_log: list[AuditRecord] = []

    async def preflight(self, context: PreflightContext) -> PreflightResult:
        """Check all governance rules before storage operation.

        Args:
            context: Operation context with parameters

        Returns:
            PreflightResult indicating allow/deny and reason
        """
        _logger.debug(
            "Storage governance preflight: %s",
            context.operation,
        )

        failed_rules: list[str] = []
        warnings: list[str] = []
        modifications: dict[str, Any] = {}

        for rule in self._rules:
            try:
                passed, reason = rule.check(context)
                if not passed:
                    failed_rules.append(f"{rule.__class__.__name__}: {reason}")
                else:
                    # Log successful checks at debug level
                    _logger.debug("Rule passed: %s", reason)
            except Exception as e:
                _logger.error("Rule %s failed with exception: %s", rule.__class__.__name__, e)
                failed_rules.append(f"{rule.__class__.__name__}: Error - {e}")

        if failed_rules:
            return PreflightResult(
                decision=Decision.DENY,
                allowed=False,
                reason="; ".join(failed_rules),
            )

        return PreflightResult(
            decision=Decision.ALLOW,
            allowed=True,
            reason="All storage governance rules passed",
            modifications=modifications if modifications else None,
            warnings=warnings if warnings else None,
        )

    async def audit(self, record: AuditRecord) -> None:
        """Record audit entry for storage operation.

        Args:
            record: Audit record to log
        """
        if self._audit_all or record.status in ("blocked", "error"):
            self._audit_log.append(record)
            _logger.info(
                "Storage audit: %s - %s (%s)",
                record.operation,
                record.status,
                record.details,
            )

    def get_audit_log(self, limit: int | None = None) -> list[AuditRecord]:
        """Get audit log entries.

        Args:
            limit: Maximum entries to return (None = all)

        Returns:
            List of audit records
        """
        if limit:
            return self._audit_log[-limit:]
        return self._audit_log.copy()

    def clear_audit_log(self) -> None:
        """Clear the audit log."""
        self._audit_log.clear()

    def add_rule(self, rule: StorageRule) -> None:
        """Add a governance rule.

        Args:
            rule: Rule to add
        """
        self._rules.append(rule)

    def remove_rule(self, rule_type: type[StorageRule]) -> bool:
        """Remove rules of a specific type.

        Args:
            rule_type: Type of rule to remove

        Returns:
            True if any rules were removed
        """
        original_count = len(self._rules)
        self._rules = [r for r in self._rules if not isinstance(r, rule_type)]
        return len(self._rules) < original_count

    def get_rules(self) -> list[StorageRule]:
        """Get all registered rules.

        Returns:
            List of governance rules
        """
        return self._rules.copy()
