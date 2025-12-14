"""
Governance engine configuration.

Provides GovernanceEngineConfig extending BaseEngineConfig for standardized
engine configuration with governance-specific settings.
"""

from dataclasses import dataclass, field

from ordinis.engines.base import BaseEngineConfig


@dataclass
class GovernanceEngineConfig(BaseEngineConfig):
    """Configuration for Governance Engine.

    Extends BaseEngineConfig with governance-specific settings including
    audit, PPI, ethics, and policy enforcement configuration.

    Attributes:
        engine_id: Unique identifier (default: "governance")
        engine_name: Display name (default: "Governance Engine")
        enable_audit: Enable audit trail logging
        enable_ppi_scanning: Enable PPI detection
        enable_ethics_checks: Enable ethics compliance checks
        enable_default_policies: Load default trading policies
        max_position_size_pct: Default max position size as percentage
        daily_trade_limit: Default daily trade limit
        max_drawdown_pct: Maximum drawdown threshold
        require_human_approval_above: Trade value requiring approval
        audit_chain_enabled: Enable hash-chained audit trail
        ppi_auto_mask: Automatically mask detected PPI
        ethics_esg_threshold: Minimum ESG score for investments
        session_id: Session identifier for audit grouping
    """

    engine_id: str = "governance"
    engine_name: str = "Governance Engine"

    # Feature flags
    enable_audit: bool = True
    enable_ppi_scanning: bool = True
    enable_ethics_checks: bool = True
    enable_default_policies: bool = True

    # Trading policy defaults
    max_position_size_pct: float = 0.10
    daily_trade_limit: int = 100
    max_drawdown_pct: float = -0.10
    require_human_approval_above: float = 100000.0

    # Audit configuration
    audit_chain_enabled: bool = True

    # PPI configuration
    ppi_auto_mask: bool = True

    # Ethics configuration
    ethics_esg_threshold: float = 40.0

    # Session tracking
    session_id: str | None = None

    # Sub-engine configuration
    sub_engines: list[str] = field(
        default_factory=lambda: ["audit", "ppi", "ethics", "broker_compliance"]
    )

    def validate(self) -> list[str]:
        """Validate governance engine configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = super().validate()

        if not 0.0 < self.max_position_size_pct <= 1.0:
            errors.append("max_position_size_pct must be between 0 and 1")

        if self.daily_trade_limit < 1:
            errors.append("daily_trade_limit must be at least 1")

        if self.max_drawdown_pct > 0:
            errors.append("max_drawdown_pct must be negative or zero")

        if self.require_human_approval_above < 0:
            errors.append("require_human_approval_above cannot be negative")

        if not 0.0 <= self.ethics_esg_threshold <= 100.0:
            errors.append("ethics_esg_threshold must be between 0 and 100")

        return errors
