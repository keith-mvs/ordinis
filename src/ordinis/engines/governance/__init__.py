"""
Governance Engines for Ordinis Trading System.

Implements OECD AI Principles (2024) for responsible automated decision-making:
1. Inclusive growth, sustainable development, and human well-being
2. Respect for rule of law, human rights, and democratic values
3. Transparency and explainability
4. Robustness, security, and safety
5. Accountability (traceability and risk management)

Reference: https://oecd.ai/en/ai-principles

The engine follows the standard Ordinis engine template with:
- core/ - Engine, config, and domain models
- UnifiedGovernanceEngine - Standardized BaseEngine wrapper

Components:
- UnifiedGovernanceEngine: Standardized BaseEngine wrapper with lifecycle
- GovernanceEngine: Policy enforcement and compliance orchestration
- AuditEngine: Immutable audit trails with hash chaining
- PPIEngine: Personal/Private Information detection and masking
- EthicsEngine: OECD-compliant ethical constraints and ESG scoring
- BrokerComplianceEngine: Broker terms of service compliance (Alpaca, IB, etc.)
"""

# Core engine components
from ordinis.engines.governance.core import (
    ApprovalRequest,
    ApprovalStatus,
    AuditEngine,
    AuditEvent,
    AuditEventType,
    EthicsCheckResult,
    EthicsEngine,
    GovernanceEngine,
    GovernanceEngineConfig,
    MaskingMethod,
    OECDPrinciple,
    Policy,
    PolicyAction,
    PolicyDecision,
    PolicyType,
    PPICategory,
    PPIEngine,
    UnifiedGovernanceEngine,
)
from ordinis.engines.governance.core.broker_compliance import (
    Broker,
    BrokerComplianceEngine,
    BrokerPolicy,
    ComplianceCategory,
    ComplianceCheckResult,
)

__all__ = [
    # Governance
    "ApprovalRequest",
    "ApprovalStatus",
    # Audit
    "AuditEngine",
    "AuditEvent",
    "AuditEventType",
    # Broker Compliance
    "Broker",
    "BrokerComplianceEngine",
    "BrokerPolicy",
    "ComplianceCategory",
    "ComplianceCheckResult",
    # Ethics
    "EthicsCheckResult",
    "EthicsEngine",
    "GovernanceEngine",
    # Unified Engine (standardized interface)
    "GovernanceEngineConfig",
    # PPI
    "MaskingMethod",
    "OECDPrinciple",
    "PPICategory",
    "PPIEngine",
    "Policy",
    "PolicyAction",
    "PolicyDecision",
    "PolicyType",
    "UnifiedGovernanceEngine",
]
