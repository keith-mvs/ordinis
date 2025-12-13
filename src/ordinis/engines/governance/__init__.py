"""
Governance Engines for Ordinis Trading System.

Implements OECD AI Principles (2024) for responsible automated decision-making:
1. Inclusive growth, sustainable development, and human well-being
2. Respect for rule of law, human rights, and democratic values
3. Transparency and explainability
4. Robustness, security, and safety
5. Accountability (traceability and risk management)

Reference: https://oecd.ai/en/ai-principles

Components:
- AuditEngine: Immutable audit trails with hash chaining
- PPIEngine: Personal/Private Information detection and masking
- EthicsEngine: OECD-compliant ethical constraints and ESG scoring
- GovernanceEngine: Policy enforcement and compliance orchestration
- BrokerComplianceEngine: Broker terms of service compliance (Alpaca, IB, etc.)
"""

from .core.audit import AuditEngine, AuditEvent, AuditEventType
from .core.broker_compliance import (
    Broker,
    BrokerComplianceEngine,
    BrokerPolicy,
    ComplianceCategory,
    ComplianceCheckResult,
)
from .core.ethics import EthicsCheckResult, EthicsEngine, OECDPrinciple
from .core.governance import GovernanceEngine, Policy, PolicyDecision
from .core.ppi import MaskingMethod, PPICategory, PPIEngine

__all__ = [
    # Audit
    "AuditEngine",
    "AuditEvent",
    "AuditEventType",
    # PPI
    "PPIEngine",
    "PPICategory",
    "MaskingMethod",
    # Ethics
    "EthicsEngine",
    "OECDPrinciple",
    "EthicsCheckResult",
    # Governance
    "GovernanceEngine",
    "Policy",
    "PolicyDecision",
    # Broker Compliance
    "BrokerComplianceEngine",
    "Broker",
    "BrokerPolicy",
    "ComplianceCategory",
    "ComplianceCheckResult",
]
