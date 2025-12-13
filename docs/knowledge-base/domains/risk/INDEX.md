# Risk Management Knowledge Base - Index

**Section**: 03_risk
**Last Updated**: 2025-12-12
**Version**: 2.0

---

## Overview

This knowledge base provides comprehensive enterprise-grade risk management documentation for algorithmic trading systems and supporting operations. It serves architects, engineers, risk professionals, and decision-makers for design decisions, audits, incident response, and strategic planning.

---

## Structure

### Trading-Specific Risk (Original Content)

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Core trading risk: position sizing, stops, drawdowns, kill switches |
| [Advanced Risk Methods](advanced_risk_methods.md) | Quantitative methods: VaR, CVaR, correlation, stress testing |
| [Fixed Income Risk](fixed_income_risk.md) | Bond pricing, duration, convexity, credit risk |

### Frameworks (Cross-Cutting)

| Document | Description |
|----------|-------------|
| [Risk Taxonomy](frameworks/risk_taxonomy.md) | Classification system, risk hierarchy, ownership model |
| [Risk Assessment Methodology](frameworks/risk_assessment_methodology.md) | Probability/impact scales, scoring, control effectiveness |
| [Risk Scoring and Heat Maps](frameworks/risk_scoring.md) | Prioritization, visualization, aggregation methods |
| [Risk Governance](frameworks/risk_governance.md) | Three lines of defense, RACI, escalation, reporting |

### Strategic Risk

| Document | Description |
|----------|-------------|
| [Strategic Risk](strategic/strategic_risk.md) | Market position, business model, capital allocation, transformation |

### Operational Risk

| Document | Description |
|----------|-------------|
| [Operational Risk](operational/operational_risk.md) | Basel categories, RCSA, loss data collection, process controls |

### Technical Risk

| Document | Description |
|----------|-------------|
| [Technical Risk](technical/technical_risk.md) | System availability, performance, data integrity, kill switches |

### Security Risk

| Document | Description |
|----------|-------------|
| [Security Risk](security/security_risk.md) | Cyber threats, attack vectors, incident response, compliance |

### Compliance Risk

| Document | Description |
|----------|-------------|
| [Compliance Risk](compliance/compliance_risk.md) | Regulatory landscape, market manipulation, best execution |

### Data and Privacy Risk

| Document | Description |
|----------|-------------|
| [Data Privacy Risk](data_privacy/data_privacy_risk.md) | GDPR, data subject rights, breach response, third-party data |

### Financial and Vendor Risk

| Document | Description |
|----------|-------------|
| [Financial and Vendor Risk](financial/financial_vendor_risk.md) | Market, credit, liquidity, vendor management |

### Human and Organizational Risk

| Document | Description |
|----------|-------------|
| [Human and Organizational Risk](human_organizational/human_organizational_risk.md) | Key person, error, misconduct, culture, governance |

### Emerging and Systemic Risk

| Document | Description |
|----------|-------------|
| [Emerging Risk](emerging/emerging_risk.md) | Technology disruption, climate, geopolitical, systemic risk |

### Publications

| Document | Description |
|----------|-------------|
| [López de Prado - Advances](publications/lopez_de_prado_advances.md) | Book notes: ML finance, purged CV, deflated Sharpe |

---

## Quick Reference

### Risk Category Summary

| Category | Primary Focus | Key Documents |
|----------|---------------|---------------|
| **Trading** | Position limits, stops, drawdowns | README, advanced_risk_methods |
| **Strategic** | Business viability, market position | strategic_risk |
| **Operational** | Process, people, system failures | operational_risk |
| **Technical** | System availability, performance | technical_risk |
| **Security** | Cyber threats, data protection | security_risk |
| **Compliance** | Regulatory requirements | compliance_risk |
| **Financial** | Market, credit, liquidity | financial_vendor_risk |
| **Human** | Key person, error, culture | human_organizational_risk |
| **Emerging** | Future threats, systemic | emerging_risk |

### Control Types by Risk

| Risk Area | Preventive | Detective | Corrective |
|-----------|------------|-----------|------------|
| Trading | Position limits | Real-time monitoring | Kill switches |
| Operational | Dual control | Reconciliation | Incident response |
| Technical | Redundancy | Health monitoring | Failover |
| Security | Access control | Intrusion detection | Incident response |
| Compliance | Policies | Surveillance | Remediation |

---

## Document Standards

All risk documents follow a consistent structure:

1. **Definition and Scope** - Clear boundaries
2. **Risk Categories** - Subcategorization
3. **Sources and Triggers** - What causes the risk
4. **Early Warning Signals** - How to detect
5. **Impact Analysis** - Consequences
6. **Mitigation Strategies** - Controls and responses
7. **Real-World Examples** - Lessons learned
8. **Residual Risk** - What remains after controls

---

## Related Knowledge Base Sections

- [01_foundations](../01_foundations/) - Core concepts
- [02_signals](../02_signals/) - Signal generation
- [04_strategy](../04_strategy/) - Strategy formulation
- [05_execution](../05_execution/) - Trade execution

---

## Maintenance

| Activity | Frequency | Owner |
|----------|-----------|-------|
| Content review | Quarterly | Risk Management |
| Regulatory updates | As needed | Compliance |
| Incident lessons | Post-incident | Risk/Operations |
| Framework updates | Annual | Risk Committee |

---

**Template**: Enterprise Risk Management v2.0
