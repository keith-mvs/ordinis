# Data and Privacy Risk

**Section**: 03_risk/data_privacy
**Last Updated**: 2025-12-12
**Version**: 1.0

---

## Definition and Scope

**Data risk** encompasses the potential for loss, harm, or liability arising from the collection, storage, processing, transmission, and disposal of data.

**Privacy risk** specifically addresses risks related to personal data protection and individual privacy rights.

---

## 1. Data Risk Categories

### 1.1 Data Quality Risk

**Definition**: Risk that data is inaccurate, incomplete, or unreliable.

| Dimension | Definition | Impact |
|-----------|------------|--------|
| **Accuracy** | Data correctly reflects reality | Wrong decisions |
| **Completeness** | All required data present | Analysis gaps |
| **Consistency** | Data agrees across sources | Reconciliation issues |
| **Timeliness** | Data available when needed | Stale information |
| **Validity** | Data conforms to rules | Processing errors |

### 1.2 Data Availability Risk

**Definition**: Risk that data is not accessible when needed.

| Scenario | Cause | Impact |
|----------|-------|--------|
| Storage failure | Hardware, corruption | Data loss |
| Access issues | Permissions, network | Operational delay |
| Vendor outage | Third-party failure | Data unavailable |
| Disaster | Physical event | Extended unavailability |

### 1.3 Data Integrity Risk

**Definition**: Risk that data is modified without authorization.

| Threat | Method | Detection |
|--------|--------|-----------|
| Unauthorized modification | Direct access | Audit logs |
| Injection attacks | Application exploit | Input validation |
| Data corruption | System error | Checksums |
| Synchronization errors | Replication issues | Reconciliation |

### 1.4 Data Leakage Risk

**Definition**: Risk of unauthorized data disclosure.

| Channel | Examples | Controls |
|---------|----------|----------|
| Network | Unencrypted transmission | TLS, VPN |
| Storage | Exposed databases | Access control |
| Physical | Lost devices | Encryption |
| Human | Social engineering | Training |
| Third party | Vendor breach | Due diligence |

---

## 2. Privacy Risk Categories

### 2.1 Regulatory Privacy Risk

**Definition**: Risk of violating data protection regulations.

| Regulation | Jurisdiction | Key Requirements |
|------------|--------------|------------------|
| **GDPR** | EU/EEA | Consent, rights, breach notification |
| **CCPA/CPRA** | California | Consumer rights, opt-out |
| **GLBA** | US (Financial) | Safeguards, privacy notices |
| **HIPAA** | US (Health) | PHI protection |
| **LGPD** | Brazil | Data protection principles |

### 2.2 Personal Data Categories

| Category | Sensitivity | Examples |
|----------|-------------|----------|
| **Identifiers** | Standard | Name, address, email |
| **Financial** | High | Account numbers, holdings |
| **Biometric** | Special | Fingerprints, voice |
| **Behavioral** | Variable | Trading patterns |
| **Derived** | Variable | Risk scores, predictions |

### 2.3 Processing Risk

| Activity | Risk | Mitigation |
|----------|------|------------|
| Collection | Over-collection | Purpose limitation |
| Storage | Retention excess | Retention policies |
| Usage | Purpose creep | Use limitation |
| Sharing | Unauthorized disclosure | Access controls |
| Disposal | Inadequate destruction | Secure deletion |

---

## 3. Common Sources and Triggers

### 3.1 Internal Sources

| Source | Examples |
|--------|----------|
| **Data governance gaps** | No ownership, unclear policies |
| **Technical debt** | Legacy systems, manual processes |
| **Access proliferation** | Too many users, roles |
| **Training deficiency** | Staff unaware of requirements |
| **Process weakness** | Manual handling, workarounds |

### 3.2 External Triggers

| Trigger | Examples |
|---------|----------|
| **Regulatory change** | New laws, enforcement actions |
| **Technology change** | Cloud adoption, AI/ML |
| **Vendor issues** | Breaches, service changes |
| **Market events** | Mergers, restructuring |
| **Cyber threats** | Targeted attacks |

---

## 4. Early Warning Signals

### 4.1 Data Risk Indicators

| Indicator | Green | Amber | Red |
|-----------|-------|-------|-----|
| Data quality score | > 95% | 85-95% | < 85% |
| Reconciliation breaks | < 10/day | 10-50 | > 50 |
| Data access requests | Tracked | Partial | Not tracked |
| Unstructured data growth | < 10%/yr | 10-30% | > 30% |
| Shadow IT data stores | 0 known | Some known | Unknown |

### 4.2 Privacy Risk Indicators

| Indicator | Green | Amber | Red |
|-----------|-------|-------|-----|
| DSAR response time | < 30 days | 30-45 days | > 45 days |
| Consent records | Complete | Partial | Missing |
| Privacy incidents | 0 | 1-5/quarter | > 5/quarter |
| Training completion | > 95% | 80-95% | < 80% |
| Third-party assessments | Current | Some overdue | Many overdue |

---

## 5. Impact Analysis

### 5.1 Regulatory Penalties

| Regulation | Maximum Penalty | Recent Examples |
|------------|-----------------|-----------------|
| GDPR | 4% global revenue or €20M | Meta: €1.2B (2023) |
| CCPA | $7,500 per violation | Sephora: $1.2M (2022) |
| GLBA | $100K per violation | Various fines |
| SEC | $10K+ per day | Data retention cases |

### 5.2 Business Impact

| Impact Type | Description |
|-------------|-------------|
| **Customer trust** | Loss of confidence, churn |
| **Competitive harm** | IP/strategy exposure |
| **Operational disruption** | Investigation, remediation |
| **Legal costs** | Defense, settlements |
| **Insurance** | Premium increases, exclusions |

---

## 6. Mitigation Strategies

### 6.1 Data Governance Framework

```
Data Governance Structure
├── Data Governance Council
│   └── Cross-functional oversight
├── Data Stewards
│   └── Domain-level ownership
├── Data Custodians
│   └── Technical management
└── Data Users
    └── Compliance with policies
```

### 6.2 Privacy Controls

| Control | Description | GDPR Article |
|---------|-------------|--------------|
| **Lawful basis** | Legal ground for processing | Art. 6 |
| **Purpose limitation** | Defined, legitimate purposes | Art. 5(1)(b) |
| **Data minimization** | Only necessary data | Art. 5(1)(c) |
| **Accuracy** | Keep data current | Art. 5(1)(d) |
| **Storage limitation** | Defined retention | Art. 5(1)(e) |
| **Security** | Appropriate protection | Art. 32 |

### 6.3 Data Protection by Design

```python
class PrivacyByDesign:
    """
    Privacy engineering principles.
    """

    PRINCIPLES = {
        'proactive': 'Anticipate and prevent privacy issues',
        'default': 'Privacy as the default setting',
        'embedded': 'Privacy embedded in design',
        'positive_sum': 'Full functionality with privacy',
        'end_to_end': 'Lifecycle protection',
        'visibility': 'Transparent operations',
        'user_centric': 'Respect user privacy'
    }

    def assess_processing(self, activity: DataActivity) -> PrivacyAssessment:
        """
        Assess processing activity for privacy risks.
        """
        assessment = PrivacyAssessment()

        # Check lawful basis
        assessment.lawful_basis = self._verify_legal_basis(activity)

        # Check data minimization
        assessment.minimized = self._verify_minimization(activity)

        # Check retention
        assessment.retention_defined = self._verify_retention(activity)

        # Check security
        assessment.security_adequate = self._verify_security(activity)

        return assessment
```

---

## 7. Data Subject Rights

### 7.1 GDPR Rights

| Right | Description | Response Time |
|-------|-------------|---------------|
| **Access** | Copy of personal data | 30 days |
| **Rectification** | Correct inaccurate data | Without delay |
| **Erasure** | Delete data ("right to be forgotten") | Without delay |
| **Portability** | Receive data in portable format | 30 days |
| **Objection** | Object to processing | Case-by-case |
| **Restriction** | Limit processing | Case-by-case |

### 7.2 DSAR Process

```
1. Request Receipt
   └── Log request, verify identity

2. Scope Assessment
   └── Determine data in scope

3. Data Collection
   └── Gather data from all systems

4. Review
   └── Legal review, redaction

5. Response
   └── Provide data or explanation

6. Documentation
   └── Retain evidence of compliance
```

---

## 8. Third-Party Data Risk

### 8.1 Vendor Categories

| Category | Data Access | Risk Level |
|----------|-------------|------------|
| Data processors | Process on behalf | High |
| Sub-processors | Downstream processing | High |
| Data recipients | Receive data | Medium-High |
| Service providers | Support services | Medium |
| Analytics providers | Aggregated data | Low-Medium |

### 8.2 Vendor Due Diligence

| Area | Assessment Elements |
|------|---------------------|
| **Security** | Controls, certifications, audits |
| **Privacy** | Policies, DPA, breach procedures |
| **Compliance** | Regulatory adherence, attestations |
| **Resilience** | Business continuity, recovery |
| **Monitoring** | Ongoing performance, incidents |

---

## 9. Breach Response

### 9.1 Breach Classification

| Category | Definition | Notification |
|----------|------------|--------------|
| **High risk** | Likely to result in harm | Regulator + individuals |
| **Risk** | Possible harm | Regulator only |
| **Low risk** | Unlikely to cause harm | Documentation only |

### 9.2 Response Timeline (GDPR)

| Milestone | Timeframe |
|-----------|-----------|
| Internal detection | Immediate |
| Breach assessment | < 24 hours |
| Regulator notification | 72 hours |
| Individual notification | Without undue delay |
| Post-incident review | 30 days |

---

## 10. Residual Risk

Data and privacy risk remains due to:

- Regulatory evolution
- Technology complexity
- Human factors
- Third-party dependencies
- Unknown data stores

**Acceptance requires**:
- Robust governance framework
- Regular assessments
- Incident response capability
- Staff training
- Adequate insurance

---

## Cross-References

- [Security Risk](../security/security_risk.md)
- [Compliance Risk](../compliance/compliance_risk.md)
- [Risk Governance](../frameworks/risk_governance.md)

---

**Template**: Enterprise Risk Management v1.0
