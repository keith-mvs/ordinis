# Risk Taxonomy and Classification

**Section**: 03_risk/frameworks
**Last Updated**: 2025-12-12
**Version**: 1.0

---

## Overview

A risk taxonomy provides a structured classification system for identifying, categorizing, and managing risks across an organization or system. This document establishes the foundational framework for all risk management activities.

---

## 1. Risk Taxonomy Structure

### 1.1 Primary Risk Categories

```
Risk Universe
├── Strategic Risk
│   ├── Market positioning
│   ├── Competitive dynamics
│   └── Business model viability
├── Operational Risk
│   ├── Process failures
│   ├── Execution errors
│   └── Business continuity
├── Technical Risk
│   ├── System availability
│   ├── Performance degradation
│   └── Integration failures
├── Security Risk
│   ├── Cyber threats
│   ├── Access control
│   └── Data protection
├── Compliance Risk
│   ├── Regulatory violations
│   ├── Legal exposure
│   └── Reporting failures
├── Financial Risk
│   ├── Market risk
│   ├── Credit risk
│   ├── Liquidity risk
│   └── Counterparty risk
├── Human Risk
│   ├── Key person dependency
│   ├── Error and negligence
│   └── Misconduct
└── Emerging Risk
    ├── Technology disruption
    ├── Climate and ESG
    └── Geopolitical shifts
```

### 1.2 Risk Taxonomy Attributes

Each risk entry should capture:

| Attribute | Description | Example |
|-----------|-------------|---------|
| **Risk ID** | Unique identifier | `OPS-001` |
| **Category** | Primary classification | Operational |
| **Subcategory** | Secondary classification | Process Failure |
| **Name** | Descriptive title | Order execution delay |
| **Description** | Detailed explanation | System latency causes delayed order fills |
| **Owner** | Accountable party | Trading Operations |
| **Probability** | Likelihood assessment | Medium (3/5) |
| **Impact** | Consequence severity | High (4/5) |
| **Controls** | Existing mitigations | Latency monitoring, failover |
| **Residual Risk** | Post-control risk level | Medium |
| **Status** | Current state | Active, Monitored |

---

## 2. Risk Classification Dimensions

### 2.1 By Source (Origin)

| Source Type | Description | Examples |
|-------------|-------------|----------|
| **Internal** | Originating within the organization | Process errors, system bugs, employee mistakes |
| **External** | Originating outside the organization | Market crashes, regulatory changes, vendor failures |
| **Interface** | Arising at boundaries | API integration issues, counterparty failures |

### 2.2 By Nature

| Nature | Description | Characteristics |
|--------|-------------|-----------------|
| **Inherent** | Risk before controls | Raw exposure level |
| **Residual** | Risk after controls | Remaining exposure |
| **Accepted** | Risk deliberately retained | Within tolerance, cost-benefit justified |
| **Transferred** | Risk shifted to third party | Insurance, hedging, outsourcing |

### 2.3 By Time Horizon

| Horizon | Timeframe | Focus |
|---------|-----------|-------|
| **Immediate** | < 1 day | Intraday trading risks |
| **Short-term** | 1-30 days | Execution, liquidity |
| **Medium-term** | 1-12 months | Strategy, portfolio |
| **Long-term** | > 1 year | Strategic, systemic |

### 2.4 By Controllability

| Level | Description | Management Approach |
|-------|-------------|---------------------|
| **Controllable** | Directly manageable | Implement controls |
| **Influenceable** | Partially manageable | Monitor, adapt |
| **Observable** | Can detect but not control | Early warning, contingency |
| **Unobservable** | Unknown until materialized | Scenario planning, reserves |

---

## 3. Risk Identification Methods

### 3.1 Systematic Approaches

| Method | Description | Best For |
|--------|-------------|----------|
| **Process Analysis** | Walk through workflows | Operational risks |
| **FMEA** | Failure Mode and Effects Analysis | Technical systems |
| **Threat Modeling** | Structured security analysis | Security risks |
| **Scenario Analysis** | Explore hypothetical events | Strategic, emerging risks |
| **Historical Analysis** | Review past incidents | Known failure modes |
| **Expert Elicitation** | Gather specialist input | Complex, novel risks |

### 3.2 Risk Identification Checklist

**Strategic:**
- [ ] Market position changes
- [ ] Competitive threats
- [ ] Technology disruption
- [ ] Business model sustainability

**Operational:**
- [ ] Process dependencies
- [ ] Single points of failure
- [ ] Manual intervention points
- [ ] Data quality issues

**Technical:**
- [ ] System availability requirements
- [ ] Performance bottlenecks
- [ ] Integration complexity
- [ ] Technical debt

**Security:**
- [ ] Attack surface
- [ ] Access controls
- [ ] Data sensitivity
- [ ] Third-party exposure

**Compliance:**
- [ ] Regulatory requirements
- [ ] Reporting obligations
- [ ] Audit findings
- [ ] Legal exposure

**Financial:**
- [ ] Market exposure
- [ ] Counterparty concentration
- [ ] Liquidity requirements
- [ ] Funding dependencies

---

## 4. Risk Hierarchy and Relationships

### 4.1 Risk Aggregation Levels

```
Enterprise Risk
    └── Domain Risk (e.g., Trading Operations)
        └── Category Risk (e.g., Execution Risk)
            └── Specific Risk (e.g., Order rejection from exchange)
```

### 4.2 Risk Interdependencies

| Relationship | Description | Example |
|--------------|-------------|---------|
| **Causal** | One risk causes another | System outage → trading losses |
| **Correlated** | Risks co-occur | Market stress → liquidity + credit risk |
| **Compounding** | Combined effect exceeds sum | Multiple simultaneous failures |
| **Offsetting** | Risks partially cancel | Long/short positions |

### 4.3 Dependency Matrix

| Risk A | Risk B | Relationship | Strength |
|--------|--------|--------------|----------|
| System outage | Trading losses | Causal | High |
| Market crash | Counterparty default | Correlated | Medium |
| Data breach | Regulatory fine | Consequential | High |

---

## 5. Risk Ownership Model

### 5.1 Three Lines of Defense

| Line | Role | Responsibilities |
|------|------|------------------|
| **First Line** | Business Operations | Own and manage day-to-day risks |
| **Second Line** | Risk Management | Oversight, frameworks, monitoring |
| **Third Line** | Internal Audit | Independent assurance |

### 5.2 RACI for Risk Management

| Activity | Responsible | Accountable | Consulted | Informed |
|----------|-------------|-------------|-----------|----------|
| Risk Identification | Business Owner | Risk Manager | SMEs | Executive |
| Risk Assessment | Risk Analyst | Risk Manager | Business Owner | Executive |
| Control Implementation | Business Owner | Business Owner | Risk Manager | Audit |
| Monitoring | Operations | Risk Manager | Business Owner | Executive |
| Reporting | Risk Analyst | Risk Manager | Business Owner | Board |

---

## 6. Risk Taxonomy Governance

### 6.1 Maintenance Responsibilities

- **Annual Review**: Full taxonomy review and update
- **Quarterly Refresh**: New risk identification cycle
- **Event-Driven Update**: Post-incident taxonomy additions
- **Continuous Monitoring**: Emerging risk surveillance

### 6.2 Change Management

| Change Type | Approval Required | Documentation |
|-------------|-------------------|---------------|
| Add new risk | Risk Manager | Risk register entry |
| Modify category | Risk Committee | Taxonomy update |
| Remove risk | Risk Manager + Business Owner | Retirement justification |
| Restructure taxonomy | Risk Committee + Executive | Full impact assessment |

---

## 7. Integration with Risk Processes

### 7.1 Linkages

| Process | Taxonomy Usage |
|---------|----------------|
| Risk Assessment | Categorization framework |
| Control Design | Risk-to-control mapping |
| Incident Management | Root cause classification |
| Reporting | Aggregation structure |
| Audit Planning | Risk-based scoping |

### 7.2 Technology Integration

```python
@dataclass
class RiskEntry:
    """Standard risk taxonomy entry."""
    risk_id: str
    category: str
    subcategory: str
    name: str
    description: str
    owner: str
    probability: int  # 1-5
    impact: int       # 1-5
    controls: List[str]
    residual_score: int
    status: str
    last_reviewed: date

    @property
    def inherent_score(self) -> int:
        return self.probability * self.impact

    @property
    def risk_level(self) -> str:
        score = self.residual_score
        if score >= 16:
            return "Critical"
        elif score >= 9:
            return "High"
        elif score >= 4:
            return "Medium"
        return "Low"
```

---

## Cross-References

- [Risk Assessment Methodology](risk_assessment_methodology.md)
- [Risk Scoring and Heat Maps](risk_scoring.md)
- [Risk Governance Framework](risk_governance.md)

---

**Template**: Enterprise Risk Management v1.0
